#!/usr/bin/env python3
"""LaBo-style CLIP concept bottleneck for SiFC and fMoW leave-one-country-out runs.

This adaptation uses the included Qwen candidate pools and greedy concept
coverage selection. It does not reproduce LaBo's GPT-3/T5 pool or modified
Apricot implementation. Held-out labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from zero_shot_baselines import CLASSES, CSV_PATHS, ROOT, read_sites

DATASETS = {
    "sifc": {
        "splits": ROOT / "metadata/labo_sifc_loco_splits.csv",
        "pool": ROOT / "descriptors/labo_sifc_candidate_pool.json",
        "n": 800, "countries": ("China", "India", "USA"),
    },
    "fmow": {
        "splits": ROOT / "metadata/labo_fmow_loco_splits.csv",
        "pool": ROOT / "descriptors/labo_fmow_candidate_pool.json",
        "n": 400, "countries": ("France", "Russia", "USA"),
    },
}
MODEL_ID = "openai/clip-vit-large-patch14"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--stage", choices=("audit", "smoke", "embed", "fold", "merge"), required=True)
    p.add_argument("--images", type=Path, help="directory of coordinate/year PNGs")
    p.add_argument("--year", type=int, default=2026, help="year in image filenames")
    p.add_argument("--csv", type=Path, help="override the released site CSV")
    p.add_argument("--candidate-pool", type=Path, help="override the included concept pool")
    p.add_argument("--heldout-country")
    p.add_argument("--splits", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/labo")
    p.add_argument("--image-shard", default="0/1", help="rank/count, for --stage embed")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-download", action="store_true", help="allow CLIP checkpoint download")
    p.add_argument("--dry-run", action="store_true", help="validate without loading CLIP")
    p.add_argument("--disable-cudnn", action="store_true",
                   help="work around cuDNN initialization failure")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--concepts-per-class", type=int, default=50)
    p.add_argument("--mi-weight", type=float, default=1.0)
    p.add_argument("--coverage-weight", type=float, default=1.0)
    p.add_argument("--learning-rates", type=float, nargs="+", default=(1e-3, 1e-2))
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = DATASETS[a.dataset]
    a.csv = a.csv or CSV_PATHS[a.dataset]
    a.splits = a.splits or cfg["splits"]
    a.candidate_pool = a.candidate_pool or cfg["pool"]
    a.stem = f"labo_{a.dataset}_loco_clip_vitl14_adaptation"
    a.cache_dir = a.cache_dir or a.output_dir / "cache" / a.dataset
    a.result_dir = a.output_dir
    if a.stage != "smoke" and a.images is None:
        p.error("--images is required except for --stage smoke")
    if a.year < 1:
        p.error("--year must be positive")
    try:
        rank, count = (int(x) for x in a.image_shard.split("/"))
    except ValueError:
        p.error("--image-shard must be rank/count")
    if count < 1 or not 0 <= rank < count:
        p.error("invalid --image-shard")
    a.rank, a.shards = rank, count
    if count > cfg["n"]:
        p.error("--image-shard count cannot exceed the number of sites")
    if a.stage == "fold" and a.heldout_country not in cfg["countries"]:
        p.error(f"--heldout-country must be one of {cfg['countries']}")
    if a.heldout_country and a.stage != "fold":
        p.error("--heldout-country is only used for --stage fold")
    if a.stage != "embed" and a.image_shard != "0/1":
        p.error("--image-shard is only used for --stage embed")
    if min(a.batch_size, a.train_batch_size, a.concepts_per_class, a.max_epochs, a.patience) < 1:
        p.error("positive batch size, concept count, epochs, and patience required")
    if not a.learning_rates or any(not math.isfinite(x) or x <= 0 for x in a.learning_rates):
        p.error("learning rates must be positive and finite")
    if (not math.isfinite(a.mi_weight) or not math.isfinite(a.coverage_weight)
            or a.mi_weight < 0 or a.coverage_weight < 0
            or a.mi_weight + a.coverage_weight == 0):
        p.error("selection weights must be finite, nonnegative, and not both zero")
    return a


def audit(a):
    sites = read_sites(a.csv, a.images, a.year, a.dataset)
    if len(sites) != DATASETS[a.dataset]["n"]:
        raise ValueError(f"{a.dataset}: expected {DATASETS[a.dataset]['n']} sites, got {len(sites)}")
    label_to_class = {item[3]: item[1] for item in CLASSES[a.dataset]}
    classes = sorted(label_to_class.values())
    rows = [{"site_key": site.filename.removesuffix(f"_{a.year}.png"),
             "image_file": site.filename, "cls": label_to_class[site.label],
             "country": site.country, "image_path": site.path} for site in sites]
    ids = [r["site_key"] for r in rows]
    if len(set(ids)) != len(rows):
        raise ValueError("duplicate site key in release CSV")
    if {r["country"] for r in rows} != set(DATASETS[a.dataset]["countries"]):
        raise ValueError("release CSV country inventory disagrees with dataset")
    if {r["cls"] for r in rows} != set(classes):
        raise ValueError("release CSV class inventory disagrees with dataset")
    splits = read_csv(a.splits)
    expected_fields = ["heldout_country", "site_key", "source_index", "country", "true_class", "role"]
    if not splits or list(splits[0]) != expected_fields:
        raise ValueError(f"split CSV requires columns {expected_fields}")
    key = {(s["heldout_country"], s["site_key"]): s for s in splits}
    if len(key) != len(rows) * 3 or len(splits) != len(key):
        raise ValueError("split file must have exactly one row per site and held-out country")
    by_id = {r["site_key"]: r for r in rows}
    order = {}
    for country in DATASETS[a.dataset]["countries"]:
        fold = [key.get((country, site)) for site in ids]
        if any(s is None for s in fold):
            raise ValueError(f"split missing sites for heldout {country}")
        for s in fold:
            r = by_id[s["site_key"]]
            if s["country"] != r["country"] or s["true_class"] != r["cls"]:
                raise ValueError("split metadata disagrees with release CSV")
            if s["heldout_country"] != country or s["role"] not in {"train", "validation", "test"}:
                raise ValueError("invalid split country or role")
            if (r["country"] == country) != (s["role"] == "test"):
                raise ValueError("held-out country leaked into train/val or source leaked into test")
            try:
                index = int(s["source_index"])
            except ValueError as error:
                raise ValueError("invalid source row index") from error
            if not 0 <= index < len(rows):
                raise ValueError("source row index outside dataset")
            previous = order.setdefault(s["site_key"], index)
            if previous != index:
                raise ValueError("source row index differs across folds")
        for role in ("train", "validation", "test"):
            present = {by_id[s["site_key"]]["cls"] for s in fold if s["role"] == role}
            if present != set(classes):
                raise ValueError(f"{country}/{role} lacks classes: {set(classes)-present}")
    if set(order.values()) != set(range(len(rows))):
        raise ValueError("source row order must cover every site once")
    rows.sort(key=lambda r: order[r["site_key"]])
    pool = json.loads(a.candidate_pool.read_text(encoding="utf-8"))
    if pool.get("dataset") != a.dataset or set(pool.get("classes", {})) != set(classes):
        raise ValueError("candidate pool dataset/class inventory disagrees with manifest")
    if not pool.get("generator"):
        raise ValueError("candidate pool requires generator provenance")
    for cls in classes:
        texts = pool["classes"][cls]
        if (not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts)
                or len({t.casefold().strip() for t in texts}) < a.concepts_per_class):
            raise ValueError(f"insufficient unique candidates for {cls}")
    missing_files = sum(not r["image_path"].is_file() for r in rows)
    if a.stage == "embed" and not a.dry_run and missing_files:
        raise FileNotFoundError(f"{missing_files} image files absent from --images")
    print(f"AUDIT OK {a.dataset}: {len(rows)} sites, {len(classes)} classes, "
          f"{len(splits)} split rows, {missing_files} images absent; "
          f"csv={sha256(a.csv)[:12]}", flush=True)
    return rows, classes, key, pool


def concepts_from_pool(pool, classes):
    texts, owners = [], []
    seen = set()
    for cls in classes:
        for raw in pool["classes"][cls]:
            t = " ".join(raw.strip().split())
            if not t or t.casefold() in seen:
                continue
            seen.add(t.casefold())
            texts.append(t)
            owners.append(cls)
    return texts, owners


def device_name(value):
    import torch
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def embed(a, rows, classes, pool):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if a.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[embed] cuDNN disabled for host compatibility", flush=True)
    dev = device_name(a.device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=not a.allow_download)
    model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=not a.allow_download).to(dev).eval()
    if model.config.vision_config.image_size != 224:
        raise ValueError("CLIP checkpoint does not use 224-pixel image inputs")
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    texts, owners = concepts_from_pool(pool, classes)
    tokenized = processor.tokenizer(texts, padding=False, truncation=False)
    if any(len(x) > model.config.text_config.max_position_embeddings for x in tokenized["input_ids"]):
        raise ValueError("a candidate phrase exceeds CLIP's context length")
    with torch.inference_mode():
        feats = []
        for start in range(0, len(texts), 64):
            t = processor.tokenizer(texts[start:start+64], padding=True, return_tensors="pt")
            f = model.get_text_features(**{k: v.to(dev) for k, v in t.items()})
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        text_features = np.concatenate(feats)
    text_path = a.cache_dir / "text.npz"
    if a.rank == 0:
        temporary = a.cache_dir / f"text.{os.getpid()}.npz"
        np.savez_compressed(temporary, texts=np.asarray(texts), owners=np.asarray(owners),
                            features=text_features, pool_sha256=sha256(a.candidate_pool), model_id=MODEL_ID)
        os.replace(temporary, text_path)
    selected = [(i, r) for i, r in enumerate(rows) if i % a.shards == a.rank]
    ids, vectors = [], []
    with torch.inference_mode():
        for start in range(0, len(selected), a.batch_size):
            chunk = selected[start:start+a.batch_size]
            images = []
            for _, row in chunk:
                with Image.open(row["image_path"]) as im:
                    images.append(im.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            if tuple(inputs["pixel_values"].shape[-2:]) != (224, 224):
                raise ValueError("CLIP processor did not output 224x224")
            f = model.get_image_features(pixel_values=inputs["pixel_values"].to(dev))
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
            vectors.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
            ids.extend(r["site_key"] for _, r in chunk)
            if (start // a.batch_size) % 10 == 0:
                print(f"[embed {a.rank}/{a.shards}] {len(ids)}/{len(selected)}", flush=True)
    out = a.cache_dir / f"image_shard_{a.rank}_of_{a.shards}.npz"
    np.savez_compressed(out, ids=np.asarray(ids), features=np.concatenate(vectors),
                        csv_sha256=sha256(a.csv), model_id=MODEL_ID)
    print(f"saved {out} ({len(ids)} images) and {text_path}", flush=True)


def load_features(a, rows, pool):
    text_path = a.cache_dir / "text.npz"
    if not text_path.is_file():
        raise FileNotFoundError(text_path)
    with np.load(text_path, allow_pickle=False) as d:
        if str(d["pool_sha256"]) != sha256(a.candidate_pool) or str(d["model_id"]) != MODEL_ID:
            raise ValueError("text embedding cache provenance mismatch")
        texts, owners, text_features = d["texts"].tolist(), d["owners"].tolist(), d["features"].copy()
    vectors = {}
    for path in sorted(a.cache_dir.glob("image_shard_*_of_*.npz")):
        with np.load(path, allow_pickle=False) as d:
            if str(d["csv_sha256"]) != sha256(a.csv) or str(d["model_id"]) != MODEL_ID:
                raise ValueError(f"image cache provenance mismatch: {path}")
            for site, feat in zip(d["ids"].tolist(), d["features"]):
                if site in vectors:
                    raise ValueError(f"duplicate site in cache: {site}")
                vectors[site] = feat
    ids = [r["site_key"] for r in rows]
    if set(vectors) != set(ids):
        raise ValueError(f"incomplete image cache: {len(vectors)}/{len(ids)}")
    image_features = np.stack([vectors[x] for x in ids]).astype(np.float32)
    if not np.isfinite(image_features).all() or not np.isfinite(text_features).all():
        raise ValueError("nonfinite cached CLIP features")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image/text feature dimensions differ")
    return image_features, text_features.astype(np.float32), texts, owners


def select_concepts(image_features, labels, text_features, owners, classes, k, mi_weight, coverage_weight):
    """Source-only per-class discriminability + facility-location adaptation.

    The released LaBo MI divides raw CLIP similarities and can be undefined
    when any are nonpositive. For this satellite adaptation, convert the
    class-mean similarities into a softmax distribution before computing
    divergence from a uniform class distribution. Modified Apricot is
    replaced by an explicit greedy cosine facility-location objective.
    """
    cls_mean = np.stack([image_features[labels == i].mean(axis=0) for i in range(len(classes))], axis=1)
    sim = text_features @ cls_mean
    centered = (sim - sim.max(axis=1, keepdims=True)) / 0.07
    norm = np.exp(centered)
    norm /= norm.sum(axis=1, keepdims=True)
    mi = (norm * np.log(norm * len(classes))).sum(axis=1)
    chosen = []
    for cls in classes:
        idx = np.flatnonzero(np.asarray(owners) == cls)
        if len(idx) < k:
            raise ValueError(f"{cls}: only {len(idx)} candidates, need {k}")
        coverage = np.maximum(0.0, text_features[idx] @ text_features[idx].T)
        best = np.zeros(len(idx), dtype=np.float32)
        local = []
        for _ in range(k):
            gain = coverage_weight * (np.maximum(coverage, best[:, None]) - best[:, None]).sum(axis=0) / len(idx)
            gain += mi_weight * mi[idx]
            gain[local] = -np.inf
            winner = int(np.argmax(gain))
            local.append(winner)
            best = np.maximum(best, coverage[:, winner])
        chosen.extend(idx[local].tolist())
    return np.asarray(chosen, dtype=int)


def fit_head(scores, labels, train_idx, val_idx, owners, classes, lr, epochs, patience, seed,
             batch_size=64):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score

    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    x = torch.tensor(scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    initial = torch.zeros((len(classes), len(owners)), dtype=torch.float32)
    for j, owner in enumerate(owners):
        initial[classes.index(owner), j] = 1.0
    weight = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([weight], lr=lr)
    best_acc, best_epoch, best_ce, best_weight, stale = -1.0, 0, math.inf, None, 0
    train = torch.tensor(train_idx, dtype=torch.long)
    val = torch.tensor(val_idx, dtype=torch.long)
    for epoch in range(1, epochs + 1):
        order = train[torch.randperm(len(train))]
        for start in range(0, len(order), batch_size):
            batch = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = 100.0 * (x[batch] @ F.softmax(weight, dim=-1).T)
            loss = F.cross_entropy(logits, y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite LaBo association loss")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            v_logits = 100.0 * (x[val] @ F.softmax(weight, dim=-1).T)
            acc = accuracy_score(y[val].numpy(), v_logits.argmax(dim=1).numpy())
            ce = float(F.cross_entropy(v_logits, y[val]))
        if acc > best_acc + 1e-12 or (abs(acc-best_acc) <= 1e-12 and ce < best_ce-1e-12):
            best_acc, best_epoch, best_ce, best_weight, stale = acc, epoch, ce, weight.detach().clone(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_weight.numpy(), {"validation_accuracy": best_acc, "validation_ce": best_ce,
                                 "best_epoch": best_epoch, "last_epoch": epoch}


def predict_scores(scores, weight):
    from scipy.special import softmax
    w = softmax(weight, axis=-1)
    logits = 100.0 * (scores @ w.T)
    return softmax(logits, axis=-1)


def fold(a, rows, classes, split, pool):
    from sklearn.metrics import accuracy_score, f1_score

    image_features, text_features, texts, owners = load_features(a, rows, pool)
    class_index = {name: i for i, name in enumerate(classes)}
    labels = np.asarray([class_index[r["cls"]] for r in rows], dtype=int)
    roles = [split[(a.heldout_country, r["site_key"])]["role"] for r in rows]
    train = np.flatnonzero(np.asarray(roles) == "train")
    val = np.flatnonzero(np.asarray(roles) == "validation")
    test = np.flatnonzero(np.asarray(roles) == "test")
    if any(rows[i]["country"] == a.heldout_country for i in np.r_[train, val]):
        raise AssertionError("held-out leakage")
    selected = select_concepts(image_features[train], labels[train], text_features, owners,
                               classes, a.concepts_per_class, a.mi_weight, a.coverage_weight)
    selected_owners = [owners[i] for i in selected]
    selected_texts = [texts[i] for i in selected]
    scores = image_features @ text_features[selected].T
    trials = []
    for lr in a.learning_rates:
        weight, info = fit_head(scores, labels, train, val, selected_owners, classes,
                                lr, a.max_epochs, a.patience, a.seed, a.train_batch_size)
        trials.append((info["validation_accuracy"], -info["validation_ce"], lr, weight, info))
    _, _, lr, weight, info = max(trials, key=lambda item: (item[0], item[1], -item[2]))
    probs = predict_scores(scores[test], weight)
    pred = probs.argmax(axis=1)
    metrics = {
        "model": "LaBo-style CLIP ViT-L/14 (LOCO)",
        "dataset": a.dataset, "heldout_country": a.heldout_country,
        "n_test": len(test), "macro_f1": float(f1_score(labels[test], pred, labels=np.arange(len(classes)), average="macro")),
        "accuracy": float(accuracy_score(labels[test], pred)), "selected_lr": lr,
        "selected_epochs": info["best_epoch"], "validation_accuracy": info["validation_accuracy"],
    }
    predicted = []
    for pos, idx in enumerate(test):
        entry = {"image_file": rows[idx]["image_file"], "country": rows[idx]["country"],
                 "true_class": rows[idx]["cls"], "pred_class": classes[pred[pos]]}
        entry.update({f"p_{cls}": float(probs[pos, j]) for j, cls in enumerate(classes)})
        predicted.append(entry)
    suffix = a.heldout_country.lower()
    write_csv(a.output_dir / f"{a.stem}__{suffix}__predictions.csv", list(predicted[0]), predicted)
    write_csv(a.result_dir / f"{a.stem}__{suffix}__macro_f1.csv", list(metrics), [metrics])
    info_path = a.output_dir / f"{a.stem}__{suffix}__selected_concepts.json"
    info_path.write_text(json.dumps({"heldout_country": a.heldout_country, "concepts":
        [{"class": o, "text": t} for o, t in zip(selected_owners, selected_texts)],
        "candidate_pool_sha256": sha256(a.candidate_pool),
        "csv_sha256": sha256(a.csv), "splits_sha256": sha256(a.splits),
        "model_id": MODEL_ID, "mi_weight": a.mi_weight,
        "coverage_weight": a.coverage_weight,
        "concepts_per_class": a.concepts_per_class,
        "training": info,
        "selection": "source-train-only softmax discriminability + cosine facility-location greedy adaptation"}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def merge(a, rows, classes):
    from sklearn.metrics import accuracy_score, f1_score

    predictions = []
    for country in DATASETS[a.dataset]["countries"]:
        path = a.output_dir / f"{a.stem}__{country.lower()}__predictions.csv"
        predictions.extend(read_csv(path))
    ids = [p["image_file"] for p in predictions]
    if len(ids) != len(rows) or set(ids) != {r["image_file"] for r in rows} or len(set(ids)) != len(ids):
        raise ValueError("fold predictions do not cover dataset exactly once")
    by_id = {r["image_file"]: r for r in rows}
    if any(by_id[p["image_file"]]["cls"] != p["true_class"] or
           by_id[p["image_file"]]["country"] != p["country"] for p in predictions):
        raise ValueError("prediction metadata mismatch")
    records = []
    for country in (*DATASETS[a.dataset]["countries"], "ALL"):
        sub = [p for p in predictions if country == "ALL" or p["country"] == country]
        records.append({"model": "LaBo-style CLIP ViT-L/14 (LOCO)", "dataset": a.dataset,
            "country": country, "n": len(sub), "macro_f1": float(f1_score(
                [p["true_class"] for p in sub], [p["pred_class"] for p in sub],
                labels=classes, average="macro")),
            "accuracy": float(accuracy_score([p["true_class"] for p in sub],
                                              [p["pred_class"] for p in sub]))})
    write_csv(a.output_dir / f"{a.stem}__predictions.csv", list(predictions[0]), predictions)
    out = a.result_dir / f"{a.stem}__macro_f1.csv"
    write_csv(out, list(records[0]), records)
    print(f"saved {out}", flush=True)
    for r in records:
        print(f"{r['country']:>7} n={r['n']:>3} F1={r['macro_f1']:.6f} acc={r['accuracy']:.6f}", flush=True)


def smoke():
    import torch  # noqa: F401

    rng = np.random.default_rng(11)
    classes = ["a", "b", "c"]
    owners = [c for c in classes for _ in range(5)]
    text = rng.normal(size=(15, 8)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    labels = np.repeat(np.arange(3), 20)
    image = np.stack([text[y*5] * 0.8 + rng.normal(size=8) * 0.05 for y in labels]).astype(np.float32)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    selected = select_concepts(image, labels, text, owners, classes, 2, 1, 1)
    assert len(selected) == 6 and Counter(owners[i] for i in selected) == {c: 2 for c in classes}
    scores = image @ text[selected].T
    weight, info = fit_head(scores, labels, np.r_[0:15, 20:35, 40:55],
                            np.r_[15:20, 35:40, 55:60], [owners[i] for i in selected],
                            classes, 0.01, 12, 5, 11)
    probs = predict_scores(scores, weight)
    assert probs.shape == (60, 3) and np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    print(f"SMOKE OK: 6 selected concepts, finite association head, probability rows sum to 1; {info}", flush=True)


def main():
    a = parse_args()
    if a.stage == "smoke":
        smoke()
        return
    rows, classes, splits, pool = audit(a)
    if a.stage == "audit" or a.dry_run:
        return
    if a.stage == "embed":
        embed(a, rows, classes, pool)
    elif a.stage == "fold":
        fold(a, rows, classes, splits, pool)
    elif a.stage == "merge":
        merge(a, rows, classes)


if __name__ == "__main__":
    main()

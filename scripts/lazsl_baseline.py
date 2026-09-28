#!/usr/bin/env python3
"""LaZSL adaptation for the released SiFC and fMoW facility datasets.

A frozen CLIP ViT-L/14 scores class names, global attributes, and LaZSL's
random-crop optimal-transport alignment. The included numeric seed map preserves
the source crop draws without publishing source image IDs. Class labels are not
used by CLIP or to choose a prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from zero_shot_baselines import CLASSES, CSV_PATHS, ROOT, Site, read_sites

DESCRIPTOR_FILES = {
    "sifc": ROOT / "descriptors" / "lazsl_sifc800_class_attributes_v1.json",
    "fmow": ROOT / "descriptors" / "lazsl_fmow400_class_attributes_v1.json",
}
SEED_MAP = ROOT / "metadata" / "lazsl_crop_seeds.csv"
SOURCE_SEED = 1
COUNTRIES = {"sifc": ("India", "USA", "China"),
             "fmow": ("USA", "France", "Russia")}
COUNTRY_COUNTS = {"sifc": {"India": 210, "USA": 295, "China": 295},
                  "fmow": {"USA": 161, "France": 149, "Russia": 90}}
CLASS_COUNTS = {
    "sifc": {"iron_still_plant": 180, "lng_terminals": 49, "nuclear_plants": 74,
             "oil_refineries": 159, "power_plants": 180, "stp_plants": 158},
    "fmow": {"shipyard": 22, "solar_farm": 46, "storage_tank": 195,
             "water_treatment_facility": 96, "wind_farm": 41},
}
METHODS = ("clip_class_name", "global_attribute_mean", "lazsl")
MODEL_ID = "openai/clip-vit-large-patch14"
UPSTREAM_REPOSITORY = "https://github.com/shiming-chen/LaZSL"
UPSTREAM_COMMIT = "77c613cfd4a66ee23a4e43aebf1b28fedc24f0af"
PROMPT_PATTERN = "{class_name}, which has {attribute}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=tuple(CLASSES))
    parser.add_argument("--images", required=True, type=Path,
                        help="directory of coordinate/year PNGs from download_arcgis_images.py")
    parser.add_argument("--csv", type=Path, help="override the release site CSV")
    parser.add_argument("--descriptors", type=Path, help="override dataset attribute JSON")
    parser.add_argument("--seed-map", type=Path, default=SEED_MAP)
    parser.add_argument("--year", type=int, default=2026, help="year in PNG filenames")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--n-crops", type=int, default=70)
    parser.add_argument("--min-crop-scale", type=float, default=0.6)
    parser.add_argument("--max-crop-scale", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--sinkhorn-tolerance", type=float, default=1e-2)
    parser.add_argument("--view-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--top-explanations", type=int, default=8)
    parser.add_argument("--limit", type=int, help="score the first N sites as a smoke test")
    parser.add_argument("--shard", help="independent inference shard as INDEX/COUNT")
    parser.add_argument("--merge-only", action="store_true", help="merge completed full shards")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lazsl")
    parser.add_argument("--dry-run", action="store_true", help="validate without loading CLIP")
    parser.add_argument("--allow-download", action="store_true", help="allow checkpoint download")
    parser.add_argument("--enable-cudnn", action="store_true",
                        help="enable cuDNN; the source runner leaves it disabled")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    pieces = value.split("/")
    if len(pieces) != 2:
        raise ValueError("--shard must be INDEX/COUNT")
    try:
        index, count = (int(part) for part in pieces)
    except ValueError as error:
        raise ValueError("--shard must be INDEX/COUNT") from error
    if count < 1 or not 0 <= index < count:
        raise ValueError("--shard requires 0 <= INDEX < COUNT")
    return index, count


def validate_args(args: argparse.Namespace) -> tuple[int, int] | None:
    shard = parse_shard(args.shard)
    if args.limit is not None and not 1 <= args.limit <= sum(COUNTRY_COUNTS[args.dataset].values()):
        raise ValueError("--limit is outside the dataset size")
    if min(args.n_crops, args.view_batch_size, args.text_batch_size,
           args.sinkhorn_iterations, args.shard_count) < 1:
        raise ValueError("crop, batch, iteration, and shard counts must be positive")
    if not (math.isfinite(args.min_crop_scale) and math.isfinite(args.max_crop_scale)
            and 0 < args.min_crop_scale <= args.max_crop_scale <= 1):
        raise ValueError("crop scales must satisfy 0 < minimum <= maximum <= 1")
    if not math.isfinite(args.theta) or not 0 <= args.theta <= 1:
        raise ValueError("--theta must be in [0, 1]")
    if not math.isfinite(args.sinkhorn_epsilon) or args.sinkhorn_epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be positive")
    if not math.isfinite(args.sinkhorn_tolerance) or args.sinkhorn_tolerance <= 0:
        raise ValueError("Sinkhorn tolerance must be positive")
    if not 1 <= args.top_explanations <= args.n_crops * 10:
        raise ValueError("--top-explanations must be between 1 and n_crops * 10")
    if args.merge_only and (shard is not None or args.limit is not None):
        raise ValueError("--merge-only requires complete shards and no --limit")
    if Path(args.model_id).is_absolute():
        raise ValueError("--model-id must be a checkpoint identifier, not a local path")
    return shard


def class_ids(dataset: str) -> list[str]:
    return [item[1] for item in CLASSES[dataset]]


def load_descriptors(dataset: str, path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = class_ids(dataset)
    if data.get("prompt_pattern") != PROMPT_PATTERN or tuple(data.get("classes", {})) != tuple(classes):
        raise ValueError("LaZSL attribute pattern or class order does not match the dataset")
    names: dict[str, str] = {}
    attributes: dict[str, list[str]] = {}
    for cls in classes:
        item = data["classes"][cls]
        name = item.get("class_name")
        values = item.get("attributes")
        if not isinstance(name, str) or not name.strip() or not isinstance(values, list):
            raise ValueError(f"Invalid LaZSL attributes for {cls}")
        if len(values) != 10 or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"Expected ten nonempty attributes for {cls}")
        cleaned = [value.strip() for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"Duplicate LaZSL attribute for {cls}")
        names[cls], attributes[cls] = name.strip(), cleaned
    return names, attributes


def site_key(site: Site) -> str:
    return Path(site.filename).stem.rsplit("_", 1)[0]


def load_crop_seeds(path: Path, dataset: str, sites: list[Site]) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["dataset", "site_key", "seed_uint64"]:
            raise ValueError("LaZSL seed-map columns changed")
        rows = list(reader)
    result = {}
    for row in rows:
        if row["dataset"] not in COUNTRIES or not row["site_key"] or not row["seed_uint64"].isdigit():
            raise ValueError("Invalid LaZSL seed-map row")
        if row["dataset"] == dataset:
            if row["site_key"] in result:
                raise ValueError("Duplicate LaZSL site key")
            value = int(row["seed_uint64"])
            if not 0 <= value < 2**64:
                raise ValueError("LaZSL crop seed is outside uint64")
            result[row["site_key"]] = value
    expected = {site_key(site) for site in sites}
    if set(result) != expected or len(set(result.values())) != len(result):
        raise ValueError("LaZSL seed map does not match release sites one to one")
    return result


def validate_sites(dataset: str, sites: list[Site]) -> None:
    labels = {item[3]: item[1] for item in CLASSES[dataset]}
    country_counts = Counter(site.country for site in sites)
    class_counts = Counter(labels[site.label] for site in sites)
    if dict(country_counts) != COUNTRY_COUNTS[dataset] or dict(class_counts) != CLASS_COUNTS[dataset]:
        raise ValueError("Release site country/class counts changed")


def stable_rng(crop_seed: int):
    import numpy as np
    return np.random.default_rng(crop_seed)


def crop_boxes(image_size: tuple[int, int], args: argparse.Namespace, crop_seed: int):
    width, height = image_size
    shortest = min(width, height)
    rng = stable_rng(crop_seed)
    boxes = []
    for index in range(args.n_crops):
        scale = float(rng.uniform(args.min_crop_scale, args.max_crop_scale))
        side = max(1, min(shortest, int(scale * shortest)))
        left = int(rng.integers(0, width - side + 1))
        top = int(rng.integers(0, height - side + 1))
        boxes.append((index, left, top, left + side, top + side, scale))
    return boxes


def output_tag(args: argparse.Namespace, shard: tuple[int, int] | None) -> str:
    model_tag = ("vitl14" if args.model_id == MODEL_ID else
                 "custom_" + hashlib.sha256(args.model_id.encode()).hexdigest()[:10])
    tag = f"{args.dataset}_lazsl_{model_tag}_{args.year}"
    if args.limit is not None:
        tag += f"_smoke{args.limit}"
    if shard is not None:
        tag += f"_shard{shard[0]}of{shard[1]}"
    return tag


def output_paths(args: argparse.Namespace, shard: tuple[int, int] | None) -> dict[str, Path]:
    tag = output_tag(args, shard)
    return {key: args.output_dir / f"{tag}_{key}.{extension}" for key, extension in
            (("predictions", "csv"), ("explanations", "csv"), ("crops", "csv"),
             ("summary", "csv"), ("class_metrics", "csv"), ("run", "json"))}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def class_statistics(rows: list[dict], method: str, country: str, classes: list[str]):
    selected = [row for row in rows if row["method"] == method
                and (country == "ALL" or row["country"] == country)]
    if not selected:
        return None
    details = []
    for cls in classes:
        tp = sum(row["true_class"] == cls and row["pred_class"] == cls for row in selected)
        fp = sum(row["true_class"] != cls and row["pred_class"] == cls for row in selected)
        fn = sum(row["true_class"] == cls and row["pred_class"] != cls for row in selected)
        support = sum(row["true_class"] == cls for row in selected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details.append((cls, support, precision, recall, f1))
    accuracy = sum(row["true_class"] == row["pred_class"] for row in selected) / len(selected)
    return sum(item[4] for item in details) / len(classes), accuracy, details, len(selected)


def metric_rows(dataset: str, predictions: list[dict], model_id: str):
    classes = class_ids(dataset)
    summary, per_class = [], []
    for method in METHODS:
        values = {country: class_statistics(predictions, method, country, classes)
                  for country in (*COUNTRIES[dataset], "ALL")}
        for country, stats in values.items():
            if stats is not None:
                for cls, support, precision, recall, f1 in stats[2]:
                    per_class.append({"method": method, "country": country,
                                      "class_name": cls, "support": support,
                                      "precision": f"{precision:.6f}", "recall": f"{recall:.6f}",
                                      "f1": f"{f1:.6f}"})
        for metric, index in (("macro_f1", 0), ("accuracy", 1)):
            row = {"dataset": dataset, "model": model_id, "method": method,
                   "metric": metric, "n_images": values["ALL"][3]}
            row.update({country: f"{values[country][index]:.6f}" if values[country] else ""
                        for country in values})
            summary.append(row)
    return summary, per_class


def unwrap_features(value):
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if hasattr(value, "image_embeds"):
        return value.image_embeds
    if hasattr(value, "text_embeds"):
        return value.text_embeds
    return value[0] if isinstance(value, tuple) else value


def sinkhorn_score(torch, global_feature, region_features, text_features, args):
    global_similarity = global_feature @ text_features.T
    region_similarity = region_features @ text_features.T
    hybrid = args.theta * region_similarity + (1.0 - args.theta) * global_similarity.unsqueeze(0)
    region_global = region_features @ global_feature
    keep = region_global >= region_global.mean()
    if not bool(keep.any()):
        keep[region_global.argmax()] = True
    row_mass = torch.zeros(region_features.shape[0], device=region_features.device,
                           dtype=torch.float32)
    row_mass[keep] = 1.0 / int(keep.sum().item())
    column_mass = torch.full((text_features.shape[0],), 1.0 / text_features.shape[0],
                              device=region_features.device, dtype=torch.float32)
    kernel = torch.exp(-(1.0 - hybrid.float()) / args.sinkhorn_epsilon)
    left, right = torch.ones_like(row_mass), torch.ones_like(column_mass)
    tiny = torch.finfo(torch.float32).tiny
    iterations = 0
    for iterations in range(1, args.sinkhorn_iterations + 1):
        previous = left
        left = row_mass / (kernel @ right).clamp_min(tiny)
        right = column_mass / (kernel.T @ left).clamp_min(tiny)
        if float((left - previous).abs().mean()) < args.sinkhorn_tolerance:
            break
    plan = left.unsqueeze(1) * right.unsqueeze(0) * kernel
    if not bool(torch.isfinite(plan).all()):
        raise FloatingPointError("non-finite Sinkhorn transport plan")
    score = (plan * hybrid.float()).sum()
    return score, plan, hybrid.float(), region_global.float(), keep, iterations


def encode_texts(torch, model, processor, texts: list[str], device, batch_size: int):
    chunks = []
    for start in range(0, len(texts), batch_size):
        inputs = processor(text=texts[start:start + batch_size], padding=True,
                           truncation=True, return_tensors="pt")
        with torch.inference_mode():
            features = unwrap_features(model.get_text_features(
                **{key: value.to(device) for key, value in inputs.items()})).float()
            features = features / features.norm(dim=-1, keepdim=True)
        chunks.append(features)
    return torch.cat(chunks, dim=0)


def encode_images(torch, model, processor, images: list, device):
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        features = unwrap_features(model.get_image_features(pixel_values=pixels)).float()
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def score_image(torch, model, processor, image, boxes, class_features,
                attribute_features, classes: list[str], args, device):
    global_feature = encode_images(torch, model, processor, [image], device)[0]
    chunks = []
    for start in range(0, len(boxes), args.view_batch_size):
        batch = boxes[start:start + args.view_batch_size]
        crops = [image.crop((left, top, right, bottom))
                 for _, left, top, right, bottom, _ in batch]
        chunks.append(encode_images(torch, model, processor, crops, device))
        for crop in crops:
            crop.close()
    region_features = torch.cat(chunks, dim=0)
    clip_scores = global_feature @ class_features.T
    global_scores, lazsl_scores, diagnostics = [], [], {}
    for cls in classes:
        text_features = attribute_features[cls]
        global_scores.append((global_feature @ text_features.T).mean())
        score, plan, hybrid, region_global, keep, iterations = sinkhorn_score(
            torch, global_feature, region_features, text_features, args)
        lazsl_scores.append(score)
        diagnostics[cls] = {"plan": plan, "hybrid": hybrid, "region_global": region_global,
                            "keep": keep, "iterations": iterations}
    return clip_scores, torch.stack(global_scores), torch.stack(lazsl_scores), diagnostics


def protocol_metadata(args, csv_path: Path, descriptor_path: Path) -> dict:
    return {
        "dataset": args.dataset, "year": args.year, "model_id": args.model_id,
        "site_csv_sha256": sha256_file(csv_path),
        "descriptors_sha256": sha256_file(descriptor_path),
        "seed_map_sha256": sha256_file(args.seed_map),
        "seed": SOURCE_SEED,
        "crop_seed_derivation": "first 8 SHA256 bytes of UTF-8 1|source_site_id, big endian",
        "n_crops": args.n_crops,
        "min_crop_scale": args.min_crop_scale, "max_crop_scale": args.max_crop_scale,
        "theta": args.theta, "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "sinkhorn_tolerance": args.sinkhorn_tolerance,
        "top_explanations": args.top_explanations,
        "image_input_pixels": 224, "true_label_used_for_class_scoring": False,
        "crop_seed_provenance": "hashed source site IDs; source IDs may encode class/country",
        "region_filter": "crop-to-global similarity >= per-image mean",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
    }


def output_fields(dataset: str) -> dict[str, list[str]]:
    classes = class_ids(dataset)
    return {
        "predictions": ["image_file", "country", "true_class", "method", "pred_class",
                        "correct", *(f"score_{cls}" for cls in classes),
                        *(f"prob_{cls}" for cls in classes), "n_crops", "n_selected_crops",
                        "seconds"],
        "explanations": ["image_file", "country", "true_class", "pred_class", "rank",
                         "class_id", "attribute_index", "attribute", "prompt", "crop_index",
                         "left", "top", "right", "bottom", "crop_scale", "selected",
                         "region_global_similarity", "transport_mass", "hybrid_similarity",
                         "contribution"],
        "crops": ["image_file", "country", "true_class", "crop_index", "left", "top",
                  "right", "bottom", "crop_scale", "selected_for_predicted_class",
                  "region_global_similarity_for_predicted_class"],
        "summary": ["dataset", "model", "method", "metric", *COUNTRIES[dataset], "ALL",
                    "n_images"],
        "class_metrics": ["method", "country", "class_name", "support", "precision",
                          "recall", "f1"],
    }


def score_sites(args, sites: list[Site], crop_seeds: dict[str, int], names: dict[str, str],
                attributes: dict[str, list[str]]):
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("LaZSL requires numpy, torch, transformers, and Pillow") from error
    random.seed(SOURCE_SEED)
    np.random.seed(SOURCE_SEED)
    torch.manual_seed(SOURCE_SEED)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        device = torch.device(args.device)
    torch.backends.cudnn.enabled = bool(args.enable_cudnn)
    processor = CLIPProcessor.from_pretrained(args.model_id,
                                               local_files_only=not args.allow_download,
                                               use_fast=False)
    model = CLIPModel.from_pretrained(args.model_id,
                                      local_files_only=not args.allow_download).to(device).eval()
    model.requires_grad_(False)
    image_size = int(model.config.vision_config.image_size)
    if image_size != 224:
        raise ValueError(f"Expected 224-pixel CLIP input, found {image_size}")
    classes = class_ids(args.dataset)
    class_features = encode_texts(torch, model, processor, [names[cls] for cls in classes],
                                   device, args.text_batch_size)
    attribute_prompts = {cls: [PROMPT_PATTERN.format(class_name=names[cls], attribute=value)
                               for value in attributes[cls]] for cls in classes}
    attribute_features = {cls: encode_texts(torch, model, processor,
                                            attribute_prompts[cls], device, args.text_batch_size)
                          for cls in classes}
    label_to_class = {item[3]: item[1] for item in CLASSES[args.dataset]}
    predictions, explanations, crops_output = [], [], []
    for number, site in enumerate(sites, start=1):
        started = time.monotonic()
        with Image.open(site.path) as source:
            image = source.convert("RGB")
        boxes = crop_boxes(image.size, args, crop_seeds[site_key(site)])
        clip, global_scores, lazsl, diagnostics = score_image(
            torch, model, processor, image, boxes, class_features, attribute_features,
            classes, args, device)
        image.close()
        elapsed = time.monotonic() - started
        true_class = label_to_class[site.label]
        for method, scores in zip(METHODS, (clip, global_scores, lazsl)):
            predicted = classes[int(scores.argmax().item())]
            probabilities = torch.softmax(scores.float(), dim=0).detach().cpu().tolist()
            selected_count = int(diagnostics[predicted]["keep"].sum().item()) if method == "lazsl" else ""
            row = {"image_file": site.filename, "country": site.country,
                   "true_class": true_class, "method": method, "pred_class": predicted,
                   "correct": int(predicted == true_class), "n_crops": args.n_crops,
                   "n_selected_crops": selected_count, "seconds": f"{elapsed:.6f}"}
            row.update({f"score_{cls}": f"{float(scores[i].item()):.9f}"
                        for i, cls in enumerate(classes)})
            row.update({f"prob_{cls}": f"{probabilities[i]:.9f}"
                        for i, cls in enumerate(classes)})
            predictions.append(row)
        predicted = classes[int(lazsl.argmax().item())]
        diagnostic = diagnostics[predicted]
        contributions = diagnostic["plan"] * diagnostic["hybrid"]
        order = torch.argsort(contributions.flatten(), descending=True)
        count = len(attributes[predicted])
        for rank, flat_index in enumerate(order[:args.top_explanations].tolist(), start=1):
            crop_index, attribute_index = divmod(flat_index, count)
            _, left, top, right, bottom, scale = boxes[crop_index]
            explanations.append({
                "image_file": site.filename, "country": site.country,
                "true_class": true_class, "pred_class": predicted, "rank": rank,
                "class_id": predicted, "attribute_index": attribute_index,
                "attribute": attributes[predicted][attribute_index],
                "prompt": attribute_prompts[predicted][attribute_index],
                "crop_index": crop_index, "left": left, "top": top, "right": right,
                "bottom": bottom, "crop_scale": f"{scale:.9f}",
                "selected": int(diagnostic["keep"][crop_index].item()),
                "region_global_similarity": f"{float(diagnostic['region_global'][crop_index].item()):.9f}",
                "transport_mass": f"{float(diagnostic['plan'][crop_index, attribute_index].item()):.9f}",
                "hybrid_similarity": f"{float(diagnostic['hybrid'][crop_index, attribute_index].item()):.9f}",
                "contribution": f"{float(contributions[crop_index, attribute_index].item()):.9f}",
            })
        for crop_index, left, top, right, bottom, scale in boxes:
            crops_output.append({
                "image_file": site.filename, "country": site.country,
                "true_class": true_class, "crop_index": crop_index,
                "left": left, "top": top, "right": right, "bottom": bottom,
                "crop_scale": f"{scale:.9f}",
                "selected_for_predicted_class": int(diagnostic["keep"][crop_index].item()),
                "region_global_similarity_for_predicted_class":
                    f"{float(diagnostic['region_global'][crop_index].item()):.9f}",
            })
        print(f"[{number}/{len(sites)}] {site.filename} pred={predicted} "
              f"kept={int(diagnostic['keep'].sum().item())}/{args.n_crops} "
              f"time={elapsed:.2f}s", flush=True)
    return predictions, explanations, crops_output, str(device)


def save_metrics(args, predictions: list[dict], paths: dict[str, Path]) -> None:
    summary, per_class = metric_rows(args.dataset, predictions, args.model_id)
    fields = output_fields(args.dataset)
    write_csv(paths["summary"], fields["summary"], summary)
    write_csv(paths["class_metrics"], fields["class_metrics"], per_class)
    for row in summary:
        if row["metric"] == "macro_f1":
            print(f"{row['method']}: ALL macro-F1={row['ALL']}", flush=True)


def infer(args, shard, selected: list[Site], crop_seeds, names, attributes,
          metadata: dict) -> None:
    paths = output_paths(args, shard)
    relevant = {key: path for key, path in paths.items()
                if shard is None or key in {"predictions", "explanations", "crops", "run"}}
    existing = [path.name for path in relevant.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}; use --overwrite")
    predictions, explanations, crops, device = score_sites(args, selected, crop_seeds, names, attributes)
    fields = output_fields(args.dataset)
    for key, rows in (("predictions", predictions), ("explanations", explanations),
                      ("crops", crops)):
        write_csv(paths[key], fields[key], rows)
    write_json(paths["run"], {**metadata, "device": device, "n_selected_images": len(selected),
                              "shard": args.shard})
    if shard is None:
        save_metrics(args, predictions, paths)


def merge_shards(args, sites: list[Site], metadata: dict) -> None:
    all_predictions, all_explanations, all_crops = [], [], []
    expected = {site.filename for site in sites}
    for index in range(args.shard_count):
        shard = (index, args.shard_count)
        paths = output_paths(args, shard)
        run = json.loads(paths["run"].read_text(encoding="utf-8"))
        if any(run.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Shard {index} protocol differs from the requested merge")
        rows = read_csv(paths["predictions"])
        shard_expected = {site.filename for position, site in enumerate(sites)
                          if position % args.shard_count == index}
        for method in METHODS:
            names = [row["image_file"] for row in rows if row["method"] == method]
            if len(names) != len(shard_expected) or set(names) != shard_expected:
                raise ValueError(f"Shard {index} has incomplete or duplicate {method} predictions")
        explanations = read_csv(paths["explanations"])
        crops = read_csv(paths["crops"])
        if len(explanations) != len(shard_expected) * args.top_explanations:
            raise ValueError(f"Shard {index} has incomplete explanations")
        if len(crops) != len(shard_expected) * args.n_crops:
            raise ValueError(f"Shard {index} has incomplete crops")
        all_predictions.extend(rows)
        all_explanations.extend(explanations)
        all_crops.extend(crops)
    if {row["image_file"] for row in all_predictions} != expected:
        raise ValueError("Merged prediction coverage differs from the release CSV")
    order = {site.filename: index for index, site in enumerate(sites)}
    all_predictions.sort(key=lambda row: (order[row["image_file"]], METHODS.index(row["method"])))
    all_explanations.sort(key=lambda row: (order[row["image_file"]], int(row["rank"])))
    all_crops.sort(key=lambda row: (order[row["image_file"]], int(row["crop_index"])))
    paths = output_paths(args, None)
    existing = [path.name for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Final outputs already exist: {existing}; use --overwrite")
    fields = output_fields(args.dataset)
    for key, rows in (("predictions", all_predictions), ("explanations", all_explanations),
                      ("crops", all_crops)):
        write_csv(paths[key], fields[key], rows)
    write_json(paths["run"], {**metadata, "n_selected_images": len(sites),
                              "merged_shards": args.shard_count})
    save_metrics(args, all_predictions, paths)


def main() -> None:
    args = parse_args()
    shard = validate_args(args)
    csv_path = args.csv or CSV_PATHS[args.dataset]
    descriptor_path = args.descriptors or DESCRIPTOR_FILES[args.dataset]
    sites = read_sites(csv_path, args.images, args.year, args.dataset)
    validate_sites(args.dataset, sites)
    names, attributes = load_descriptors(args.dataset, descriptor_path)
    crop_seeds = load_crop_seeds(args.seed_map, args.dataset, sites)
    selected = sites[:args.limit]
    if shard is not None:
        selected = [site for position, site in enumerate(selected)
                    if position % shard[1] == shard[0]]
    if not selected and not args.merge_only:
        raise ValueError("Selected shard contains no images")
    missing = [] if args.merge_only else [site.filename for site in selected if not site.path.is_file()]
    metadata = protocol_metadata(args, csv_path, descriptor_path)
    print(f"dataset={args.dataset} sites={len(sites)} selected={len(selected)} "
          f"attributes/class={len(attributes[class_ids(args.dataset)[0]])} "
          f"missing_images={len(missing)}")
    print(f"model={args.model_id} outputs={output_tag(args, shard)}")
    if missing:
        print(f"first missing image={missing[0]}")
    if args.dry_run:
        return
    if args.merge_only:
        merge_shards(args, sites, metadata)
        return
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected images are missing")
    infer(args, shard, selected, crop_seeds, names, attributes, metadata)


if __name__ == "__main__":
    main()

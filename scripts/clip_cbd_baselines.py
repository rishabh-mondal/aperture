#!/usr/bin/env python3
"""Class-name CLIP and description-based CbD on SiFC or fMoW.

Both methods share frozen CLIP ViT-L/14 image embeddings. CbD here follows the
Menon--Vondrick description-scoring method using the included independently
drafted descriptions; these are not the original paper's GPT-3 descriptions.
Labels are used only for the final evaluation, never for scoring or prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from zero_shot_baselines import CLASSES, CSV_PATHS, ROOT, Site, read_sites

DESCRIPTOR_FILES = {
    "sifc": ROOT / "descriptors" / "menon_vondrick_facility6_class_descriptions_v1.json",
    "fmow": ROOT / "descriptors" / "menon_vondrick_fmow5_class_descriptions_v1.json",
}
COUNTRIES = {"sifc": ("India", "USA", "China"),
             "fmow": ("USA", "France", "Russia")}
EXPECTED_SITES = {"sifc": 800, "fmow": 400}
METHODS = ("class_name_clip", "description_clip")
MODEL_ID = "openai/clip-vit-large-patch14"
PROMPT_PATTERN = "{class_name} which has {descriptor}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=tuple(CLASSES))
    parser.add_argument("--images", required=True, type=Path,
                        help="directory of coordinate/year PNGs from download_arcgis_images.py")
    parser.add_argument("--csv", type=Path, help="override the release site CSV")
    parser.add_argument("--descriptors", type=Path, help="override this dataset's description JSON")
    parser.add_argument("--year", type=int, default=2026, help="year in downloaded PNG filenames")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="score the first N sites as a smoke test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "clip_cbd")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without loading CLIP")
    parser.add_argument("--allow-download", action="store_true",
                        help="allow checkpoint download if it is not cached")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_descriptions(dataset: str, path: Path) -> tuple[list[str], list[str], list[str], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [item[1] for item in CLASSES[dataset]]
    if set(data.get("classes", {})) != set(keys):
        raise ValueError("Description classes do not match the selected dataset")
    if data.get("description_prompt_pattern") != PROMPT_PATTERN:
        raise ValueError("Unexpected description prompt pattern")

    names: list[str] = []
    prompts: list[str] = []
    owners: list[str] = []
    ordinals: list[int] = []
    for key in keys:
        entry = data["classes"][key]
        name = entry.get("class_name")
        descriptions = entry.get("descriptors")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing class name for {key}")
        if not isinstance(descriptions, list) or len(descriptions) < 2:
            raise ValueError(f"At least two descriptions are required for {key}")
        if any(not isinstance(value, str) or not value.strip() for value in descriptions):
            raise ValueError(f"Blank description for {key}")
        cleaned = [value.strip() for value in descriptions]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"Duplicate description for {key}")
        name = name.strip()
        names.append(name)
        for ordinal, description in enumerate(cleaned, start=1):
            prompts.append(PROMPT_PATTERN.format(class_name=name, descriptor=description))
            owners.append(key)
            ordinals.append(ordinal)
    if len(names) != len(set(names)):
        raise ValueError("Class names must be distinct")
    return names, prompts, owners, ordinals


def output_paths(dataset: str, output_dir: Path, limit: int | None) -> dict[str, Path]:
    stem = f"{dataset}_clip_cbd_vitl14" + (f"_smoke{limit}" if limit is not None else "")
    return {key: output_dir / f"{stem}_{key}.{extension}" for key, extension in
            (("predictions", "csv"), ("descriptor_scores", "csv"),
             ("summary", "csv"), ("run", "json"))}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluate_group(rows: list[dict], method: str, country: str,
                   classes: list[str]) -> tuple[float, float] | None:
    group = [row for row in rows if row["method"] == method
             and (country == "ALL" or row["country"] == country)]
    if not group:
        return None
    f1 = []
    for cls in classes:
        tp = sum(row["true_class"] == cls and row["pred_class"] == cls for row in group)
        fp = sum(row["true_class"] != cls and row["pred_class"] == cls for row in group)
        fn = sum(row["true_class"] == cls and row["pred_class"] != cls for row in group)
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    accuracy = sum(row["pred_class"] == row["true_class"] for row in group) / len(group)
    return sum(f1) / len(classes), accuracy


def summary_rows(dataset: str, predictions: list[dict], model_id: str) -> list[dict]:
    classes = [item[1] for item in CLASSES[dataset]]
    result = []
    for method in METHODS:
        scores = {country: evaluate_group(predictions, method, country, classes)
                  for country in (*COUNTRIES[dataset], "ALL")}
        for metric, index in (("macro_f1", 0), ("accuracy", 1)):
            row = {"dataset": dataset, "model": model_id, "method": method,
                   "metric": metric, "n_images": len(predictions) // len(METHODS)}
            row.update({country: f"{scores[country][index]:.6f}" if scores[country] else ""
                        for country in scores})
            result.append(row)
    return result


def score_images(args: argparse.Namespace, sites: list[Site], names: list[str],
                 prompts: list[str], owners: list[str], ordinals: list[int]
                 ) -> tuple[list[dict], list[dict], str]:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    processor = CLIPProcessor.from_pretrained(args.model_id,
                                               local_files_only=not args.allow_download,
                                               use_fast=False)
    model = CLIPModel.from_pretrained(args.model_id,
                                      local_files_only=not args.allow_download).to(device).eval()
    max_tokens = model.config.text_config.max_position_embeddings

    def encode_text(strings: list[str]):
        vectors = []
        for start in range(0, len(strings), args.text_batch_size):
            chunk = strings[start:start + args.text_batch_size]
            tokens = processor.tokenizer(chunk, padding=True, truncation=False,
                                         return_tensors="pt")
            if tokens["input_ids"].shape[1] > max_tokens:
                raise ValueError(f"CLIP text exceeds {max_tokens} tokens: {chunk}")
            with torch.inference_mode():
                features = model.get_text_features(**{k: v.to(device) for k, v in tokens.items()})
            vectors.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(vectors)

    classes = [item[1] for item in CLASSES[args.dataset]]
    label_to_class = {item[3]: item[1] for item in CLASSES[args.dataset]}
    class_vectors = encode_text(names)
    description_vectors = encode_text(prompts)
    class_indices = {cls: [i for i, owner in enumerate(owners) if owner == cls]
                     for cls in classes}
    predictions: list[dict] = []
    descriptor_scores: list[dict] = []
    started = time.monotonic()
    for start in range(0, len(sites), args.batch_size):
        batch = sites[start:start + args.batch_size]
        images = []
        for site in batch:
            with Image.open(site.path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixels)
        image_vectors = F.normalize(features.float(), dim=-1).cpu()
        plain = (image_vectors @ class_vectors.T).numpy()
        described = (image_vectors @ description_vectors.T).numpy()
        for j, site in enumerate(batch):
            true_class = label_to_class[site.label]
            means = [float(described[j, class_indices[cls]].mean()) for cls in classes]
            for method, scores in ((METHODS[0], [float(value) for value in plain[j]]),
                                   (METHODS[1], means)):
                best = max(range(len(classes)), key=lambda i: scores[i])
                row = {"image_file": site.filename, "country": site.country,
                       "true_class": true_class, "method": method,
                       "pred_class": classes[best], "correct": int(classes[best] == true_class)}
                row.update({f"score_{cls}": f"{scores[i]:.9f}"
                            for i, cls in enumerate(classes)})
                predictions.append(row)
            for i, prompt in enumerate(prompts):
                descriptor_scores.append({"image_file": site.filename, "class": owners[i],
                                          "description_index": ordinals[i], "prompt": prompt,
                                          "cosine_similarity": f"{float(described[j, i]):.9f}"})
        print(f"processed {min(start + len(batch), len(sites))}/{len(sites)}; "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    return predictions, descriptor_scores, device


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.text_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    csv_path = args.csv or CSV_PATHS[args.dataset]
    descriptor_path = args.descriptors or DESCRIPTOR_FILES[args.dataset]
    sites = read_sites(csv_path, args.images, args.year, args.dataset)
    if len(sites) != EXPECTED_SITES[args.dataset]:
        raise ValueError(f"Expected {EXPECTED_SITES[args.dataset]} {args.dataset} sites, found {len(sites)}")
    if any(site.country not in COUNTRIES[args.dataset] for site in sites):
        raise ValueError("Unexpected country in site CSV")
    names, prompts, owners, ordinals = load_descriptions(args.dataset, descriptor_path)
    selected = sites[:args.limit]
    missing = [site.filename for site in selected if not site.path.is_file()]
    paths = output_paths(args.dataset, args.output_dir, args.limit)
    print(f"dataset={args.dataset} sites={len(sites)} selected={len(selected)} "
          f"descriptions={len(prompts)} missing_images={len(missing)}")
    print(f"model={args.model_id} outputs={paths['predictions'].name}, {paths['summary'].name}")
    if missing:
        print(f"first missing image={missing[0]}")
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected images are missing")
    existing = [path.name for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}; use --overwrite to replace them")

    predictions, descriptor_scores, device = score_images(args, selected, names, prompts,
                                                            owners, ordinals)
    summary = summary_rows(args.dataset, predictions, args.model_id)
    classes = [item[1] for item in CLASSES[args.dataset]]
    write_csv(paths["predictions"],
              ["image_file", "country", "true_class", "method", "pred_class", "correct",
               *(f"score_{cls}" for cls in classes)], predictions)
    write_csv(paths["descriptor_scores"],
              ["image_file", "class", "description_index", "prompt", "cosine_similarity"],
              descriptor_scores)
    write_csv(paths["summary"],
              ["dataset", "model", "method", "metric", *COUNTRIES[args.dataset],
               "ALL", "n_images"], summary)
    metadata = {
        "dataset": args.dataset, "model_id": args.model_id, "device": device,
        "n_images": len(selected), "n_classes": len(classes), "n_descriptions": len(prompts),
        "site_csv_sha256": sha256_file(csv_path),
        "descriptors_sha256": sha256_file(descriptor_path),
        "class_name_prompt": "{class_name}", "description_prompt": PROMPT_PATTERN,
        "class_score": "mean of separate normalized image-text cosine similarities",
        "methods": list(METHODS), "labels_used_for_prediction": False,
        "description_provenance": "independently drafted; method adaptation, not original GPT-3 descriptions",
        "torch_version": __import__("torch").__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    paths["run"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()

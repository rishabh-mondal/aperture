"""Score the 90 original SiFC images and their three cumulative removal steps.

The manifest stores image paths relative to --data-root. Zero-shot and selective
inference reuse the release runners' prompts, image preparation, and models.
Labels are used only for saved predictions and the separate evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import selective_multiscale as selective
import zero_shot_baselines as zero_shot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "SiFC data" / "intervention_90_manifest.csv"
DEFAULT_OUTPUT = ROOT / "results" / "intervention_90"
FIELDS = ("site_id", "original_site_id", "cls", "country", "step", "step_name", "image_path")
COUNTRIES = ("India", "USA", "China")
MODELS = ("gemma", "qwen", "glm")


@dataclass(frozen=True)
class InterventionRecord:
    site_id: str
    original_site_id: str
    cls: str
    country: str
    step: int
    step_name: str
    image_path: str


def read_manifest(path: Path) -> list[InterventionRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Manifest columns must be {FIELDS}")
        raw = list(reader)
    classes = {item[1] for item in zero_shot.CLASSES["sifc"]}
    records = []
    for row in raw:
        site_id = row["site_id"]
        original_id = row["original_site_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", site_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", original_id):
            raise ValueError(f"Invalid site ID: {site_id}")
        try:
            step = int(row["step"])
        except ValueError as error:
            raise ValueError(f"Invalid step for {site_id}") from error
        relative = PurePosixPath(row["image_path"])
        if (not row["image_path"] or relative.is_absolute() or ".." in relative.parts
                or "\\" in row["image_path"] or step not in (0, 1, 2, 3)
                or row["cls"] not in classes or row["country"] not in COUNTRIES):
            raise ValueError(f"Invalid manifest row: {site_id}")
        expected_folder = "cbm_data" if step == 0 else "cbm_data_inpainted_bbox_additive"
        if relative.parts[0] != expected_folder or relative.suffix.lower() != ".png":
            raise ValueError(f"Image path does not match step {step}: {site_id}")
        if step == 0 and (site_id != original_id or row["step_name"] != "original"):
            raise ValueError(f"Invalid original row: {site_id}")
        if step > 0 and (site_id != f"{original_id}__step{step}" or not row["step_name"]):
            raise ValueError(f"Invalid cumulative step row: {site_id}")
        records.append(InterventionRecord(site_id, original_id, row["cls"],
                                          row["country"], step, row["step_name"],
                                          relative.as_posix()))
    if len(records) != 360 or len({row.site_id for row in records}) != 360:
        raise ValueError("Expected 360 unique image records")
    if Counter(row.step for row in records) != dict.fromkeys(range(4), 90):
        raise ValueError("Expected 90 originals and 90 images at each removal step")
    original = {row.site_id: row for row in records if row.step == 0}
    if Counter(row.cls for row in original.values()) != dict.fromkeys(classes, 15):
        raise ValueError("Expected 15 original images per class")
    if Counter(row.country for row in original.values()) != dict.fromkeys(COUNTRIES, 30):
        raise ValueError("Expected 30 original images per country")
    if set(Counter((row.cls, row.country) for row in original.values()).values()) != {5}:
        raise ValueError("Expected five original images per class and country")
    for row in records:
        parent = original.get(row.original_site_id)
        if parent is None or (row.cls, row.country) != (parent.cls, parent.country):
            raise ValueError(f"Removal step has inconsistent original metadata: {row.site_id}")
    if any(len({row.step_name for row in records if row.cls == cls and row.step == step}) != 1
           for cls in classes for step in (1, 2, 3)):
        raise ValueError("Cumulative removal names differ within a class")
    return records


def image_sites(records: list[InterventionRecord], data_root: Path) -> list[zero_shot.Site]:
    from PIL import Image

    root = data_root.resolve(strict=True)
    labels = {item[1]: item[3] for item in zero_shot.CLASSES["sifc"]}
    sites = []
    for record in records:
        image = (root / record.image_path).resolve(strict=True)
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Image is outside the data root: {record.site_id}")
        with Image.open(image) as opened:
            if opened.size != (4096, 4096):
                raise ValueError(f"Expected a 4096 x 4096 image: {record.site_id}")
        sites.append(zero_shot.Site(record.site_id, image, record.country, labels[record.cls]))
    return sites


def score_zero_shot(args: argparse.Namespace, sites: list[zero_shot.Site], model_id: str) -> None:
    run_dir = args.output_dir / args.model
    output = run_dir / "zero_shot_predictions.csv"
    prompt = zero_shot.prompt_for("sifc")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    columns = [*zero_shot.output_columns("sifc"), "image_sha256"]
    saved = zero_shot.load_checkpoint(output, sites, model_id, prompt_hash, columns, "sifc")
    for site in sites:
        if site.filename in saved and saved[site.filename]["image_sha256"] != selective.sha256_file(site.path):
            raise ValueError(f"Saved zero-shot image changed: {site.filename}")
    pending = [site for site in sites if site.filename not in saved]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"zero-shot: completed={len(saved)}/{len(sites)} pending={len(pending)}")
    if args.dry_run or not pending:
        return
    letters = [item[0] for item in zero_shot.CLASSES["sifc"]]
    label_by_letter = {item[0]: item[3] for item in zero_shot.CLASSES["sifc"]}
    for site, letter, probabilities, raw, top, mass, seconds in zero_shot.run_vllm(
            args, pending, prompt, letters, model_id):
        row = {"image_file": site.filename, "country": site.country, "true": site.label,
               "pred": label_by_letter[letter], "model": model_id,
               "prompt_sha256": prompt_hash, "seconds": round(seconds, 3), "raw": raw,
               "top_token": top, "letter_mass": mass,
               "image_sha256": selective.sha256_file(site.path)}
        row.update({f"p_{item[1]}": value for item, value in zip(zero_shot.CLASSES["sifc"], probabilities)})
        saved[site.filename] = row
        zero_shot.write_checkpoint(output, sites, saved, columns)
        print(f"zero-shot scored {len(saved)}/{len(sites)}: {site.filename}", flush=True)


def score_selective(args: argparse.Namespace, sites: list[zero_shot.Site], model_id: str) -> None:
    descriptions = selective.load_descriptors("sifc")
    identity = {"dataset": "sifc_intervention_90", "model": args.model,
                "model_id": model_id, "route_temperature": args.route_temperature,
                "home_padding": args.home_padding,
                "tensor_parallel_size": args.tensor_parallel_size,
                "manifest_sha256": selective.sha256_file(args.manifest),
                "descriptor_sha256": selective.sha256_file(selective.DESCRIPTOR_FILES["sifc"]),
                "scorer_sha256": selective.sha256_file(Path(selective.__file__).resolve()),
                "runner_sha256": selective.sha256_file(Path(__file__).resolve()),
                "scoring": "fresh_global_top1_home_v1"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run_dir = args.output_dir / args.model / "selective"
    run_info = {"identity": identity, "fingerprint": fingerprint}
    info_path = run_dir / "run_info.json"
    if info_path.exists() and json.loads(info_path.read_text(encoding="utf-8")) != run_info:
        raise ValueError("Saved selective run metadata does not match current inputs")
    saved = {}
    for site in sites:
        path = selective.shard_path(run_dir, site.filename)
        if path.exists():
            saved[site.filename] = selective.read_shard(
                path, site, descriptions[site.country], "sifc", fingerprint)
    pending = [site for site in sites if site.filename not in saved]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"selective: completed={len(saved)}/{len(sites)} pending={len(pending)}")
    print(f"global/route/new-home requests per image: "
          f"{selective.expected_calls('sifc', descriptions[sites[0].country])}")
    if args.dry_run:
        return
    if not info_path.exists():
        selective.write_json_atomic(info_path, run_info)
    if pending:
        engine = selective.VllmEngine(args, model_id)
        for site in pending:
            shard = selective.score_site(site, descriptions[site.country], "sifc",
                                         engine, args.home_padding, fingerprint)
            shard["image_sha256"] = selective.sha256_file(site.path)
            selective.write_json_atomic(selective.shard_path(run_dir, site.filename), shard)
            saved[site.filename] = shard
            print(f"selective scored {len(saved)}/{len(sites)}: {site.filename}", flush=True)
    rows = selective.prediction_rows(sites, saved, descriptions, "sifc", model_id)
    selective.write_predictions(run_dir / "predictions.csv", rows, "sifc")
    print(f"selective prediction rows={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory containing cbm_data and cbm_data_inpainted_bbox_additive")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", required=True, choices=("validate", "zero-shot", "selective"))
    parser.add_argument("--model", choices=MODELS, default="gemma")
    parser.add_argument("--model-id", help="override the checkpoint for both inference stages")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and saved outputs without loading a model")
    parser.add_argument("--limit", type=int, help="score at most this many pending images")
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES for this process")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--home-padding", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.gpus is None:
        args.gpus = "0,1,2,3" if args.model in {"qwen", "glm"} else "0"
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 4 if args.model in {"qwen", "glm"} else 1
    if args.batch < 1 or args.tensor_parallel_size < 1:
        raise ValueError("Batch and tensor parallel size must be positive")
    if not math.isfinite(args.route_temperature) or args.route_temperature <= 0:
        raise ValueError("Route temperature must be positive and finite")
    if not 0 <= args.home_padding <= 0.25:
        raise ValueError("Home padding must be between 0 and 0.25")
    records = read_manifest(args.manifest)
    sites = image_sites(records, args.data_root)
    print(f"validated {len(sites)} images: 90 original + 270 cumulative removals")
    if args.stage == "validate":
        return
    model_id = args.model_id or selective.MODEL_IDS[args.model]
    if args.stage == "zero-shot":
        score_zero_shot(args, sites, model_id)
    else:
        score_selective(args, sites, model_id)


if __name__ == "__main__":
    main()

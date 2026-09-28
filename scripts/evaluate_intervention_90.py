"""Report fixed-protocol true-class score drops for SiFC part removal.

Read all 90 original images and all 270 cumulative removal scores produced by
intervention_90.py. For each concept view, the original image chooses concepts
with p(yes) > 0.5; the same concepts are scored after every removal. The
APERTURE score blends probabilities at tau=30 with alpha_home=0.25.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import intervention_90 as intervention
import selective_multiscale as selective
import zero_shot_baselines as zero_shot

METHODS = ("zero_shot", "global", "home", "aperture")
COLUMNS = ("class", "method", "n_images", "concept_set", "temperature", "alpha_home",
           "step1_removed", "step2_removed", "step3_removed", "score_original",
           "score_step1", "drop_step1", "score_step2", "drop_step2",
           "score_step3", "drop_step3")


def load_run_info(run_dir: Path, manifest: Path) -> dict:
    path = run_dir / "selective" / "run_info.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    if (identity.get("dataset") != "sifc_intervention_90"
            or identity.get("manifest_sha256") != selective.sha256_file(manifest)
            or identity.get("descriptor_sha256") != selective.sha256_file(
                selective.DESCRIPTOR_FILES["sifc"])):
        raise ValueError("Selective run metadata does not match this manifest and descriptor inventory")
    if not payload.get("fingerprint") or not identity.get("model_id"):
        raise ValueError("Selective run metadata is incomplete")
    return payload


def load_zero_shot(path: Path, records: list[intervention.InterventionRecord],
                   model_id: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_file", "country", "true", "model", "pred", "prompt_sha256",
                    "image_sha256", *(f"p_{item[1]}" for item in zero_shot.CLASSES["sifc"])}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Zero-shot predictions are missing required columns")
        rows = list(reader)
    expected = {record.site_id: record for record in records}
    labels = {item[1]: item[3] for item in zero_shot.CLASSES["sifc"]}
    prompt_hash = hashlib.sha256(zero_shot.prompt_for("sifc").encode("utf-8")).hexdigest()
    saved = {}
    for row in rows:
        key = row["image_file"]
        record = expected.get(key)
        if (record is None or key in saved or row["country"] != record.country
                or row["true"] != labels[record.cls] or row["model"] != model_id
                or row["pred"] not in labels.values() or row["prompt_sha256"] != prompt_hash
                or not row["image_sha256"]):
            raise ValueError(f"Invalid zero-shot prediction: {key}")
        probabilities = [float(row[f"p_{cls}"]) for cls in labels]
        if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)):
            raise ValueError(f"Invalid zero-shot class probabilities: {key}")
        saved[key] = row
    if set(saved) != set(expected):
        raise ValueError(f"Expected 360 zero-shot predictions, found {len(saved)}")
    return saved


def load_concepts(run_dir: Path, records: list[intervention.InterventionRecord],
                  fingerprint: str, descriptions: dict[str, dict[str, dict]]) -> dict[str, dict]:
    folder = run_dir / "selective" / "shards"
    paths = list(folder.glob("*.json"))
    expected = {record.site_id: record for record in records}
    if {path.stem for path in paths} != set(expected):
        raise ValueError(f"Expected 360 selective evidence shards, found {len(paths)}")
    saved = {}
    for site_id, record in expected.items():
        shard = json.loads((folder / f"{site_id}.json").read_text(encoding="utf-8"))
        entries = descriptions[record.country]
        selected = selective.selected_entries("sifc", entries)
        if (shard.get("fingerprint") != fingerprint or shard.get("image_file") != site_id
                or shard.get("country") != record.country or shard.get("source_size") != [4096, 4096]
                or not shard.get("image_sha256") or set(shard.get("global", {})) != set(entries)
                or set(shard.get("home", {})) != set(selected)
                or set(shard.get("routes", {})) != set(selected)):
            raise ValueError(f"Incomplete or mismatched selective evidence: {site_id}")
        for key in selected:
            if len(shard["routes"][key]) != len(entries[key]["question_route_by_scale"]):
                raise ValueError(f"Incomplete routing evidence: {site_id}/{key}")
            for evidence in (shard["global"][key], shard["home"][key]):
                kind = evidence.get("kind")
                value = float(evidence.get("value", float("nan")))
                if (kind not in {"logit", "confidence"} or not math.isfinite(value)
                        or (kind == "confidence" and not 0 <= value <= 1)):
                    raise ValueError(f"Invalid concept evidence: {site_id}/{key}")
        saved[site_id] = shard
    return saved


def concept_probability(shard: dict, key: str, method: str) -> float:
    temperature = selective.TEMPERATURES[0]
    global_score = selective.score_at_temperature(shard["global"][key], temperature)
    if method == "global":
        return global_score
    home_score = selective.score_at_temperature(shard["home"][key], temperature)
    if method == "home":
        return home_score
    return (1 - selective.ALPHA_HOME) * global_score + selective.ALPHA_HOME * home_score


def score_sequence(original: intervention.InterventionRecord, method: str,
                   concept_set: str, by_id: dict[str, dict],
                   descriptions: dict[str, dict[str, dict]],
                   zero_rows: dict[str, dict]) -> list[float]:
    ids = [original.site_id, *(f"{original.site_id}__step{step}" for step in (1, 2, 3))]
    if method == "zero_shot":
        return [float(zero_rows[site_id][f"p_{original.cls}"]) for site_id in ids]
    entries = selective.selected_entries("sifc", descriptions[original.country])
    keys = [key for key, row in entries.items() if row["class"] == original.cls]
    if concept_set == "present":
        keys = [key for key in keys if concept_probability(by_id[original.site_id], key, method) > 0.5]
    if not keys:
        raise ValueError(f"No {method} concepts selected on original image: {original.site_id}")
    return [sum(concept_probability(by_id[site_id], key, method) for key in keys) / len(keys)
            for site_id in ids]


def summarize(records: list[intervention.InterventionRecord], by_id: dict[str, dict],
              descriptions: dict[str, dict[str, dict]], zero_rows: dict[str, dict],
              concept_set: str) -> list[dict]:
    if selective.TEMPERATURES != (30,) or selective.ALPHA_HOME != 0.25:
        raise ValueError("This report requires the paper's tau=30, alpha_home=0.25 settings")
    originals = [row for row in records if row.step == 0]
    classes = [item[1] for item in zero_shot.CLASSES["sifc"]]
    steps = {(row.cls, row.step): row.step_name for row in records if row.step > 0}
    if len(steps) != len(classes) * 3:
        raise ValueError("Missing cumulative removal names")
    summary = []
    for cls in classes:
        sites = [row for row in originals if row.cls == cls]
        if len(sites) != 15:
            raise ValueError(f"Expected 15 original images for {cls}")
        for method in METHODS:
            sequences = [score_sequence(row, method, concept_set, by_id, descriptions, zero_rows)
                         for row in sites]
            means = [sum(sequence[index] for sequence in sequences) / len(sequences)
                     for index in range(4)]
            row = {"class": cls, "method": method, "n_images": len(sites),
                   "concept_set": "class_probability" if method == "zero_shot" else concept_set,
                   "temperature": "" if method == "zero_shot" else 30,
                   "alpha_home": "" if method == "zero_shot" else
                                 0.0 if method == "global" else
                                 1.0 if method == "home" else selective.ALPHA_HOME,
                   **{f"step{step}_removed": steps[cls, step] for step in (1, 2, 3)},
                   "score_original": f"{means[0]:.8f}"}
            for step in (1, 2, 3):
                row[f"score_step{step}"] = f"{means[step]:.8f}"
                row[f"drop_step{step}"] = f"{means[0] - means[step]:.8f}"
            summary.append(row)
    for method in METHODS:
        rows = [row for row in summary if row["method"] == method]
        aggregate = {"class": "Mean", "method": method, "n_images": 90,
                     "concept_set": rows[0]["concept_set"],
                     "temperature": rows[0]["temperature"],
                     "alpha_home": rows[0]["alpha_home"],
                     "step1_removed": "", "step2_removed": "", "step3_removed": ""}
        for column in ("score_original", "score_step1", "drop_step1",
                       "score_step2", "drop_step2", "score_step3", "drop_step3"):
            aggregate[column] = f"{sum(float(row[column]) for row in rows) / len(rows):.8f}"
        summary.append(aggregate)
    if Counter(row["method"] for row in summary) != dict.fromkeys(METHODS, 7):
        raise ValueError("Expected six class rows and one mean row per method")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=intervention.DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path,
                        default=intervention.DEFAULT_OUTPUT / "gemma")
    parser.add_argument("--concept-set", choices=("present", "all"), default="present",
                        help="originally present concepts (paper intervention protocol), or all")
    parser.add_argument("--output", type=Path, help="output summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = intervention.read_manifest(args.manifest)
    run_info = load_run_info(args.results_dir, args.manifest)
    descriptions = selective.load_descriptors("sifc")
    zero_rows = load_zero_shot(args.results_dir / "zero_shot_predictions.csv", records,
                               run_info["identity"]["model_id"])
    concepts = load_concepts(args.results_dir, records, run_info["fingerprint"], descriptions)
    if any(zero_rows[site_id]["image_sha256"] != concepts[site_id]["image_sha256"]
           for site_id in zero_rows):
        raise ValueError("Zero-shot and selective evidence were scored on different image bytes")
    rows = summarize(records, concepts, descriptions, zero_rows, args.concept_set)
    output = args.output or args.results_dir / f"part_removal_{args.concept_set}_tau30_alpha0p25.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"validated 360 zero-shot predictions and selective shards; wrote {len(rows)} rows")
    for row in rows[-len(METHODS):]:
        print(f"{row['method']}: mean drops S1={row['drop_step1']} "
              f"S2={row['drop_step2']} S3={row['drop_step3']}")


if __name__ == "__main__":
    main()

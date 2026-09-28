"""Run the same zero-shot facility baselines on SiFC or fMoW imagery.

Images must use the coordinate/year PNG names produced by download_arcgis_images.py.
The CSV supplies labels for evaluation only; labels are never added to a model
prompt or used to select a prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP_SIZE = 224
TOP_LOGPROBS = 64
MISSING_LOGPROB = -30.0

# Letter order and prompt wording follow the saved zero-shot baselines.
CLASSES = {
    "sifc": (
        ("A", "iron_still_plant", "iron and steel plant", "Iron and steel plant"),
        ("B", "lng_terminals", "LNG terminal", "LNG terminal"),
        ("C", "nuclear_plants", "nuclear power plant", "Nuclear power plant"),
        ("D", "oil_refineries", "oil refinery", "Oil refinery"),
        ("E", "power_plants", "coal or thermal power plant", "Power plant"),
        ("F", "stp_plants", "sewage treatment plant", "Sewage treatment plant"),
    ),
    "fmow": (
        ("A", "shipyard", "shipyard", "Shipyard"),
        ("B", "solar_farm", "solar farm", "Solar farm"),
        ("C", "storage_tank", "storage tank facility", "Storage tank"),
        ("D", "water_treatment_facility", "water treatment facility", "Water treatment facility"),
        ("E", "wind_farm", "wind farm", "Wind farm"),
    ),
}
CSV_PATHS = {
    "sifc": ROOT / "SiFC data" / "sifc_dataset.csv",
    "fmow": ROOT / "fMoW data" / "fmow_dataset.csv",
}
MODEL_IDS = {
    "gemma": "google/gemma-4-31B-it",
    "gemini": "models/gemini-3.5-flash",
    "qwen": "Qwen/Qwen3.8-27B",
    "glm": "zai-org/GLM-4.6V",
}


@dataclass(frozen=True)
class Site:
    filename: str
    path: Path
    country: str
    label: str


def prompt_for(dataset: str) -> str:
    options = "\n".join(f"{letter}) {name}" for letter, _, name, _ in CLASSES[dataset])
    if dataset == "sifc":
        opening = "This is an aerial satellite image of an industrial facility. Classify its type."
    else:
        opening = "This is an aerial satellite image of a facility. Classify its type."
    last_letter = CLASSES[dataset][-1][0]
    return f"{opening}\n\n{options}\n\nAnswer with exactly one letter (A-{last_letter})."


def read_sites(csv_path: Path, images: Path, year: int, dataset: str) -> list[Site]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"country", "industry", "center_latitude", "center_longitude"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no sites")
    allowed_labels = {item[3] for item in CLASSES[dataset]}
    sites: list[Site] = []
    filenames: set[str] = set()
    for line, row in enumerate(rows, start=2):
        label = row["industry"].strip()
        country = row["country"].strip()
        if label not in allowed_labels or not country:
            raise ValueError(f"Unsupported label or empty country on CSV line {line}")
        try:
            latitude = float(row["center_latitude"])
            longitude = float(row["center_longitude"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid coordinates on CSV line {line}") from error
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError(f"Non-finite coordinates on CSV line {line}")
        if not (-85.05112878 < latitude < 85.05112878 and -180 <= longitude < 180):
            raise ValueError(f"Coordinates outside imagery bounds on CSV line {line}")
        filename = f"{latitude:.6f}_{longitude:.6f}_{year}.png"
        if filename in filenames:
            raise ValueError(f"Duplicate image filename on CSV line {line}: {filename}")
        filenames.add(filename)
        sites.append(Site(filename, images / filename, country, label))
    return sites


def output_columns(dataset: str) -> list[str]:
    return ["image_file", "country", "true", "pred", "model", "prompt_sha256",
            "seconds", "raw", "top_token", "letter_mass",
            *(f"p_{item[1]}" for item in CLASSES[dataset])]


def load_checkpoint(path: Path, sites: list[Site], model_id: str,
                    prompt_hash: str, columns: list[str], dataset: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    expected = {site.filename: site for site in sites}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Checkpoint columns do not match this dataset/model: {path.name}")
        rows = list(reader)
    result = {}
    valid = {item[3] for item in CLASSES[dataset]}
    for row in rows:
        name = row["image_file"]
        site = expected.get(name)
        if (site is None or row["country"] != site.country or row["true"] != site.label
                or row["model"] != model_id or row["prompt_sha256"] != prompt_hash):
            raise ValueError(f"Checkpoint does not match current inputs: {name}")
        if name in result:
            raise ValueError(f"Duplicate checkpoint image: {name}")
        if row["pred"] in valid:
            result[name] = row
    return result


def write_checkpoint(path: Path, sites: list[Site], rows: dict[str, dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows[site.filename] for site in sites if site.filename in rows)
    os.replace(temporary, path)


def resize_image(path: Path):
    from PIL import Image
    with Image.open(path) as source:
        return source.convert("RGB").resize((PREP_SIZE, PREP_SIZE), Image.Resampling.LANCZOS)


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def letter_token_ids(tokenizer, letters: list[str], model: str) -> dict[str, list[int]]:
    result = {}
    for letter in letters:
        spellings = [letter] if model == "gemma" else [letter, " " + letter]
        token_ids = set()
        for spelling in spellings:
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
        if not token_ids:
            raise ValueError(f"No single-token spelling for letter {letter}")
        result[letter] = sorted(token_ids)
    return result


def vllm_prompt(tokenizer, prompt: str, model: str) -> str:
    message = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt},
    ]}]
    if model == "gemma":
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = "Answer: <|begin_of_box|>" if model == "glm" else "Answer:"
    return rendered + prefix


def score_vllm_output(output, letter_ids: dict[str, list[int]], letters: list[str]):
    first = output.outputs[0]
    if not first.logprobs or not first.logprobs[0]:
        raise ValueError("No first-token log probabilities returned")
    logprobs = first.logprobs[0]
    values = []
    present = 0
    mass = 0.0
    for letter in letters:
        found = [logprobs[token].logprob for token in letter_ids[letter] if token in logprobs]
        if found:
            present += 1
            values.append(max(found))
            mass += sum(math.exp(value) for value in found)
        else:
            values.append(MISSING_LOGPROB)
    if not present:
        raise ValueError("No answer letters found in first-token log probabilities")
    top = min(logprobs.values(), key=lambda entry: entry.rank)
    return letters[max(range(len(letters)), key=lambda index: values[index])], softmax(values), first.text, top.decoded_token, mass


def run_vllm(args, pending: list[Site], prompt: str, letters: list[str], model_id: str):
    # Import and initialize the model only after input and checkpoint validation.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              max_logprobs=TOP_LOGPROBS, enable_prefix_caching=True,
              trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    ids = letter_token_ids(tokenizer, letters, args.model)
    rendered = vllm_prompt(tokenizer, prompt, args.model)
    kwargs = {"max_tokens": 1, "temperature": 0.0, "logprobs": TOP_LOGPROBS}
    if args.model == "gemma":
        kwargs["allowed_token_ids"] = sorted({token for group in ids.values() for token in group})
    sampling = SamplingParams(**kwargs)
    for start in range(0, len(pending), args.batch):
        batch = pending[start:start + args.batch]
        started = time.monotonic()
        images = [resize_image(site.path) for site in batch]
        outputs = llm.generate(
            [{"prompt": rendered, "multi_modal_data": {"image": image}} for image in images],
            sampling, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned the wrong number of outputs")
        seconds = (time.monotonic() - started) / len(batch)
        for site, output in zip(batch, outputs):
            letter, probabilities, raw, top, mass = score_vllm_output(output, ids, letters)
            yield site, letter, probabilities, raw, top, mass, seconds


def run_gemini(args, pending: list[Site], prompt: str, letters: list[str], model_id: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY before running Gemini inference")
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    from typing import Literal

    class FiveLetterAnswer(BaseModel):
        letter: Literal["A", "B", "C", "D", "E"]

    class SixLetterAnswer(BaseModel):
        letter: Literal["A", "B", "C", "D", "E", "F"]

    schema = FiveLetterAnswer if len(letters) == 5 else SixLetterAnswer
    config = types.GenerateContentConfig(temperature=0.0,
                                         response_mime_type="application/json",
                                         response_schema=schema)
    client = genai.Client(api_key=api_key)
    for site in pending:
        started = time.monotonic()
        image = resize_image(site.path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        parts = [types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
                 types.Part(text=prompt)]
        last_error = None
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=model_id, contents=[types.Content(role="user", parts=parts)],
                    config=config)
                raw = (response.text or "").strip()
                letter = json.loads(raw).get("letter")
                if letter not in letters:
                    raise ValueError("Gemini returned an invalid answer letter")
                probabilities = [float(candidate == letter) for candidate in letters]
                yield site, letter, probabilities, raw, "", "", time.monotonic() - started
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                last_error = error
                if attempt < 3:
                    time.sleep(2 ** attempt)
        else:
            # No response content or credentials are saved on failure.
            print(f"Gemini request failed for {site.filename}: {type(last_error).__name__}", flush=True)


def macro_f1(rows: list[dict], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(row["true"] == label and row["pred"] == label for row in rows)
        fp = sum(row["true"] != label and row["pred"] == label for row in rows)
        fn = sum(row["true"] == label and row["pred"] != label for row in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def print_summary(rows: dict[str, dict], sites: list[Site], dataset: str) -> None:
    labels = [item[3] for item in CLASSES[dataset]]
    print(f"completed: {len(rows)}/{len(sites)}")
    if not rows:
        return
    for country in [*dict.fromkeys(site.country for site in sites), "ALL"]:
        subset = list(rows.values()) if country == "ALL" else [row for row in rows.values() if row["country"] == country]
        if subset:
            accuracy = sum(row["true"] == row["pred"] for row in subset) / len(subset)
            print(f"{country}: n={len(subset)} accuracy={accuracy:.4f} macro-F1={macro_f1(subset, labels):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=tuple(CLASSES), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--images", type=Path, required=True,
                        help="directory of coordinate/year PNGs from download_arcgis_images.py")
    parser.add_argument("--csv", type=Path, help="override the release dataset CSV")
    parser.add_argument("--year", type=int, default=2026, help="year used in PNG filenames")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-id", help="override the default model ID")
    parser.add_argument("--dry-run", action="store_true", help="check inputs without loading a model")
    parser.add_argument("--show-prompt", action="store_true", help="print the exact dataset prompt")
    parser.add_argument("--limit", type=int, help="process at most this many pending images")
    parser.add_argument("--batch", type=int, default=4, help="images per vLLM call")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES for local models")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.batch < 1 or args.tensor_parallel_size < 1:
        raise ValueError("--batch and --tensor-parallel-size must be positive")
    csv_path = args.csv or CSV_PATHS[args.dataset]
    model_id = args.model_id or MODEL_IDS[args.model]
    prompt = prompt_for(args.dataset)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    sites = read_sites(csv_path, args.images, args.year, args.dataset)
    columns = output_columns(args.dataset)
    output = args.output_dir / f"{args.dataset}_{args.model}_predictions.csv"
    rows = load_checkpoint(output, sites, model_id, prompt_hash, columns, args.dataset)
    pending = [site for site in sites if site.filename not in rows]
    if args.limit is not None:
        pending = pending[:args.limit]
    missing = [site.filename for site in pending if not site.path.is_file()]
    print(f"dataset={args.dataset} model={model_id} sites={len(sites)} pending={len(pending)}")
    print(f"images={args.images.name} output={output.name}")
    print(f"missing pending images={len(missing)}")
    if missing:
        print(f"first missing image={missing[0]}")
    if args.show_prompt:
        print(prompt)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(f"{len(missing)} pending images are missing from {args.images.name}")
    if not pending:
        print_summary(rows, sites, args.dataset)
        return
    letters = [item[0] for item in CLASSES[args.dataset]]
    label_by_letter = {item[0]: item[3] for item in CLASSES[args.dataset]}
    runner = run_gemini if args.model == "gemini" else run_vllm
    for site, letter, probabilities, raw, top, mass, seconds in runner(args, pending, prompt, letters, model_id):
        row = {"image_file": site.filename, "country": site.country, "true": site.label,
               "pred": label_by_letter[letter], "model": model_id,
               "prompt_sha256": prompt_hash, "seconds": round(seconds, 3), "raw": raw,
               "top_token": top, "letter_mass": mass}
        row.update({f"p_{item[1]}": value for item, value in zip(CLASSES[args.dataset], probabilities)})
        rows[site.filename] = row
        write_checkpoint(output, sites, rows, columns)
        print(f"{site.filename}: {row['pred']} ({len(rows)}/{len(sites)})", flush=True)
    print_summary(rows, sites, args.dataset)


if __name__ == "__main__":
    main()

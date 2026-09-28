"""Run country-selected, top-1 multiscale concept inference on SiFC or fMoW.

All global questions are scored fresh. The true class is used only when writing
predictions and metrics, never for question selection, routing, or scoring.
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
from collections import Counter
from pathlib import Path
from typing import Any

from zero_shot_baselines import CLASSES, ROOT, Site, read_sites

DESCRIPTOR_FILES = {
    "sifc": ROOT / "descriptors" / "cccd_descriptors.json",
    "fmow": ROOT / "descriptors" / "fmow_cccd_descriptors.json",
}
CSV_FILES = {
    "sifc": ROOT / "SiFC data" / "sifc_dataset.csv",
    "fmow": ROOT / "fMoW data" / "fmow_dataset.csv",
}
MODEL_IDS = {
    "gemma": "google/gemma-4-31B-it",
    "gemini": "models/gemini-3.5-flash",
    "qwen": "Qwen/Qwen3.5-122B-A10B",
    "glm": "zai-org/GLM-4.6V",
}
TEMPERATURES = (30,)  # Fixed yes/no concept-score temperature (paper: tau = 30).
ALPHA_HOME = 0.25  # Paper's home-scale contribution to each concept probability.
PREP_SIZE = 224
LEVELS = (4096, 2048, 1024, 512)
YES_SURFACES = ("yes", "Yes", " yes", " Yes")
NO_SURFACES = ("no", "No", " no", " No")
DIGIT_SURFACES = {digit: (digit, " " + digit, "\n" + digit) for digit in "1234"}
ROUTE_SUFFIX = "Answer with exactly one digit: 1, 2, 3, or 4."
PRESENCE_SUFFIX = "Answer with exactly one word: yes or no."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_descriptors(dataset: str) -> dict[str, dict[str, dict]]:
    raw = json.loads(DESCRIPTOR_FILES[dataset].read_text(encoding="utf-8"))
    classes = {item[1] for item in CLASSES[dataset]}
    countries = {"sifc": {"India", "USA", "China"},
                 "fmow": {"USA", "France", "Russia"}}[dataset]
    expected = 113 if dataset == "sifc" else 25
    selected_expected = 100 if dataset == "sifc" else 25
    result: dict[str, dict[str, dict]] = {country: {} for country in countries}
    for key, entry in raw.items():
        country, cls, concept = entry["country"], entry["class"], entry["concept"]
        if key != f"{country}::{cls}::{concept}" or country not in countries or cls not in classes:
            raise ValueError(f"Invalid descriptor identity: {key}")
        if not entry["question_global"].endswith(PRESENCE_SUFFIX):
            raise ValueError(f"Invalid global question: {key}")
        if not entry["question_home"].endswith(PRESENCE_SUFFIX):
            raise ValueError(f"Invalid home question: {key}")
        routes = entry["question_route_by_scale"]
        if dataset == "sifc":
            home = entry["home_scale_px"]
            if home not in (*LEVELS, 256):
                raise ValueError(f"Invalid SiFC home width: {key}")
            expected_routes = [str(level) for level in LEVELS if level > home]
        else:
            expected_routes = ["source", "half"]
            weight = float(entry["shared_idf_weight"])
            membership = entry["shared_concept_classes"]
            if not membership or cls not in membership or not math.isclose(weight, 1 / len(membership)):
                raise ValueError(f"Invalid fMoW shared-concept weight: {key}")
        if list(routes) != expected_routes or any(not q.endswith(ROUTE_SUFFIX) for q in routes.values()):
            raise ValueError(f"Invalid routing questions: {key}")
        result[country][key] = entry
    for country, entries in result.items():
        if len(entries) != expected:
            raise ValueError(f"Expected {expected} descriptors for {country}, found {len(entries)}")
        selected = selected_entries(dataset, entries)
        if len(selected) != selected_expected:
            raise ValueError(f"Unexpected selected concept count for {country}")
        if dataset == "fmow" and Counter(row["class"] for row in entries.values()) != dict.fromkeys(classes, 5):
            raise ValueError(f"Expected five fMoW descriptors per class for {country}")
        if dataset == "sifc":
            if any(row["use_for_global_class_score_human"] != row["use_for_multiscale_search_human"]
                   for row in entries.values()):
                raise ValueError(f"SiFC selection flags disagree for {country}")
            expected_by_class = {"iron_still_plant": 26, "lng_terminals": 16,
                                 "nuclear_plants": 12, "oil_refineries": 18,
                                 "power_plants": 18, "stp_plants": 10}
            if Counter(row["class"] for row in selected.values()) != expected_by_class:
                raise ValueError(f"SiFC selected class counts changed for {country}")
            expected_home = {4096: 4, 2048: 7, 1024: 41, 512: 32, 256: 16}
            if Counter(row["home_scale_px"] for row in selected.values()) != expected_home:
                raise ValueError(f"SiFC selected home widths changed for {country}")
    return result


def selected_entries(dataset: str, entries: dict[str, dict]) -> dict[str, dict]:
    if dataset == "fmow":
        return entries
    return {key: row for key, row in entries.items() if row["use_for_multiscale_search_human"]}


def expected_calls(dataset: str, entries: dict[str, dict]) -> tuple[int, int, int]:
    selected = selected_entries(dataset, entries)
    routes = sum(len(row["question_route_by_scale"]) for row in selected.values())
    home = sum(dataset == "fmow" or row["home_scale_px"] != 4096 for row in selected.values())
    return len(entries), routes, home


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    return [value / sum(weights) for value in weights]


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def score_at_temperature(evidence: dict, temperature: int) -> float:
    if evidence["kind"] == "logit":
        return sigmoid(float(evidence["value"]) / temperature)
    p = float(evidence["value"])
    if temperature == 1:
        return p
    p = min(1 - 1e-6, max(1e-6, p))
    return sigmoid((math.log(p) - math.log1p(-p)) / temperature)


def split_rect(rect: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    x, y, width, height = rect
    if width < 2 or height < 2:
        raise ValueError("Image crop is too small to route")
    left, top = width // 2, height // 2
    return {
        "1": (x, y, left, top),
        "2": (x + left, y, width - left, top),
        "3": (x, y + top, left, height - top),
        "4": (x + left, y + top, width - left, height - top),
    }


def home_view(dataset: str, rect: tuple[int, int, int, int],
              source_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    source_w, source_h = source_size
    if dataset == "sifc" and width == 4096:
        return rect
    if dataset == "sifc":
        view_w = view_h = min(source_w, round(width * (1 + 2 * padding)))
    else:
        view_w = min(source_w, max(1, round(source_w * 0.25 * (1 + 2 * padding))))
        view_h = min(source_h, max(1, round(source_h * 0.25 * (1 + 2 * padding))))
    view_x = min(max(round(x + width / 2 - view_w / 2), 0), source_w - view_w)
    view_y = min(max(round(y + height / 2 - view_h / 2), 0), source_h - view_h)
    return view_x, view_y, view_w, view_h


def crop_224(source: Any, rect: tuple[int, int, int, int]):
    from PIL import Image
    x, y, width, height = rect
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > source.width or y + height > source.height:
        raise ValueError(f"Invalid crop {rect} for image size {source.size}")
    crop = source.crop((x, y, x + width, y + height))
    try:
        return crop.resize((PREP_SIZE, PREP_SIZE), Image.Resampling.LANCZOS)
    finally:
        crop.close()


def router_overlay(clean: Any):
    from PIL import ImageDraw, ImageFont
    image = clean.copy()
    draw = ImageDraw.Draw(image)
    half = PREP_SIZE // 2
    for line in ((half, 0, half, PREP_SIZE), (0, half, PREP_SIZE, half)):
        draw.line(line, fill=(0, 0, 0), width=4)
        draw.line(line, fill=(255, 230, 0), width=2)
    font_file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_file), 22) if font_file.is_file() else ImageFont.load_default()
    for digit, position in {"1": (7, 4), "2": (half + 7, 4),
                            "3": (7, half + 4), "4": (half + 7, half + 4)}.items():
        draw.text(position, digit, fill="white", font=font, stroke_width=2, stroke_fill="black")
    return image


def image_requests(source: Any, specs: list[tuple[str, str, tuple[int, int, int, int]]],
                   overlay: bool = False) -> list[tuple[str, Any]]:
    cache = {}
    requests = []
    for _, prompt, rect in specs:
        if rect not in cache:
            clean = crop_224(source, rect)
            if overlay:
                cache[rect] = router_overlay(clean)
                clean.close()
            else:
                cache[rect] = clean
        requests.append((prompt, cache[rect]))
    return requests


def close_requests(requests: list[tuple[str, Any]]) -> None:
    for image in {id(image): image for _, image in requests}.values():
        image.close()

class VllmEngine:
    def __init__(self, args: argparse.Namespace, model_id: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model = args.model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = self.processor.tokenizer

        def token_group(surfaces: tuple[str, ...]) -> list[int]:
            ids = {encoded[0] for surface in surfaces
                   if len(encoded := tokenizer.encode(surface, add_special_tokens=False)) == 1}
            if not ids:
                raise ValueError(f"No one-token answer surface among {surfaces}")
            return sorted(ids)

        self.yes_ids = token_group(YES_SURFACES)
        self.no_ids = token_group(NO_SURFACES)
        self.digit_ids = {digit: token_group(surfaces) for digit, surfaces in DIGIT_SURFACES.items()}
        if set(self.yes_ids) & set(self.no_ids):
            raise ValueError("Yes and no token groups overlap")
        all_digits = [value for ids in self.digit_ids.values() for value in ids]
        if len(all_digits) != len(set(all_digits)):
            raise ValueError("Quadrant token groups overlap")
        presence_ids = [*self.yes_ids, *self.no_ids]
        logprobs = max(64, len(presence_ids), len(all_digits))
        self.llm = LLM(model=model_id, tensor_parallel_size=args.tensor_parallel_size,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_model_len=4096, limit_mm_per_prompt={"image": 1},
                       max_logprobs=logprobs, logprobs_mode="processed_logprobs",
                       enable_prefix_caching=True, trust_remote_code=True)
        self.presence_sampling = SamplingParams(
            temperature=0.0, max_tokens=1, allowed_token_ids=presence_ids,
            logprobs=logprobs)
        self.route_sampling = SamplingParams(
            temperature=0.0, max_tokens=1, allowed_token_ids=all_digits,
            logprobs=max(4, len(all_digits)))
        self.batch = args.batch
        self.route_temperature = args.route_temperature

    def wrap(self, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question},
        ]}]
        extra = {} if self.model == "gemma" else {"enable_thinking": False}
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)
        prefix = "Answer: <|begin_of_box|>" if self.model == "glm" else "Answer:"
        return rendered + prefix

    def _generate(self, requests: list[tuple[str, Any]], sampling: Any) -> list[Any]:
        outputs = []
        for start in range(0, len(requests), self.batch):
            chunk = requests[start:start + self.batch]
            packed = [{"prompt": self.wrap(question), "multi_modal_data": {"image": image}}
                      for question, image in chunk]
            outputs.extend(self.llm.generate(packed, sampling, use_tqdm=False))
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an unexpected number of responses")
        return outputs

    @staticmethod
    def _logprobs(output: Any) -> dict:
        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError("Missing first-token log probabilities")
        return output.outputs[0].logprobs[0]

    @staticmethod
    def _group(logprobs: dict, ids: list[int]) -> float:
        if any(token not in logprobs for token in ids):
            raise RuntimeError("An allowed answer token is missing from log probabilities")
        return logsumexp([float(logprobs[token].logprob) for token in ids])

    def presence(self, requests: list[tuple[str, Any]]) -> list[dict]:
        result = []
        for output in self._generate(requests, self.presence_sampling):
            logprobs = self._logprobs(output)
            yes = self._group(logprobs, self.yes_ids)
            no = self._group(logprobs, self.no_ids)
            result.append({"kind": "logit", "value": yes - no})
        return result

    def route(self, requests: list[tuple[str, Any]]) -> list[dict[str, float]]:
        result = []
        for output in self._generate(requests, self.route_sampling):
            logprobs = self._logprobs(output)
            values = [self._group(logprobs, self.digit_ids[digit]) for digit in "1234"]
            result.append(dict(zip("1234", softmax(values, self.route_temperature))))
        return result


class GeminiEngine:
    def __init__(self, args: argparse.Namespace, model_id: str):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY before Gemini inference")
        from google import genai
        from google.genai import types
        from pydantic import BaseModel, Field
        from typing import Literal

        class ConfidenceAnswer(BaseModel):
            status: Literal["Present", "Absent"]
            confidence: float = Field(ge=0.0, le=1.0)
            alternative: str | None = None

        class RouteAnswer(BaseModel):
            quadrant_ranking: list[Literal["1", "2", "3", "4"]] = Field(min_length=4, max_length=4)

        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.dataset = args.dataset
        self.route_temperature = args.route_temperature
        self.confidence_schema = ConfidenceAnswer
        self.route_schema = RouteAnswer
        if args.dataset == "sifc":
            self.token_config = types.GenerateContentConfig(
                max_output_tokens=8, response_logprobs=True, logprobs=20,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
                thinking_config=types.ThinkingConfig(thinking_level="minimal", include_thoughts=False))
        else:
            common = {"temperature": 0.0, "max_output_tokens": 96,
                      "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
                      "thinking_config": types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
                      "response_mime_type": "application/json"}
            self.presence_config = types.GenerateContentConfig(**common, response_schema=ConfidenceAnswer)
            self.route_config = types.GenerateContentConfig(**common, response_schema=RouteAnswer)

    @staticmethod
    def _png(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    def _call(self, question: str, image: Any, config: Any) -> Any:
        types = self.types
        parts = [types.Part.from_bytes(data=self._png(image), mime_type="image/png"),
                 types.Part(text=question)]
        for attempt in range(6):
            try:
                return self.client.models.generate_content(
                    model=self.model_id,
                    contents=[types.Content(role="user", parts=parts)], config=config)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                if attempt == 5:
                    raise RuntimeError(f"Gemini request failed: {type(error).__name__}") from None
                time.sleep(2 ** attempt)
        raise RuntimeError("Gemini request failed")

    @staticmethod
    def _parse_json(response: Any, schema: Any) -> Any:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(json.loads((response.text or "").strip()))

    @staticmethod
    def _top_scores(response: Any, labels: tuple[str, ...]) -> dict[str, float]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError("Gemini returned no answer candidate")
        result = getattr(candidates[0], "logprobs_result", None)
        positions = getattr(result, "top_candidates", None) if result else None
        if not positions:
            raise RuntimeError("Gemini returned no top-candidate log probabilities")
        for position in positions:
            grouped = {label: [] for label in labels}
            for candidate in getattr(position, "candidates", None) or []:
                label = str(candidate.token).replace("▁", " ").replace("Ġ", " ").strip().lower()
                if label in grouped:
                    grouped[label].append(float(candidate.log_probability))
            if all(grouped.values()):
                return {label: logsumexp(grouped[label]) for label in labels}
        raise RuntimeError("Gemini omitted one or more required answer alternatives")

    def presence(self, requests: list[tuple[str, Any]]) -> list[dict]:
        result = []
        for question, image in requests:
            if self.dataset == "sifc":
                response = self._call(question.rstrip() + "\nAnswer:", image, self.token_config)
                scores = self._top_scores(response, ("yes", "no"))
                result.append({"kind": "logit", "value": scores["yes"] - scores["no"]})
            else:
                if not question.endswith(PRESENCE_SUFFIX):
                    raise ValueError("Invalid fMoW presence question")
                suffix = ('\nClassify the requested concept as "Present" when its visible evidence '
                          'is present, or "Absent" otherwise. Give your confidence from 0.0 to '
                          '1.0 in that chosen status. If Absent, an alternative visible object '
                          'may be named briefly. Respond only as JSON: '
                          '{"status":"Present" or "Absent","confidence":0.85,'
                          '"alternative":null or "short description"}')
                rewritten = question[:-len(PRESENCE_SUFFIX)].rstrip() + suffix
                response = self._call(rewritten, image, self.presence_config)
                answer = self._parse_json(response, self.confidence_schema)
                p = answer.confidence if answer.status == "Present" else 1 - answer.confidence
                result.append({"kind": "confidence", "value": float(p),
                               "status": answer.status, "confidence": float(answer.confidence)})
        return result

    def route(self, requests: list[tuple[str, Any]]) -> list[dict[str, float]]:
        result = []
        for question, image in requests:
            if self.dataset == "sifc":
                response = self._call(question.rstrip() + "\nAnswer:", image, self.token_config)
                scores = self._top_scores(response, tuple("1234"))
                result.append(dict(zip("1234", softmax([scores[d] for d in "1234"], self.route_temperature))))
            else:
                if not question.endswith(ROUTE_SUFFIX):
                    raise ValueError("Invalid fMoW routing question")
                rewritten = question[:-len(ROUTE_SUFFIX)].rstrip() + (
                    "\nRank quadrants 1, 2, 3, and 4 from most to least likely to contain "
                    "the requested visible evidence. Return each quadrant exactly once "
                    "in quadrant_ranking.")
                response = self._call(rewritten, image, self.route_config)
                ranking = [str(value) for value in self._parse_json(response, self.route_schema).quadrant_ranking]
                if len(ranking) != 4 or set(ranking) != set("1234"):
                    raise RuntimeError("Gemini returned an invalid quadrant ranking")
                base = {digit: weight / 10.0 for digit, weight in zip(ranking, (4, 3, 2, 1))}
                result.append(dict(zip("1234", softmax([math.log(base[d]) for d in "1234"],
                                                      self.route_temperature))))
        return result

def score_site(site: Site, entries: dict[str, dict], dataset: str,
               engine: Any, padding: float, fingerprint: str) -> dict:
    from PIL import Image

    with Image.open(site.path) as opened:
        source = opened.convert("RGB")
    try:
        if dataset == "sifc" and source.size != (4096, 4096):
            raise ValueError(f"SiFC image {site.filename} must be 4096 x 4096 pixels")
        full = (0, 0, source.width, source.height)
        global_specs = [(key, row["question_global"], full) for key, row in entries.items()]
        requests = image_requests(source, global_specs)
        try:
            global_values = engine.presence(requests)
        finally:
            close_requests(requests)
        if len(global_values) != len(global_specs):
            raise RuntimeError("Global scorer returned the wrong number of answers")
        global_by_key = dict(zip(entries, global_values))

        selected = selected_entries(dataset, entries)
        state = {key: (full, "") for key in selected}
        route_by_key: dict[str, list[dict]] = {key: [] for key in selected}
        stages = [str(level) for level in LEVELS] if dataset == "sifc" else ["source", "half"]
        for stage in stages:
            specs = []
            for key, row in selected.items():
                prompt = row["question_route_by_scale"].get(stage)
                if prompt is None:
                    continue
                rect, _ = state[key]
                if dataset == "sifc" and rect[2:] != (int(stage), int(stage)):
                    raise RuntimeError(f"Route state is at the wrong level for {key}")
                specs.append((key, prompt, rect))
            if not specs:
                continue
            requests = image_requests(source, specs, overlay=True)
            try:
                answers = engine.route(requests)
            finally:
                close_requests(requests)
            if len(answers) != len(specs):
                raise RuntimeError("Router returned the wrong number of answers")
            for (key, _, rect), probabilities in zip(specs, answers):
                if set(probabilities) != set("1234") or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
                    raise RuntimeError(f"Invalid quadrant distribution for {key}")
                digit = max("1234", key=lambda value: probabilities[value])
                child = split_rect(rect)[digit]
                _, previous_path = state[key]
                state[key] = (child, previous_path + digit)
                route_by_key[key].append({"stage": stage, "digit": digit,
                                          "probabilities": probabilities})

        home_by_key = {}
        home_specs = []
        for key, row in selected.items():
            rect, path = state[key]
            if dataset == "sifc" and rect[2] != row["home_scale_px"]:
                raise RuntimeError(f"Home crop has the wrong width for {key}")
            view = home_view(dataset, rect, source.size, padding)
            if dataset == "sifc" and row["home_scale_px"] == 4096:
                if row["question_home"] != row["question_global"]:
                    raise ValueError(f"Full-image home/global questions differ for {key}")
                home_by_key[key] = {**global_by_key[key], "path": path,
                                    "rect": rect, "view": view, "source": "global_reuse"}
            else:
                home_specs.append((key, row["question_home"], view))
                home_by_key[key] = {"path": path, "rect": rect, "view": view,
                                    "source": "home_model"}
        requests = image_requests(source, home_specs)
        try:
            answers = engine.presence(requests) if requests else []
        finally:
            close_requests(requests)
        if len(answers) != len(home_specs):
            raise RuntimeError("Home scorer returned the wrong number of answers")
        for (key, _, _), evidence in zip(home_specs, answers):
            home_by_key[key].update(evidence)
        return {"fingerprint": fingerprint, "image_file": site.filename,
                "country": site.country, "source_size": list(source.size),
                "global": global_by_key, "routes": route_by_key, "home": home_by_key}
    finally:
        source.close()


def shard_path(output_dir: Path, filename: str) -> Path:
    return output_dir / "shards" / f"{filename}.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_shard(path: Path, site: Site, entries: dict[str, dict],
               dataset: str, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = selected_entries(dataset, entries)
    if (data.get("fingerprint") != fingerprint or data.get("image_file") != site.filename
            or data.get("country") != site.country or set(data.get("global", {})) != set(entries)
            or set(data.get("home", {})) != set(selected)
            or set(data.get("routes", {})) != set(selected)):
        raise ValueError(f"Saved evidence does not match current inputs: {site.filename}")
    if data.get("image_sha256") != sha256_file(site.path):
        raise ValueError(f"Image changed after scoring: {site.filename}")
    for key, row in selected.items():
        if len(data["routes"][key]) != len(row["question_route_by_scale"]):
            raise ValueError(f"Incomplete routing evidence: {site.filename}/{key}")
        for evidence in (data["global"][key], data["home"][key]):
            if evidence.get("kind") not in {"logit", "confidence"} or not math.isfinite(float(evidence["value"])):
                raise ValueError(f"Invalid presence evidence: {site.filename}/{key}")
    return data


def class_scores(dataset: str, entries: dict[str, dict],
                 evidence: dict[str, dict], temperature: int,
                 home_evidence: dict[str, dict] | None = None) -> dict[str, float]:
    selected = selected_entries(dataset, entries)
    scores = {}
    for _, cls, _, _ in CLASSES[dataset]:
        terms = [(key, float(row["shared_idf_weight"]) if dataset == "fmow" else 1.0)
                 for key, row in selected.items() if row["class"] == cls]
        if not terms:
            raise ValueError(f"No selected concepts for {cls}")
        total = 0.0
        for key, weight in terms:
            concept_score = score_at_temperature(evidence[key], temperature)
            if home_evidence is not None:
                home_score = score_at_temperature(home_evidence[key], temperature)
                concept_score = (1 - ALPHA_HOME) * concept_score + ALPHA_HOME * home_score
            total += concept_score * weight
        scores[cls] = total / sum(weight for _, weight in terms)
    return scores


def prediction_rows(sites: list[Site], shards: dict[str, dict],
                    descriptions: dict[str, dict[str, dict]], dataset: str,
                    model_id: str) -> list[dict]:
    rows = []
    class_to_label = {item[1]: item[3] for item in CLASSES[dataset]}
    for site in sites:
        shard = shards.get(site.filename)
        if shard is None:
            continue
        entries = descriptions[site.country]
        for temperature in TEMPERATURES:
            for method, evidence, home_evidence, alpha_home in (
                ("global_mean", shard["global"], None, 0.0),
                ("selective_home_mean", shard["home"], None, 1.0),
                ("aperture_blend", shard["global"], shard["home"], ALPHA_HOME),
            ):
                scores = class_scores(dataset, entries, evidence, temperature, home_evidence)
                best = max(scores, key=scores.get)
                rows.append({"image_file": site.filename, "country": site.country,
                             "true": site.label, "method": method,
                             "temperature": temperature, "alpha_home": alpha_home,
                             "pred": class_to_label[best],
                             "model": model_id,
                             **{f"score_{cls}": f"{value:.8f}" for cls, value in scores.items()}})
    return rows


def write_predictions(path: Path, rows: list[dict], dataset: str) -> None:
    columns = ["image_file", "country", "true", "method", "temperature", "alpha_home",
               "pred", "model", *(f"score_{item[1]}" for item in CLASSES[dataset])]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=("sifc", "fmow"))
    parser.add_argument("--model", required=True, choices=tuple(MODEL_IDS))
    parser.add_argument("--images", type=Path, required=True,
                        help="directory of coordinate/year PNGs from download_arcgis_images.py")
    parser.add_argument("--csv", type=Path, help="override the release site CSV")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "selective")
    parser.add_argument("--model-id", help="override the default model ID")
    parser.add_argument("--dry-run", action="store_true", help="validate without loading a model or writing outputs")
    parser.add_argument("--limit", type=int, help="score at most this many pending images")
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--home-padding", type=float, default=0.10)
    parser.add_argument("--batch", type=int, default=32, help="maximum local vLLM requests per call")
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES for local models")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpus is None:
        args.gpus = "0,1,2,3" if args.model in {"qwen", "glm"} else "0"
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 4 if args.model in {"qwen", "glm"} else 1
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.batch < 1 or args.tensor_parallel_size < 1:
        raise ValueError("Batch size and tensor parallel size must be positive")
    if not (0 < args.route_temperature and math.isfinite(args.route_temperature)):
        raise ValueError("Route temperature must be positive and finite")
    if not 0 <= args.home_padding <= 0.25:
        raise ValueError("Home padding must be between 0 and 0.25")
    csv_file = args.csv or CSV_FILES[args.dataset]
    sites = read_sites(csv_file, args.images, args.year, args.dataset)
    descriptions = load_descriptors(args.dataset)
    unknown = {site.country for site in sites} - set(descriptions)
    if unknown:
        raise ValueError(f"Countries missing from descriptor inventory: {sorted(unknown)}")
    model_id = args.model_id or MODEL_IDS[args.model]
    identity = {"dataset": args.dataset, "model": args.model, "model_id": model_id,
                "year": args.year, "route_temperature": args.route_temperature,
                "home_padding": args.home_padding,
                "tensor_parallel_size": args.tensor_parallel_size,
                "csv_sha256": sha256_file(csv_file),
                "descriptor_sha256": sha256_file(DESCRIPTOR_FILES[args.dataset]),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "scoring": "fresh_global_top1_home_v1"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run_dir = args.output_dir / f"{args.dataset}_{args.model}"
    existing = {}
    for site in sites:
        path = shard_path(run_dir, site.filename)
        if path.exists():
            if not site.path.is_file():
                raise FileNotFoundError(f"Scored image is missing: {site.filename}")
            existing[site.filename] = read_shard(
                path, site, descriptions[site.country], args.dataset, fingerprint)
    pending = [site for site in sites if site.filename not in existing]
    if args.limit is not None:
        pending = pending[:args.limit]
    missing = [site.filename for site in pending if not site.path.is_file()]
    counts = {country: expected_calls(args.dataset, rows) for country, rows in descriptions.items()}
    print(f"dataset={args.dataset} model={model_id} sites={len(sites)} completed={len(existing)} pending={len(pending)}")
    print(f"global/route/new-home requests per image: {counts}")
    print(f"missing pending images={len(missing)}")
    if missing:
        print(f"first missing={missing[0]}")
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(f"{len(missing)} pending images are missing")
    if pending:
        engine = GeminiEngine(args, model_id) if args.model == "gemini" else VllmEngine(args, model_id)
        for index, site in enumerate(pending, start=1):
            result = score_site(site, descriptions[site.country], args.dataset,
                                engine, args.home_padding, fingerprint)
            result["image_sha256"] = sha256_file(site.path)
            write_json_atomic(shard_path(run_dir, site.filename), result)
            existing[site.filename] = result
            print(f"[{index}/{len(pending)}] {site.country} {site.filename} scored", flush=True)
    predictions = prediction_rows(sites, existing, descriptions, args.dataset, model_id)
    output = run_dir / "predictions.csv"
    write_predictions(output, predictions, args.dataset)
    print(f"completed={len(existing)}/{len(sites)} prediction_rows={len(predictions)} output={output.name}")


if __name__ == "__main__":
    main()

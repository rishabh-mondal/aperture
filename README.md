# aperture
Training-Free CBM

## SiFC imagery retrieval

The SiFC site list is in `SiFC data/sifc_dataset.csv`. The downloader uses its
`center_latitude` and `center_longitude` columns to retrieve 4096 × 4096 pixel
images at zoom 18 from Esri World Imagery Wayback. It processes every CSV row;
the corner columns are provided as geographic metadata.

Install the two Python dependencies and run from the repository root:

```bash
python -m pip install requests Pillow
python scripts/download_arcgis_images.py --output downloaded_images --dry-run
python scripts/download_arcgis_images.py --output downloaded_images
```

The default Wayback year is 2026. Use `--year`, `--zoom`, `--crop-size`, or
`--workers` to change the download settings. Completed images are skipped on
subsequent runs, and `download_report.csv` records the status of each CSV row.
Downloaded imagery stays in the ignored `downloaded_images/` directory.

The same downloader accepts the fMoW site list through `--csv`. Keep the two
image sets in separate directories:

```bash
python scripts/download_arcgis_images.py --csv "SiFC data/sifc_dataset.csv" --output downloaded_images/sifc
python scripts/download_arcgis_images.py --csv "fMoW data/fmow_dataset.csv" --output downloaded_images/fmow
```

## General concept descriptors (GCD)

`descriptors/gcd_descriptors.json` contains the 44 structured general concept
descriptors described in Appendix A.1 of the paper. Each entry includes the
concept, its applicable classes, visual appearance, likely confusers, and a
verification question template with a `{scale_text}` placeholder. The same
concept-level descriptor is used across countries.

## Class-conditioned descriptors

`descriptors/class_conditioned_descriptors.json` contains the 113
`class::concept` entries from the class-conditioning stage in Appendix A.1,
Section C. Each value has exactly the three generated fields: `appearance`,
`confusers`, and `placement`. The class and concept are identified by the key;
these descriptions are not country-conditioned.

## Home-scale assignments

`descriptors/home_scale_assignments.json` records the 44 concept-level crop
widths used for the home-scale assignment in Appendix A.1, Section G. Each
entry contains `home_scale_px`, one of 256, 512, 1024, 2048, or 4096 source
pixels. The selected crop is resized to 224 × 224 pixels for verification.
The assignments are shared across classes and countries; they are assigned
settings, not experimentally established optima.

## Class- and country-conditioned descriptors (CCCD)

`descriptors/cccd_descriptors.json` contains the CCCD inventory described in
Appendix A.1: 113 class–concept entries for each of China, India, and the USA
(339 entries in total). Each entry gives appearance, confusers, placement,
home scale, and the saved global, routing, and home verification questions.
The concept-selection flags are included. Image country selects the descriptor
inventory. Question text is preserved from the saved prompts; internal run
status and provenance fields are omitted from this release JSON.

## Final inference prompts

`descriptors/final_inference_prompts.json` contains the saved questions described
in Appendix A.1, Sections I–K: global yes/no verification, numbered-quadrant
routing, and home-scale yes/no verification. Each of its 339
`country::class::concept` entries has `question_global`,
`question_route_by_scale`, and `question_home`. Routing questions are indexed
by source crop width and stop before the assigned home scale. The questions
are copied from the corresponding CCCD entries, including the distinct home
wording for object- and tile-level concepts.

## Zero-shot baselines

`scripts/zero_shot_baselines.py` runs the original 224 × 224 pixel,
letter-choice baseline prompt for either dataset with `--dataset sifc` or
`--dataset fmow`. Select `--model gemma`, `gemini`, `qwen`, or `glm`. The SiFC
prompt has six choices; the fMoW prompt has five. GLM uses the Qwen-style
first-token scoring path with its saved boxed-answer prefix. The script reads
coordinate/year PNG filenames from the downloader, so `--year` must match the
download year. It does not use the CSV label to construct the prompt.

Check inputs without loading a model or making API calls:

```bash
python scripts/zero_shot_baselines.py --dataset sifc --model gemma --images downloaded_images/sifc --dry-run --show-prompt
python scripts/zero_shot_baselines.py --dataset fmow --model glm --images downloaded_images/fmow --dry-run --show-prompt
```

Remove `--dry-run` to infer. Gemma, Qwen, and GLM require Pillow and vLLM with
the respective model available. GLM can use `--gpus 0,1,2,3 --tensor-parallel-size 4`. Gemini requires Pillow, `google-genai`, `pydantic`,
and `GEMINI_API_KEY` in the environment. Gemini returns one selected class;
local models also record normalized answer-letter scores. Predictions resume
from `results/<dataset>_<model>_predictions.csv`, with one row per image. Use
`--limit` for a small run and `--model-id` to override a default model ID.
No previous prediction files are reused.

## Selective multiscale inference

`scripts/selective_multiscale.py` runs the country-selected concept search with
`--dataset sifc` or `--dataset fmow` and `--model gemma`, `gemini`, `qwen`, or
`glm`. It scores all global questions fresh, routes each selected concept by
top-1 quadrant choice on an image with a numbered grid, and verifies the home
crop from the original image without the grid. A global “no” does not stop
routing. Saved site evidence resumes safely, and predictions use the fixed
yes/no concept-score temperature τ = 30 for both datasets and all four models.
Results are written under the ignored `results/selective/`. Quadrant ranking
uses a separate routing temperature (default 1); model answer generation uses
its own decoding settings.

SiFC uses the 339-entry `descriptors/cccd_descriptors.json`, 100 selected
class–concept pairs per country, and the saved concept-specific home widths.
The 4096-pixel home questions reuse same-run global evidence. fMoW uses the
75-entry `descriptors/fmow_cccd_descriptors.json`, 25 pairs per country, and
a relative home window at one quarter of each image dimension with 10% context
padding. Its home width is provisional; the fMoW descriptors do not define a
calibrated pixel width. fMoW class scores use the saved shared-concept weights.

```bash
python scripts/selective_multiscale.py --dataset sifc --model gemma --images downloaded_images/sifc --dry-run
python scripts/selective_multiscale.py --dataset fmow --model glm --images downloaded_images/fmow --dry-run
```

Remove `--dry-run` to run inference. `--limit 1` processes one pending image;
`--gpus` and `--tensor-parallel-size` configure local vLLM models.
Qwen and GLM default to one model spread across four GPUs; Gemma defaults to
one GPU. The selective Qwen model is Qwen3.5-122B-A10B, matching its source
runner. SiFC Gemini uses answer-token log probabilities; fMoW Gemini uses a
structured quadrant ranking and self-reported presence confidence, matching
its separate source protocol. Gemini calls require `GEMINI_API_KEY` and use
the standard API.

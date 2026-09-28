# APERTURE: Training-Free Multiscale Concept Bottlenecks for Remote Sensing

## Abstract

While earth observation models have advanced substantially, they still lack
interpretability. While concept-bottleneck models provide interpretability and
expert interaction, they are either too expensive to train for the remote sensing
domain or perform poorly without annotation. We posit that in expert domains like
remote sensing, such training-free models require both fine details in both image
and concept space. In image space, we propose a multiscale concept bottleneck
using greedy quadtree routing to locate small concepts. In concept space, we
replace contrastive vision language models with pre-trained MLLMs and present a
way to get reliable concept scores from them. We introduce APERTURE that blends
concept scores at the global image and native concept-scale level to give
state-of-the-art training-free model performance. To test these models, introduce
SiFC, a fine-grained concept-centric dataset across three countries, with
human-reviewed class-level concept maps. On SiFC, APERTURE outperforms the best
training-free baselines by more than 10 percentage points in macro F1-score, and
notably also outperforms supervised concept bottleneck models. Targeted
component-removal tests examine whether concept scores respond to changes in
visual evidence, while temporal experiments show that descriptor updates improve
recognition of technological changes without retraining. Our data and code are
available at https://anonymous.4open.science/r/aperture-C1984/.

## What is here

| Path | Contents |
| --- | --- |
| `SiFC data/sifc_dataset.csv` | SiFC sites and coordinates |
| `fMoW data/fmow_dataset.csv` | fMoW sites and center coordinates |
| `descriptors/gcd_descriptors.json` | 44 general concepts |
| `descriptors/class_conditioned_descriptors.json` | 113 class–concept descriptions |
| `descriptors/home_scale_assignments.json` | 44 assigned home widths |
| `descriptors/cccd_descriptors.json` | 339 country–class–concept entries for SiFC |
| `descriptors/fmow_cccd_descriptors.json` | 75 country–class–concept entries for fMoW |
| `descriptors/final_inference_prompts.json` | Global, routing, and home questions for SiFC |

## Run from the repository root

Download 4096 × 4096 imagery from Esri World Imagery Wayback (zoom 18, default year 2026). Check the CSV first, then remove `--dry-run` to download:

```bash
python -m pip install requests Pillow
python scripts/download_arcgis_images.py --csv "SiFC data/sifc_dataset.csv" --output downloaded_images/sifc --dry-run
python scripts/download_arcgis_images.py --csv "fMoW data/fmow_dataset.csv" --output downloaded_images/fmow --dry-run
```

After downloading, check a baseline and APERTURE run:

```bash
python scripts/zero_shot_baselines.py --dataset sifc --model gemma --images downloaded_images/sifc --dry-run
python scripts/selective_multiscale.py --dataset sifc --model gemma --images downloaded_images/sifc --dry-run
```

Both runners also accept `--dataset fmow --images downloaded_images/fmow`. Choose `gemma`, `gemini`, `qwen`, or `glm` with `--model`. Remove `--dry-run` to infer; `--limit 1` is a small first run. The download `--year` and inference `--year` must agree. Local models use vLLM; Gemini needs `google-genai`, `pydantic`, and `GEMINI_API_KEY`.

APERTURE uses the paper's fixed concept temperature **τ = 30** and blends each concept's global and home probabilities as **0.75 × global + 0.25 × home**. It then averages concept scores per class (weighted by the saved shared-concept weights for fMoW). The selective output also reports global-only and home-only comparisons. SiFC uses assigned pixel widths; fMoW uses a provisional quarter-image home window with 10% padding.

## SiFC part removal

`SiFC data/intervention_90_manifest.csv` maps 90 intact images to three cumulative removal steps each (360 images). Place the edited images under a data root with `cbm_data/` and `cbm_data_inpainted_bbox_additive/`, then run:

```bash
python scripts/intervention_90.py --data-root /path/to/intervention_90 --stage validate
python scripts/intervention_90.py --data-root /path/to/intervention_90 --stage zero-shot --dry-run
python scripts/intervention_90.py --data-root /path/to/intervention_90 --stage selective --dry-run
python scripts/evaluate_intervention_90.py
```

Run both scoring stages without `--dry-run` before evaluation. The report measures true-class score drops from the intact image, using original-image present concepts (`p(yes) > 0.5`) across all steps. It uses τ = 30 and a home weight of 0.25;
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

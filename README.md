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

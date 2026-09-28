#!/usr/bin/env python3
"""Download centered SiFC crops from Esri World Imagery Wayback.

The input CSV must contain center_latitude and center_longitude. The default
settings match the SiFC image geometry: zoom 18 and 4096 by 4096 pixels.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


YEAR_TO_TIME_ID = {
    2014: 10,
    2016: 6984,
    2018: 2168,
    2019: 4756,
    2020: 18289,
    2021: 9812,
    2022: 10321,
    2023: 25982,
    2024: 41468,
    2025: 34007,
    2026: 49059,
}
TILE_SIZE = 256
WEB_MERCATOR_MAX_LATITUDE = 85.05112878
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "SiFC data" / "sifc_dataset.csv"
_thread_state = threading.local()


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    filename: str


def read_sites(csv_path: Path, year: int) -> list[Site]:
    """Read every site and reject coordinates that cannot be downloaded."""
    sites: list[Site] = []
    filenames: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"center_latitude", "center_longitude"} <= set(reader.fieldnames):
            raise ValueError("CSV needs center_latitude and center_longitude columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                latitude = float(row["center_latitude"])
                longitude = float(row["center_longitude"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinate on CSV line {line_number}") from exc
            if not (math.isfinite(latitude) and math.isfinite(longitude)):
                raise ValueError(f"Non-finite coordinate on CSV line {line_number}")
            if not (-WEB_MERCATOR_MAX_LATITUDE < latitude < WEB_MERCATOR_MAX_LATITUDE):
                raise ValueError(f"Latitude outside Web Mercator on CSV line {line_number}")
            if not (-180.0 <= longitude < 180.0):
                raise ValueError(f"Longitude outside [-180, 180) on CSV line {line_number}")
            filename = f"{latitude:.6f}_{longitude:.6f}_{year}.png"
            if filename in filenames:
                raise ValueError(f"Duplicate image filename on CSV line {line_number}: {filename}")
            filenames.add(filename)
            sites.append(Site(latitude, longitude, filename))
    if not sites:
        raise ValueError("CSV contains no image rows")
    return sites


def lonlat_to_global_pixel(longitude: float, latitude: float, zoom: int) -> tuple[float, float]:
    """Use the same Web Mercator pixel projection as the source downloader."""
    world_pixels = (2**zoom) * TILE_SIZE
    latitude_radians = math.radians(latitude)
    x = (longitude + 180.0) / 360.0 * world_pixels
    y = (
        1.0
        - math.log(math.tan(latitude_radians) + 1.0 / math.cos(latitude_radians)) / math.pi
    ) / 2.0 * world_pixels
    return x, y


def crop_bounds(site: Site, zoom: int, crop_size: int) -> tuple[int, int, int, int]:
    """Return the crop's outer pixel edges, matching the source's rounded center."""
    x, y = lonlat_to_global_pixel(site.longitude, site.latitude, zoom)
    left = round(x) - crop_size // 2
    top = round(y) - crop_size // 2
    return left, top, left + crop_size, top + crop_size


def session_for_thread() -> requests.Session:
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_state.session = session
    return session


def fetch_tile(time_id: int, zoom: int, tile_x: int, tile_y: int) -> Image.Image:
    """Fetch one tile; a missing or corrupt tile fails the image download."""
    tile_count = 2**zoom
    if not (0 <= tile_y < tile_count):
        raise ValueError("Crop extends beyond the Web Mercator tile grid")
    url = (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
        "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
        f"{time_id}/{zoom}/{tile_y}/{tile_x % tile_count}"
    )
    response = session_for_thread().get(url, timeout=(10, 30))
    if response.status_code != 200 or not response.content:
        raise RuntimeError(f"Tile returned HTTP {response.status_code}")
    with Image.open(BytesIO(response.content)) as image:
        tile = image.convert("RGB")
    if tile.size != (TILE_SIZE, TILE_SIZE):
        raise ValueError("Tile has an unexpected size")
    return tile


def valid_existing_image(path: Path, crop_size: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (crop_size, crop_size):
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def download_site(site: Site, output_dir: Path, time_id: int, zoom: int, crop_size: int) -> str:
    destination = output_dir / site.filename
    if valid_existing_image(destination, crop_size):
        return "skipped"

    left, top, right, bottom = crop_bounds(site, zoom, crop_size)
    min_tile_x, max_tile_x = left // TILE_SIZE, (right - 1) // TILE_SIZE
    min_tile_y, max_tile_y = top // TILE_SIZE, (bottom - 1) // TILE_SIZE
    stitched = Image.new(
        "RGB",
        ((max_tile_x - min_tile_x + 1) * TILE_SIZE,
         (max_tile_y - min_tile_y + 1) * TILE_SIZE),
    )
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile = fetch_tile(time_id, zoom, tile_x, tile_y)
            stitched.paste(tile, ((tile_x - min_tile_x) * TILE_SIZE,
                                  (tile_y - min_tile_y) * TILE_SIZE))

    crop = stitched.crop((left - min_tile_x * TILE_SIZE,
                          top - min_tile_y * TILE_SIZE,
                          right - min_tile_x * TILE_SIZE,
                          bottom - min_tile_y * TILE_SIZE))
    if crop.size != (crop_size, crop_size):
        raise ValueError("Cropped image has an unexpected size")

    temporary = output_dir / f".{site.filename}.{threading.get_ident()}.part"
    try:
        crop.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with center coordinates")
    parser.add_argument("--output", type=Path, required=True, help="Directory for downloaded PNGs")
    parser.add_argument("--year", type=int, choices=sorted(YEAR_TO_TIME_ID), default=2026)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--crop-size", type=int, default=4096, help="Square crop width in pixels")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Download only the first N CSV rows")
    parser.add_argument("--dry-run", action="store_true", help="Validate input without downloading")
    args = parser.parse_args()
    if args.zoom < 1 or args.crop_size < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("zoom, crop size, workers, and limit must be positive")
    try:
        all_sites = read_sites(args.csv, args.year)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sites = all_sites[:args.limit] if args.limit is not None else all_sites
    print(f"Validated {len(all_sites)} CSV rows; selected {len(sites)} image(s).")
    print(f"Wayback year {args.year}, zoom {args.zoom}, crop {args.crop_size} x {args.crop_size} pixels.")
    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, str] | None] = [None] * len(sites)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(download_site, site, args.output, YEAR_TO_TIME_ID[args.year],
                            args.zoom, args.crop_size): index
            for index, site in enumerate(sites)
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            index = pending[future]
            try:
                status, error = future.result(), ""
            except Exception as exc:
                status, error = "failed", type(exc).__name__
            results[index] = (sites[index].filename, status, error)
            if completed % 25 == 0 or completed == len(sites):
                print(f"Processed {completed}/{len(sites)} images.")

    report_path = args.output / "download_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_file", "status", "error_type"))
        writer.writerows(result for result in results if result is not None)
    counts = {status: sum(result[1] == status for result in results if result) for status in
              ("downloaded", "skipped", "failed")}
    print(f"Downloaded: {counts['downloaded']}; skipped: {counts['skipped']}; failed: {counts['failed']}.")
    print(f"Report: {report_path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

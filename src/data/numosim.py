"""NUMOSIM stay-point loading, including the OSF download.

NUMOSIM ships a native train/test split: four weeks of each agent's stays
(``train``) followed by four more (``test_truth``). The dataset also ships
``test_anomalous``, the test window with single unusual stays injected; it is
not used here, because those are stay-level anomalies while module 2 studies
whole-day behavioural ones, which it splices in exactly as for Geolife.

Demographics are not read: they carry no per-slot signal.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import ROOT

TRAIN_FILES = ("stay_points_train.parquet", "poi.parquet")
TEST_FILES = ("stay_points_test_truth.parquet",)
CORE_FILES = TRAIN_FILES

# Activity-type codes from the NUMOSIM data dictionary (osf.io/sjyfr).
ACTIVITY_CODES = {
    0: "Transportation",
    1: "Home",
    2: "Work",
    3: "School",
    4: "Childcare",
    5: "BuyGoods",
    6: "Services",
    7: "EatOut",
    8: "Errands",
    9: "Recreation",
    10: "Excercise",
    11: "Visit",
    12: "HealthCare",
    13: "Religious",
    14: "SomethingElse",
    15: "DropOff",
}


def _list_osf_files(api_url: str, found: dict | None = None) -> dict:
    """Recursively map filename -> download URL for an OSF storage endpoint."""
    if found is None:
        found = {}
    url = api_url
    while url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("data", []):
            attrs = item["attributes"]
            if attrs["kind"] == "file":
                found[attrs["name"]] = item["links"]["download"]
            elif attrs["kind"] == "folder":
                _list_osf_files(
                    item["relationships"]["files"]["links"]["related"]["href"], found
                )
        url = payload.get("links", {}).get("next")
    return found


def _stream_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"  {dest.name}: {100 * downloaded / total:.1f}% "
                        f"({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)",
                        end="\r",
                    )
    print()


def download(
    raw_dir: str | Path,
    osf_project: str = "sjyfr",
    files: tuple[str, ...] = TRAIN_FILES,
) -> None:
    """Fetch NUMOSIM parquet files from OSF. Existing files are kept.

    Template discovery needs only ``TRAIN_FILES``; anomaly detection also needs
    ``TEST_FILES``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in files if not (raw_dir / name).exists()]
    if not missing:
        shown = raw_dir.relative_to(ROOT) if raw_dir.is_relative_to(ROOT) else raw_dir
        print(f"NUMOSIM: all {len(files)} files already in {shown}")
        return

    print(f"Listing OSF project {osf_project} ...")
    available = _list_osf_files(
        f"https://api.osf.io/v2/nodes/{osf_project}/files/osfstorage/"
    )
    for name in missing:
        if name not in available:
            raise FileNotFoundError(f"{name} not found in OSF project {osf_project}")
        print(f"  downloading {name} ...")
        _stream_download(available[name], raw_dir / name)


def load_stays(
    raw_dir: str | Path,
    train_start: date | None = None,
    train_end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stay points (optionally date-filtered) and the POI table."""
    raw_dir = Path(raw_dir)
    stays = pd.read_parquet(raw_dir / "stay_points_train.parquet")
    pois = pd.read_parquet(raw_dir / "poi.parquet")

    if train_start is not None and train_end is not None:
        days = pd.to_datetime(stays["start_datetime"]).dt.date
        stays = stays[(days >= train_start) & (days <= train_end)].reset_index(drop=True)

    return stays, pois

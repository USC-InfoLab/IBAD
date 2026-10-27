"""Load the labelled Geolife splits produced by module 0 (``split_dataset``)."""

from pathlib import Path

import pandas as pd

# Geolife labels onto the NHTS vocabulary. Identity for every category except
# the spelling of Exercise, which NHTS writes 'Excercise'; aligning here keeps a
# single shared column order across both datasets.
GEOLIFE_TO_NHTS = {
    "Home": "Home",
    "Work": "Work",
    "Education": "Education",
    "Childcare": "Childcare",
    "Adultcare": "Adultcare",
    "Shopping": "Shopping",
    "Services": "Services",
    "Errands": "Errands",
    "Recreation": "Recreation",
    "Restaurant": "Restaurant",
    "Religious": "Religious",
    "Social": "Social",
    "Exercise": "Excercise",
    "Excercise": "Excercise",
    "Health": "Health",
    "Transportation": "Transportation",
    "Other": "Other",
    "Pickup_Dropoff": "Pickup_Dropoff",
    "Volunteer": "Volunteer",
}


def find_split(processed_dir: str | Path, split: str) -> Path:
    """Locate one split TSV. The agent count in the filename is set at split time."""
    processed_dir = Path(processed_dir)
    matches = sorted(processed_dir.glob(f"{split}-*-agents-*.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"No '{split}' split in {processed_dir}. "
            "Run: python -m src.data.geolife.split_dataset"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple '{split}' splits in {processed_dir}: {matches}")
    return matches[0]


def load_split(processed_dir: str | Path, split: str) -> pd.DataFrame:
    """Read one split and map its activity labels onto the NHTS vocabulary."""
    df = pd.read_csv(find_split(processed_dir, split), sep="\t")
    df["ArrivingTime"] = pd.to_datetime(df["ArrivingTime"], utc=False)
    df["LeavingTime"] = pd.to_datetime(df["LeavingTime"], utc=False)

    unmapped = set(df["ActivityCategory"].unique()) - set(GEOLIFE_TO_NHTS)
    if unmapped:
        raise ValueError(f"Unmapped Geolife categories: {sorted(unmapped)}")

    df["nhts_category"] = df["ActivityCategory"].map(GEOLIFE_TO_NHTS)
    df["split"] = split
    return df


def activity_vocabulary(stays: pd.DataFrame) -> list[str]:
    """Sorted activity categories present in the split, always including Home."""
    return sorted(set(stays["nhts_category"].unique()) | {"Home"})

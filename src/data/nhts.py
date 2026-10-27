"""NHTS 2017 trip loading and activity-code preprocessing.

Only the trip file is read: template discovery encodes travel days, and the
person/household files carry no per-slot signal.
"""

from pathlib import Path

import pandas as pd

# WHYFROM / WHYTO survey codes.
# Reference: NHTS 2017 codebook — https://nhts.ornl.gov/assets/codebook_v1.2.pdf
ACTIVITY_CODES = {
    1: "HOME",
    2: "WORK_HOME",
    3: "WORK",
    4: "WORK_RELATED",
    5: "VOLUNTEER",
    6: "PICKUP_DROPOFF",
    7: "CHANGE_MODE",
    8: "SCHOOL",
    9: "CHILDCARE",
    10: "ADULTCARE",
    11: "SHOPPING",
    12: "SERVICES",
    13: "MEALS",
    14: "ERRANDS",
    15: "RECREATION",
    16: "EXCERCISE",
    17: "SOCIAL",
    18: "HEALTHCARE",
    19: "RELIGIOUS",
    97: "OTHER",
    -7: "REFUSED",
    -8: "DONT_KNOW",
    -9: "NOT_ASCERTAINED",
}

# Survey codes collapsed into the 18 canonical activity categories. 'Excercise'
# keeps the NHTS spelling; the Geolife loader normalises onto it so both
# datasets share one vocabulary and column order.
ACTIVITY_CATEGORIES = {
    "Home": ["HOME"],
    "Work": ["WORK", "WORK_HOME", "WORK_RELATED"],
    "Education": ["SCHOOL"],
    "Childcare": ["CHILDCARE"],
    "Adultcare": ["ADULTCARE"],
    "Shopping": ["SHOPPING"],
    "Services": ["SERVICES"],
    "Errands": ["ERRANDS"],
    "Recreation": ["RECREATION"],
    "Restaurant": ["MEALS"],
    "Religious": ["RELIGIOUS"],
    "Social": ["SOCIAL"],
    "Excercise": ["EXCERCISE"],
    "Health": ["HEALTHCARE"],
    "Pickup_Dropoff": ["PICKUP_DROPOFF"],
    "Transportation": ["CHANGE_MODE"],
    "Volunteer": ["VOLUNTEER"],
    "Other": ["OTHER", "REFUSED", "DONT_KNOW", "NOT_ASCERTAINED"],
}

ACTIVITY_TO_CATEGORY = {
    activity: category
    for category, activities in ACTIVITY_CATEGORIES.items()
    for activity in activities
}

TRIP_COLUMNS = [
    "HOUSEID", "PERSONID", "TDTRPNUM",
    "STRTTIME", "ENDTIME", "DWELTIME",
    "WHYFROM", "WHYTO",
]


def load_trips(raw_dir: str | Path) -> pd.DataFrame:
    """Read trippub.csv, keeping only the columns template discovery needs."""
    path = Path(raw_dir) / "trippub.csv"
    return pd.read_csv(path, usecols=TRIP_COLUMNS)


def preprocess_trips(trips: pd.DataFrame, keep_other: bool = True) -> pd.DataFrame:
    """Map activity codes to categories and attach a person-day identifier.

    Trips with a non-positive dwell time are dropped as invalid. When
    ``keep_other`` is False, trips whose origin or destination is an
    Other/Unknown code are dropped rather than folded into 'Other'.
    """
    df = trips[trips["DWELTIME"] > 0].copy()

    df["ORIGIN_ACTIVITY"] = df["WHYFROM"].map(ACTIVITY_CODES).fillna("UNKNOWN")
    df["DEST_ACTIVITY"] = df["WHYTO"].map(ACTIVITY_CODES).fillna("UNKNOWN")

    if not keep_other:
        other = ACTIVITY_CATEGORIES["Other"] + ["UNKNOWN"]
        df = df[
            ~df["ORIGIN_ACTIVITY"].isin(other) & ~df["DEST_ACTIVITY"].isin(other)
        ].copy()

    df["ORIGIN_CATEGORY"] = df["ORIGIN_ACTIVITY"].map(
        lambda a: ACTIVITY_TO_CATEGORY.get(a, "Other")
    )
    df["DEST_CATEGORY"] = df["DEST_ACTIVITY"].map(
        lambda a: ACTIVITY_TO_CATEGORY.get(a, "Other")
    )

    # Each NHTS person is surveyed for exactly one travel day, so household +
    # person already identifies a person-day.
    df["PERSON_DAY_ID"] = (
        df["HOUSEID"].astype(str) + "_" + df["PERSONID"].astype(str)
    )
    return df


def activity_vocabulary(trips_df: pd.DataFrame) -> list[str]:
    """Sorted activity categories present in the data, always including Home.

    Built from the data rather than from all 18 categories so that filtered-out
    categories do not become dead all-zero columns. Both origin and destination
    categories count, because travel slots are encoded with the origin activity.
    """
    present = set(trips_df["ORIGIN_CATEGORY"]) | set(trips_df["DEST_CATEGORY"])
    present.add("Home")
    return sorted(present)


def person_day_ids(trips_df: pd.DataFrame) -> list[str]:
    """Ordered person-day identifiers, one per document row."""
    return sorted(trips_df["PERSON_DAY_ID"].unique())

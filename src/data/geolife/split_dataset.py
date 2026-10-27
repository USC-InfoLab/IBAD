"""Stage 6 — 80/10/10 per-agent chronological split (clean, no needle swap).

Output schema (each of train/val/test):
    AgentID, ArrivingTime, LeavingTime,
    Latitude, Longitude, Duration_Min, ActivityCategory

Run from repo root:
    python -m src.data.geolife.split_dataset
"""

import os

import pandas as pd

from src.utils.config import ROOT, load_cfg

def main():
    cfg = load_cfg("configs/data/geolife_poi.yaml")

    input_tsv   = ROOT / cfg["labeled_tsv"]
    output_dir  = ROOT / cfg["processed_dir"]
    min_records = cfg["min_records_split"]
    train_frac  = cfg["train_frac"]
    val_frac    = cfg["val_frac"]

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_tsv, sep="\t")
    df["ArrivingTime"] = pd.to_datetime(df["ArrivingTime"])
    df = df.sort_values(["AgentID", "ArrivingTime"]).reset_index(drop=True)

    counts = df["AgentID"].value_counts()
    eligible = counts[counts >= min_records]
    total_agents = len(eligible)
    if total_agents < 1:
        raise RuntimeError(
            f"Only {total_agents} agent(s) with >= {min_records} staypoints — "
            "lower min_records_split or check upstream stages."
        )
    df = df[df["AgentID"].isin(eligible.index)].reset_index(drop=True)
    print(f"eligible agents: {total_agents}  retained staypoints: {len(df):,}")

    # Per-agent chronological split.
    train_parts, val_parts, test_parts = [], [], []
    for _, g in df.groupby("AgentID", sort=False):
        n = len(g)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train_parts.append(g.iloc[:t1])
        val_parts.append(g.iloc[t1:t2])
        test_parts.append(g.iloc[t2:])
    train = pd.concat(train_parts, ignore_index=True)
    val   = pd.concat(val_parts,   ignore_index=True)
    test  = pd.concat(test_parts,  ignore_index=True)

    cols = ["AgentID", "ArrivingTime", "LeavingTime",
            "Latitude", "Longitude", "Duration_Min", "ActivityCategory"]
    if "LLMReasoning" in train.columns:
        cols.append("LLMReasoning")
    test_frac = round(1 - train_frac - val_frac, 2)
    suffix = f"{total_agents}-agents-{train_frac:g}-{val_frac:g}-{test_frac:g}"
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(output_dir, f"{name}-{suffix}.tsv")
        part[cols].to_csv(path, sep="\t", index=False)
        print(f"  {name}: {len(part):,} rows -> {path}")

    print("\n=== Row counts ===")
    print(f"  train: {len(train):,}")
    print(f"  val:   {len(val):,}")
    print(f"  test:  {len(test):,}")


if __name__ == "__main__":
    main()

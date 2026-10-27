"""Stage 2 — Stay-point extraction via trackintel.

Reads the raw Geolife PLT tree, runs trackintel's sliding-window stay-point
detector (Li et al. 2008), normalizes columns, filters to the Beijing bbox,
and writes the staypoints TSV.

Run from repo root:
    python -m src.data.geolife.extract_staypoints
"""

import os
import time

import trackintel as ti

from src.utils.config import ROOT, load_cfg

def main():
    cfg = load_cfg("configs/data/geolife_poi.yaml")

    dist_threshold_m        = cfg["dist_threshold_m"]
    time_threshold_min      = cfg["time_threshold_min"]
    gap_threshold_min       = cfg["gap_threshold_min"]
    min_records_per_user    = cfg["min_records_per_user"]
    min_staypoints_per_user = cfg["min_staypoints_per_user"]
    geolife_path            = ROOT / cfg["raw_path"]
    output_tsv              = ROOT / cfg["staypoints_tsv"]
    bbox                    = cfg["bbox"]

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Reading Geolife from {geolife_path} ...")
    pfs, _ = ti.io.read_geolife(geolife_path, print_progress=True)

    print(f"  positionfixes: {len(pfs):,}  agents: {pfs['user_id'].nunique()}")

    if bbox is not None:
        before = len(pfs)
        lats = pfs.geometry.y
        lons = pfs.geometry.x

        pfs = pfs[
            lats.between(bbox["min_lat"], bbox["max_lat"])
            & lons.between(bbox["min_lon"], bbox["max_lon"])
        ].copy()

        # filter users with fewer than x records.
        pfs = pfs.groupby("user_id").filter(lambda g: len(g) > min_records_per_user).copy()
        print(f"  positionfixes inside bbox: {len(pfs):,} / {before:,} "
              f"({100 * len(pfs) / max(before, 1):.1f}%)  "
              f"agents now: {pfs['user_id'].nunique()}")
    else:
        print("  positionfix bbox filter: OFF")

    print(f"[{time.strftime('%H:%M:%S')}] Generating staypoints "
          f"(dist={dist_threshold_m} m, time={time_threshold_min} min, "
          f"gap={gap_threshold_min} min) ...")
    pfs, sp = pfs.generate_staypoints(
        method="sliding",
        dist_threshold=dist_threshold_m,
        time_threshold=time_threshold_min,
        gap_threshold=gap_threshold_min,
        print_progress=True,
    )
    print(f"  raw staypoints: {len(sp):,}")

    sp["Latitude"]     = sp.geometry.y
    sp["Longitude"]    = sp.geometry.x
    sp["Duration_Min"] = (sp["finished_at"] - sp["started_at"]).dt.total_seconds() / 60.0
    sp = sp.rename(columns={
        "user_id":     "AgentID",
        "started_at":  "ArrivingTime",
        "finished_at": "LeavingTime",
    })

    sp['ArrivingTime'] = sp['ArrivingTime'].dt.tz_convert('Asia/Shanghai') # Beijing local time is UTC+8.
    sp['LeavingTime'] = sp['LeavingTime'].dt.tz_convert('Asia/Shanghai') # Beijing local time is UTC+8.

    out = sp[[
        "AgentID", "ArrivingTime", "LeavingTime",
        "Latitude", "Longitude", "Duration_Min",
    ]].sort_values(["AgentID", "ArrivingTime"]).reset_index(drop=True)

    out = out.groupby("AgentID").filter(lambda g: len(g) > min_staypoints_per_user).copy()
    out.to_csv(output_tsv, sep="\t", index=False)
    print(f"  wrote {output_tsv}")

    counts = out["AgentID"].value_counts()
    print("\n=== Stay-point summary ===")
    print(f"  total staypoints       : {len(out):,}")
    print(f"  unique agents          : {counts.size}")

    print(f"  duration_min  mean/med : "
            f"{out['Duration_Min'].mean():.1f} / {out['Duration_Min'].median():.1f}")
    print(f"  per-agent     mean/med : "
            f"{counts.mean():.1f} / {counts.median():.1f}")
    print(f"  elapsed                : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

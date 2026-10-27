"""Stage 3b — Top-k nearest activity-relevant OSM POIs per centroid.

For every unique 4 dp centroid in `staypoints.tsv`, query the
Overpass API for activity-relevant POIs within `R_PRIMARY` metres
(extended to `R_FALLBACK` metres on cells that come back empty).
Keep the `k` nearest by geodesic distance and cache them in a sibling
table `osm_neighbors` inside the osm_db so the
join with the existing `osm_cache` is by `(lat_r, lon_r)`.

This stage replaces the passive grid-scan in
`LabelWithLLM.neighbor_pois()` with a true spatial k-NN against the
full OSM POI set. Stage 4 consumes the result.

Run from repo root:
    python -m src.data.geolife.fetch_osm_neighbors

Re-runnable — centroids already in the cache are skipped.
"""

import json
import sys
import time

import pandas as pd
import requests

from src.utils.config import ROOT, load_cfg
from src.utils.geometry import haversine_m
from src.utils.sqlite_utils import open_sqlite

# --- Overpass ----------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# --- Activity-relevant filter set (must mirror the prompt cheat sheet
#     in NHTSLLMJudge.SYSTEM_PROMPT and the LabelWithLLM exclusion list)

# Top-level OSM keys whose presence makes a feature an "activity-relevant"
# candidate. `highway`, `man_made`, `place`, `boundary`, and `building=yes`
# are deliberately omitted — those are the noisy primaries the neighbour
# list is meant to disambiguate.
KEEP_KEYS = ("amenity", "shop", "office", "leisure", "tourism",
             "healthcare", "craft", "sport")

# `building=*` is a special case: we keep it ONLY for the activity-bearing
# subset below, and never on its own (`building=yes` is low-confidence).
BUILDING_KEEP = (
    "apartments", "house", "residential", "dormitory", "detached", "terrace",
    "school", "university", "college", "kindergarten",
    "hospital",
    "office", "commercial", "industrial", "retail", "warehouse", "factory",
    "church", "mosque", "temple", "synagogue", "cathedral", "chapel", "shrine",
)

# Sub-filter inside `amenity`: drop GPS-snap targets that aren't really
# destinations.
AMENITY_DROP = frozenset({
    "yes"
})


def _overpass_query(lat, lon, radius_m):
    """Return the Overpass QL body for one (lat, lon, radius)."""
    bldg_re = "|".join(BUILDING_KEEP)
    parts = [
        f'  nwr(around:{radius_m},{lat},{lon})[{k}];'
        for k in KEEP_KEYS
    ]
    parts.append(
        f'  nwr(around:{radius_m},{lat},{lon})[building~"^({bldg_re})$"];'
    )
    body = "\n".join(parts)
    return (
        "[out:json][timeout:25];\n"
        f"(\n{body}\n);\n"
        "out tags center 50;\n"
    )


# --- DB ----------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS osm_neighbors (
    lat_r       REAL NOT NULL,
    lon_r       REAL NOT NULL,
    rank        INTEGER NOT NULL,
    distance_m  REAL NOT NULL,
    osm_type    TEXT NOT NULL,
    osm_id      INTEGER NOT NULL,
    name        TEXT,
    primary_key TEXT,
    primary_val TEXT,
    tags_json   TEXT,
    radius_m    INTEGER NOT NULL,
    fetched_at  REAL NOT NULL,
    PRIMARY KEY (lat_r, lon_r, rank)
);
"""


def open_cache(path):
    return open_sqlite(path, [SCHEMA_SQL])


# --- Overpass call -----------------------------------------------------------


def overpass_around(lat, lon, radius_m, session, *,
                    min_interval_s, max_retries, timeout_s, headers):
    """POST one Overpass query. Returns parsed JSON dict or None on
    permanent failure. Retries on 429/5xx with exponential backoff.
    """
    body = _overpass_query(lat, lon, radius_m)
    wait = min_interval_s
    for attempt in range(max_retries):
        try:
            r = session.post(OVERPASS_URL, data={"data": body},
                             headers=headers, timeout=timeout_s)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(wait)
                wait *= 2
                continue
            print(f"  HTTP {r.status_code} for {lat},{lon}: {r.text[:160]}",
                  file=sys.stderr)
            return None
        except requests.RequestException as e:
            print(f"  request error {e!r}; retry {attempt + 1}",
                  file=sys.stderr)
            time.sleep(wait)
            wait *= 2
    return None


# --- Element parsing ---------------------------------------------------------


def _element_centroid(el):
    """Return (lat, lon) for one Overpass element, or None."""
    if el.get("type") == "node":
        return el.get("lat"), el.get("lon")
    center = el.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def _pick_primary(tags):
    """Return (primary_key, primary_val) for a tag dict, applying the
    same precedence as the prompt cheat sheet. Returns (None, None)
    if no activity-relevant tag remains after filtering.
    """
    if not tags:
        return None, None
    # Top-level keys in declared order — the FIRST present non-trivial
    # value wins, matching the cheat-sheet precedence.
    for k in KEEP_KEYS:
        v = tags.get(k)
        if not v or v == "yes":
            continue
        if k == "amenity" and v in AMENITY_DROP:
            continue
        return k, v
    # Fall back to building, but only for the activity-bearing subset.
    b = tags.get("building")
    if b and b in BUILDING_KEEP:
        return "building", b
    return None, None


def parse_overpass(payload):
    """Return a list of dicts: {osm_type, osm_id, lat, lon, primary_key,
    primary_val, name, tags}. Filters out elements whose primary tag is
    not activity-relevant.
    """
    if not payload:
        return []
    out = []
    for el in payload.get("elements", []):
        lat, lon = _element_centroid(el)
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {}) or {}
        p_key, p_val = _pick_primary(tags)
        if p_key is None:
            continue
        out.append({
            "osm_type":    el.get("type"),
            "osm_id":      int(el.get("id", 0)),
            "lat":         float(lat),
            "lon":         float(lon),
            "primary_key": p_key,
            "primary_val": p_val,
            "name":        tags.get("name"),
            "tags":        tags,
        })
    return out


# --- Per-centroid lookup -----------------------------------------------------


def fetch_neighbors(lat, lon, session, k, r_primary, r_fallback, *,
                    min_interval_s, max_retries, timeout_s, headers):
    """Return list of (rank, distance_m, radius_m, element_dict).
    Empty list when both radii return nothing.
    """
    for r_m in (r_primary, r_fallback):
        payload = overpass_around(lat, lon, r_m, session,
                                  min_interval_s=min_interval_s,
                                  max_retries=max_retries,
                                  timeout_s=timeout_s,
                                  headers=headers)
        elements = parse_overpass(payload)
        if not elements:
            continue
        for e in elements:
            e["distance_m"] = haversine_m(lat, lon, e["lat"], e["lon"])
        elements.sort(key=lambda e: e["distance_m"])
        top = elements[:k]
        return [(rank, e["distance_m"], r_m, e)
                for rank, e in enumerate(top, 1)]
    return []


# --- Main --------------------------------------------------------------------


def main():
    cfg = load_cfg("configs/data/geolife_poi.yaml")

    staypoints_tsv = ROOT / cfg["staypoints_tsv"]
    cache_db       = ROOT / cfg["osm_db"]
    round_decimals = cfg["round_decimals"]
    top_k          = cfg["top_k_neighbors"]
    r_primary_m    = cfg["radius_primary_m"]
    r_fallback_m   = cfg["radius_fallback_m"]
    req_headers    = {"User-Agent": cfg["user_agent"]}
    min_interval_s = cfg["osm_min_interval_s"]
    max_retries    = cfg["osm_max_retries"]
    timeout_s      = cfg["osm_timeout_s"]

    sp = pd.read_csv(staypoints_tsv, sep="\t")
    sp["lat_r"] = sp["Latitude"].round(round_decimals)
    sp["lon_r"] = sp["Longitude"].round(round_decimals)
    unique = sp[["lat_r", "lon_r"]].drop_duplicates().reset_index(drop=True)
    print(f"staypoints: {len(sp):,}  unique rounded centroids: {len(unique):,}")

    conn = open_cache(cache_db)
    cached = {
        (round(r, round_decimals), round(c, round_decimals))
        for r, c in conn.execute("SELECT DISTINCT lat_r, lon_r FROM osm_neighbors")
    }
    todo = unique[~unique.apply(
        lambda row: (row["lat_r"], row["lon_r"]) in cached, axis=1
    )].reset_index(drop=True)
    print(f"already cached: {len(cached):,}  to query: {len(todo):,}")
    if len(todo) == 0:
        print("nothing to do.")
        conn.close()
        return

    print(f"[{time.strftime('%H:%M:%S')}] querying Overpass "
          f"(k={top_k}, R={r_primary_m}/{r_fallback_m} m, "
          f"min interval {min_interval_s:.2f}s) ...")

    session = requests.Session()
    t_prev  = 0.0
    t_start = time.time()
    n_sparse = 0
    n_empty  = 0

    overpass_kwargs = dict(min_interval_s=min_interval_s, max_retries=max_retries,
                           timeout_s=timeout_s, headers=req_headers)

    for i, row in todo.iterrows():
        # Respect Overpass rate limit.
        elapsed = time.time() - t_prev
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        t_prev = time.time()

        lat_r = float(row["lat_r"])
        lon_r = float(row["lon_r"])
        nbrs = fetch_neighbors(lat_r, lon_r, session,
                               top_k, r_primary_m, r_fallback_m,
                               **overpass_kwargs)

        if not nbrs:
            n_empty += 1
            conn.execute(
                "INSERT OR REPLACE INTO osm_neighbors "
                "(lat_r, lon_r, rank, distance_m, osm_type, osm_id, "
                "name, primary_key, primary_val, tags_json, radius_m, "
                "fetched_at) "
                "VALUES (?, ?, 1, 0, '', 0, NULL, NULL, NULL, '{}', 0, ?)",
                (lat_r, lon_r, time.time()),
            )
        else:
            radius_used = nbrs[0][2]
            if radius_used == r_fallback_m:
                n_sparse += 1
            for rank, dist_m, r_m, e in nbrs:
                conn.execute(
                    "INSERT OR REPLACE INTO osm_neighbors "
                    "(lat_r, lon_r, rank, distance_m, osm_type, osm_id, "
                    "name, primary_key, primary_val, tags_json, radius_m, "
                    "fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        lat_r, lon_r, rank, dist_m,
                        str(e["osm_type"]),
                        int(e["osm_id"]),
                        e.get("name"),
                        str(e["primary_key"]),
                        str(e["primary_val"]),
                        json.dumps(e["tags"], ensure_ascii=False),
                        int(radius_used),
                        time.time(),
                    ),
                )

        if (i + 1) % 50 == 0:
            conn.commit()
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta_min = (len(todo) - i - 1) / max(rate, 1e-6) / 60.0
            print(f"  [{i + 1}/{len(todo)}] {lat_r},{lon_r}  "
                  f"sparse {n_sparse}  empty {n_empty}  "
                  f"ETA {eta_min:.1f} min")

    conn.commit()
    conn.close()

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print(f"  centroids needing fallback radius (sparse): {n_sparse:,}")
    print(f"  centroids with no POI within {r_fallback_m} m: {n_empty:,}")


if __name__ == "__main__":
    main()

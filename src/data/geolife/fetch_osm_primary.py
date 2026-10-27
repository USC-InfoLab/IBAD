"""Stage 3 — OSM reverse-lookup with SQLite caching.

For every unique (rounded) staypoint centroid in staypoints.tsv,
query Nominatim reverse with extratags=1 (at 1 req/sec per TOS) and stash
the full GeoJSON + parsed (category, type, class, extratags) in a local
SQLite database. Safe to re-run — cached keys are skipped.

Output: osm_db path configured in configs/data/geolife_poi.yaml (table `osm_cache`)

Run from repo root:
    python -m src.data.geolife.fetch_osm_primary
"""

import json
import sys
import time

import pandas as pd
import requests

from src.utils.config import ROOT, load_cfg
from src.utils.sqlite_utils import open_sqlite

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


_OSM_CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS osm_cache (
        lat_r     REAL NOT NULL,
        lon_r     REAL NOT NULL,
        category  TEXT,
        type      TEXT,
        class     TEXT,
        extratags TEXT,
        raw       TEXT,
        fetched_at REAL NOT NULL,
        PRIMARY KEY (lat_r, lon_r)
    )
"""


def open_cache(path):
    return open_sqlite(path, [_OSM_CACHE_DDL])


def parse_geojson(data):
    try:
        props = data["features"][0]["properties"]
    except (KeyError, IndexError, TypeError):
        return None, None, None, None
    extratags = props.get("extratags") or {}
    return (props.get("category"), props.get("type"),
            props.get("class"), extratags)


def nominatim_reverse(lat, lon, session, *, min_interval_s, max_retries, headers):
    params = {
        "format":    "geojson",
        "extratags": 1,
        "zoom":      18,
        "lat":       lat,
        "lon":       lon,
    }
    wait = min_interval_s
    for attempt in range(max_retries):
        try:
            r = session.get(NOMINATIM_URL, params=params,
                            headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(wait)
                wait *= 2
                continue
            print(f"  HTTP {r.status_code} for {lat},{lon}: {r.text[:120]}",
                  file=sys.stderr)
            return None
        except requests.RequestException as e:
            print(f"  request error {e!r}; retry {attempt + 1}", file=sys.stderr)
            time.sleep(wait)
            wait *= 2
    return None


def main():
    cfg = load_cfg("configs/data/geolife_poi.yaml")

    staypoints_tsv = ROOT / cfg["staypoints_tsv"]
    cache_db       = ROOT / cfg["osm_db"]
    round_decimals = cfg["round_decimals"]
    req_headers    = {"User-Agent": cfg["user_agent"]}
    min_interval_s = cfg["osm_min_interval_s"]
    max_retries    = cfg["osm_max_retries"]

    sp = pd.read_csv(staypoints_tsv, sep="\t")
    sp["lat_r"] = sp["Latitude"].round(round_decimals)
    sp["lon_r"] = sp["Longitude"].round(round_decimals)

    unique = sp[["lat_r", "lon_r"]].drop_duplicates().reset_index(drop=True)
    print(f"staypoints: {len(sp):,}  unique rounded centroids: {len(unique):,}")

    conn = open_cache(cache_db)
    cached = set(
        (round(r, round_decimals), round(c, round_decimals))
        for r, c in conn.execute("SELECT lat_r, lon_r FROM osm_cache")
    )
    todo = unique[~unique.apply(
        lambda row: (row["lat_r"], row["lon_r"]) in cached, axis=1
    )].reset_index(drop=True)
    print(f"cached already: {len(cached):,}  to fetch: {len(todo):,}")

    session = requests.Session()
    t_prev = 0.0
    t_start = time.time()
    for i, row in todo.iterrows():
        elapsed = time.time() - t_prev
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        t_prev = time.time()

        lat_r, lon_r = float(row["lat_r"]), float(row["lon_r"])
        data = nominatim_reverse(lat_r, lon_r, session,
                                 min_interval_s=min_interval_s,
                                 max_retries=max_retries,
                                 headers=req_headers)
        category = type_ = cls = None
        extratags = {}
        if data:
            category, type_, cls, extratags = parse_geojson(data)

        conn.execute(
            "INSERT OR REPLACE INTO osm_cache "
            "(lat_r, lon_r, category, type, class, extratags, raw, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (lat_r, lon_r, category, type_, cls,
             json.dumps(extratags), json.dumps(data) if data else None,
             time.time()),
        )
        if (i + 1) % 50 == 0:
            conn.commit()
            eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            print(f"  [{i + 1}/{len(todo)}] {lat_r},{lon_r} "
                  f"-> {category},{type_}   ETA {eta / 60:.1f} min")

    conn.commit()
    conn.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Stage 5b — LLM-as-judge NHTS labelling of Geolife stay-points (v2).

v2 changes vs. v1 (see research/geolife_poi_attribution.md for the
full design):

- Joins against `osm_neighbors` (populated by `src/data/fetch_osm_neighbors.py`)
  to attach the top-3 nearest activity-relevant OSM POIs per stay.
  Replaces the passive grid-scan over `osm_cache` cells from v1.
- Builds a per-frequent-location *temporal fingerprint* (median
  arrival, IQR, median duration, weekday/weekend split, date span)
  and feeds it to the LLM alongside visit count.
- Computes transit distance/time to the previous and next stay so the
  rubric's drop-off / transit rules have explicit signals.
- After pass 1 finishes, runs a *pass-2 `Other`-recovery loop*:
  for every agent-day that contains at least one stay labelled
  `Other`, re-call the LLM with each
  frequent-location entry annotated with the modal label assigned to
  it during pass 1. Merge: keep pass-2 result for flagged stays;
  keep pass-1 for the rest. A `pass: 1|2` audit column records which
  call produced each row.

Authentication (pick one):
  - API key:   export GEMINI_API_KEY=...   (real-time mode, no GCS needed)
  - Vertex AI: configure gcp_project/gcp_location in geolife_poi.yaml
               and set GCS_BUCKET for batch mode (50% cheaper).

Run from repo root:
    python -m src.data.geolife.label_with_llm
"""

import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from google import genai
from google.genai import types

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    gcs_storage = None

from src.utils.config import ROOT, load_cfg
from src.utils.geometry import haversine_km
from src.utils.sqlite_utils import open_sqlite
from src.utils.nhts_llm_judge import (
    classify_day, DailyQuotaExhausted,
    build_prompt, build_batch_request_line, cache_key as make_cache_key,
)

PASS2_TRIGGER_LABELS = frozenset({"Other"})

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


# ============================================================================
# Logging + checkpoint helpers
# ============================================================================


def _setup_logging(log_file) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def _save_checkpoint(results: dict, path) -> None:
    """Atomically write pass results to JSON keyed by df_index (str)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in results.items()}, fh)
    os.replace(tmp, path)
    logging.info("checkpoint saved → %s  (%d entries)", path, len(results))


def _load_checkpoint(path) -> dict:
    """Load a checkpoint written by _save_checkpoint; returns {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    result = {int(k): v for k, v in raw.items()}
    logging.info("checkpoint loaded ← %s  (%d entries)", path, len(result))
    return result


# ============================================================================
# OSM cache loading (primary tag + neighbors)
# ============================================================================


def load_osm_primary(db_path, round_decimals):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT lat_r, lon_r, category, type, class, extratags "
        "FROM osm_cache"
    ).fetchall()
    conn.close()
    cache = {}
    for lat_r, lon_r, cat, typ, cls, extra_json in rows:
        try:
            extra = json.loads(extra_json) if extra_json else {}
        except json.JSONDecodeError:
            extra = {}
        cache[(round(lat_r, round_decimals),
               round(lon_r, round_decimals))] = (cat, typ, cls, extra)
    return cache


def load_osm_neighbors(db_path, round_decimals):
    """Return dict mapping (lat_r, lon_r) -> list of neighbor dicts
    sorted by rank ascending. Sentinel rows (primary_key IS NULL) yield
    an empty list (sparse area).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT lat_r, lon_r, rank, distance_m, primary_key, "
            "primary_val, name, radius_m FROM osm_neighbors "
            "ORDER BY lat_r, lon_r, rank"
        ).fetchall()
    except sqlite3.OperationalError:
        # Table not present — pipeline can still run with empty neighbors
        # (the rubric will fall back to primary tag + history).
        conn.close()
        logging.warning("osm_neighbors table not found; "
                        "run src/data/fetch_osm_neighbors.py first for full v2 behaviour.")
        return {}
    conn.close()

    out = defaultdict(list)
    for lat_r, lon_r, rank, dist_m, p_key, p_val, name, radius_m in rows:
        key = (round(lat_r, round_decimals), round(lon_r, round_decimals))
        if p_key is None:
            # Sentinel row — sparse area, leave list empty.
            out.setdefault(key, [])
            continue
        out[key].append({
            "rank": int(rank),
            "distance_m": float(dist_m),
            "primary_key": p_key,
            "primary_val": p_val,
            "name": name,
            "radius_m": int(radius_m) if radius_m is not None else None,
        })
    return dict(out)


# ============================================================================
# Frequent-location aggregation with temporal fingerprint
# ============================================================================


def _format_hhmm(td_seconds: float) -> str:
    """Format a number of seconds (or a fractional hour ×3600) as
    'HH:MM' on the clock. Negative wraps modulo 24 h.
    """
    seconds = int(td_seconds) % (24 * 3600)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _format_iqr(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"±{h}:{m:02d}"


def _circular_median_seconds(arrivals_sec: pd.Series) -> float:
    """Median of arrivals on a 24 h clock. Naive (linear) median
    distorts at midnight; here we just take the linear median because
    Beijing-active hours rarely wrap past midnight in our corpus.
    Good enough for prompt-level summary.
    """
    return float(arrivals_sec.median())


def frequent_locations_for_agent(df_agent, k, decimals):
    """Return a list of fl-dicts (length up to k) for one agent.

    Each dict has keys consumed by NHTSLLMJudge._format_freq_loc:
      lat, lon, count, schedule_kind, median_arr, arr_iqr,
      median_dur_h, dur_iqr, wk, we, span_first, span_last.
    """
    df = df_agent.copy()
    df["lat_r"] = df["Latitude"].round(decimals)
    df["lon_r"] = df["Longitude"].round(decimals)

    arr = pd.to_datetime(df["ArrivingTime"])
    df["arr_seconds"] = (
        arr.dt.hour * 3600 + arr.dt.minute * 60 + arr.dt.second
    )
    df["weekday"] = arr.dt.weekday   # 0..6 (Mon..Sun)
    df["is_weekend"] = df["weekday"].isin([5, 6])
    df["year_month"] = arr.dt.strftime("%Y-%m")
    df["dur_sec"] = df["Duration_Min"] * 60.0

    grp = df.groupby(["lat_r", "lon_r"], sort=False)
    counts = grp.size().rename("n")
    top = counts.sort_values(ascending=False).head(k).reset_index()

    out = []
    for _, row in top.iterrows():
        sub = df[(df["lat_r"] == row["lat_r"]) &
                 (df["lon_r"] == row["lon_r"])]
        arr_med = _circular_median_seconds(sub["arr_seconds"])
        arr_q1 = sub["arr_seconds"].quantile(0.25)
        arr_q3 = sub["arr_seconds"].quantile(0.75)
        arr_iqr = max(arr_q3 - arr_q1, 0)
        dur_med = float(sub["dur_sec"].median())
        dur_q1 = sub["dur_sec"].quantile(0.25)
        dur_q3 = sub["dur_sec"].quantile(0.75)
        dur_iqr = max(dur_q3 - dur_q1, 0)
        wk = int((~sub["is_weekend"]).sum())
        we = int(sub["is_weekend"].sum())
        span_first = sub["year_month"].min()
        span_last  = sub["year_month"].max()

        if wk > 0 and we == 0:
            kind = "weekday"
        elif we > 0 and wk == 0:
            kind = "weekend"
        else:
            kind = "mixed"

        out.append({
            "lat":            float(row["lat_r"]),
            "lon":            float(row["lon_r"]),
            "count":          int(row["n"]),
            "schedule_kind":  kind,
            "median_arr":     _format_hhmm(arr_med),
            "arr_iqr":        _format_iqr(arr_iqr),
            "median_dur_h":   dur_med / 3600.0,
            "dur_iqr":        _format_iqr(dur_iqr),
            "wk":             wk,
            "we":             we,
            "span_first":     span_first,
            "span_last":      span_last,
        })
    return out


def freq_match_index(stay_lat, stay_lon, frequent_locations, decimals):
    """Return 1-based rank of the frequent location that matches the
    stay's centroid at `decimals` resolution, or None.
    """
    s_key = (round(stay_lat, decimals), round(stay_lon, decimals))
    for rank, fl in enumerate(frequent_locations, 1):
        if (round(fl["lat"], decimals), round(fl["lon"], decimals)) == s_key:
            return rank
    return None


# ============================================================================
# Geometry helpers
# ============================================================================


# ============================================================================
# Per-agent-day stay assembly
# ============================================================================


def _build_stays(day_df, freq_locs, osm_primary, osm_neighbors,
                 flagged_indices=None, *, decimals):
    """Build the list of stay-dicts for one agent-day's prompt.

    `flagged_indices` is a set of DataFrame row indices that must be
    marked `flagged_for_pass2=True` in the prompt. Pass `None` for
    pass 1.
    """
    flagged_indices = flagged_indices or set()

    rows = list(day_df.itertuples(index=True))
    stays = []
    df_indices = []
    for i, r in enumerate(rows):
        cat, typ, cls, extra = osm_primary.get(
            (r.lat_r, r.lon_r), (None, None, None, {})
        )
        nbrs = osm_neighbors.get((r.lat_r, r.lon_r), [])

        prev = rows[i - 1] if i > 0 else None
        nxt  = rows[i + 1] if i + 1 < len(rows) else None

        prev_km = (haversine_km(prev.Latitude, prev.Longitude,
                                r.Latitude, r.Longitude) if prev else None)
        prev_min = ((r.ArrivingTime - prev.LeavingTime).total_seconds() / 60.0
                    if prev else None)
        next_km = (haversine_km(r.Latitude, r.Longitude,
                                nxt.Latitude, nxt.Longitude) if nxt else None)
        next_min = ((nxt.ArrivingTime - r.LeavingTime).total_seconds() / 60.0
                    if nxt else None)

        stays.append({
            "lat":         r.Latitude,
            "lon":         r.Longitude,
            "arrived":     r.ArrivingTime.strftime("%Y-%m-%d %H:%M:%S%z"),
            "left":        r.LeavingTime.strftime("%Y-%m-%d %H:%M:%S%z"),
            "dur_min":     float(r.Duration_Min),
            "category":    cat,
            "type_":       typ,
            "class_":      cls,
            "extratags":   extra,
            "neighbors":   nbrs,
            "transit_from_prev_km":  prev_km,
            "transit_from_prev_min": prev_min,
            "transit_to_next_km":    next_km,
            "transit_to_next_min":   next_min,
            "freq_match":  freq_match_index(r.Latitude, r.Longitude, freq_locs, decimals),
            "flagged_for_pass2": r.Index in flagged_indices,
        })
        df_indices.append(r.Index)

    return df_indices, stays


# ============================================================================
# Worker tasks
# ============================================================================


def _process_day(task, client, pass_num=1, *, model_name, cache_db, decimals):
    aid, date, day_df, weekday, flocs, osm_primary, osm_neighbors, \
        flagged_indices = task

    df_indices, stays = _build_stays(
        day_df, flocs, osm_primary, osm_neighbors,
        flagged_indices=flagged_indices,
        decimals=decimals,
    )

    try:
        result = classify_day(
            agent_id=str(aid),
            weekday=weekday,
            date=str(date),
            frequent_locations=flocs,
            stays=stays,
            client=client,
            model=model_name,
            cache_db=str(cache_db),
            pass_num=pass_num,
        )
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        logging.warning("[err pass=%d agent=%s date=%s] %r", pass_num, aid, date, e)
        result = [
            {"stay": i + 1, "label": "Other",
             "reasoning": f"error: {e!r}",
             "candidates": []}
            for i in range(len(stays))
        ]

    return df_indices, result


# ============================================================================
# Pass orchestration
# ============================================================================


def _run_pass(tasks, pass_num, client, total_days_msg="", *,
             max_workers, model_name, cache_db, decimals):
    """Execute a list of agent-day tasks in a thread pool.
    Returns a dict: {df_index: classification_dict}.
    """
    results = {}
    done_days = 0
    n_tasks = len(tasks)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {
            pool.submit(_process_day, task, client, pass_num,
                        model_name=model_name, cache_db=cache_db,
                        decimals=decimals): (task[0], task[1])
            for task in tasks
        }

        for fut in as_completed(future_to_key):
            aid, date = future_to_key[fut]
            try:
                df_indices, result = fut.result()
            except DailyQuotaExhausted as e:
                logging.warning("[stop pass %d] %s", pass_num, e)
                logging.warning("[stop] Labeled %d/%d agent-days. "
                                "Rerun after quota resets to resume (cache preserves progress).",
                                done_days, n_tasks)
                for f in future_to_key:
                    f.cancel()
                break

            by_idx = {int(x["stay"]): x for x in result}
            for i, df_idx in enumerate(df_indices, 1):
                results[df_idx] = by_idx.get(i, {
                    "label": "Other",
                    "reasoning": "missing from LLM output",
                    "candidates": [],
                })

            done_days += 1
            if done_days % 25 == 0 or done_days == n_tasks:
                elapsed = time.time() - t0
                rate = done_days / elapsed
                eta_min = (n_tasks - done_days) / rate / 60
                logging.info("[pass %d %d/%d%s] last: agent %s %s   rate %.1f days/s   ETA %.1f min",
                             pass_num, done_days, n_tasks, total_days_msg, aid, date, rate, eta_min)

    return results


def _run_pass_batch(tasks, pass_num, client, total_days_msg="", *,
                   max_workers, model_name, cache_db, decimals,
                   gcs_bucket, gcs_batch_prefix, batch_min_tasks):
    """Batch-prediction variant of _run_pass.

    1. Build prompts + cache keys for every task.
    2. Return cached results immediately; collect uncached tasks.
    3. If uncached count < batch_min_tasks, fall back to _run_pass.
    4. Otherwise: write JSONL → GCS, submit batch job, poll, download
       results, write to SQLite cache, return combined results dict.
    5. Any task missing from batch output falls back to real-time.
    """
    results = {}
    uncached = []   # (task, df_indices, prompt, key)

    # ── Step 1: bulk cache check ──────────────────────────────────────────
    conn = open_sqlite(cache_db, timeout=30, wal=True)

    for task in tasks:
        aid, date, day_df, weekday, flocs, osm_primary, osm_neighbors, \
            flagged = task
        df_indices, stays = _build_stays(
            day_df, flocs, osm_primary, osm_neighbors,
            flagged_indices=flagged,
            decimals=decimals,
        )
        prompt = build_prompt(str(aid), weekday, str(date), flocs, stays,
                              pass_num=pass_num)
        key = make_cache_key(str(aid), str(date), prompt, model_name)

        row = conn.execute(
            "SELECT response FROM nhts_llm_cache WHERE key=?", (key,)
        ).fetchone()
        if row:
            try:
                labels = json.loads(row[0])["labels"]
                by_idx = {int(x["stay"]): x for x in labels}
                for i, df_idx in enumerate(df_indices, 1):
                    results[df_idx] = by_idx.get(i, {
                        "label": "Other", "reasoning": "missing", "candidates": [],
                    })
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache row — re-fetch
        uncached.append((task, df_indices, stays, prompt, key))

    conn.close()

    if not uncached:
        logging.info("[batch pass=%d] all %d tasks served from cache", pass_num, len(tasks))
        return results

    if len(uncached) < batch_min_tasks:
        logging.info("[batch pass=%d] %d uncached < batch_min_tasks (%d) — real-time fallback",
                     pass_num, len(uncached), batch_min_tasks)
        rt = _run_pass([t[0] for t in uncached], pass_num, client, total_days_msg,
                       max_workers=max_workers, model_name=model_name,
                       cache_db=cache_db, decimals=decimals)
        results.update(rt)
        return results

    # ── Step 2: build JSONL and task map ─────────────────────────────────
    # task_map: (agent_id_str, date_str) → (cache_key, df_indices, n_stays)
    # We parse agent_id and date back out of the batch output's request text.
    task_map = {}
    lines = []
    for task, df_indices, stays, prompt, key in uncached:
        aid_str, date_str = str(task[0]), str(task[1])
        weekday = task[3]
        task_map[(aid_str, date_str)] = (key, df_indices, len(stays), prompt)
        req_line = build_batch_request_line(
            aid_str, weekday, date_str,
            task[4],   # frequent_locations (flocs)
            stays,
            pass_num=pass_num,
        )
        lines.append(json.dumps(req_line, ensure_ascii=False))

    jsonl_bytes = ("\n".join(lines) + "\n").encode()
    run_id      = uuid.uuid4().hex[:8]
    input_blob  = f"{gcs_batch_prefix}/pass{pass_num}/{run_id}/input.jsonl"
    output_pfx  = f"{gcs_batch_prefix}/pass{pass_num}/{run_id}/output/"
    input_uri   = f"gs://{gcs_bucket}/{input_blob}"
    output_uri  = f"gs://{gcs_bucket}/{output_pfx}"

    # ── Step 3: upload input JSONL ────────────────────────────────────────
    gcs = gcs_storage.Client()
    bucket_obj = gcs.bucket(gcs_bucket)
    bucket_obj.blob(input_blob).upload_from_string(
        jsonl_bytes, content_type="application/x-ndjson"
    )
    print(f"  [batch pass={pass_num}] {len(uncached):,} uncached tasks → {input_uri}",
          file=sys.stderr)

    # ── Step 4: submit batch job ──────────────────────────────────────────
    batch = client.batches.create(
        model=model_name,
        src=input_uri,
        config=types.CreateBatchJobConfig(dest=output_uri),
    )
    print(f"  [batch pass={pass_num}] job: {batch.name}  state: {batch.state}",
          file=sys.stderr)

    # ── Step 5: poll for completion ───────────────────────────────────────
    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    poll_s   = 60
    while True:
        state_str = str(batch.state)
        # JobState enum __str__ may give 'JobState.JOB_STATE_SUCCEEDED' or
        # just the name; normalise to the bare name.
        state_name = state_str.split(".")[-1]
        if state_name in terminal:
            break
        time.sleep(poll_s)
        batch = client.batches.get(name=batch.name)
        logging.info("[batch pass=%d] state: %s", pass_num, batch.state)
        poll_s = min(poll_s * 2, 300)  # back off up to 5 min

    state_name = str(batch.state).split(".")[-1]
    if state_name != "JOB_STATE_SUCCEEDED":
        logging.warning("[batch pass=%d] FAILED (%s) — real-time fallback for %d tasks",
                        pass_num, batch.state, len(uncached))
        rt = _run_pass([t[0] for t in uncached], pass_num, client, total_days_msg)
        results.update(rt)
        return results

    # ── Step 6: download and parse output JSONL ───────────────────────────
    blobs = list(bucket_obj.list_blobs(prefix=output_pfx))
    conn = open_sqlite(cache_db, timeout=30, wal=True)

    n_ok = n_err = 0
    processed_keys = set()

    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue
        for raw_line in blob.download_as_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)

                # Check for per-request errors
                status = obj.get("status") or {}
                if status.get("code", 0) != 0:
                    n_err += 1
                    continue

                # Identify the task from the request prompt's first line
                prompt_text = (obj["request"]["contents"][0]
                               ["parts"][0]["text"])
                m = re.match(r"Agent (\S+) — \w+, (\d{4}-\d{2}-\d{2})",
                             prompt_text)
                if not m:
                    n_err += 1
                    continue
                aid_str, date_str = m.group(1), m.group(2)
                if (aid_str, date_str) not in task_map:
                    n_err += 1
                    continue
                key, df_indices, n_stays, orig_prompt = task_map[
                    (aid_str, date_str)
                ]

                resp_text = (obj["response"]["candidates"][0]
                             ["content"]["parts"][0]["text"])
                labels = json.loads(resp_text)["labels"]

                conn.execute(
                    "INSERT OR REPLACE INTO nhts_llm_cache "
                    "(key, agent_id, date, n_stays, prompt, response, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, aid_str, date_str, n_stays,
                     orig_prompt, resp_text, time.time()),
                )

                by_idx = {int(x["stay"]): x for x in labels}
                for i, df_idx in enumerate(df_indices, 1):
                    results[df_idx] = by_idx.get(i, {
                        "label": "Other",
                        "reasoning": "missing from batch output",
                        "candidates": [],
                    })
                processed_keys.add(key)
                n_ok += 1

            except Exception as e:
                n_err += 1
                print(f"  [batch parse error] {e!r}", file=sys.stderr)

    conn.commit()
    conn.close()

    logging.info("[batch pass=%d] parsed %d ok, %d errors", pass_num, n_ok, n_err)

    # ── Step 7: real-time fallback for tasks absent from batch output ─────
    missing = [
        t for t in uncached
        if t[4] not in processed_keys  # t[4] is the cache key
    ]
    if missing:
        logging.warning("[batch pass=%d] %d tasks absent from batch output — real-time fallback",
                        pass_num, len(missing))
        rt = _run_pass([t[0] for t in missing], pass_num, client,
                       max_workers=max_workers, model_name=model_name,
                       cache_db=cache_db, decimals=decimals)
        results.update(rt)

    return results


def _build_pass2_priors(sp, pass1_results):
    """Build {(agent, lat_r, lon_r) -> 'Label (n/total, p%)'} from pass-1
    results, excluding rows that pass 1 itself labelled 'Other'.
    """
    counter = defaultdict(Counter)
    for df_idx, res in pass1_results.items():
        label = res.get("label", "Other")
        if label == "Other":
            continue
        row = sp.loc[df_idx]
        key = (row["AgentID"], row["lat_r"], row["lon_r"])
        counter[key][label] += 1

    priors = {}
    for key, c in counter.items():
        if not c:
            continue
        total = sum(c.values())
        label, n = c.most_common(1)[0]
        priors[key] = f"{label} ({n}/{total}, {100 * n / total:.0f}%)"
    return priors


def _attach_priors(flocs, agent_id, priors, decimals):
    """Return a copy of `flocs` with `prior_label` filled in for each
    frequent-location entry that has a pass-1 prior.
    """
    out = []
    for fl in flocs:
        copy = dict(fl)
        key = (agent_id, round(fl["lat"], decimals),
               round(fl["lon"], decimals))
        if key in priors:
            copy["prior_label"] = priors[key]
        out.append(copy)
    return out


# ============================================================================
# Main
# ============================================================================


def main():
    cfg = load_cfg("configs/data/geolife_poi.yaml")

    staypoints_tsv   = ROOT / cfg["staypoints_tsv"]
    osm_db           = ROOT / cfg["osm_db"]
    output_tsv       = ROOT / cfg["labeled_tsv"]
    cache_db         = ROOT / cfg["llm_cache_db"]
    run_dir          = output_tsv.parent
    log_file         = run_dir / "label_llm_run.log"
    pass1_checkpoint = run_dir / "pass1_checkpoint.json"
    model_name       = cfg["llm_model"]
    round_decimals   = cfg["round_decimals"]
    top_k_frequent   = cfg["top_k_frequent"]
    # Concurrent Gemini workers.  5 workers × ~4 s/call ≈ 75 RPM — safely
    # under the default Vertex AI quota (60–120 RPM for Gemini 2.5 Flash).
    max_workers      = cfg["max_workers"]
    gcp_project      = os.environ.get("GOOGLE_CLOUD_PROJECT") or cfg.get("gcp_project") or ""
    gcp_location     = cfg["gcp_location"]

    # Set GCS_BUCKET to activate batch mode (≈50% cheaper, ~15–60 min latency).
    gcs_bucket       = os.environ.get("GCS_BUCKET", "")
    gcs_batch_prefix = os.environ.get("GCS_BATCH_PREFIX", "geolife-nhts-batch")
    batch_min_tasks  = int(os.environ.get("BATCH_MIN_TASKS", "20"))

    _setup_logging(log_file)

    pass_kwargs = dict(max_workers=max_workers, model_name=model_name,
                       cache_db=str(cache_db), decimals=round_decimals)

    # ── Client + mode selection ───────────────────────────────────────────
    use_api_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    if use_api_key:
        client = genai.Client()
        def run_pass(tasks, pass_num, client_, msg=""):
            return _run_pass(tasks, pass_num, client_, msg, **pass_kwargs)
        logging.info("mode: real-time (API key)  model: %s", model_name)
    elif gcs_bucket:
        if gcs_storage is None:
            logging.error(
                "google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            )
            sys.exit(1)
        if not gcp_project:
            logging.error("Vertex AI batch mode needs a GCP project: set GOOGLE_CLOUD_PROJECT "
                          "or gcp_project in geolife_poi.yaml.")
            sys.exit(1)
        client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        batch_kwargs = dict(**pass_kwargs, gcs_bucket=gcs_bucket,
                            gcs_batch_prefix=gcs_batch_prefix,
                            batch_min_tasks=batch_min_tasks)
        def run_pass(tasks, pass_num, client_, msg=""):
            return _run_pass_batch(tasks, pass_num, client_, msg, **batch_kwargs)
        logging.info("mode: batch (Vertex AI)  bucket: %s  prefix: %s  min_tasks: %d  model: %s",
                     gcs_bucket, gcs_batch_prefix, batch_min_tasks, model_name)
    else:
        logging.error(
            "No credentials found. Set GEMINI_API_KEY for real-time mode, "
            "or configure gcp_project in geolife_poi.yaml and set GCS_BUCKET "
            "for Vertex AI batch mode."
        )
        sys.exit(1)

    # ── Data loading ──────────────────────────────────────────────────────
    sp = pd.read_csv(staypoints_tsv, sep="\t")
    sp["ArrivingTime"] = pd.to_datetime(sp["ArrivingTime"])
    sp["LeavingTime"]  = pd.to_datetime(sp["LeavingTime"])
    sp["Date"]         = sp["ArrivingTime"].dt.date
    sp["lat_r"]        = sp["Latitude"].round(round_decimals)
    sp["lon_r"]        = sp["Longitude"].round(round_decimals)
    sp = sp.sort_values(["AgentID", "ArrivingTime"]).reset_index(drop=True)
    logging.info("loaded %d staypoints  agents: %d  agent-days: %d",
                 len(sp), sp["AgentID"].nunique(),
                 sp.groupby(["AgentID", "Date"]).ngroups)

    osm_primary   = load_osm_primary(osm_db, round_decimals)
    osm_neighbors = load_osm_neighbors(osm_db, round_decimals)
    logging.info("osm primary entries: %d   neighbor entries: %d",
                 len(osm_primary), len(osm_neighbors))

    freq_by_agent = {
        aid: frequent_locations_for_agent(grp, top_k_frequent, round_decimals)
        for aid, grp in sp.groupby("AgentID", sort=False)
    }

    # ── Pass 1 with checkpoint ────────────────────────────────────────────
    pass1_tasks = []
    for (aid, date), day_df in sp.groupby(["AgentID", "Date"], sort=True):
        day_df = day_df.sort_values("ArrivingTime")
        weekday = WEEKDAYS[pd.Timestamp(date).weekday()]
        pass1_tasks.append((
            aid, date, day_df, weekday,
            freq_by_agent[aid], osm_primary, osm_neighbors,
            None,
        ))

    p1_checkpoint = _load_checkpoint(pass1_checkpoint)
    covered = set(p1_checkpoint)
    pass1_tasks_needed = [
        t for t in pass1_tasks
        if not all(idx in covered for idx in t[2].index)
    ]
    n_skip = len(pass1_tasks) - len(pass1_tasks_needed)
    logging.info("=== Pass 1 ===  total: %d  checkpoint-covered: %d  to-run: %d",
                 len(pass1_tasks), n_skip, len(pass1_tasks_needed))

    pass1_results = dict(p1_checkpoint)
    if pass1_tasks_needed:
        new = run_pass(pass1_tasks_needed, 1, client)
        pass1_results.update(new)
        _save_checkpoint(pass1_results, pass1_checkpoint)
        logging.info("Pass 1 complete — %d total entries", len(pass1_results))
    else:
        logging.info("Pass 1 fully covered by checkpoint — zero LLM calls made")

    # ── Pass 2 ───────────────────────────────────────────────────────────
    priors = _build_pass2_priors(sp, pass1_results)
    flagged_per_day = defaultdict(set)
    for df_idx, res in pass1_results.items():
        if res.get("label", "Other") in PASS2_TRIGGER_LABELS:
            row = sp.loc[df_idx]
            flagged_per_day[(row["AgentID"], row["Date"])].add(df_idx)

    pass2_tasks = []
    for (aid, date), flagged in flagged_per_day.items():
        day_df = (sp[(sp["AgentID"] == aid) & (sp["Date"] == date)]
                  .sort_values("ArrivingTime"))
        weekday = WEEKDAYS[pd.Timestamp(date).weekday()]
        flocs_with_prior = _attach_priors(freq_by_agent[aid], aid, priors, round_decimals)
        pass2_tasks.append((
            aid, date, day_df, weekday,
            flocs_with_prior, osm_primary, osm_neighbors,
            flagged,
        ))
    logging.info("=== Pass 2 ===  agent-days flagged: %d  priors: %d pairs",
                 len(pass2_tasks), len(priors))
    pass2_results = run_pass(pass2_tasks, 2, client) if pass2_tasks else {}
    logging.info("Pass 2 complete — %d entries", len(pass2_results))

    # ── Merge ────────────────────────────────────────────────────────────
    labels_col     = [""] * len(sp)
    reasons_col    = [""] * len(sp)
    pass_col       = [1]  * len(sp)
    confidence_col = ["unknown"] * len(sp)
    n_recovered    = 0

    def _confidence(res):
        n = len(res.get("candidates") or [])
        if n == 1:
            return "high"
        if n >= 2:
            return "low"
        return "unknown"

    for df_idx in range(len(sp)):
        p1 = pass1_results.get(df_idx)
        if p1 is None:
            labels_col[df_idx]  = "Other"
            reasons_col[df_idx] = "no pass-1 result (likely quota stop)"
            continue
        triggered = p1.get("label") in PASS2_TRIGGER_LABELS
        if triggered and df_idx in pass2_results:
            p2 = pass2_results[df_idx]
            if p2.get("label") != "Other" or p1.get("label") == "Other":
                labels_col[df_idx]     = p2.get("label", "Other")
                reasons_col[df_idx]    = p2.get("reasoning", "")
                pass_col[df_idx]       = 2
                confidence_col[df_idx] = _confidence(p2)
                if p1.get("label") == "Other" and p2.get("label") != "Other":
                    n_recovered += 1
                continue
        labels_col[df_idx]     = p1.get("label", "Other")
        reasons_col[df_idx]    = p1.get("reasoning", "")
        confidence_col[df_idx] = _confidence(p1)

    sp["ActivityCategory"] = labels_col
    sp["LLMReasoning"]     = reasons_col
    sp["Pass"]             = pass_col
    sp["Confidence"]       = confidence_col

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    cols = ["AgentID", "ArrivingTime", "LeavingTime",
            "Latitude", "Longitude", "Duration_Min",
            "ActivityCategory", "LLMReasoning",
            "Confidence", "Pass"]
    sp[cols].to_csv(output_tsv, sep="\t", index=False)
    logging.info("wrote %s", output_tsv)

    logging.info("Pass-2 recovery: %d stays moved out of 'Other'", n_recovered)
    logging.info("LLM-derived activity distribution:\n%s",
                 sp["ActivityCategory"].value_counts().to_string())
    logging.info("Confidence distribution:\n%s",
                 sp["Confidence"].value_counts().to_string())
    logging.info("Pass distribution:\n%s",
                 sp["Pass"].value_counts().to_string())


if __name__ == "__main__":
    main()

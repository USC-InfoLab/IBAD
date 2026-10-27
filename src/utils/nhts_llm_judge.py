"""LLM-as-judge NHTS for Geolife stay-points (v2).

v2 changes vs. v1:
- `candidates` field with NHTS hypotheses considered.
- `candidates` field with the top-2 NHTS hypotheses considered.
- `confidence: high|medium|low`.
- Per-stay top-3 nearest activity-relevant OSM POIs (from
  `osm_neighbors`, populated by `code/GetOsmNeighbors.py`).
- Per-frequent-location temporal fingerprint (median arrival/duration,
  weekday/weekend split, date span) and an optional pass-1 modal-label
  prior consumed on pass 2.
- "Other-as-last-resort" rubric and explicit Volunteer / Pickup_Dropoff
  detection rules to surface those classes when the pattern fits.

The cache key in `_cache_key()` already hashes the prompt body, so v2
entries naturally do not collide with v1 entries — re-runs against the
same `nhts_llm_cache.sqlite` only re-use cached responses for prompts
that match byte-for-byte.
"""

import hashlib
import json
import time
from typing import Optional

from google import genai
from google.genai import types

from src.utils.sqlite_utils import open_sqlite



NHTS_LABELS = (
    "Home", "Work", "Education", "Childcare", "Adultcare",
    "Shopping", "Services", "Errands", "Recreation", "Restaurant",
    "Religious", "Social", "Exercise", "Health", "Pickup_Dropoff",
    "Transportation", "Volunteer", "Other",
)


# OSM (category, type) tuples that carry no activity signal on their own.
# Used by LabelWithLLM to decide whether the per-stay neighbor block
# should be emphasised in the prompt.
_LOW_CONFIDENCE_CATEGORIES = frozenset({
    "highway", "man_made", "place", "boundary", None, "",
})
_LOW_CONFIDENCE_TYPES = frozenset({
    "yes", None, "",
})


def is_low_confidence_tag(category: Optional[str],
                          type_: Optional[str]) -> bool:
    """True iff the OSM (category, type) tells us almost nothing about
    what the agent is doing — i.e. the LLM should rely on neighbors and
    spatiotemporal pattern instead.
    """
    if category in _LOW_CONFIDENCE_CATEGORIES:
        return True
    if category == "building" and type_ in (None, "", "yes"):
        return True
    if type_ in _LOW_CONFIDENCE_TYPES:
        return True
    return False


class DailyQuotaExhausted(RuntimeError):
    """Raised by classify_day when the LLM provider returns a quota
    error that won't recover within this run. Lets the caller stop the
    pool cleanly and resume after the quota window resets.
    """


# ============================================================================
# SYSTEM PROMPT — kept verbatim across runs so the paper's appendix
# references a stable artifact. Edit deliberately.
# ============================================================================
# Look at https://wiki.openstreetmap.org/wiki/Map_features
SYSTEM_PROMPT = """\
You classify GPS stay-points into NHTS 2017 activity categories. Each
input describes one agent-day: a chronological sequence of stay-points
on a single calendar date in Beijing, derived from raw GPS fixes via
the Li et al. (SIGSPATIAL 2008) sliding-window detector and reverse-
geocoded against OpenStreetMap.

# 1. NHTS labels — use exactly these strings

  Home, Work, Education, Childcare, Adultcare, Shopping, Services,
  Errands, Recreation, Restaurant, Religious, Social, Exercise,
  Health, Pickup_Dropoff, Transportation, Volunteer, Other

# 2. Label descriptions (NHTS 2017)

  Home:           Home  (chores, sleep)
  Work:           Work — including paid work-from-home and work trips
  Education:      Attend school as a student
  Childcare:      Attend child care
  Adultcare:      Attend adult care
  Shopping:       Buy goods (groceries, clothes, appliances, gas)
  Services:       Buy services (banking, dry cleaners, car/pet care)
  Errands:        Other general errands (post office, library)
  Recreation:     Recreation (parks, movies, museums, hotels, zoos)
  Restaurant:     Buy meals (restaurants, cafes, fast food, carry-out)
  Religious:      Religious or other community activities
  Social:         Visit friends or relatives (bars, community centres)
  Exercise:       Jog, walk the dog, gym, sports
  Health:         Medical, dental, therapy
  Pickup_Dropoff: Drop off or pick up someone (school run, etc.)
  Transportation: Change type of transportation (transit hubs)
  Volunteer:      Unpaid volunteer activities (charities, NGOs)
  Other:          Last-resort label

# 3. OSM-key → candidate-NHTS cheat sheet

  amenity=restaurant|cafe|fast_food|food_court|ice_cream     -> Restaurant
  amenity=school|university|college|library|*_school         -> Education
  amenity=hospital|clinic|dentist|doctors|pharmacy|blood_bank -> Health
  amenity=cinema|theatre|arts_centre|museum|aquarium|zoo     -> Recreation
  amenity=bar|pub|nightclub|biergarten|community_centre|
          events_venue|casino|social_centre                  -> Social
  amenity=bank|atm|post_office|bureau_de_change|car_*|
          dry_cleaning|veterinary                             -> Services
  amenity=bus_station|bus_stop|ferry_terminal|taxi|parking|
          fuel|charging_station                               -> Transportation
  amenity=place_of_worship|monastery                          -> Religious
  amenity=kindergarten|childcare                              -> Childcare
  amenity=nursing_home                                        -> Adultcare
  amenity=social_facility[for=senior|elderly|disabled]        -> Adultcare
  amenity=social_facility[operator:type=non_profit|ngo|charity] -> Volunteer
  amenity=marketplace|vending_machine                         -> Shopping
  amenity=townhall|courthouse|police|fire_station|courier     -> Errands
  shop=*                                                      -> Shopping
  office=charity|non_profit|ngo                               -> Volunteer
  office=*  (otherwise)                                       -> Work
  leisure=park|playground|garden|nature_reserve|marina|
          dog_park|picnic_site                                -> Recreation
  leisure=sports_centre|fitness_centre|stadium|pitch|
          swimming_pool|track|golf_course|ice_rink            -> Exercise
  building=apartments|house|residential|dormitory|detached    -> Home
  building=office|commercial|industrial|warehouse|factory     -> Work
  building=school|university|college|kindergarten             -> Education
  building=hospital                                           -> Health
  building=church|mosque|temple|synagogue|cathedral|
           chapel|shrine                                      -> Religious
  tourism=*                                                   -> Recreation
  healthcare=*                                                -> Health
  craft=*                                                     -> Services
  sport=*                                                     -> Exercise
  railway=station|subway_entrance|tram_stop|halt|platform     -> Transportation or Pickup_Dropoff (reason with visit pattern)
  aeroway=*                                                   -> Transportation or Pickup_Dropoff (reason with visit pattern)
  highway=*  (LOW-CONFIDENCE — use neighbors and visit pattern instead)
  man_made=* (LOW-CONFIDENCE — use visit pattern and neighbors instead)
  building=yes|other|unclassified|fixme|missed|undefined|unknown|maybe  (LOW-CONFIDENCE — use visit pattern and neighbors instead)

# 4. Neighbor guidance

When the label is unclear from the agent's daily activity or visit
history, look at the top-3 nearby POIs to reason from it. If they consistently
point to one NHTS category, prefer that. If they are uninformative
(highway, building=yes, man_made, or missing), do not force a label —
use Other as a last resort only.

# 5. Output — strict JSON

  {"labels": [
    {
      "stay": <int, 1-indexed>,
      "candidates": [
        {"label": "<NHTS>", "justification": "<= 20 words"}
        // 1 candidate when clear; 2 when a real alternative exists.
      ],
      "label": "<NHTS — must equal one of the candidates>",
      "reasoning": "<one sentence, <= 25 words>"
    },
    ...
  ]}
"""


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

_LABEL_SCHEMA = {"type": "STRING", "enum": list(NHTS_LABELS)}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "labels": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "stay":      {"type": "INTEGER"},
                    "candidates": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "label":         _LABEL_SCHEMA,
                                "justification": {"type": "STRING"},
                            },
                            "required": ["label", "justification"],
                        },
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "label":     _LABEL_SCHEMA,
                    "reasoning": {"type": "STRING"},
                },
                "required": ["stay", "candidates", "label", "reasoning"],
            },
        },
    },
    "required": ["labels"],
}


# ============================================================================
# CACHE
# ============================================================================


_CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS nhts_llm_cache (
        key        TEXT PRIMARY KEY,
        agent_id   TEXT,
        date       TEXT,
        n_stays    INTEGER,
        prompt     TEXT,
        response   TEXT,
        fetched_at REAL NOT NULL
    )
"""


def _open_cache(cache_db: str):
    return open_sqlite(cache_db, [_CACHE_DDL], timeout=30, wal=True)


def _cache_key(agent_id, date, prompt, model: str):
    h = hashlib.sha256()
    h.update(f"{model}|{agent_id}|{date}|".encode())
    h.update(prompt.encode())
    return h.hexdigest()


def cache_key(agent_id, date, prompt, model: str) -> str:
    """Public wrapper around _cache_key for use by the batch pipeline."""
    return _cache_key(agent_id, date, prompt, model)


def _to_rest_schema(schema):
    """Recursively convert schema type names to lowercase for the Vertex AI REST API.

    The SDK accepts uppercase ('OBJECT', 'ARRAY', …) but raw batch-prediction
    JSONL must use the OpenAPI lowercase form ('object', 'array', …).
    The 'enum' values (NHTS label strings) are intentionally left untouched.
    """
    if isinstance(schema, dict):
        return {
            k: (v.lower() if k == "type" and isinstance(v, str) else _to_rest_schema(v))
            for k, v in schema.items()
        }
    if isinstance(schema, list):
        return [_to_rest_schema(x) for x in schema]
    return schema


def build_batch_request_line(agent_id, weekday, date, frequent_locations,
                             stays, pass_num: int = 1) -> dict:
    """Return a dict for one line of a Vertex AI Gemini batch-prediction JSONL.

    Callers serialise with json.dumps and concatenate with newlines.
    The top-level key is 'request', matching the Vertex AI batch input spec.
    """
    prompt = build_prompt(agent_id, weekday, date, frequent_locations,
                          stays, pass_num=pass_num)
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": _to_rest_schema(RESPONSE_SCHEMA),
            },
        }
    }


# ============================================================================
# PROMPT BUILDING
# ============================================================================


def _format_freq_loc(rank, fl):
    """Render one frequent-location entry into a prompt-ready line.

    `fl` is a dict with keys: lat, lon, count, median_arr ('HH:MM'),
    arr_iqr ('±H:MM'), median_dur_h, dur_iqr ('±H:MM'), wk, we,
    span_first ('YYYY-MM'), span_last ('YYYY-MM'), and optionally
    `prior_label` ('Education (35/38, 92%)').
    """
    line = (
        f"  #{rank} ({fl['lat']}, {fl['lon']}) — visited {fl['count']} times, "
        f"{fl.get('schedule_kind', 'mixed')} {fl['median_arr']} {fl['arr_iqr']}, "
        f"duration {fl['median_dur_h']:.1f} h {fl['dur_iqr']}, "
        f"weekday/weekend {fl['wk']}/{fl['we']}, "
        f"{fl['span_first']} → {fl['span_last']}."
    )
    if fl.get("prior_label"):
        line += f"\n     Pass-1 modal label: {fl['prior_label']}."
    return line


def _format_neighbors(neighbors):
    """`neighbors` is a list of dicts with keys: rank, distance_m,
    primary_key, primary_val, name, radius_m. Empty list => 'none'.
    """
    if not neighbors:
        return "  Top-3 neighbors: none (sparse area)."
    lines = ["  Top-3 neighbors:"]
    for n in neighbors:
        kv = f"{n['primary_key']}={n['primary_val']}"
        nm = f" ('{n['name']}')" if n.get("name") else ""
        lines.append(f"    #{n['rank']} {kv}{nm} at {n['distance_m']:.0f}m")
    if neighbors and neighbors[0].get("radius_m") == 500:
        lines.append("    (sparse area — fallback radius 500 m)")
    return "\n".join(lines)


def _format_stay(idx, s):
    """Render one stay's data block. `s` is a dict with keys:
    lat, lon, arrived, left, dur_min, category, type_, extratags,
    neighbors (list), transit_to_next_km, transit_to_next_min,
    transit_from_prev_km, transit_from_prev_min, freq_match.
    """
    parts = [
        f"Stay {idx}: ({s['lat']}, {s['lon']}), arrived {s['arrived']}, "
        f"left {s['left']}, {s['dur_min']:.1f} min "
        f"({s['dur_min'] / 60:.2f} h).",
        f"  Primary OSM: category={s.get('category')!r}, "
        f"type={s.get('type_')!r}, "
        f"extratags={json.dumps(s.get('extratags') or {}, ensure_ascii=False)}.",
        _format_neighbors(s.get("neighbors") or []),
    ]

    transit_bits = []
    if s.get("transit_from_prev_km") is not None:
        transit_bits.append(
            f"from previous: {s['transit_from_prev_km']:.1f} km, "
            f"{s['transit_from_prev_min']:.0f} min"
        )
    if s.get("transit_to_next_km") is not None:
        transit_bits.append(
            f"to next: {s['transit_to_next_km']:.1f} km, "
            f"{s['transit_to_next_min']:.0f} min"
        )
    if transit_bits:
        parts.append("  Transit " + "; ".join(transit_bits) + ".")

    fm = s.get("freq_match")
    if fm:
        parts.append(f"  Frequent-location match: #{fm}.")
    else:
        parts.append("  Frequent-location match: none.")

    if s.get("flagged_for_pass2"):
        parts.append(
            "  [!] Pass-1 labelled this stay 'Other' or low-confidence — "
            "reconsider using rubric step 4 with the prior."
        )
    return "\n".join(parts)


def build_prompt(agent_id, weekday, date, frequent_locations, stays,
                 pass_num: int = 1):
    """Assemble the user prompt for one agent-day.

    `frequent_locations` is a list of fl-dicts (see _format_freq_loc),
    not the raw (lat, lon, count) tuples used by v1. Up to 5 entries.

    `stays` is a list of stay-dicts (see _format_stay).

    `pass_num` is 1 or 2; 2 only changes the header. The per-stay
    `flagged_for_pass2` flag is set by the caller.
    """
    head = (
        f"Agent {agent_id} — {weekday}, {date}."
        if pass_num == 1 else
        f"Agent {agent_id} — {weekday}, {date} (Pass 2: re-evaluating "
        f"stays that pass 1 labelled 'Other' or low-confidence)."
    )

    lines = [head, ""]

    if frequent_locations:
        lines.append(
            "This agent's most-visited centroids (top-5; counts and "
            "temporal fingerprint):"
        )
        for rank, fl in enumerate(frequent_locations, 1):
            lines.append(_format_freq_loc(rank, fl))
        lines.append("")

    lines.append("Today's stay-points, in chronological order:")
    lines.append("")
    for i, s in enumerate(stays, 1):
        lines.append(_format_stay(i, s))
        lines.append("")

    lines.append("Apply the 6-step rubric to each stay. Return the JSON "
                 "described in the system instructions.")
    return "\n".join(lines)


# ============================================================================
# MAIN CALL
# ============================================================================


def _is_quota_error(exc: BaseException) -> bool:
    """Heuristic — Gemini surfaces quota exhaustion as a 429 with
    `RESOURCE_EXHAUSTED` in the message. We don't parse the
    structured error type because google-genai's Python surface
    for that is still in flux.
    """
    msg = str(exc).lower()
    return ("resource_exhausted" in msg or
            "quota" in msg or
            "429" in msg)


def classify_day(agent_id, weekday, date, frequent_locations, stays,
                 client: Optional[genai.Client] = None,
                 model: str = "",
                 cache_db: str = "",
                 pass_num: int = 1,
                 max_retries: int = 4) -> list:
    """Classify one agent-day. Returns a list of dicts:
    {stay, candidates, label, confidence, reasoning}.

    Cached by sha256(model | agent | date | full prompt body), so v1
    and v2 prompts coexist in the same SQLite cache without collision.
    """
    prompt = build_prompt(agent_id, weekday, date, frequent_locations,
                          stays, pass_num=pass_num)

    key = _cache_key(agent_id, date, prompt, model)
    conn = _open_cache(cache_db)
    row = conn.execute(
        "SELECT response FROM nhts_llm_cache WHERE key=?", (key,)
    ).fetchone()
    if row:
        conn.close()
        try:
            return json.loads(row[0])["labels"]
        except (json.JSONDecodeError, KeyError):
            pass  # fall through and re-query


    last_exc = None
    wait = 1.0
    for _ in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1,
                ),
            )
            data = json.loads(resp.text)
            labels = data["labels"]
            conn.execute(
                "INSERT OR REPLACE INTO nhts_llm_cache "
                "(key, agent_id, date, n_stays, prompt, response, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, str(agent_id), str(date), len(stays),
                 prompt, resp.text, time.time()),
            )
            conn.commit()
            conn.close()
            return labels
        except Exception as e:
            last_exc = e
            if _is_quota_error(e):
                conn.close()
                raise DailyQuotaExhausted(str(e)) from e
            time.sleep(wait)
            wait *= 2

    conn.close()
    print(f"  [llm-error agent={agent_id} date={date}] {last_exc!r}")
    return [
        {
            "stay": i + 1,
            "candidates": [{"label": "Other", "justification": f"LLM error: {last_exc!r}"}],
            "label": "Other",
            "reasoning": f"LLM error: {last_exc!r}",
        }
        for i in range(len(stays))
    ]

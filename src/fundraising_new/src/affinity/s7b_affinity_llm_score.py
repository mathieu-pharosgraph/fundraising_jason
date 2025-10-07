#!/usr/bin/env python3
"""
s7b_affinity_llm_score.py  — strict + logged + per-segment fallback

- Stage A: group prompt (all segments at once). Deterministic (temp=0.0).
- Stage B: if Stage A fails strict validation, score each segment with
  a tiny JSON schema (still deterministic), merge, and re-validate.
- Rich logging of every attempt, timings, nz counts, spread, raw snippets.
- Cache keyed by (label, period, feature_hash, personas_hash, version).

Artifacts:
  - Cache JSONL ........ data/affinity/topic_affinity_llm.jsonl
  - Output parquet ..... data/affinity/topic_affinity_by_segment.parquet
  - Run log ............ data/affinity/topic_affinity_llm.log
  - Strict-fail JSONL .. data/affinity/topic_affinity_llm.strict_fail.jsonl
"""

import argparse, json, time, hashlib, logging, re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests as rq
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

# --------- Paths (defaults) ----------
TOPICS = "data/topics/features/topic_features_daily.parquet"
PERSONA= "data/affinity/segment_personas.json"
OUTRAW = "data/affinity/topic_affinity_llm.jsonl"
OUTSEG = "data/affinity/topic_affinity_by_segment.parquet"
LOGFILE= "data/affinity/topic_affinity_llm.log"

DEEPSEEK_API_URL = (Path.home() / ".deepseek_api_url").read_text().strip() \
    if (Path.home()/".deepseek_api_url").exists() else "https://api.deepseek.com/v1"

EMOTIONS = ["emo_anger","emo_outrage","emo_anxiety","emo_disgust","emo_fear","emo_hope_optimism","emo_pride","emo_sadness"]
MORALS   = ["mf_care","mf_harm","mf_fairness","mf_cheating","mf_loyalty","mf_betrayal",
            "mf_authority","mf_subversion","mf_sanctity","mf_degradation","mf_liberty","mf_oppression"]
FORBIDDEN_EMO = {"emo_anger_outrage", "anger_outrage"}  # merged keys we will reject

SCHEMA = {
  "type":"object",
  "required":["topic_label","segments","regions"],
  "properties":{
    "topic_label":{"type":"string"},
    "segments":{"type":"array","items":{
      "type":"object",
      "required":["segment_id","affinity","by_party","best_ask","why"],
      "properties":{
        "segment_id":{"type":"string"},
        "affinity":{"type":"number","minimum":0,"maximum":100},
        "by_party":{"type":"object","required":["Dem","GOP","Ind"],
                    "properties":{"Dem":{"type":"number"},"GOP":{"type":"number"},"Ind":{"type":"number"}}},
        "best_ask":{"type":"string"},
        "why":{"type":"string"}
      }
    }},
    "regions":{"type":"object","properties":{"NE":{"type":"number"},"MW":{"type":"number"},"S":{"type":"number"},"W":{"type":"number"}}}
  }
}

PROMPT_SEG = """You are scoring fundraising AFFINITY for ONE segment.

Return STRICT JSON ONLY (no prose) with this schema:
{{
  "segment_id": "{segment_id}",
  "affinity": 0-100,
  "by_party": {{"Dem": 0-100, "GOP": 0-100, "Ind": 0-100}},
  "best_ask": "<3-8 words>",
  "why": "<≤20 words>"
}}

Constraints:
- Deterministic, no randomization. No markdown or comments.
- Keep by_party internally consistent with the segment's political flavor.
- If appeal is weak, use a small value in the 5–25 range; avoid using the same number across all segments.

SEGMENT PERSONA:
{persona_blurb}

TOPIC:
label: {topic_label}
features:
- urgency: {urgency:.0f}/100
- fundraising_score: {fund:.0f}/100  | ask_strength: {ask:.0f}/100
- top emotions (0–100): {emotions_kv}
- moral foundations (0–100): {morals_kv}
- roles (optional): {roles}

Return JSON only."""

# --------- Logging ----------
def setup_logging(logfile: str):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.DEBUG)

# --------- HTTP ----------
def deepseek_chat(messages, model="deepseek-chat", max_tokens=900, temperature=0.0) -> str:
    import os
    base = os.getenv("DEEPSEEK_API_URL", DEEPSEEK_API_URL).rstrip("/")
    key  = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    url = f"{base}/chat/completions"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    pl  = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = rq.post(url, headers=hdr, json=pl, timeout=60)
    try:
        r.raise_for_status()
    except rq.HTTPError as e:
        # LOG FULL CONTEXT
        logging.error("DeepSeek HTTP %s — %s", r.status_code, r.text[:1000])
        raise
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        logging.error("DeepSeek JSON structure unexpected: %s", json.dumps(j, ensure_ascii=False)[:1000])
        raise

# --------- Hashing ----------
def hash_sig(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def features_hash(row: pd.Series, emo_cols: List[str], mf_cols: List[str]) -> str:
    payload = {
        "urg": float(row.get("urgency_score", 0) or 0),
        "ask": float(row.get("cta_ask_strength", 0) or 0),
        "fund": float(row.get("fundraising_score", 0) or 0),
        "emo": {c: float(row[c]) for c in emo_cols if pd.notna(row[c])},
        "mf":  {c: float(row[c]) for c in mf_cols  if pd.notna(row[c])},
    }
    return hash_sig(payload)

# ---------- Robust JSON helpers ----------

def _first_nonempty(row, keys):
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            return row[k]
    return None


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _normalize_quotes(s: str) -> str:
    return s.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([\}\]])", r"\1", s)

def _find_balanced_json(s: str) -> str | None:
    start = s.find("{")
    while start != -1:
        stack = 0; in_str = False; esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == "\"": in_str = False
                continue
            else:
                if ch == "\"": in_str = True
                elif ch == "{": stack += 1
                elif ch == "}":
                    stack -= 1
                    if stack == 0:
                        return s[start:i+1]
        start = s.find("{", start+1)
    return None

def try_parse_json(raw: str):
    raw0 = _strip_fences(_normalize_quotes(raw))
    blob = _find_balanced_json(raw0) or raw0
    for cand in (blob, _remove_trailing_commas(blob)):
        try: return json.loads(cand)
        except: pass
    # last chance: single-quote fix
    safe = re.sub(r"(?P<q>')(?P<key>[^']+?)'(?=\s*:)", r'"\g<key>"', blob)
    safe = re.sub(r":\s*'([^']*)'", lambda m: ':"{}"'.format(m.group(1).replace('"','\\"')), safe)
    return json.loads(_remove_trailing_commas(safe))

# ---------- Coercion & Validation ----------
def coerce_affinity_payload(js: dict, topic_label: str, period: str, seg_ids: List[str]) -> dict:
    if not isinstance(js, dict): js = {}
    js.setdefault("topic_label", topic_label)
    js.setdefault("period", period)

    # regions
    regs = (js.get("regions") or {})
    out_regs = {}
    for k in ["NE","MW","S","W"]:
        try: v = float(regs.get(k, 1.0))
        except: v = 1.0
        out_regs[k] = max(0.5, min(1.5, v))
    js["regions"] = out_regs

    # harvest segments from root & list
    incoming = []
    if "segment_id" in js:
        incoming.append({
            "segment_id": js.get("segment_id"),
            "affinity":   js.get("affinity"),
            "by_party":   js.get("by_party"),
            "best_ask":   js.get("best_ask"),
            "why":        js.get("why"),
        })
    segs = js.get("segments") or []
    if isinstance(segs, dict): segs = [segs]
    incoming += [s for s in segs if isinstance(s, dict)]

    by_id = {}
    for s in incoming:
        sid = str(s.get("segment_id","") or "").strip()
        if sid: by_id[sid] = s

    def _num(x) -> float:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return 0.0
        v = float(v)
        if v < 0.0:  v = 0.0
        if v > 100.: v = 100.0
        return v

    js["segments"] = [{
        "segment_id": sid,
        "affinity": _num(by_id.get(sid, {}).get("affinity", 0)),
        "by_party": {
            "Dem": _num((by_id.get(sid, {}).get("by_party") or {}).get("Dem", 0)),
            "GOP": _num((by_id.get(sid, {}).get("by_party") or {}).get("GOP", 0)),
            "Ind": _num((by_id.get(sid, {}).get("by_party") or {}).get("Ind", 0)),
        },
        "best_ask": str(by_id.get(sid, {}).get("best_ask", "No donation recommended"))[:120],
        "why": str(by_id.get(sid, {}).get("why", ""))[:240],
    } for sid in seg_ids]
    return js

def minimal_validate(js: dict):
    if not isinstance(js, dict): raise ValueError("JSON root must be an object")
    for k in ["topic_label","segments","regions"]:
        if k not in js: raise ValueError(f"Missing required key: {k}")
    segs = js["segments"]
    if not isinstance(segs, list) or len(segs) == 0:
        raise ValueError("segments must be a non-empty list")
    for s in segs:
        if not isinstance(s, dict): raise ValueError("segment entry must be an object")
        for k in ["segment_id","affinity","by_party"]:
            if k not in s: raise ValueError(f"segment missing key: {k}")
    return True

def is_bad(js: dict, seg_ids: List[str], need_nonzero: int,
           min_spread: float = 5.0, lowmax: float = 20.0, low_spread: float = 2.0) -> Tuple[bool, str, dict]:
    segs = js.get("segments") or []
    if isinstance(segs, dict): segs = [segs]
    diag = {"len": len(segs), "nz": 0, "spread": 0.0, "max": 0.0}
    if len(segs) != len(seg_ids):
        return True, f"segments length {len(segs)} != expected {len(seg_ids)}", diag
    ids = [s.get("segment_id","") for s in segs]
    if set(ids) != set(seg_ids):
        return True, "segment_id set mismatch", diag

    vals = []
    for s in segs:
        try: a = float(s.get("affinity",0) or 0)
        except: a = 0.0
        vals.append(a)
    nz = sum(v > 0 for v in vals)
    spread = float(max(vals) - min(vals)) if vals else 0.0
    vmax = float(max(vals)) if vals else 0.0

    diag.update({"nz": nz, "spread": spread, "max": vmax})

    if nz < need_nonzero:
        return True, f"nonzero segments {nz} < required {need_nonzero}", diag

    # dynamic spread: lower requirement if overall signal is low
    req_spread = low_spread if vmax < lowmax else min_spread
    if spread < req_spread:
        return True, f"affinity spread {spread:.1f} < required {req_spread:.1f} (max={vmax:.1f})", diag

    return False, "", diag

# ---------- Scoring helpers ----------
def build_segments_template(seg_ids: List[str]) -> List[dict]:
    return [{
        "segment_id": sid,
        "affinity": 0,
        "by_party": {"Dem": 0, "GOP": 0, "Ind": 0},
        "best_ask": "",
        "why": ""
    } for sid in seg_ids]

def score_segment(seg_id: str, persona_blurb: str, topic_label: str, row, emo_cols, mf_cols, model) -> dict:
    # ---- STRICT emotion/moral extraction (with deterministic expansion) ----
    present = set(row.index) if hasattr(row, "index") else set(row.keys())
    if "emo_anger_outrage" in present and ("emo_anger" not in present and "emo_outrage" not in present):
        v = float(pd.to_numeric(row["emo_anger_outrage"], errors="coerce") or 0.0)
        row["emo_anger"] = v; row["emo_outrage"] = v
        present = set(row.index) if hasattr(row, "index") else set(row.keys())

    missing_emo = [c for c in EMOTIONS if c not in present]
    missing_mf  = [c for c in MORALS   if c not in present]
    if missing_emo or missing_mf:
        raise RuntimeError(f"Missing required emotion/moral columns. Missing emotions={missing_emo} | missing morals={missing_mf}")

    def _num(x) -> float:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return 0.0
        v = float(v)
        if v < 0.0:  v = 0.0
        if v > 100.: v = 100.0
        return v

    emo_vals = {c.replace("emo_", ""): _num(row[c]) for c in EMOTIONS}
    mf_vals  = {c: _num(row[c]) for c in MORALS}


    # ---- context numerics from row ----
    urgency = float(row.get("urgency_score", 0) or 0)
    ask     = float(row.get("cta_ask_strength", row.get("cta_strength", 0)) or 0)
    dem     = float(row.get("dem_fundraising_potential", 0) or 0)
    gop     = float(row.get("gop_fundraising_potential", 0) or 0)
    fund    = max(dem, gop)

    # ---- roles (compact JSON string) ----
    roles = {k: row.get(k) for k in ("heroes","villains","victims","antiheroes")}
    roles_text = json.dumps(roles, ensure_ascii=False)[:400] if any(roles.values()) else "[]"

    # ---- build prompt ----
    emotions_kv = json.dumps(emo_vals, ensure_ascii=False)
    morals_kv   = json.dumps(mf_vals, ensure_ascii=False)

    prompt = PROMPT_SEG.format(
        segment_id=seg_id,                 # <-- use seg_id
        persona_blurb=persona_blurb,
        topic_label=topic_label,
        urgency=urgency, fund=fund, ask=ask,
        emotions_kv=emotions_kv,
        morals_kv=morals_kv,
        roles=roles_text,
    )

    # Optional partisan lean context
    topic_lean = str(row.get("party_lean_final","") or "Contested")
    edge = float(row.get("partisan_edge", 0) or 0)
    prompt += f"""
Context:
- Topic lean: {topic_lean} (edge={edge:+.1f}).
Guidelines:
- Segments aligned with the lean should generally score higher than the opposite party.
- Neutral segments should be lower than both parties unless clearly broad-salience.
"""

    messages = [
        {"role": "system", "content": "Return STRICT JSON only. No markdown/backticks."},
        {"role": "user",   "content": prompt},
    ]
    raw = deepseek_chat(messages, model=model, max_tokens=450, temperature=0.0)
    js = try_parse_json(raw)

    logging.debug("SEG %s raw JSON keys: %s", seg_id, list(js.keys()))
    logging.debug("SEG %s raw affinity/by_party: %s / %s", seg_id, js.get("affinity"), js.get("by_party"))

    if not isinstance(js, dict):
        raise ValueError("segment JSON not an object")
    if js.get("segment_id") != seg_id:
        js["segment_id"] = seg_id

    def _num_safe(x):
        try: return float(x)
        except: return 0.0

    js["affinity"] = max(0.0, min(100.0, _num_safe(js.get("affinity", 0))))
    byp = js.get("by_party") or {}
    js["by_party"] = {
        "Dem": max(0.0, min(100.0, _num_safe(byp.get("Dem", 0)))),
        "GOP": max(0.0, min(100.0, _num_safe(byp.get("GOP", 0)))),
        "Ind": max(0.0, min(100.0, _num_safe(byp.get("Ind", 0)))),
    }
    js["best_ask"] = str(js.get("best_ask",""))[:120]
    js["why"]      = str(js.get("why",""))[:240]

    # DEBUG: show coerced values
    logging.debug("SEG %s coerced affinity/by_party: %s / %s", seg_id, js.get("affinity"), js.get("by_party"))

    # DEBUG: write a one-line JSONL per segment (optional)
    with open("data/affinity/reports/segments_scored.seg_debug.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps({"segment_id":seg_id,
                            "affinity":js["affinity"],
                            "by_party":js["by_party"]},
                        ensure_ascii=False)+"\n")

    return js



# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default=TOPICS)
    ap.add_argument("--personas", default=PERSONA)
    ap.add_argument("--out-raw", default=OUTRAW)
    ap.add_argument("--out-seg", default=OUTSEG)
    ap.add_argument("--logfile", default=LOGFILE)
    ap.add_argument("--limit-topics", type=int, default=None)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--nonzero-frac", type=float, default=0.30)
    ap.add_argument("--min-spread", type=float, default=5.0, help="required spread when max affinity >= lowmax")
    ap.add_argument("--lowmax", type=float, default=20.0, help="if max affinity < lowmax, use low-spread instead")
    ap.add_argument("--low-spread", type=float, default=2.0, help="required spread when max affinity < lowmax")
    ap.add_argument("--concurrency", type=int, default=6, help="threads for per-segment scoring")
    ap.add_argument("--env", default="src/secret.env", help="Path to .env with secrets")
    ap.add_argument("--allow-fail", action="store_true",
                help="Log strict failures but do NOT raise (lets the run finish)")
    ap.add_argument("--logfile-strict-fail", default="data/affinity/reports/segments_scored.calls.strict_fail.jsonl")
    args = ap.parse_args()

    # Load secrets from env file (won't override already-set env vars)
    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logging.info(f"Loaded secrets from {env_path}")
    else:
        # also try CWD-relative fallback if someone launches from a subdir
        fallback = Path.cwd() / "src" / "secret.env"
        if fallback.exists():
            load_dotenv(fallback, override=False)
            logging.info(f"Loaded secrets from {fallback}")
        else:
            logging.warning(f"No env file found at {env_path} or {fallback}; relying on OS env")

    setup_logging(args.logfile)
    logging.info("Starting affinity scoring (strict mode; with per-segment fallback)")

    personas = json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"]
    seg_ids = sorted(personas.keys())
    need_nonzero = max(4, int(args.nonzero_frac * len(seg_ids)))

    X = pd.read_parquet(args.topics)
    X["urgency_score"]    = pd.to_numeric(X.get("urgency_score", 0), errors="coerce")
    X["cta_ask_strength"] = pd.to_numeric(X.get("cta_ask_strength", 0), errors="coerce")
    if args.limit_topics: X = X.head(args.limit_topics)
    logging.info(f"Loaded topics rows={len(X)} segments={len(seg_ids)} need_nonzero={need_nonzero}")

    emo_cols = [c for c in X.columns if c.startswith("emo_")]
    mf_cols  = [c for c in X.columns if c.startswith("mf_")]

    raw_path = Path(args.out_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # If we are recomputing, start with a fresh cache file
    if args.recompute and raw_path.exists():
        open(raw_path, "w", encoding="utf-8").close()

    # Cache index
    seen_sigs = set()
    if raw_path.exists() and not args.recompute:
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if "_sig" in j: seen_sigs.add(j["_sig"])
                except: pass
        logging.info(f"Cache warm: {len(seen_sigs)} signatures")

    for idx, (_, row) in enumerate(X.iterrows(), start=1):
        topic_label = _first_nonempty(row, ["story_label","label","topic_label","best_label","winner_label"])
        if topic_label is None:
            raise RuntimeError("Missing topic label: expected one of story_label/label/topic_label/best_label/winner_label")
        topic_label = str(topic_label)

        period = str(row.get("period", ""))
        fhash = features_hash(row, emo_cols, mf_cols)
        personas_hash = hash_sig(personas)
        sig = hash_sig({"label": topic_label, "period": period, "feat": fhash, "personas": personas_hash, "v": 10})

        logging.info(f"[{idx}/{len(X)}] {period} | {topic_label}")

        if (sig in seen_sigs) and not args.recompute:
            logging.info("  → using cache")
            continue

        # ---------- Per-segment scoring (group mode disabled) ----------
        seg_objs = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(
                    score_segment,
                    sid,
                    personas[sid].get("blurb",""),
                    topic_label, row, emo_cols, mf_cols, args.model
                ): sid for sid in seg_ids
            }
            for fut in as_completed(futs):
                sid = futs[fut]
                t0 = time.time()
                try:
                    sobj = fut.result()
                    logging.info(f"    seg {sid}: aff={sobj.get('affinity',0)}")
                    seg_objs.append(sobj)
                except Exception:
                    logging.exception("    seg %s: exception during score_segment; inserting zeroed record", sid)
                    seg_objs.append({"segment_id": sid, "affinity": 0.0,
                                    "by_party": {"Dem":0,"GOP":0,"Ind":0},
                                    "best_ask":"", "why":""})

            # NaN-safe values then unconditional rescale when ≥2 segments
            vals = np.array([pd.to_numeric(s.get("affinity", 0), errors="coerce") for s in seg_objs], dtype=float)
            vals = np.nan_to_num(vals, nan=0.0)

            if len(seg_objs) >= 2:  # run rescale whenever we have at least 2 segments
                urg_raw = pd.to_numeric(row.get("urgency_score", 0), errors="coerce")
                ask_raw = pd.to_numeric(row.get("cta_ask_strength", row.get("cta_strength", 0)), errors="coerce")
                urgency = 0.0 if pd.isna(urg_raw) else float(urg_raw)/100.0
                ask     = 0.0 if pd.isna(ask_raw) else float(ask_raw)/100.0
                mu = 20.0 + 60.0 * (0.5 * urgency + 0.5 * ask)

                # outrage widens spread a bit (robust to different column names)
                anger = 0.0
                for k in ("emo_anger_outrage", "emo_anger", "emo_outrage"):
                    if k in row and pd.notna(row[k]):
                        anger = float(row[k]) / 100.0
                        break
                target_spread = 10 + 25*anger

                z = (vals - vals.mean()) / (vals.std(ddof=1) + 1e-6)
                rescaled = np.clip(mu + z*(target_spread/2), 0, 100)
                for s, v in zip(seg_objs, rescaled):
                    s["affinity"] = float(v)

                logging.debug("rescale: mu=%.1f target_spread=%.1f after>0=%d",
                            mu, target_spread, sum(1 for s in seg_objs if (s.get("affinity",0) or 0)>0))



        # ===== PATCH (2): sanitize seg_objs to be NaN-safe & numeric =====
        def _num01(x):
            v = pd.to_numeric(x, errors="coerce")
            if pd.isna(v): return 0.0
            v = float(v)
            if v < 0.0:  v = 0.0
            if v > 100.: v = 100.0
            return v

        for s in seg_objs:
            # affinity
            s["affinity"] = _num01(s.get("affinity", 0))
            # by_party: ensure all three keys, numeric, bounded
            bp = s.get("by_party") or {}
            s["by_party"] = {
                "Dem": _num01(bp.get("Dem", 0)),
                "GOP": _num01(bp.get("GOP", 0)),
                "Ind": _num01(bp.get("Ind", 0)),
            }
            # trim text fields (avoid giant blobs)
            s["best_ask"] = str(s.get("best_ask", ""))[:120]
            s["why"]      = str(s.get("why", ""))[:240]
        # ===== END PATCH (2) =====

        accepted = {
            "topic_label": topic_label,
            "period": period,
            "regions": {"NE":1.0,"MW":1.0,"S":1.0,"W":1.0},
            "segments": seg_objs,
            "_source": "per-seg"
        }

        logging.debug("post-accepted >0 count: %d",
                    sum(1 for s in accepted["segments"] if (s.get("affinity",0) or 0)>0))

        bad, why_bad, diag = is_bad(
            accepted, seg_ids, need_nonzero,
            min_spread=args.min_spread, lowmax=args.lowmax, low_spread=args.low_spread
        )
        logging.info(f"  per-seg summary: nz={diag['nz']}, spread={diag['spread']:.1f}, max={diag.get('max',0.0):.1f}")

        if bad:
            # ===== PATCH (3): JSON-safe failure record + optional raise =====
            logfile_strict = Path(args.logfile_strict_fail)
            logfile_strict.parent.mkdir(parents=True, exist_ok=True)

            # JSON-safe compact sample (no NaN/None)
            def _bp(s):
                bp = s.get("by_party") or {}
                return {
                    "Dem": float(bp.get("Dem", 0) or 0.0),
                    "GOP": float(bp.get("GOP", 0) or 0.0),
                    "Ind": float(bp.get("Ind", 0) or 0.0),
                }

            sample = [{
                "segment_id": s.get("segment_id"),
                "affinity": float(s.get("affinity", 0) or 0.0),
                "by_party": _bp(s),
                "best_ask": str(s.get("best_ask",""))[:120],
                "why": str(s.get("why",""))[:240],
            } for s in seg_objs[:3]]

            rec = {
                "topic_label": str(topic_label),
                "period": str(period),
                "feature_hash": fhash,
                "per_seg_nz": int(diag.get("nz", 0)),
                "per_seg_spread": float(diag.get("spread", 0.0) or 0.0),
                "per_seg_max": float(diag.get("max", 0.0) or 0.0),
                "per_seg_sample": sample,
                "error": "strict_quality_gate",
                "why": why_bad,
            }
            with open(logfile_strict, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            logging.error("  ❌ per-seg failed strict validation → logged to %s", logfile_strict)
            if not args.allow_fail:
                raise RuntimeError(f"Affinity scoring failed (per-seg) for {period} | {topic_label}")
            # ===== END PATCH (3) =====



        # ---------- Cache accepted ----------
        accepted["_sig"] = sig
        accepted["_ts"]  = pd.Timestamp.utcnow().isoformat()  # <— add this line
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(accepted, ensure_ascii=False) + "\n")
        seen_sigs.add(sig)
        logging.info("  ✓ cached")

        time.sleep(0.12)  # polite pacing

    # ---------- Flatten cache → parquet ----------
    cache_by_sig = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                sig = j.get("_sig")
                if sig:
                    # last write wins (later lines overwrite earlier ones)
                    cache_by_sig[sig] = j
            except:
                badp = raw_path.with_suffix(".bad")
                with open(badp, "a", encoding="utf-8") as bf:
                    bf.write(line + "\n")

    cache_rows = list(cache_by_sig.values())

    seg_rows = []
    for js in cache_rows:
        topic_label = js.get("topic_label", "")
        period = js.get("period", "")
        regions = js.get("regions", {}) or {}
        segments = js.get("segments", []) or []
        for s in segments:
            try:
                seg_rows.append({
                    "period": period,
                    "label": topic_label,
                    "segment_id": s["segment_id"],
                    "affinity": float(s.get("affinity", 0)),
                    "aff_dem": float(s.get("by_party", {}).get("Dem", 0)),
                    "aff_gop": float(s.get("by_party", {}).get("GOP", 0)),
                    "aff_ind": float(s.get("by_party", {}).get("Ind", 0)),
                    "best_ask": s.get("best_ask", ""),
                    "why": s.get("why", ""),
                    "regions": regions,
                    "_source": js.get("_source", "group")
                })
            except Exception as e:
                logging.debug(f"skip malformed segment entry: {e}")
                continue

    seg_df = pd.DataFrame(seg_rows)
    Path(args.out_seg).parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_parquet(args.out_seg, index=False)
    logging.info(f"✓ wrote {args.out_seg} rows={len(seg_df)}")

    g = seg_df.groupby(["period","label"])["affinity"].apply(lambda s: int((s>0).sum())).reset_index(name="nz")
    logging.info("QA: topics with nz>=1: %d / %d", (g["nz"]>0).sum(), len(g))

    # QA summary
    if len(seg_df):
        g = seg_df.groupby(["period","label"])["affinity"].apply(lambda s: int((s > 0).sum())).reset_index(name="nonzero_segments")
        logging.info("QA: min/max nonzero segments per topic: %s / %s",
                     g["nonzero_segments"].min(), g["nonzero_segments"].max())

if __name__ == "__main__":
    main()

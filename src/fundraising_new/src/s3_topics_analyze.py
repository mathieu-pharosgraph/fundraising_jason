#!/usr/bin/env python3
"""
Analyze accepted topic clusters (from data/topics) with DeepSeek to produce
fundraising-relevant metrics. Incremental via hash cache.

Inputs  (from scripts/topics_build.py):
  - data/topics/items.parquet
  - data/topics/clusters.parquet
  - data/topics/cluster_meta.parquet
  - data/topics/topic_events.parquet (optional)

Outputs:
  - data/topics/metrics/topic_metrics.parquet        (wide table)
  - data/topics/metrics/topic_metrics.jsonl          (row-wise JSON, audit)
  - data/topics/metrics/_cache_signatures.parquet    (signatures processed)

CLI:
  python scripts/topics_analyze.py \
    --topics-dir data/topics \
    --out-dir data/topics/metrics \
    --by day \
    --start 2025-08-27 --end 2025-09-07 \
    --threshold 60 \
    --max-snips 12
"""
from __future__ import annotations
import os, re, json, argparse, hashlib, textwrap, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

import requests as rq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- env ----------
from pathlib import Path
from dotenv import load_dotenv, find_dotenv




# Try: (1) explicit CLI later, (2) ../../secret.env (repo/src/secret.env), (3) ../../secret.env at repo root, (4) any .env upward
HERE = Path(__file__).resolve()
CANDIDATES = [
    HERE.parents[2] / "secret.env",       # .../repo/src/secret.env
    HERE.parents[3] / "secret.env",       # .../repo/secret.env (just in case)
]
loaded = False
for p in CANDIDATES:
    if p.exists():
        load_dotenv(p)
        loaded = True
        break
if not loaded:
    # fallback: load any .env found upward (optional)
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")


# ---------- LLM ----------
_session = None
def _get_session():
    global _session
    if _session is None:
        _session = rq.Session()
        retry = Retry(
            total=5, connect=5, read=5, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(['POST'])
        )
        _session.mount("https://", HTTPAdapter(max_retries=retry))
        _session.mount("http://", HTTPAdapter(max_retries=retry))
    return _session

def deepseek_chat(messages, model=DEEPSEEK_MODEL, max_tokens=600, temperature=0.2, timeout=90):
    base = DEEPSEEK_API_URL.rstrip("/")
    url  = f"{base}/chat/completions"
    key  = DEEPSEEK_API_KEY
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env (topics_analyze)")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    s = _get_session()
    r = s.post(url, headers=hdr, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def _normalize_cta(obj):
    """
    Accepts cta dicts with variant keys and returns a normalized dict:
    {"ask_type": <str>, "ask_strength": <num>, "copy": <str>}
    """
    import re
    if not isinstance(obj, dict):
        return {}
    def kfix(k):
        k = str(k).lower().strip()
        k = k.replace("-", "_").replace(" ", "_")
        return k
    d = {kfix(k): v for k, v in obj.items()}

    ask_type     = d.get("ask_type") or d.get("type") or d.get("cta_type")
    ask_strength = d.get("ask_strength") or d.get("strength") or d.get("score")
    copy         = d.get("copy") or d.get("cta_copy") or d.get("text") or d.get("message")

    try:
        ask_strength = float(ask_strength) if ask_strength is not None else None
    except Exception:
        ask_strength = None

    if isinstance(ask_type, str): ask_type = ask_type.strip()
    if isinstance(copy, str):     copy     = copy.strip()

    out = {}
    if ask_type not in ("", None): out["ask_type"] = ask_type
    if ask_strength is not None:   out["ask_strength"] = ask_strength
    if copy not in ("", None):     out["copy"] = copy
    return out


def normalize_roles(x):
    # accept dict/string/None → dict of lists
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: x = {}
    if not isinstance(x, dict): x = {}
    out = {}
    for k in ["heroes","villains","victims","antiheroes"]:
        v = x.get(k, [])
        if isinstance(v, str): v = [v] if v.strip() else []
        if not isinstance(v, list): v = []
        v = [s.strip() for s in v if isinstance(s,str) and s.strip()]
        out[k] = list(dict.fromkeys(v))[:5]
    return out

ROLES_ONLY_PROMPT = """Extract ROLES from these snippets. Return strict JSON only:
{{"heroes":[], "villains":[], "victims":[], "antiheroes":[]}}
Snippets:
---
{snips}
---"""

# ---------- prompt ----------
VERIFY_PROMPT = """You score a US political TOPIC for fundraising relevance.

Return STRICT JSON with this schema:
{{
  "label": "<topic label 3-6 words>",
  "urgency_score": 0-100,
  "urgency_rationale": "<1 sentence why>",
  "emotions": {{
    "fear": 0-100, "anger_outrage": 0-100, "sadness": 0-100,
    "hope_optimism": 0-100, "pride": 0-100, "disgust": 0-100, "anxiety": 0-100
  }},
  "emotions_top": "<one of: fear|anger_outrage|sadness|hope_optimism|pride|disgust|anxiety>",
  "moral_foundations": {{
    "harm": 0-100, "care": 0-100, "fairness": 0-100, "cheating": 0-100,
    "loyalty": 0-100, "betrayal": 0-100, "authority": 0-100, "subversion": 0-100,
    "sanctity": 0-100, "degradation": 0-100, "liberty": 0-100, "oppression": 0-100
  }},
  "roles": {{
    "heroes":   ["<entity/person/org>", ...],
    "villains": ["<entity/person/org>", ...],
    "victims":  ["<entity/person/org>", ...],
    "antiheroes": ["<entity/person/org>", ...]
  }},
  "fundraising_hooks": {{
    "threat_or_loss": true|false,
    "deadline_or_timing": true|false,
    "identity_activation": true|false,
    "clear_villain": true|false,
    "actionability": true|false
  }},
  "cta": {{
    "ask_strength": 0-100,
    "ask_type": "<one of: donate_now|end_of_month_match|legal_defense_fund|field_program|ad_buy|aid_relief|petition_then_donate>",
    "copy": "<2 short sentences of donation copy>"
  }},
  "notes": "<OPTIONAL 1 sentence>"
}}

Guidelines:
- Scores are relative to US political fundraising norms.
- "Urgency": imminent vote/court/ban/deadline/emergency increases score.
- Emotions: distribute realistically; do not set all high.
- Moral foundations: score both sides of each pair separately (e.g., harm vs care).
- Roles: list proper nouns or groups if explicit; else leave empty lists.
- Hooks: set true if present in the snippets (not speculation).
- CTA: concise, actionable, non-generic; reflect the topic framing.

SNIPPETS (bullet list, representative for ONE topic):
---
{snips}
---

Return JSON ONLY.
"""

# ---------- util ----------
def safe_json_load(s: str) -> Dict[str, Any]:
    s = s.strip().strip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError("LLM returned non-JSON")

def hash_signature(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def truncate_words(s: str, n: int = 80) -> str:
    toks = re.split(r"\s+", s or "")
    return " ".join(toks[:n])

# ---------- core ----------
def build_groups(items: pd.DataFrame,
                 clusters: pd.DataFrame,
                 meta: pd.DataFrame,
                 mode: str = "day",
                 start: str | None = None,
                 end: str | None = None,
                 threshold: int = 60) -> pd.DataFrame:
    """Return groups to score: one row per (cluster_id, period) with label and item_ids."""
    # accepted clusters
    acc = meta[(meta["us_relevance"]) & (meta["fundraising_usable"]) &
               (meta["fundraising_score"] >= threshold)][["cluster_id","label"]].drop_duplicates()

    df = (items[["item_id","title","source","url","published_at","text"]]
          .merge(clusters[["item_id","cluster_id","cluster_prob"]], on="item_id", how="inner")
          .merge(acc, on="cluster_id", how="inner"))
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if start:
        df = df[df["published_at"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["published_at"] <= pd.to_datetime(end, utc=True)]

    if mode == "day":
        df["period"] = df["published_at"].dt.date.astype(str)
        key_cols = ["cluster_id","label","period"]
    else:
        df["period"] = "all"
        key_cols = ["cluster_id","label","period"]

    # aggregate: keep top-N by cluster_prob per group (decided later)
    df = df.sort_values(["cluster_id","published_at","cluster_prob"], ascending=[True, False, False])
    return df, key_cols

def make_snippets(sub: pd.DataFrame, max_snips: int = 12) -> List[str]:
    sub = sub.sort_values(["cluster_prob","published_at"], ascending=[False, False]).head(max_snips)
    out = []
    for _, r in sub.iterrows():
        title = (r.get("title") or "").strip()
        src   = (r.get("source") or "").strip()
        url   = (r.get("url") or "").strip()
        txt   = truncate_words((r.get("text") or "").strip(), 120)
        piece = f"- {title} — {src}\n  {txt}\n  {url}"
        out.append(piece)
    return out

def score_group(label: str, snippets: List[str]) -> Dict[str, Any]:
    snips = "\n\n".join(snippets)
    prompt = VERIFY_PROMPT.format(snips=snips)
    messages = [
        {"role":"system","content":"You are a strict JSON scoring assistant for US political fundraising."},
        {"role":"user","content": prompt}
    ]
    raw = deepseek_chat(messages, max_tokens=800, temperature=0.15)
    return safe_json_load(raw)

# ---------- IO helpers ----------
def load_topics_dir(topics_dir: Path) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    items = pd.read_parquet(topics_dir/"items.parquet")
    clusters = pd.read_parquet(topics_dir/"clusters.parquet")
    meta = pd.read_parquet(topics_dir/"cluster_meta.parquet")
    return items, clusters, meta

def load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    return pd.DataFrame(columns=["signature","cluster_id","label","period"])

def save_metrics(out_dir: Path, rows: List[Dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet table (wide)
    df = pd.DataFrame(rows)
    p_path = out_dir/"topic_metrics.parquet"
    if p_path.exists():
        prev = pd.read_parquet(p_path)
        df = pd.concat([prev, df], ignore_index=True)
        # dedup on signature keep last
        df = df.sort_values("created_at").drop_duplicates("signature", keep="last")
    # ensure nested dicts survive parquet as JSON strings
    for col in ["emotions","moral_foundations","roles","fundraising_hooks","cta"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x,(dict,list)) else x)
    df.to_parquet(p_path, index=False)
    # JSONL audit
    j_path = out_dir/"topic_metrics.jsonl"
    with open(j_path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _materialize_spiders_from_metrics(metrics_parquet: Path, out_csv: Path):
    """Flatten emotions/moral/cta/roles from topic_metrics.parquet to per-(period,label_key) CSV."""
    import json, ast, re
    import pandas as pd

    def _nkey(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

    if not metrics_parquet.exists():
        print(f"[spiders] metrics parquet missing: {metrics_parquet}"); return

    df = pd.read_parquet(metrics_parquet)  # expected: label, period, emotions, moral_foundations, cta, roles
    df["period_norm"] = pd.to_datetime(df.get("period",""), errors="coerce").dt.date.astype("string")
    df["label_key"]   = df.get("label","").astype(str).apply(_nkey)

    def _to_obj(x):
        if isinstance(x, dict): return x
        s = str(x or "").strip()
        if not s: return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                o = parser(s); return o if isinstance(o, dict) else {}
            except Exception: pass
        return {}

    emo   = pd.json_normalize(df["emotions"].map(_to_obj)).add_prefix("emo_")
    mf    = pd.json_normalize(df["moral_foundations"].map(_to_obj)).add_prefix("mf_")
    cta   = pd.json_normalize(df["cta"].map(_to_obj).map(_normalize_cta)).add_prefix("cta_")
    roles = pd.json_normalize(df["roles"].map(_to_obj))[["heroes","villains","victims","antiheroes"]]

    tall = pd.concat([df[["period_norm","label_key"]], emo, mf, cta, roles], axis=1)

    num_cols = [c for c in tall.columns if c.startswith(("emo_","mf_","cta_ask_strength"))]

    def _first_nonempty(series):
        s = series.dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        return s.iloc[0] if not s.empty else None


    agg = (tall.groupby(["period_norm","label_key"], as_index=False)
            .agg({**{c:"mean" for c in num_cols},
                    **{c:_first_nonempty for c in ["cta_ask_type","cta_copy","heroes","villains","victims","antiheroes"]
                    if c in tall.columns}}))



    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"[spiders] wrote {out_csv} rows={len(agg)}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-dir", default="data/topics")
    ap.add_argument("--out-dir", default="data/topics/metrics")
    ap.add_argument("--by", choices=["day","all"], default="day")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--threshold", type=int, default=60)
    ap.add_argument("--max-snips", type=int, default=12)
    ap.add_argument("--recompute", action="store_true", help="ignore cache and recompute")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    topics_dir = Path(args.topics_dir); out_dir = Path(args.out_dir)
    items, clusters, meta = load_topics_dir(topics_dir)
    df, key_cols = build_groups(items, clusters, meta,
                                mode=args.by, start=args.start, end=args.end,
                                threshold=args.threshold)

    # build cache key per group
    cache_path = out_dir/"_cache_signatures.parquet"
    cache = load_cache(cache_path)
    new_rows = []
    processed = 0

    for (cid, label, period), sub in df.groupby(["cluster_id", "label", "period"]):
        snippets = make_snippets(sub, max_snips=args.max_snips)
        payload = {"cluster_id": int(cid), "label": str(label), "period": str(period), "snippets": snippets}
        sig = hash_signature(payload)

        # skip if in cache (unless recompute)
        if not args.recompute and ((cache["signature"] == sig).any()):
            continue

        if args.skip_llm:
            row = {
                "signature": sig,
                "cluster_id": int(cid),
                "label": label,
                "period": period,
                "urgency_score": np.nan,
                "emotions": None,
                "moral_foundations": None,
                "roles": None,
                "fundraising_hooks": None,
                "cta": None,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
        else:
            try:
                # 1) main scoring call
                res = score_group(label, snippets)

                # normalize roles (you already do)
                roles = normalize_roles(res.get("roles"))

                # normalize CTA (NEW)
                cta_norm = _normalize_cta(res.get("cta", {}))

                row = {
                    "signature": sig,
                    "cluster_id": int(cid),
                    "label": res.get("label", label),
                    "period": period,
                    "urgency_score": res.get("urgency_score"),
                    "urgency_rationale": res.get("urgency_rationale"),
                    "emotions": res.get("emotions"),
                    "emotions_top": res.get("emotions_top"),
                    "moral_foundations": res.get("moral_foundations"),
                    "roles": roles,
                    "fundraising_hooks": res.get("fundraising_hooks"),
                    "cta": cta_norm,                # <— write the normalized CTA dict
                    "notes": res.get("notes"),
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }


            except Exception as e:
                # On any failure, write an error row; don't reference res/roles here
                row = {
                    "signature": sig,
                    "cluster_id": int(cid),
                    "label": label,
                    "period": period,
                    "error": f"{type(e).__name__}: {e}",
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }

        new_rows.append(row)
        processed += 1
        time.sleep(0.4)


    # write out
    if new_rows:
        save_metrics(out_dir, new_rows)
        # update cache
        cache_new = pd.DataFrame([{"signature": r["signature"],
                                   "cluster_id": r.get("cluster_id"),
                                   "label": r.get("label"),
                                   "period": r.get("period")} for r in new_rows])
        cache_all = pd.concat([cache, cache_new], ignore_index=True).drop_duplicates("signature", keep="last")
        cache_all.to_parquet(cache_path, index=False)

    # small summary
    p = out_dir/"topic_metrics.parquet"
    if p.exists():
        tdf = pd.read_parquet(p)
        print(f"✓ wrote/updated {len(new_rows)} rows; metrics total={len(tdf)} → {p}")
    else:
        print("No new rows; nothing written.")

    try:
        _materialize_spiders_from_metrics(out_dir/"topic_metrics.parquet",
                                        Path("data/affinity/reports/topics_enriched_spiders.csv"))
    except Exception as e:
        print(f"[spiders] materialization failed: {e}")



if __name__ == "__main__":
    main()

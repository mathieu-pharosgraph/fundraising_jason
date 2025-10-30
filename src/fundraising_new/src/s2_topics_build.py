#!/usr/bin/env python3
"""
Build topical clusters from collected news/Reddit JSON dumps, verify relevance,
and assign topic labels.

Inputs  : content_*.json (your existing files)
Outputs : data/topics/
          - items.parquet                (normalized items + fulltext)
          - embeddings.npz               (float32 [N, D])
          - clusters.parquet             (item_id → cluster_id [+ scores])
          - cluster_meta.parquet         (per-cluster summary: label, flags, reps)
          - topic_events.parquet         (daily intensity per topic label)
"""
import os, re, json, time, argparse, datetime as dt
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from fundraising_new.src.utils.keys import nkey

# Load DeepSeek key from repo's src/secret.env
ENV_PATH = Path(__file__).resolve().parents[2] / "secret.env"   # .../src/secret.env
load_dotenv(ENV_PATH)

# Optional: allow custom endpoint, else default
os.environ.setdefault("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

import numpy as np
import pandas as pd

# ---------- fulltext ----------
import trafilatura
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- resilient HTTP session for DeepSeek
import requests as rq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_DS_SESSION = None
def _ds_session():
    global _DS_SESSION
    if _DS_SESSION is None:
        s = rq.Session()
        retry = Retry(
            total=5,                 # total retries
            connect=5, read=5,      # per phase
            backoff_factor=0.8,     # 0.8, 1.6, 3.2, ...
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
        _DS_SESSION = s
    return _DS_SESSION

# ---------- embeddings ----------
EMB_BACKENDS = ("openai", "sentence-transformers")
try:
    import openai  # v1 SDK
except Exception:
    openai = None

# ---------- sbert fallback ----------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------- clustering ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import umap
import hdbscan

# ---------- LLM (DeepSeek) ----------
import requests as rq
import hashlib



# =============== helpers ===============
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

SNAP_DIR = Path("data/topics/snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)
SNAP_MANIFEST = SNAP_DIR / "_manifest.csv"

SNAP_DIR = Path("data/topics/snapshots")

def load_items(glob_pat: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path().glob(glob_pat)):
        try:
            obj = json.loads(Path(p).read_text())
            for it in obj.get("content", []):
                rows.append({
                    "file": str(p),
                    "source": it.get("source"),
                    "subreddit": it.get("subreddit"),
                    "title": it.get("title") or "",
                    "content": it.get("content") or "",
                    "description": it.get("description") or "",
                    "url": it.get("url") or "",
                    "published_at": it.get("published_at"),
                    "source_name": it.get("source_name"),
                    "author": it.get("author"),
                    "query": it.get("query"),
                })
        except Exception as e:
            print(f"⚠️  failed to read {p}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No input items found.")
    # normalize timestamps
    def parse_ts(x):
        if pd.isna(x): return pd.NaT
        try:
            if isinstance(x, (int, float)):  # reddit epoch
                return pd.to_datetime(x, unit="s", utc=True)
            return pd.to_datetime(str(x), utc=True, errors="coerce")
        except Exception:
            return pd.NaT
    df["published_at"] = df["published_at"].apply(parse_ts)
    df["date"] = df["published_at"].dt.date
    # id
    df["item_id"] = (df["source"].fillna("na").str[:3] + "_" +
                     pd.util.hash_pandas_object(df[["url","title"]], index=False).astype(str).str[-10:])
    # keep the best text we have initially
    df["text_seed"] = df[["title","description","content"]].fillna("").agg(" ".join, axis=1).str.strip()
    return df.drop_duplicates("item_id")

def fetch_fulltext_url(url: str, timeout=15) -> str:
    if not url or not isinstance(url, str): return ""
    try:
        downloaded = trafilatura.fetch_url(url, user_agent=UA, timeout=timeout)
        if downloaded:
            txt = trafilatura.extract(downloaded,
                                      favor_recall=True,
                                      include_comments=False,
                                      include_tables=False)
            return (txt or "").strip()
    except Exception:
        pass
    return ""

def hydrate_fulltext(df: pd.DataFrame, max_workers=12) -> pd.DataFrame:
    need = df["url"].fillna("") != ""
    urls = df.loc[need, "url"].tolist()
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(fetch_fulltext_url, u): u for u in urls}
        for i, f in enumerate(as_completed(fut), 1):
            u = fut[f]
            try:
                out[u] = f.result()
            except Exception:
                out[u] = ""
            if i % 200 == 0:
                print(f"  fetched {i}/{len(urls)}")
    df["fulltext"] = df["url"].map(out).fillna("")
    # choose best available text
    df["text"] = df["fulltext"]
    empty = df["text"].str.len() < 400  # if short, fallback to seed
    df.loc[empty, "text"] = df.loc[empty, "text_seed"]
    # prune garbage
    df["text"] = df["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    return df

# ---------- embeddings ----------
def embed_texts(texts: List[str], backend="sentence-transformers",
                model_name="sentence-transformers/all-mpnet-base-v2",
                batch=64) -> np.ndarray:
    if backend == "openai":
        if openai is None:
            raise SystemExit("openai SDK not available. pip install openai")
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        vecs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            resp = client.embeddings.create(
                model=os.environ.get("OPENAI_EMBED_MODEL","text-embedding-3-large"),
                input=chunk
            )
            vecs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
        return np.vstack(vecs)
    else:
        if SentenceTransformer is None:
            raise SystemExit("pip install sentence-transformers")
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
        return emb.astype(np.float32)

def _hash_list(xs) -> str:
    b = json.dumps(sorted(list(xs)), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]

def _load_manifest() -> pd.DataFrame:
    if SNAP_MANIFEST.exists():
        try:
            return pd.read_csv(SNAP_MANIFEST, dtype=str)
        except Exception:
            pass
    return pd.DataFrame(columns=["day","n_clusters","items_hash","written_at"])

def _save_manifest_row(day: str, n_clusters: int, items_hash: str):
    mf = _load_manifest()
    row = {"day": day, "n_clusters": str(n_clusters),
           "items_hash": items_hash, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    # upsert by (day, items_hash) — keep the last record
    mf = pd.concat([mf, pd.DataFrame([row])], ignore_index=True)
    mf = (mf.sort_values("written_at")
            .drop_duplicates(["day"], keep="last"))
    mf.to_csv(SNAP_MANIFEST, index=False)

def _date_str(ts):
    if pd.isna(ts): return None
    return pd.to_datetime(ts).date().isoformat()

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _collect_rep_titles(items_for_cluster: pd.DataFrame, k=5) -> str:
    g = (items_for_cluster
         .sort_values(["cluster_prob","published_at"], ascending=[False, True])
         .head(k))
    titles = [str(t)[:140] for t in g["title"].fillna("").tolist() if str(t).strip()]
    return " | ".join(titles)

def build_daily_snapshot(day: str,
                         items: pd.DataFrame,
                         clusters: pd.DataFrame,
                         meta: pd.DataFrame,
                         std_topics: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return one row per (cluster) that has >=1 item on `day` AND is accepted by meta gates.
    Columns are stable for apps.
    """
    # items on this day
    it = items.copy()
    it["published_at"] = pd.to_datetime(it["published_at"], errors="coerce", utc=True)
    it["day"] = it["published_at"].dt.date.astype(str)
    it = it[it["day"] == str(day)]

    if it.empty:
        return it.iloc[0:0]

    # join cluster_id + meta acceptance
    tall = (it.merge(clusters[["item_id","cluster_id","cluster_prob"]],
                     on="item_id", how="inner")
              .merge(meta, on="cluster_id", how="left"))

    # accepted gates (match your topics_build acceptance)
    acc = tall[(tall["fundraising_us_relevance"]) &
               (tall["fundraising_usable"]) &
               (pd.to_numeric(tall["fundraising_score"], errors="coerce").fillna(0) >= 60)].copy()

    if acc.empty:
        return acc.iloc[0:0]

    # attach standardized topics if provided (from s5)
    if std_topics is not None and not std_topics.empty:
        acc = acc.merge(std_topics[["cluster_id","standardized_topic_names"]],
                        on="cluster_id", how="left")
    else:
        acc["standardized_topic_names"] = ""

    # per-cluster aggregation for the day
    def agg_cluster(g):
        rid = g["cluster_id"].iloc[0]
        rep_ids = (g.sort_values(["cluster_prob","published_at"], ascending=[False, True])["item_id"]
                    .dropna().astype(str).head(5).tolist())

        return pd.Series({
            "cluster_id": int(rid),
            "label": str(g.get("label", "").iloc[0]) if "label" in g.columns else "",
            "standardized_topic_names": str(g.get("standardized_topic_names", "").iloc[0]),
            "us_relevance": bool(g.get("fundraising_us_relevance", False).max()),
            "fundraising_usable": bool(g["fundraising_usable"].max()),
            "fundraising_score": float(pd.to_numeric(g["fundraising_score"], errors="coerce").max()),
            "party_lean": str(g.get("party_lean", "").iloc[0]),
            "n_items": int(g["item_id"].nunique()),
            "first_date": str(pd.to_datetime(g["published_at"]).dt.date.min()),
            "last_date":  str(pd.to_datetime(g["published_at"]).dt.date.max()),
            "rep_titles": _collect_rep_titles(g),
            "rep_item_ids": json.dumps(rep_ids, ensure_ascii=False)
        })

    snap = (acc.groupby("cluster_id", as_index=False)
              .apply(agg_cluster)
              .reset_index(drop=True))

    snap.insert(0, "day", str(day))
    return snap

# ---------- clustering ----------
def cluster_embeddings(X: np.ndarray, min_cluster_size=12, min_samples=None,
                       umap_n=5, n_neighbors=15, min_dist=0.1, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    reducer = umap.UMAP(n_components=umap_n, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="cosine", random_state=random_state)
    U = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples, metric='euclidean',
                                cluster_selection_epsilon=0.0,
                                cluster_selection_method='eom', prediction_data=True)
    labels = clusterer.fit_predict(U)
    probs = clusterer.probabilities_
    return labels, probs

def cluster_representatives(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray,
                            top_k=5) -> pd.DataFrame:
    reps = []
    for cid in sorted(set(labels)):
        if cid == -1:  # noise
            continue
        idx = np.where(labels == cid)[0]
        # pick medoid(s) by average cosine distance
        D = cosine_distances(X[idx])
        medoid_ix = idx[np.argsort(D.mean(axis=1))[:top_k]]
        reps.append({"cluster_id": int(cid),
                     "n_items": int(len(idx)),
                     "rep_item_ids": df.iloc[medoid_ix]["item_id"].tolist()})
    return pd.DataFrame(reps)

# ---------- LLM verify/label (DeepSeek) ----------
def deepseek_chat(messages: List[Dict[str, str]], model="deepseek-chat",
                  max_tokens=300, temperature=0.1, timeout=120):
    base = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
    url  = f"{base.rstrip('/')}/chat/completions"
    key  = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    s = _ds_session()
    try:
        # tuple timeout: (connect_timeout, read_timeout)
        r = s.post(url, headers=hdr, json=payload, timeout=(10, timeout))
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except (rq.exceptions.ReadTimeout, rq.exceptions.ConnectTimeout) as e:
        # surface a clear, typed error so callers can fallback
        raise TimeoutError(f"deepseek timeout: {e}") from e
    except rq.exceptions.RequestException as e:
        # other transient HTTP issues
        raise RuntimeError(f"deepseek request error: {e}") from e



VERIFY_PROMPT = """You are vetting a TOPIC CLUSTER for a US political fundraising dashboard.

Return STRICT JSON only with this schema:
{{
  "us_relevance": true|false,
  "fundraising_usable": true|false,
  "fundraising_score": 0-100,
  "party_lean": "Dem"|"GOP"|"Neutral",
  "label": "3-6 word topic label",
  "rationale": "1-2 sentences (why this topic can/cannot motivate donations, who is moved, what frame/urgency)"
}}

Definition:
- "fundraising_usable" = The topic can plausibly be turned into a donation ask (threat/opportunity, urgency, identity/values, clear villain/beneficiary, time-boxed action).
- NOT required to literally mention fundraising.

Rubric (tick ≥2 to set fundraising_usable=true, else false):
- THREAT/LOSS or RIGHTS at stake (e.g., access, bans, safety, court decision)
- URGENCY/DEADLINES (e.g., vote, hearing, matching deadline, emergency aid)
- IDENTITY/VALUE activation (core constituency or moral foundation)
- CLEAR VILLAIN or BENEFICIARY (who/what donations would confront or protect)
- ACTIONABILITY (concrete next steps a campaign/NGO could fund: legal fight, field, ads, aid)

Also decide party_lean (Dem/GOP/Neutral) if it obviously benefits one side’s fundraising more.

Now judge these cluster snippets:
---
{snips}
---
Return ONLY the JSON."""


VOTING_PROMPT = """You are vetting a TOPIC CLUSTER for US political voter engagement and mobilization.

Return STRICT JSON only with this schema:
{
  "us_relevance": true|false,
  "voting_usable": true|false,
  "voting_score": 0-100,
  "party_lean": "Dem"|"GOP"|"Neutral",
  "label": "3-6 word topic label",
  "rationale": "1-2 sentences (why this topic can/cannot motivate voter support/engagement)"
}

Definition:
- "voting_usable" = The topic can plausibly be turned into voter mobilization (emotional resonance, values alignment, threat/opportunity, clear stakes for voters).

Rubric (tick ≥2 to set voting_usable=true, else false):
- EMOTIONAL RESONANCE (evokes strong feelings - anger, hope, fear, pride)
- VALUES ALIGNMENT (connects to core voter values/identity)
- CLEAR STAKES FOR VOTERS (personal impact on voters' lives/rights/communities)
- URGENCY/TIMELINESS (election relevance, immediate consequences)
- SHARED NARRATIVE (fits broader political narrative that motivates base)

Also decide party_lean (Dem/GOP/Neutral) if it obviously benefits one side's voter mobilization more.

Now judge these cluster snippets:
---
{snips}
---
Return ONLY the JSON."""

def llm_verify_label_voting(snippets: List[str]) -> Dict[str, Any]:
    """LLM evaluation specifically for voting engagement potential"""
    snips = "\n\n".join(f"- {s[:5000]}" for s in snippets[:10])
    
    msg = [
        {"role":"system","content": "Return STRICT JSON only. No backticks, no markdown, no commentary."},
        {"role":"user","content": VOTING_PROMPT.replace("{snips}", snips)}
    ]
    
    try:
        txt = deepseek_chat(msg, timeout=150)  # longer read timeout
    except Exception as e:
        Path("data/topics/metrics").mkdir(parents=True, exist_ok=True)
        with open("data/topics/metrics/llm_label_voting_failures.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"{type(e).__name__}: {e}",
                "kind": "voting"
            }, ensure_ascii=False) + "\n")
        return _default_voting()

    
    # Use the same robust JSON parsing as before
    try:
        js = safe_json_load(
            txt,
            schema={
                "us_relevance": False,
                "voting_usable": False,
                "voting_score": 0,
                "party_lean": "Neutral",
                "label": "",
                "rationale": ""
            }
        )
    except Exception as e:
        # Log failure and return safe fallback
        Path("data/topics/metrics").mkdir(parents=True, exist_ok=True)
        with open("data/topics/metrics/llm_label_voting_failures.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "raw": txt[:4000]
            }, ensure_ascii=False) + "\n")
        js = {
            "us_relevance": False,
            "voting_usable": False,
            "voting_score": 0,
            "party_lean": "Neutral",
            "label": "Unknown",
            "rationale": ""
        }
    
    # Coercions / bounds
    try:
        vs = float(js.get("voting_score", 0))
        js["voting_score"] = int(max(0, min(100, vs)))
    except Exception:
        js["voting_score"] = 0

    js["party_lean"] = str(js.get("party_lean", "Neutral") or "Neutral").strip()[:12]
    js["label"] = str(js.get("label", "Unknown") or "Unknown").strip()[:80]
    js["rationale"] = str(js.get("rationale", "") or "")[:500]
    js["us_relevance"] = bool(js.get("us_relevance", False))
    js["voting_usable"] = bool(js.get("voting_usable", False))
    
    return js


# ----- Robust JSON parsing helpers (drop-in) -----
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)  # e.g., ```json
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("’", "'").replace("‘", "'"))

def _find_balanced_json(s: str) -> str | None:
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        start = s.find("{", start + 1)
    return None

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([\}\]])", r"\1", s)

def _repair_single_quotes(s: str) -> str:
    # Keys: 'key': → "key":
    s = re.sub(r"(?P<pre>[\{\s,])'(?P<key>[^']+?)'\s*:", r'\g<pre>"\g<key>":', s)
    # Values: :'text' → :"text"
    s = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'',
               lambda m: ':"{}"'.format(m.group(1).replace('"','\\"')), s)
    return s

def safe_json_load(raw: str, schema: dict | None = None) -> dict:
    """
    Aggressively repair common LLM JSON issues (fences, smart quotes, trailing commas, single quotes).
    Return a dict or raise ValueError.
    """
    text = _normalize_quotes(_strip_fences(raw))
    # try candidates in order
    candidates = []
    bal = _find_balanced_json(text)
    if bal: candidates.append(bal)
    candidates.append(text)
    # try fixups
    candidates += [_remove_trailing_commas(c) for c in list(candidates)]
    candidates += [_repair_single_quotes(c) for c in list(candidates)]

    last_err = None
    for c in candidates:
        try:
            js = json.loads(c)
            if isinstance(js, dict):
                # ensure schema keys exist
                if schema:
                    for k, v in schema.items():
                        js.setdefault(k, v)
                return js
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Could not parse JSON after repairs: {last_err}")

# ---- default fallbacks if the LLM call times out or errors ----
def _default_fundraising():
    return {
        "us_relevance": False,
        "fundraising_usable": False,
        "fundraising_score": 0,
        "party_lean": "Neutral",
        "label": "Unknown",
        "rationale": ""
    }

def _default_voting():
    return {
        "us_relevance": False,
        "voting_usable": False,
        "voting_score": 0,
        "party_lean": "Neutral",
        "label": "Unknown",
        "rationale": ""
    }

def llm_verify_label(snippets: List[str]) -> Dict[str, Any]:
    """
    Ask the LLM to vet/label a topic cluster.
    Always returns a dict with fields:
      us_relevance (bool), fundraising_usable (bool), fundraising_score (int 0-100),
      party_lean ("Dem"|"GOP"|"Neutral"), label (str), rationale (str)
    """
    snips = "\n\n".join(f"- {s[:5000]}" for s in snippets[:10])

    # Harden the system message to reduce junk
    sys_msg = "Return STRICT JSON only. No backticks, no markdown, no commentary."

    msg = [
        {"role":"system","content": sys_msg},
        {"role":"user","content": VERIFY_PROMPT.replace("{snips}", snips)}

    ]

    try:
        txt = deepseek_chat(msg, timeout=150)  # longer read timeout
    except Exception as e:
        Path("data/topics/metrics").mkdir(parents=True, exist_ok=True)
        with open("data/topics/metrics/llm_label_failures.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"{type(e).__name__}: {e}",
                "kind": "fundraising"
            }, ensure_ascii=False) + "\n")
        return _default_fundraising()


    # Try robust parse
    try:
        js = safe_json_load(
            txt,
            schema={
                "us_relevance": False,
                "fundraising_usable": False,
                "fundraising_score": 0,
                "party_lean": "Neutral",
                "label": "",
                "rationale": ""
            }
        )
    except Exception as e:
        # Log the raw failure for later inspection and return a safe fallback
        Path("data/topics/metrics").mkdir(parents=True, exist_ok=True)
        with open("data/topics/metrics/llm_label_failures.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "raw": txt[:4000],
                "snippet_preview": snips[:1000]
            }, ensure_ascii=False) + "\n")
        js = {
            "us_relevance": False,
            "fundraising_usable": False,
            "fundraising_score": 0,
            "party_lean": "Neutral",
            "label": "Unknown",
            "rationale": ""
        }

    # Coercions / bounds
    try:
        fs = float(js.get("fundraising_score", 0))
        js["fundraising_score"] = int(max(0, min(100, fs)))
    except Exception:
        js["fundraising_score"] = 0

    js["party_lean"] = str(js.get("party_lean", "Neutral") or "Neutral").strip()[:12]
    js["label"]      = str(js.get("label", "Unknown") or "Unknown").strip()[:80]
    js["rationale"]  = str(js.get("rationale", "") or "")[:500]

    # booleans
    js["us_relevance"]      = bool(js.get("us_relevance", False))
    js["fundraising_usable"]= bool(js.get("fundraising_usable", False))

    return js


# =============== main ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="src/fundraising_new/src/data/content_*.json")
    ap.add_argument("--outdir", default="data/topics")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--embed-backend", choices=EMB_BACKENDS, default="sentence-transformers")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=20000)
    ap.add_argument("--write-snapshots", action="store_true",
                help="Write append-only daily snapshots of accepted clusters.")
    ap.add_argument("--snapshot-start", default=None,
                    help="YYYY-MM-DD; if set, write snapshots for days >= this.")
    ap.add_argument("--snapshot-end", default=None,
                    help="YYYY-MM-DD; if set, write snapshots for days <= this.")
    ap.add_argument("--snapshot-force", action="store_true",
                    help="Overwrite an existing daily snapshot if present (default: skip).")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("⏳ loading items...")
    df = load_items(args.input_glob)
    if args.start:
        df = df[df["date"] >= pd.to_datetime(args.start).date()]
    if args.end:
        df = df[df["date"] <= pd.to_datetime(args.end).date()]
    df = df.sort_values("published_at").tail(args.max_items).reset_index(drop=True)

    print("items date min/max:",
        df["published_at"].min(), "→", df["published_at"].max(),
        "rows=", len(df))

    print("⏳ fetching full text (trafilatura)...")
    df = hydrate_fulltext(df)
    n_has = (df["text"].str.len() >= 400).sum()
    print(f"✓ fulltext coverage >=400 chars: {n_has}/{len(df)}")

    items_path = outdir / "items.parquet"
    df.to_parquet(items_path, index=False)
    print(f"✓ wrote {items_path} rows={len(df)}")

    print("⏳ computing embeddings...")
    texts = df["text"].fillna("").tolist()
    if args.embed_backend == "openai":
        X = embed_texts(texts, backend="openai")
    else:
        X = embed_texts(texts, backend="sentence-transformers", model_name=args.sbert_model)
    np.savez_compressed(outdir / "embeddings.npz", X=X)
    print(f"✓ embeddings shape={X.shape}")

    print("⏳ clustering (UMAP + HDBSCAN)...")
    labels, probs = cluster_embeddings(X, min_cluster_size=args.min_cluster_size)
    df["cluster_id"] = labels
    df["cluster_prob"] = probs
    df.to_parquet(outdir / "clusters.parquet", index=False)

    # Build tidy frames for snapshot use
    items_df = df[["item_id","title","source","url","published_at"]].copy()
    clusters_df = df[["item_id","cluster_id","cluster_prob"]].copy()

    reps = cluster_representatives(df, X, labels, top_k=5)
    # build cluster snippets for LLM
    meta_rows = []
    for _, row in reps.iterrows():
        cid = int(row["cluster_id"])
        ridx = df.set_index("item_id").loc[row["rep_item_ids"]]
        snippets = ridx["text"].fillna("").tolist()
        
        # Get BOTH evaluations
        fundraising_verdict = llm_verify_label(snippets)
        voting_verdict = llm_verify_label_voting(snippets)
        
        meta_rows.append({
            "cluster_id": cid,
            "n_items": int(row["n_items"]),
            "rep_item_ids": row["rep_item_ids"],
            # Fundraising metrics
            "fundraising_us_relevance": bool(fundraising_verdict.get("us_relevance", False)),
            "fundraising_usable": bool(fundraising_verdict.get("fundraising_usable", False)),
            "fundraising_score": int(fundraising_verdict.get("fundraising_score", 0)),
            "fundraising_party_lean": str(fundraising_verdict.get("party_lean", "Neutral")),
            "fundraising_label": str(fundraising_verdict.get("label","Unknown")).strip()[:80],
            "fundraising_rationale": str(fundraising_verdict.get("rationale",""))[:500],
            # Voting metrics
            "voting_us_relevance": bool(voting_verdict.get("us_relevance", False)),
            "voting_usable": bool(voting_verdict.get("voting_usable", False)),
            "voting_score": int(voting_verdict.get("voting_score", 0)),
            "voting_party_lean": str(voting_verdict.get("party_lean", "Neutral")),
            "voting_label": str(voting_verdict.get("label","Unknown")).strip()[:80],
            "voting_rationale": str(voting_verdict.get("rationale",""))[:500],
        })
        
        time.sleep(1.0)  # Increase delay since we're making 2 API calls

    meta = pd.DataFrame(meta_rows).sort_values(
        ["fundraising_usable","fundraising_score","n_items"],
        ascending=[False,False,False]
    )
    meta["label"] = meta.get("fundraising_label", meta.get("voting_label", "")).astype(str)


    need = {"cluster_id","fundraising_us_relevance","fundraising_usable","fundraising_score",
            "fundraising_label","fundraising_party_lean","fundraising_rationale",
            "voting_us_relevance","voting_usable","voting_score"}
    missing = need - set(meta.columns)
    if missing:
        raise RuntimeError(f"[meta] missing expected columns: {sorted(missing)}")

    meta.to_parquet(outdir / "cluster_meta.parquet", index=False)
    print(f"✓ wrote {outdir/'cluster_meta.parquet'} clusters={len(meta)}")

    # Build daily topic intensity for BOTH fundraising AND voting relevant clusters
    fundraising_rel = meta[(meta.fundraising_us_relevance) & (meta.fundraising_usable) & (meta.fundraising_score >= 60)]
    voting_rel = meta[(meta.voting_us_relevance) & (meta.voting_usable) & (meta.voting_score >= 60)]

    # Combine both sets
    rel_cids = set(fundraising_rel["cluster_id"].tolist() + voting_rel["cluster_id"].tolist())

    df_rel = (df[df["cluster_id"].isin(rel_cids)]
          .merge(meta[["cluster_id","fundraising_label"]], on="cluster_id", how="left")
          .rename(columns={"fundraising_label": "label"}))
    topic_events = (df_rel
        .assign(day=pd.to_datetime(df_rel["published_at"]).dt.date)
        .groupby(["day","label"], as_index=False)
        .agg(items=("item_id","count")))
    topic_events["label_key"] = topic_events["label"].astype(str).map(nkey)
    topic_events.to_parquet(outdir / "topic_events.parquet", index=False)

    # Optional standardized topics from s5 (to carry into snapshots)
    S5_PATH = Path("data/topics/merged_data_with_topics.parquet")
    std_topics = pd.DataFrame()
    if S5_PATH.exists():
        s5 = pd.read_parquet(S5_PATH)
        keep = [c for c in ["cluster_id","standardized_topic_ids","standardized_topic_names"] if c in s5.columns]
        if keep:
            s5 = s5[keep].drop_duplicates("cluster_id")
            # prefer names if already present; else leave as-is
            if "standardized_topic_names" in s5.columns:
                s5["standardized_topic_names"] = s5["standardized_topic_names"].astype(str)
            std_topics = s5[["cluster_id","standardized_topic_names"]]


    # -------------------- DAILY SNAPSHOT(S) (append-only) --------------------
    if args.write_snapshots:
        snap_dir = SNAP_DIR
        _ensure_dir(snap_dir)

        # days present in items
        all_days = (pd.to_datetime(items_df["published_at"], errors="coerce", utc=True)
                    .dt.date.dropna().astype(str).sort_values().unique().tolist())

        if not all_days:
            print("[snapshot] no dates found in items — skipping")
        else:
            # restrict to window if provided
            if args.snapshot_start:
                all_days = [d for d in all_days if d >= args.snapshot_start]
            if args.snapshot_end:
                all_days = [d for d in all_days if d <= args.snapshot_end]

            for snap_date in all_days:
                out_path = snap_dir / f"{snap_date}_clusters.parquet"
                out_all  = snap_dir / f"{snap_date}_clusters_all.parquet"

                if (out_path.exists() or out_all.exists()) and not args.snapshot_force:
                    print(f"[snapshot] exists (strict or all), skip: {snap_date}")
                    continue

                # items on that day
                it = items_df.copy()
                it["day"] = pd.to_datetime(it["published_at"], errors="coerce", utc=True).dt.date.astype(str)
                it = it[it["day"] == snap_date]
                if it.empty:
                    print(f"[snapshot] no items on {snap_date}")
                    continue

                def _write_snap(meta_set: pd.DataFrame, suffix: str):
                    """Write one snapshot file for this day using the given meta set."""
                    if meta_set.empty:
                        print(f"[snapshot{suffix}] no clusters in meta set on {snap_date}")
                        return

                    # active clusters that day
                    act = (it.merge(clusters_df, on="item_id", how="inner")
                            [["cluster_id","item_id","cluster_prob"]]
                            .drop_duplicates("item_id"))

                    meta_act = meta_set.merge(act[["cluster_id"]].drop_duplicates(), on="cluster_id", how="inner")
                    if meta_act.empty:
                        print(f"[snapshot{suffix}] no clusters active on {snap_date}")
                        return
                    
                    # representatives (titles of the day)
                    rep = (it.merge(clusters_df, on="item_id", how="inner")
                            .merge(meta_act[["cluster_id"]], on="cluster_id", how="inner")
                            .sort_values(["cluster_prob","published_at"], ascending=[False, True])
                            .groupby("cluster_id")["title"]
                            .apply(lambda s: " | ".join([str(t)[:140] for t in s.head(5) if str(t).strip()]))
                            .reset_index(name="rep_titles_day"))

                    # rename fundraising_* → generic names for snapshot
                    meta_view = (meta_act.rename(columns={
                                    "fundraising_label": "label",
                                    "fundraising_party_lean": "party_lean",
                                    "fundraising_rationale": "rationale"
                                })
                                [["cluster_id","label","party_lean","fundraising_score","voting_score","rationale"]])

                    snap = (meta_view.merge(rep, on="cluster_id", how="left")
                            .assign(snapshot_date=snap_date)
                            [["snapshot_date","cluster_id","label","party_lean",
                            "fundraising_score","voting_score","rep_titles_day","rationale"]])

                    snap = snap.loc[:, ~snap.columns.duplicated()].copy()

                    # if label got split into label_x/label_y by an upstream change, normalize back to 'label'
                    if "label" not in snap.columns:
                        if "label_x" in snap.columns:
                            snap = snap.rename(columns={"label_x": "label"})
                        elif "label_y" in snap.columns:
                            snap = snap.rename(columns={"label_y": "label"})

                    # now safely compute the standardized key
                    snap["label_key"] = snap.apply(lambda r: nkey(str(r["label"])), axis=1)

                    if not std_topics.empty:
                        snap = snap.merge(std_topics, on="cluster_id", how="left")

                    out_file = snap_dir / f"{snap_date}_clusters{suffix}.parquet"
                    snap.to_parquet(out_file, index=False)
                    print(f"[snapshot{suffix}] wrote {out_file} rows={len(snap)}")


                # meta sets
                meta_accepted = meta[(meta["fundraising_us_relevance"]) &
                     (meta["fundraising_usable"]) &
                     (pd.to_numeric(meta["fundraising_score"], errors="coerce").fillna(0) >= 60)].copy()

                meta_all = meta.copy()

                # write both snapshots
                snap = _write_snap(meta_accepted, "")
                snap_all = _write_snap(meta_all, "_all")

            # update manifest after writing all days
            rows = []
            for p in sorted(snap_dir.glob("*_clusters.parquet")):
                d = p.stem.split("_clusters")[0]
                try:
                    n = len(pd.read_parquet(p))
                except Exception:
                    n = 0
                rows.append({"snapshot_date": d, "path": str(p), "n_clusters": str(n)})
            man = pd.DataFrame(rows).sort_values("snapshot_date")
            man.to_csv(snap_dir / "_manifest.csv", index=False)
            print(f"[snapshot] updated {snap_dir / '_manifest.csv'} with {len(man)} days")


    print(f"✓ wrote {outdir/'topic_events.parquet'} rows={len(topic_events)}")

    print("✅ done.")

if __name__ == "__main__":
    main()

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

# =============== helpers ===============
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

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
                  max_tokens=300, temperature=0.1):
    base = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
    url = f"{base.rstrip('/')}/chat/completions"   # <-- ensure correct endpoint
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    r = rq.post(url, headers=hdr, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]


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
        {"role":"user","content": VERIFY_PROMPT.format(snips=snips)}
    ]

    txt = deepseek_chat(msg)

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

    reps = cluster_representatives(df, X, labels, top_k=5)
    # build cluster snippets for LLM
    meta_rows = []
    for _, row in reps.iterrows():
        cid = int(row["cluster_id"])
        ridx = df.set_index("item_id").loc[row["rep_item_ids"]]
        snippets = ridx["text"].fillna("").tolist()
        verdict = llm_verify_label(snippets)
        meta_rows.append({
            "cluster_id": cid,
            "n_items": int(row["n_items"]),
            "rep_item_ids": row["rep_item_ids"],
            "us_relevance": bool(verdict.get("us_relevance", False)),
            "fundraising_usable": bool(verdict.get("fundraising_usable", False)),
            "fundraising_score": int(verdict.get("fundraising_score", 0)),
            "party_lean": str(verdict.get("party_lean", "Neutral")),
            "label": str(verdict.get("label","Unknown")).strip()[:80],
            "rationale": str(verdict.get("rationale",""))[:500],
        })

        time.sleep(0.5)  # be gentle to the API
    meta = pd.DataFrame(meta_rows).sort_values(
        ["fundraising_usable","fundraising_score","n_items"],
        ascending=[False,False,False]
    )

    meta.to_parquet(outdir / "cluster_meta.parquet", index=False)
    print(f"✓ wrote {outdir/'cluster_meta.parquet'} clusters={len(meta)}")

    # Build daily topic intensity table for relevant clusters only
    rel = meta[(meta.us_relevance) & (meta.fundraising_usable) & (meta.fundraising_score >= 60)]
    rel_cids = set(rel["cluster_id"].tolist())
    df_rel = df[df["cluster_id"].isin(rel_cids)].merge(rel[["cluster_id","label"]], on="cluster_id", how="left")
    topic_events = (df_rel
        .assign(day=pd.to_datetime(df_rel["published_at"]).dt.date)
        .groupby(["day","label"], as_index=False)
        .agg(items=("item_id","count")))
    topic_events.to_parquet(outdir / "topic_events.parquet", index=False)

    print(f"✓ wrote {outdir/'topic_events.parquet'} rows={len(topic_events)}")

    print("✅ done.")

if __name__ == "__main__":
    main()

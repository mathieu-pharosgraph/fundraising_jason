#!/usr/bin/env python3
import os, re, json, time, argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests as rq
from dotenv import load_dotenv

# ---------- robust session with retries ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_DS_SESSION = None
def _ds_session():
    global _DS_SESSION
    if _DS_SESSION is None:
        s = rq.Session()
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
        _DS_SESSION = s
    return _DS_SESSION

# ---------- env loader (robust for your repo layout) ----------
def ensure_env():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "src" / "secret.env",   # .../repo/src/secret.env
        here.parents[2] / "secret.env",           # .../repo/secret.env
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break
    os.environ.setdefault("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

# ---------- llm ----------
def deepseek_chat(messages: List[Dict[str,str]], model: str = None,
                  max_tokens=400, temperature=0.1, timeout=150) -> str:
    base = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    key  = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    if model is None:
        model = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

    hdr = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    pl  = {"model": model, "messages": messages,
           "max_tokens": max_tokens, "temperature": temperature}

    s = _ds_session()
    try:
        # tuple timeout: (connect, read)
        r = s.post(url, headers=hdr, json=pl, timeout=(10, timeout))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except (rq.exceptions.ReadTimeout, rq.exceptions.ConnectTimeout) as e:
        raise TimeoutError(f"deepseek timeout: {e}") from e
    except rq.exceptions.RequestException as e:
        raise RuntimeError(f"deepseek request error: {e}") from e


# ---------- Dual Prompts for Fundraising vs Voting ----------
FUNDRAISING_ROLES_PROMPT = """Extract role lists from these US political news/reddit snippets for FUNDRAISING context.

Return STRICT JSON only:
{"heroes":[], "villains":[], "victims":[], "antiheroes":[]}

Rules for FUNDRAISING:
- Focus on entities that drive donation motivations (threats, opportunities, clear antagonists)
- Heroes: Who donors would want to support/protect
- Villains: Who donors would want to oppose/defeat  
- Victims: Who donors would want to help/rescue
- Antiheroes: Complex figures who might motivate donations
- Prefer proper nouns or clear groups (e.g., "CDC", "Texas AG", "DACA recipients", "insurers").
- If implicit but obvious, infer category names (e.g., "state legislature", "federal court").
- Limit each list to at most 5 items.
- If truly absent, leave list empty; do NOT invent.

SNIPPETS:
---
{snips}
---"""

VOTING_ROLES_PROMPT = """Extract role lists from these US political news/reddit snippets for VOTER ENGAGEMENT context.

Return STRICT JSON only:
{"heroes":[], "villains":[], "victims":[], "antiheroes":[]}

Rules for VOTER ENGAGEMENT:
- Focus on entities that drive voter motivation and mobilization
- Heroes: Who voters would support at the polls
- Villains: Who voters would oppose at the polls  
- Victims: Whose plight would motivate voter turnout
- Antiheroes: Complex figures who might influence voting behavior
- Prefer proper nouns or clear groups (e.g., "election officials", "voting rights advocates", "state legislatures").
- If implicit but obvious, infer category names (e.g., "state legislature", "federal court").
- Limit each list to at most 5 items.
- If truly absent, leave list empty; do NOT invent.

SNIPPETS:
---
{snips}
---"""

def parse_json(s: str) -> Dict:
    s = s.strip().strip("`").strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m: s = m.group(0)
    try:
        obj = json.loads(s)
    except Exception:
        obj = {"heroes":[], "villains":[], "victims":[], "antiheroes":[]}
    # normalize
    out = {}
    for k in ["heroes","villains","victims","antiheroes"]:
        v = obj.get(k, [])
        if isinstance(v, str): v = [v] if v.strip() else []
        if not isinstance(v, list): v = []
        v = [x.strip() for x in v if isinstance(x,str) and x.strip()]
        out[k] = list(dict.fromkeys(v))[:5]
    return out

def make_snips(items: pd.DataFrame, clusters: pd.DataFrame, cid: int, period: str, max_snips=12) -> str:
    items = items.copy()
    items["published_at"] = pd.to_datetime(items["published_at"], utc=True, errors="coerce")
    assign = (items.merge(clusters[["item_id","cluster_id","cluster_prob"]], on="item_id", how="inner")
                   .assign(period = items["published_at"].dt.date.astype(str)))
    sub = (assign[(assign["cluster_id"]==cid) & (assign["period"]==period)]
                  .sort_values(["cluster_prob","published_at"], ascending=[False, False])
                  .head(max_snips))
    snips = []
    for _, r in sub.iterrows():
        title = (r.get("title") or "").strip()
        src   = (r.get("source") or "").strip()
        url   = (r.get("url") or "").strip()
        text  = " ".join((r.get("text") or "").split()[:120])
        snips.append(f"- {title} — {src}\n  {text}\n  {url}")
    return "\n\n".join(snips)

def roles_is_empty(roles_dict: Dict) -> bool:
    """Check if roles dictionary is effectively empty"""
    if not isinstance(roles_dict, dict):
        return True
    return not any(len(roles_dict.get(k, [])) > 0 for k in ["heroes", "villains", "victims", "antiheroes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-dir", default="data/topics")
    ap.add_argument("--metrics", default="data/topics/metrics/topic_metrics.parquet")
    ap.add_argument("--max-snips", type=int, default=12)
    ap.add_argument("--recompute-all", action="store_true",
                    help="force recompute roles for all rows (not just empty)")
    ap.add_argument("--context", choices=["both", "fundraising", "voting"], default="both",
                    help="which role contexts to compute")
    args = ap.parse_args()

    ensure_env()

    items   = pd.read_parquet(Path(args.topics_dir)/"items.parquet")[["item_id","title","source","url","text","published_at"]]
    clusters= pd.read_parquet(Path(args.topics_dir)/"clusters.parquet")[["item_id","cluster_id","cluster_prob"]]
    df      = pd.read_parquet(args.metrics)

    # Determine which columns need processing
    columns_to_process = []
    if args.context in ["both", "fundraising"]:
        if "fundraising_roles" in df.columns:
            columns_to_process.append(("fundraising_roles", FUNDRAISING_ROLES_PROMPT))
        else:
            print("⚠️  fundraising_roles column not found - skipping fundraising context")
    
    if args.context in ["both", "voting"]:
        if "voting_roles" in df.columns:
            columns_to_process.append(("voting_roles", VOTING_ROLES_PROMPT))
        else:
            print("⚠️  voting_roles column not found - skipping voting context")

    if not columns_to_process:
        raise SystemExit("No valid role columns found to process")

    # Process each column
    total_updates = 0
    for column_name, prompt_template in columns_to_process:
        print(f"Processing {column_name}...")
        
        # Identify rows that need processing for this column
        if args.recompute_all:
            need_mask = pd.Series([True] * len(df), index=df.index)
        else:
            need_mask = df[column_name].apply(roles_is_empty)
        
        need = df[need_mask].copy()
        if need.empty:
            print(f"No rows need {column_name} backfill.")
            continue

        updates = 0
        for idx, r in need.iterrows():
            cid = int(r["cluster_id"])
            per = str(r["period"])
            snips = make_snips(items, clusters, cid, per, max_snips=args.max_snips)
            if not snips:
                continue
            
            msg = [
                {"role":"system","content":"Return strict JSON only."},
                {"role":"user","content": prompt_template.replace("{snips}", snips)}
            ]
            try:
                roles = parse_json(deepseek_chat(msg))
                df.at[idx, column_name] = roles
                updates += 1
                total_updates += 1
            except Exception as e:
                print(f"Error processing {column_name} for cluster {cid}, period {per}: {e}")
                # log to file
                Path("data/topics/metrics").mkdir(parents=True, exist_ok=True)
                with open("data/topics/metrics/roles_backfill_failures.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "column": column_name,      # "fundraising_roles" or "voting_roles"
                        "cluster_id": cid,
                        "period": per,
                        "error": f"{type(e).__name__}: {e}"
                    }, ensure_ascii=False) + "\n")
                # Set empty roles on error
                df.at[idx, column_name] = {"heroes":[], "villains":[], "victims":[], "antiheroes":[]}

            
            time.sleep(0.2)  # Rate limiting

        print(f"✓ {column_name} updated for {updates} rows")

    # Save results
    if total_updates > 0:
        df.to_parquet(args.metrics, index=False)
        print(f"✓ All roles updated for {total_updates} total rows → {args.metrics}")
    else:
        print("No role updates needed.")

if __name__ == "__main__":
    main()

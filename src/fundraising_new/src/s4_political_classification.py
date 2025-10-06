#!/usr/bin/env python3
"""
Enhanced political classification with hashing for cache management.
Standalone version that doesn't depend on topics_build module.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import hashlib
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
import pyarrow.dataset as ds

# Load environment variables
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[2] / "secret.env"
load_dotenv(ENV_PATH)

# DeepSeek API setup
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

MAP_DIR = "data/topics/merged_data_with_topics.parquet"

# Political classification prompt

POLITICAL_CLASSIFICATION_PROMPT = """
You are a political analyst specializing in US politics and fundraising. Today is {current_date}.
Donald Trump is the CURRENT President of the United States, leading a Republican administration.
Republicans control the White House and executive branch (including the DOJ).

Topic label: "{topic_label}"
Standardized categories: {standardized_topics}

For EACH party, produce a concise fundraising angle that would motivate **donors** of that party,
and numeric fundraising potentials 0–100 (integers).

Guidance
- GOP angles: attack Democratic positions, defend Republican actions, highlight threats from Democrats/deep state/external foes.
  Since GOP controls the DOJ, avoid alleging corruption **by** GOP-controlled DOJ unless framed as deep-state sabotage.
- DEM angles: attack GOP actions/abuse/overreach; defend norms/rights; highlight threats to democracy/corruption under Trump admin.
- Keep each angle ≤160 chars. Be realistic to **today's** context.

Return ONLY strict JSON with this exact schema (no extra keys, no text):
{{
  "story_label": "<short human label>",
  "classification": "dem-owned" | "gop-owned" | "contested" | "dem-avoided" | "gop-avoided" | "non-political",
  "dem_angle": "<string>",
  "gop_angle": "<string>",
  "dem_fundraising_potential": <0-100>,
  "gop_fundraising_potential": <0-100>
}}

Cluster items:
{items_text}
"""

def nkey(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

def deepseek_chat(messages: List[Dict[str, str]], model="deepseek-chat",
                  max_tokens=300, temperature=0.1):
    """Direct implementation of deepseek_chat function"""
    base = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
    url = f"{base.rstrip('/')}/chat/completions"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model, 
        "messages": messages,
        "max_tokens": max_tokens, 
        "temperature": temperature
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 30))
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# Add this helper function near the top of your script
def get_current_date_context():
    """Get the current date and format it for context priming"""
    now = pd.Timestamp.now()
    current_date = now.strftime("%B %d, %Y")
    
    # Calculate how long Trump has been in office (assuming Jan 20, 2025 inauguration)
    inauguration_date = pd.Timestamp("2025-01-20")
    days_in_office = (now - inauguration_date).days
    months_in_office = days_in_office // 30
    
    return f"{current_date} (Trump has been in office for {months_in_office} months since January 20, 2025)"

def _clip01(x):
    try:
        return float(max(0, min(100, float(x))))
    except Exception:
        return None

def safe_json_load(s: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM response"""
    s = s.strip().strip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        # Try to extract JSON from text if it's wrapped
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    # Return a default error response if parsing fails
    return {
        "classification": "error",
        "gop_angle": "",
        "dem_angle": "",
        "reasoning": "Failed to parse JSON response",
        "fundraising_potential": {"gop": 0, "dem": 0}
    }

def compute_content_hash(items_df: pd.DataFrame) -> str:
    """Compute a hash of the content to detect changes."""
    content_str = ""
    for _, row in items_df.iterrows():
        content_str += f"{row.get('title', '')}|{row.get('text', '')}|"
    return hashlib.sha256(content_str.encode()).hexdigest()

# Fixed retry decorator
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def classify_topic_politically(topic_label: str, items_text: str, standardized_topics: List[str]) -> Dict[str, Any]:
    """Classify a topic's political positioning using DeepSeek."""
    topics_str = ", ".join(standardized_topics) if isinstance(standardized_topics, list) else str(standardized_topics or "")
    prompt = POLITICAL_CLASSIFICATION_PROMPT.format(
        current_date=get_current_date_context(),    # <-- add this
        topic_label=topic_label,
        items_text=items_text,
        standardized_topics=topics_str
    )

    messages = [
        {"role": "system", "content": "You are a political analyst specializing in US politics and fundraising. Be concise and objective."},
        {"role": "user", "content": prompt}
    ]
    response = deepseek_chat(messages, max_tokens=400, temperature=0.1)
    return safe_json_load(response)


def format_items_text(items_df: pd.DataFrame, max_items: int = 8) -> str:
    """Format representative items for the LLM prompt."""
    items_text = ""
    for i, (_, row) in enumerate(items_df.iterrows()):
        if i >= max_items:
            break
            
        title = row.get('title', '')[:100]  # Truncate title
        content = row.get('text', '')[:300]  # Truncate content
        source = row.get('source', 'unknown')
        
        items_text += f"Item {i+1} (Source: {source}):\n"
        items_text += f"Title: {title}\n"
        items_text += f"Content: {content}\n\n"
    
    return items_text

def load_topic_data(topics_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items    = pd.read_parquet(topics_dir / "items.parquet")
    clusters = pd.read_parquet(topics_dir / "clusters.parquet")
    meta     = pd.read_parquet(topics_dir / "cluster_meta.parquet")  # <-- bring back rep_item_ids, etc.

    # Optionally enrich meta with standardized topics (if present)
    std_path = topics_dir / "merged_data_with_topics.parquet"
    if std_path.exists():
        std = pd.read_parquet(std_path)
        # keep only id/label + topic columns; and avoid blowing away meta rows
        keep = [c for c in ["cluster_id","label","standardized_topic_names","standardized_topic_ids"] if c in std.columns]
        if keep:
            meta = meta.merge(std[keep], on=["cluster_id","label"], how="left")
    return items, clusters, meta


def load_or_create_cache(cache_path: Path) -> pd.DataFrame:
    """Load existing cache or create a new one."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    else:
        return pd.DataFrame(columns=[
            "cluster_id", "content_hash", "classification", "gop_angle", 
            "dem_angle", "reasoning", "gop_fundraising_potential", 
            "dem_fundraising_potential", "processed_at"
        ])


def validate_angle(topic_label: str, angle: str, party: str) -> str:
    """Validate a political angle using LLM. Returns 'VALID' or error message."""
    current_context = get_current_date_context()
    
    validation_prompt = f"""
Validate this political fundraising angle for {party}. Today is {current_context}:

TOPIC: {topic_label}
ANGLE: {angle}

Check for:
1. Factual plausibility given current political reality (Republicans control White House/DOJ).
2. Relevance to the topic.
3. Strategic soundness for fundraising (highlights threat/opportunity for the party).
4. Absence of contradictions (e.g., GOP alleging DOJ corruption under Trump doesn't make sense).
5. Correct temporal context (Trump is the CURRENT president, not former president).

If valid, respond with "VALID". Otherwise, respond with "INVALID: [reason]".
"""
    messages = [
        {"role": "system", "content": "You are a political strategist. Be concise and strict."},
        {"role": "user", "content": validation_prompt}
    ]
    try:
        response = deepseek_chat(messages, max_tokens=150, temperature=0)
        if "VALID" in response:
            return "VALID"
        else:
            return response.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"
    
def is_empty_rep_item_ids(rep_item_ids) -> bool:
    """Safely check if rep_item_ids is empty."""
    if isinstance(rep_item_ids, (list, np.ndarray)):
        return len(rep_item_ids) == 0
    elif pd.isna(rep_item_ids):
        return True
    else:
        # Handle other cases (e.g., string representation of list)
        try:
            if isinstance(rep_item_ids, str):
                parsed = json.loads(rep_item_ids)
                return len(parsed) == 0
        except:
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Add political classification to topic analysis")
    parser.add_argument("--topics-dir", default="data/topics", help="Directory with topic data")
    parser.add_argument("--output-dir", default="data/topics", help="Output directory")
    parser.add_argument("--threshold", type=int, default=60, 
                       help="Minimum fundraising score to include (0-100)")
    parser.add_argument("--max-items", type=int, default=8,
                       help="Maximum number of representative items to use per cluster")
    parser.add_argument("--cache-file", default="political_classification_cache.parquet",
                       help="Cache file to avoid reprocessing")
    args = parser.parse_args()
    
    topics_dir = Path(args.topics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading topic data...")
    items, clusters, meta = load_topic_data(topics_dir)
    
    # Filter to only relevant clusters with sufficient fundraising potential
    relevant_meta = meta[
        (meta["us_relevance"]) & 
        (meta["fundraising_usable"]) & 
        (meta["fundraising_score"] >= args.threshold)
    ].copy()
    
    print(f"Found {len(relevant_meta)} relevant clusters with fundraising score >= {args.threshold}")
    
    # Load cache
    cache_path = output_dir / args.cache_file
    cache_df = load_or_create_cache(cache_path)
    print(f"Cache contains {len(cache_df)} entries")
    
    # Get representative items for each cluster
    clusters_to_process = []
    for _, row in relevant_meta.iterrows():
        cluster_id = row["cluster_id"]
        
        rep_item_ids = row.get("rep_item_ids", [])
        
        # Check if rep_item_ids is empty using our safe function
        if is_empty_rep_item_ids(rep_item_ids):
            continue
            
        # Convert to list if it's a string representation
        if isinstance(rep_item_ids, str):
            try:
                rep_item_ids = json.loads(rep_item_ids)
            except:
                continue
                
        # Get the representative items
        rep_items = items[items["item_id"].isin(rep_item_ids)]
        if len(rep_items) == 0:
            continue
        
        # Compute content hash to check if we need to reprocess
        content_hash = compute_content_hash(rep_items)
        
        # Check if we have a cached result for this cluster with the same content
        cached_result = cache_df[
            (cache_df["cluster_id"] == cluster_id) & 
            (cache_df["content_hash"] == content_hash)
        ]
        
        if not cached_result.empty:
            # Skip if we have a cached result for this content
            print(f"Skipping cluster {cluster_id} (already processed with same content)")
            continue

        # Get standardized topics
        standardized_topics = row.get('standardized_topic_names', [])
        if isinstance(standardized_topics, str):
            try:
                standardized_topics = json.loads(standardized_topics)
            except:
                standardized_topics = [standardized_topics]
        elif not isinstance(standardized_topics, list):
            standardized_topics = []
            
        clusters_to_process.append({
            "cluster_id": cluster_id,
            "label": row["label"],
            "rep_items": rep_items,
            "content_hash": content_hash,
            "standardized_topics": standardized_topics
        })
    
    seen_keys = set()
    deduped = []
    for _c in clusters_to_process:
        k = (int(_c["cluster_id"]), str(_c["content_hash"]))
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(_c)
    clusters_to_process = deduped
    print(f"Processing {len(clusters_to_process)} clusters for political classification")
    
    # Process each cluster
    results = []
    for cluster in clusters_to_process:
        cluster_id = cluster["cluster_id"]
        label = cluster["label"]
        rep_items = cluster["rep_items"]
        content_hash = cluster["content_hash"]
        standardized_topics = cluster["standardized_topics"]
        
        print(f"Processing cluster {cluster_id}: {label}")
        
        # Format items for LLM
        items_text = format_items_text(rep_items, args.max_items)
        
        try:
            # Get political classification
            classification_result = classify_topic_politically(
                label, 
                items_text, 
                standardized_topics
            )
            # ---- normalize/fallback schema (accept nested or flat) ----
            fp = classification_result.get("fundraising_potential", {}) if isinstance(classification_result, dict) else {}

            # prefer explicit flat fields; fallback to nested
            dem_raw = classification_result.get("dem_fundraising_potential", fp.get("dem"))
            gop_raw = classification_result.get("gop_fundraising_potential", fp.get("gop"))

            classification_result["dem_fundraising_potential"] = _clip01(dem_raw)
            classification_result["gop_fundraising_potential"] = _clip01(gop_raw)

            classification_result["classification"] = str(classification_result.get("classification", "unknown"))
            classification_result["dem_angle"] = str(classification_result.get("dem_angle", ""))
            classification_result["gop_angle"] = str(classification_result.get("gop_angle", ""))
            # -----------------------------------------------------------

            # Extract angles
            gop_angle = classification_result.get("gop_angle", "")
            dem_angle = classification_result.get("dem_angle", "")

            # Validate angles
            gop_validation = validate_angle(label, gop_angle, "GOP")
            if "VALID" not in gop_validation:
                print(f"GOP angle invalid for cluster {cluster_id}: {gop_validation}")
                # Optionally, you can set a flag or skip saving this angle
            dem_validation = validate_angle(label, dem_angle, "DEM")
            if "VALID" not in dem_validation:
                print(f"DEM angle invalid for cluster {cluster_id}: {dem_validation}")

            # Add to results
            # Map both old (nested) and new (flat) schemas
            fp = classification_result.get("fundraising_potential") or {}
            gop_fp_old = fp.get("gop")
            dem_fp_old = fp.get("dem")

            gop_fp_new = classification_result.get("gop_fundraising_potential")
            dem_fp_new = classification_result.get("dem_fundraising_potential")

            def _num(x):
                try:
                    return int(float(x))
                except Exception:
                    return None

            gop_fp = next(v for v in [_num(gop_fp_new), _num(gop_fp_old)] if v is not None) if any([gop_fp_new is not None, gop_fp_old is not None]) else None
            dem_fp = next(v for v in [_num(dem_fp_new), _num(dem_fp_old)] if v is not None) if any([dem_fp_new is not None, dem_fp_old is not None]) else None

            result = {
                "cluster_id": cluster_id,
                "content_hash": content_hash,
                "label": label,
                "story_label": label,
                "classification": classification_result.get("classification", "unknown"),
                "gop_angle": classification_result.get("gop_angle", ""),
                "dem_angle": classification_result.get("dem_angle", ""),
                "reasoning": classification_result.get("reasoning", ""),
                "gop_fundraising_potential": classification_result.get("gop_fundraising_potential"),
                "dem_fundraising_potential": classification_result.get("dem_fundraising_potential"),
                "processed_at": pd.Timestamp.now()
            }

            
            results.append(result)
            # incremental cache write — ensures we skip repeats in this run
            try:
                cache_df = pd.concat([
                    cache_df,
                    pd.DataFrame([{"cluster_id": int(cluster_id), "content_hash": content_hash}])
                ], ignore_index=True).drop_duplicates(["cluster_id","content_hash"])
                cache_df.to_parquet(cache_path, index=False)
            except Exception:
                pass
            print(f"  Classification: {result['classification']}")
            
            # Be kind to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {str(e)}")
            # Add error record
            results.append({
                "cluster_id": cluster_id,
                "content_hash": content_hash,
                "label": label,
                "classification": "error",
                "error": str(e),
                "processed_at": pd.Timestamp.now()
            })
    
    # Update cache and save results
    if results:
        results_df = pd.DataFrame(results)
        
        # ---- attach period via cluster_id from merged_data_with_topics.parquet
        mp = ds.dataset(MAP_DIR, format="parquet").to_table(columns=["cluster_id","period"]).to_pandas()
        mp["period"] = mp["period"].astype(str)
        mp = mp.drop_duplicates(["cluster_id","period"])

        results_df["story_label"] = results_df["label"].astype(str)
        results_df = results_df.merge(mp, on="cluster_id", how="left") 

        # ---- normalize numeric types and clamp to 0..100
        for c in ["dem_fundraising_potential","gop_fundraising_potential"]:
            results_df[c] = pd.to_numeric(results_df.get(c, np.nan), errors="coerce").clip(0,100)

        # ---- label_key for stable merges
        results_df["label_key"] = results_df["story_label"].str.lower().str.replace(r"[^a-z0-9]+","", "", regex=True)
        results_df["period"] = results_df["period"].astype(str)
        results_df["cluster_id"] = pd.to_numeric(results_df["cluster_id"], errors="coerce").astype("Int64")
        # ---- write an enriched parquet for downstream merges
        enriched_out = output_dir / "political_classification_enriched.parquet"
        results_df[[
            "period","cluster_id","story_label","label_key",
            "classification","dem_angle","gop_angle",
            "dem_fundraising_potential","gop_fundraising_potential",
            "processed_at"
        ]].to_parquet(enriched_out, index=False)
        print(f"Saved political classification (enriched) to {enriched_out}")

        # Update cache with new results
        cache_df = pd.concat([cache_df, results_df], ignore_index=True)
        cache_df.to_parquet(cache_path, index=False)
        print(f"Updated cache with {len(results_df)} new entries: {cache_path}")
        
        # Create a comprehensive results file with all classifications
        all_classifications_path = output_dir / "political_classification.parquet"
        
        # Load existing results if they exist
        if all_classifications_path.exists():
            existing_results = pd.read_parquet(all_classifications_path)
            # Remove any existing records for these clusters
            existing_results = existing_results[~existing_results["cluster_id"].isin(results_df["cluster_id"])]
            # Combine with new results
            results_df = pd.concat([existing_results, results_df], ignore_index=True)
        
        results_df.to_parquet(all_classifications_path, index=False)
        print(f"Saved political classification to {all_classifications_path}")
        
        # Create a simplified angles table for easy use
        angles_df = results_df[["cluster_id", "label", "classification", "gop_angle", "dem_angle"]].copy()
        angles_path = output_dir / "political_angles.parquet"
        angles_df.to_parquet(angles_path, index=False)
        print(f"Saved political angles to {angles_path}")
        
        # Update the cluster_meta with political classification
        meta_with_politics = meta.merge(
            results_df[["cluster_id", "classification"]], 
            on="cluster_id", 
            how="left"
        )
        meta_with_politics_path = output_dir / "cluster_meta_with_politics.parquet"
        meta_with_politics.to_parquet(meta_with_politics_path, index=False)
        print(f"Updated cluster_meta with political classification: {meta_with_politics_path}")
    else:
        print("No new clusters processed")

if __name__ == "__main__":
    main()
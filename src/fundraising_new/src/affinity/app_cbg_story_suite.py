#!/usr/bin/env python3
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

import pyarrow as pa
import pyarrow.dataset as ds
from pandas.api.types import is_numeric_dtype
import hashlib
from collections import Counter
import plotly.graph_objects as go
import ast, re
from functools import lru_cache


# -------------------- CONFIG --------------------
FEATUREDOT_PATH = "data/affinity/topic_affinity_by_cbg_featuredot.parquet"
FEATURES_PATH = "fundraising_participation/data/geo/cbg_features.parquet"
# ENRICHED_CSV  = "data/affinity/reports/topics_enriched.csv"   # period,label,standardized_topic_names
ENRICHED_CSV  = "data/affinity/reports/topics_enriched_fixed_period.csv" # temporary fix
RECO_DIR = Path("data/affinity/reports/reco")

LAT_CANDS = ["lat","latitude","cbg_lat","centroid_lat"]
LON_CANDS = ["lon","longitude","cbg_lon","centroid_lon"]

SUBGROUP_CANDIDATES = [
    "share_black","pct_black",
    "share_hisp","pct_hispanic",
    "pct_bachelors_plus","pct_broadband",
    "share_spanish","pct_spanish",
]

# ----- STORY METRICS CONFIG -----
# Friendly label -> CSV column
EMOTION_COLS = {
    "Anger / Outrage": "emo_anger_outrage",
    "Anxiety": "emo_anxiety",
    "Disgust": "emo_disgust",
    "Fear": "emo_fear",
    "Hope & Optimism": "emo_hope_optimism",
    "Pride": "emo_pride",
    "Sadness": "emo_sadness",
}
MORAL_COLS = {
    "Care": "mf_care",
    "Fairness": "mf_fairness",
    "Harm": "mf_harm",
    "Liberty": "mf_liberty",
    "Loyalty": "mf_loyalty",
    "Authority": "mf_authority",
    "Sanctity": "mf_sanctity",
    "Subversion": "mf_subversion",
    "Cheating": "mf_cheating",
    "Betrayal": "mf_betrayal",
    "Degradation": "mf_degradation",
    "Oppression": "mf_oppression",
}
BUBBLE_OPTIONS = {**EMOTION_COLS, **MORAL_COLS}

COMPACT_DIR = Path("data/affinity/compact")
TIMELINE_PRE = COMPACT_DIR / "topic_timeline.parquet"
BEST_PRE = {"Dem": COMPACT_DIR / "best_per_cbg_Dem.parquet",
            "GOP": COMPACT_DIR / "best_per_cbg_GOP.parquet"}

USE_PRECOMPUTED_ONLY = True     # <--- force fast path

# If you switched to the packed dataset:
# SURFACES_PATH = "data/affinity/surfaces_partitioned"
PACKED = False  # set True if you use s7g_pack_surfaces.py output
SCALE = 10000   # keep in sync with _meta.json if PACKED=True

# -------------------- HELPERS --------------------

def norm_label(s: str) -> str:
    return re.sub(r"\s+"," ", re.sub(r"[^a-z0-9]+"," ", str(s).lower())).strip()

def _safe_eval_list(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [s.strip(" '[]") for s in str(x).split("' '") if s]

NUM_COLS = [
    "dem_fundraising_potential","gop_fundraising_potential",
    "urgency_score","anger_outrage","cta_strength"
]

@lru_cache(maxsize=1)
def load_enriched_topics(csv_path: str = "data/affinity/reports/topics_enriched.csv") -> pd.DataFrame:
    en = pd.read_csv(csv_path)
    if "standardized_topic_names" in en.columns:
        en["standardized_topic_names"] = en["standardized_topic_names"].apply(_safe_eval_list)
    for c in NUM_COLS:
        if c in en.columns:
            en[c] = pd.to_numeric(en[c], errors="coerce").fillna(0.0)
    if "story_label" in en.columns:
        en["story_key"] = en["story_label"].apply(norm_label)
    return en

def _coerce_best(df: pd.DataFrame) -> pd.DataFrame:
    pr = df.get("period", df.get("date"))
    df = df[pr.astype(str).str.lower() != "all"].copy()
    df["period"] = pd.to_datetime(pr, errors="coerce").dt.date
    df = df.dropna(subset=["period"])
    df["cbg_id"] = df.get("cbg_id", df.get("cbg")).astype(str)
    df["margin"] = pd.to_numeric(df.get("margin"), errors="coerce").fillna(0.0)
    df["story_key"] = df["best_label"].apply(norm_label)
    return df

@lru_cache(maxsize=4)
def load_best_per_cbg(party: str, path_map: dict) -> pd.DataFrame:
    # path_map like {"Dem": Path(...), "GOP": Path(...)}
    df = pd.read_parquet(str(path_map[party]))
    return _coerce_best(df)

@lru_cache(maxsize=4)
def load_horizons(party: str, data_dir: str, which: str = "best") -> pd.DataFrame:
    # which: "best" uses best_horizons_*.parquet (winner_label)
    fp = Path(data_dir) / f"best_horizons_{party}.parquet"
    h = pd.read_parquet(str(fp))
    h["period_end"] = pd.to_datetime(h["period_end"], errors="coerce").dt.date
    h["story_key"] = h["winner_label"].apply(norm_label)
    return h

@st.cache_data(show_spinner=False)
def load_best_for_period(party: str, period: str, columns=None):
    """Read only the requested period from precomputed best_per_cbg_<party>.parquet using Arrow filters."""
    path = str(BEST_PRE[party])
    dataset = ds.dataset(path, format="parquet")
    flt = (ds.field("period") == str(period))
    cols = columns
    if cols is None:
        cols = [c for c in dataset.schema.names]  # or a fixed list
    # keep only existing columns
    cols = [c for c in cols if c in dataset.schema.names]
    tbl = dataset.to_table(columns=cols, filter=flt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype).copy()
    # tidy dtypes
    if "cbg_id" in df.columns: df["cbg_id"] = df["cbg_id"].astype(str)
    return df

@st.cache_data(ttl=600, show_spinner=False)
def _threads_means_dem_gop_cta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-period means, all-period means) for dem/gop/cta by story_label."""
    import pyarrow.parquet as pq
    P = "/opt/pharos/storysuite/data/affinity/compact/story_threads.parquet"
    pf = pq.ParquetFile(P)
    need = {"period","story_label","label","winner_label",
            "dem_fundraising_potential","gop_fundraising_potential","cta_ask_strength"}
    cols = [c for c in pf.schema_arrow.names if c in need]
    if not cols:
        return pd.DataFrame(columns=["period","story_label"]), pd.DataFrame(columns=["story_label"])

    T = pf.read(columns=cols).to_pandas().copy()

    # choose any available label text column
    lab = "story_label" if "story_label" in T.columns else ("label" if "label" in T.columns else "winner_label")
    T["story_label"] = T[lab].astype(str).str.strip()
    T["period"]      = T["period"].astype(str).str.strip()

    for c in ["dem_fundraising_potential","gop_fundraising_potential","cta_ask_strength"]:
        if c in T.columns:
            T[c] = pd.to_numeric(T[c], errors="coerce")

    # per-period means
    per = (T.groupby(["period","story_label"], as_index=False)
             [["dem_fundraising_potential","gop_fundraising_potential","cta_ask_strength"]]
             .mean())

    # all-period means
    allp = (T.groupby("story_label", as_index=False)
              [["dem_fundraising_potential","gop_fundraising_potential","cta_ask_strength"]]
              .mean())

    return per, allp

@st.cache_data(show_spinner=False)
def _load_topic_features_sc():
    """
    Load the richer per-story metrics (emotions, moral foundations) produced by spiders.
    Tries a few likely files and returns the first that exists and has emo_/mf_ columns.
    """
    candidates = [
        "data/affinity/reports/topics_enriched_spiders.csv",
        "data/affinity/reports/topics_enriched.csv",                 # fallback
        ENRICHED_CSV,                                                # last resort
    ]
    for p in candidates:
        try:
            df = pd.read_csv(p)
            # only keep if it actually brings emo_/mf_ columns
            has_emo = any(str(c).startswith("emo_") for c in df.columns)
            has_mf  = any(str(c).startswith("mf_")  for c in df.columns)
            if has_emo or has_mf:
                return df
        except Exception:
            pass
    return pd.DataFrame()
    
@st.cache_data(ttl=600, show_spinner=False)
def _load_topic_features_sc_compact():
    tf = _load_topic_features_sc()
    drop = [c for c in tf.columns if str(c).startswith("text_") or c in {"raw","html"}]
    return tf.drop(columns=drop, errors="ignore")

def _fmt_num(x):
    try:
        x = float(x)
        return "—" if not np.isfinite(x) else f"{x:.1f}"
    except Exception:
        return "—"
    
def _norm_key(x: str) -> str:
    s = (str(x) if x is not None else "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def _scorecard_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric score columns and normalize CTA strength alias."""
    out = df.copy()
    if "cta_strength" not in out.columns and "cta_ask_strength" in out.columns:
        out["cta_strength"] = pd.to_numeric(out["cta_ask_strength"], errors="coerce")
    num_cols = (
        ["urgency_score","dem_fundraising_potential","gop_fundraising_potential","cta_strength"]
        + list(EMOTION_COLS.values()) + list(MORAL_COLS.values())
    )
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    return out

def _select_rows_for_label(df_en, selected_label):
    rows = df_en.copy()
    if "label_key" not in rows.columns:
        rows["label_key"] = rows["story_label"].astype(str).apply(_norm_key)
    sel_key = _norm_key(selected_label)
    return rows[rows["label_key"] == sel_key].copy()

def _inject_spider_features(rows: pd.DataFrame) -> pd.DataFrame:
    # If we already have any emo_/mf_ values, keep as is
    emo_present = [c for c in rows.columns if str(c).startswith("emo_")]
    mf_present  = [c for c in rows.columns if str(c).startswith("mf_")]
    if emo_present or mf_present:
        if rows[emo_present + mf_present].notna().any(axis=None):
            return rows

    # Load topic-features with emo_/mf_
    tf = _load_topic_features_sc()
    if tf.empty:
        return rows

    # Which column in tf holds the label text
    lab_candidates = ("story_label","label","topic_name","topic_label","topic")
    labcol_f = next((c for c in lab_candidates if c in tf.columns), None)
    if not labcol_f or rows.empty or "story_label" not in rows.columns:
        return rows

    def _key(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())
    tf = tf.copy()
    tf["label_key"] = tf[labcol_f].astype(str).map(_key)

    sel_key = _key(rows["story_label"].iloc[0])

    # 1) Try exact key match with same-period rows if available
    tf_match = tf[tf["label_key"] == sel_key]

    # 2) If empty, try contains on raw label
    if tf_match.empty:
        _lbl = str(rows["story_label"].iloc[0])
        tf_match = tf[tf[labcol_f].astype(str).str.contains(_lbl, case=False, na=False)]

    # 3) Still empty? nothing to inject
    if tf_match.empty:
        return rows

    # Inject mean values into missing columns
    emo_cols = [c for c in tf_match.columns if str(c).startswith("emo_")]
    mf_cols  = [c for c in tf_match.columns if str(c).startswith("mf_")]
    for c in emo_cols + mf_cols:
        if c not in rows.columns or rows[c].isna().all():
            rows[c] = pd.to_numeric(tf_match[c], errors="coerce").mean()

    return rows

def add_label_key(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col in df.columns and "label_key" not in df.columns:
        df["label_key"] = df[label_col].apply(_norm_key)
    return df

# state FIPS -> USPS two-letter
_FIPS2USPS = {
 "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
 "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
 "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
 "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
 "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
 "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
 "54":"WV","55":"WI","56":"WY","60":"AS","66":"GU","69":"MP","72":"PR","78":"VI"
}

# Build emotion menu dynamically from columns
def discover_emotions(df):
    cols = set(df.columns)
    choices = []
    # prefer separate anger/outrage
    if "emo_anger" in cols:    choices.append(("Anger","emo_anger"))
    if "emo_outrage" in cols:  choices.append(("Outrage","emo_outrage"))
    # allow merged if separate not present
    if not any(k in cols for k in ("emo_anger","emo_outrage")) and "emo_anger_outrage" in cols:
        choices.append(("Anger / Outrage","emo_anger_outrage"))
    # rest
    for label,key in [
        ("Anxiety","emo_anxiety"), ("Disgust","emo_disgust"), ("Fear","emo_fear"),
        ("Hope & Optimism","emo_hope_optimism"), ("Pride","emo_pride"), ("Sadness","emo_sadness")
    ]:
        if key in cols: choices.append((label,key))
    return dict(choices)

def _derive_state_from_any(row: pd.Series) -> str:
    # 1) if 'state' looks like USPS, use it
    s = str(row.get("state", "") or "").strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    # 2) if 'state' looks like 2-digit FIPS, map it
    if len(s) == 2 and s.isdigit():
        return _FIPS2USPS.get(s, "")
    # 3) derive from cbg_id (first 2 digits)
    cid = str(row.get("cbg_id", "") or "")
    if len(cid) >= 2 and cid[:2].isdigit():
        return _FIPS2USPS.get(cid[:2], "")
    return ""

def party_col(party: str) -> str:
    return {"Dem":"aff_dem","GOP":"aff_gop","Ind":"aff_ind"}[party]

def _featuredot_party_col(party: str, cols: list[str]) -> str:
    # prefer blended columns when present
    wants = {
        "Dem": ["final_affinity_cbg_dem", "affinity_cbg_dem"],
        "GOP": ["final_affinity_cbg_gop", "affinity_cbg_gop"],
        "Ind": ["affinity_cbg_ind"]  # only if you ever emit it
    }[party]
    for c in wants:
        if c in cols: return c
    # fallback: raise a clear error
    raise KeyError(f"No party affinity column found for {party}. Available: {cols}")

def read_featuredot_period(period: str, party: str, with_geo=True, labels=True, states=None):
    cols = ["period","cbg_id","label"]
    dataset = ds.dataset(FEATUREDOT_PATH, format="parquet")
    # discover which party column to read
    pt_col = _featuredot_party_col(party, dataset.schema.names)
    cols.append(pt_col)
    if with_geo:
        for g in ["state","zcta"]:
            if g in pd.read_parquet(FEATURES_PATH, nrows=0).columns:
                cols.append(g)

    flt = (ds.field("period") == str(period))
    if states and "state" in dataset.schema.names and len(states):
        flt = flt & (ds.field("state").isin(states))

    tbl = dataset.to_table(columns=[c for c in cols if c in dataset.schema.names], filter=flt)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype).copy()
    if "cbg_id" in df.columns: df["cbg_id"] = df["cbg_id"].astype(str)
    if "label" in df.columns:  df["label"]  = df["label"].astype(str)
    df["period"] = str(period)
    df = add_label_key(df, "label")
    df = df.rename(columns={pt_col: "aff_party"})  # normalize name for downstream
    return df

def normalize_share_like(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.quantile(0.95) > 1.0:  # looks like 0..100
        x = x / 100.0
    return x.clip(0.0, 1.0)

def color_for_label(label: str):
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return [50 + r % 180, 50 + g % 180, 50 + b % 180, 180]

@st.cache_data(show_spinner=False)
def load_enriched_metrics():
    base_cols = [
        "period","story_label","standardized_topic_names","urgency_score",
        "dem_fundraising_potential","gop_fundraising_potential",
        "gop_angle","dem_angle","classification",          # <— added
        "cta_ask_type","cta_copy","cta_ask_strength",
        "heroes","villains","victims","antiheroes",
    ]
    num_cols = list(EMOTION_COLS.values()) + list(MORAL_COLS.values())

    usecols = list(dict.fromkeys(base_cols + num_cols))
    try:
        df = pd.read_csv(ENRICHED_CSV, usecols=usecols)
    except Exception:
        df = pd.read_csv(ENRICHED_CSV)

    # make sure every referenced column exists
    for c in base_cols + num_cols:
        if c not in df.columns:
            df[c] = np.nan

    # parse date (not used here, but handy if you later want a range)
    df["period_dt"] = pd.to_datetime(df["period"], errors="coerce", infer_datetime_format=True)

    # numeric coercions
    num_all = ["urgency_score","dem_fundraising_potential","gop_fundraising_potential",
               "cta_ask_strength"] + num_cols
    for c in num_all:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # strings
    for c in ["story_label","standardized_topic_names","gop_angle","dem_angle",
                "classification",                           # <— added
                "cta_ask_type","cta_copy","heroes","villains","victims","antiheroes"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
    return df

def _parse_std_topics(raw: str) -> list[str]:
    if not isinstance(raw, str): return []
    s = raw.strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                parts = [str(x).strip() for x in val]
            else:
                parts = [str(val).strip()]
        except Exception:
            inner = s[1:-1].strip().replace("', '", "|").replace('"', "").replace("'", "")
            parts = re.split(r"\s*\|\s*|\s{2,}|;", inner)
    else:
        t = s.replace(";", "|").replace(",", "|").replace(" | ", "|").replace("', '","|").replace('"', "").replace("'", "")
        parts = re.split(r"\s*\|\s*", t)
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def _explode_topics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["standardized_topic_names"] = d["standardized_topic_names"].astype("string").fillna("")
    d = d.assign(std_topic=d["standardized_topic_names"].apply(_parse_std_topics)).explode("std_topic", ignore_index=True)
    d["std_topic"] = d["std_topic"].astype("string").fillna("").str.strip()
    d = d[d["std_topic"] != ""]
    return d

def _explode_and_fill_std_topics(d2: pd.DataFrame) -> pd.DataFrame:
    d2 = d2.copy()
    d2["label"] = d2.get("label", "").astype("string")
    d2["standardized_topic_names"] = d2.get("standardized_topic_names", "").astype("string").fillna("")
    d2 = d2.assign(std_topic=d2["standardized_topic_names"].apply(_parse_std_topics)).explode("std_topic", ignore_index=True)
    d2["std_topic"] = d2["std_topic"].astype("string").fillna("").str.strip()
    d2["std_topic"] = d2["std_topic"].mask(d2["std_topic"] == "", d2["label"])
    return d2

def _topic_agg(df, fund_col: str, bubble_col: str):
    # One row per standardized topic
    g = (df.groupby("std_topic", as_index=False)
           .agg(urgency_mean=("urgency_score","mean"),
                fund_mean=(fund_col,"mean"),
                bubble_mean=(bubble_col,"mean"),
                n_stories=("std_topic","size")))
    return g


def _story_agg(df, fund_col: str, bubble_col: str):
    # One row per story label
    g = (df.groupby("story_label", as_index=False)
           .agg(urgency_mean=("urgency_score","mean"),
                fund_mean=(fund_col,"mean"),
                bubble_mean=(bubble_col,"mean"),
                n_rows=("story_label","size")))
    return g

def _jitter_by_key(keys: pd.Series, amplitude: float = 0.4) -> pd.Series:
    """
    Deterministic jitter in [-amplitude, +amplitude] per key (label/topic),
    so points don't shift on reruns.
    """
    def jit_one(k):
        h = hashlib.md5(str(k).encode("utf-8")).digest()
        u = int.from_bytes(h[:4], "little") / 2**32  # [0,1)
        return (u * 2 - 1) * amplitude
    return keys.apply(jit_one)


def _make_bubble_size(values: pd.Series,
                      contrast: float = 1.6,
                      min_size: float = 80.0,
                      max_size: float = 2200.0) -> pd.Series:
    """
    Map metric → visual area (px^2). Higher 'contrast' exaggerates differences.
    """
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        norm = pd.Series(1.0, index=x.index)
    else:
        norm = (x - lo) / (hi - lo)
    norm = np.power(norm, contrast)  # contrast >1 → more pop; <1 → flatter
    return (min_size + norm * (max_size - min_size)).astype(float)


def _mode_str(s: pd.Series) -> str:
    vals = [x.strip() for x in s.dropna().astype(str) if str(x).strip() not in ("", "nan", "None")]
    if not vals: return ""
    return Counter(vals).most_common(1)[0][0]

def _to_list(s: str) -> list[str]:
    if not isinstance(s, str): return []
    parts = [p.strip() for p in re.split(r"[;|,]", s) if p.strip()]
    # de-dup in order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def _radar_fig(series: pd.Series, title: str) -> go.Figure:
    cats = series.index.tolist()
    vals = [float(series.get(k, 0) or 0) for k in cats]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=title))
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",     # transparent around the plot
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="white",               # white polar panel (so black text reads well)
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="black"),
                gridcolor="#e5e7eb", linecolor="#cbd5e1"
            ),
            angularaxis=dict(
                tickfont=dict(color="black"),
                gridcolor="#eef2f7", linecolor="#cbd5e1"
            ),
        ),
        showlegend=False,
        height=380, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig



# -------------------- DATA ACCESS (LAZY) --------------------
@st.cache_data(show_spinner=False)
def load_features_small():
    feat = pd.read_parquet(FEATURES_PATH)
    feat["cbg_id"] = feat["cbg_id"].astype(str)
    # harmonize adults
    if "adults" not in feat.columns and "adults_18plus" in feat.columns:
        feat = feat.rename(columns={"adults_18plus":"adults"})
    lat = next((c for c in LAT_CANDS if c in feat.columns), None)
    lon = next((c for c in LON_CANDS if c in feat.columns), None)
    has_points = lat is not None and lon is not None

    keep = ["cbg_id","state","zcta"]
    if "adults" in feat.columns: keep.append("adults")
    for c in SUBGROUP_CANDIDATES:
        if c in feat.columns: keep.append(c)
    if has_points: keep += [lat, lon]

    small = feat[keep].copy()
    if has_points:
        small = small.rename(columns={lat:"lat", lon:"lon"})
    return small, has_points

@st.cache_data(show_spinner=False)
def load_enriched():
    """Return a compact label_key → standardized_topic_names mapping."""
    try:
        df = pd.read_csv(ENRICHED_CSV)
        for c in ("period","label","story_label","standardized_topic_names"):
            if c in df.columns: df[c] = df[c].astype(str)

        maps = []
        if "label" in df.columns:
            m = df[["label","standardized_topic_names"]].dropna(subset=["label"]).copy()
            m["label_key"] = m["label"].apply(_norm_key)
            maps.append(m[["label_key","standardized_topic_names"]])
        if "story_label" in df.columns:
            m = df[["story_label","standardized_topic_names"]].dropna(subset=["story_label"]).copy()
            m["label_key"] = m["story_label"].apply(_norm_key)
            maps.append(m[["label_key","standardized_topic_names"]])

        if maps:
            map_df = (pd.concat(maps, ignore_index=True)
                        .dropna(subset=["label_key"])
                        .drop_duplicates("label_key"))
        else:
            map_df = pd.DataFrame(columns=["label_key","standardized_topic_names"])
        return map_df
    except Exception:
        return pd.DataFrame(columns=["label_key","standardized_topic_names"])

@st.cache_data(show_spinner=False)
def list_periods():
    ds_path = FEATUREDOT_PATH
    if not Path(ds_path).exists():
        return []
    dataset = ds.dataset(ds_path, format="parquet")
    if "period" not in dataset.schema.names:
        return []
    tbl = dataset.to_table(columns=["period"])
    s = pd.Series(tbl.column("period").to_pylist(), dtype="string").dropna().astype(str)
    return sorted(s.unique().tolist())


    # 3) Fallback to surfaces (will likely be just 'all')
    dataset = ds.dataset(SURFACES_PATH, format="parquet")
    if "period" in dataset.schema.names:
        tbl = dataset.to_table(columns=["period"])
        s = pd.Series(tbl.column("period").to_pylist(), dtype="string").dropna().astype(str)
        return sorted(s.unique().tolist())
    return []


@st.cache_data(show_spinner=False)
def load_best_csv(party: str):
    p = RECO_DIR / f"best_story_per_cbg_{party}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, dtype={"cbg_id":str})
    needed = {"period","cbg_id","best_label","best_score","second_label","second_score","margin"}
    if not needed.issubset(df.columns): return None
    return df

# -------------------- COMPUTATIONS (PER PERIOD) --------------------
def compute_best_per_cbg(period: str, party: str, states=None) -> pd.DataFrame:
    df = read_featuredot_period(period, party, with_geo=True, labels=True, states=states)
    if df.empty:
        return pd.DataFrame(columns=["period","cbg_id","best_label","best_score","second_label","second_score","margin"])
    df["__r"] = df.groupby("cbg_id")["aff_party"].rank(ascending=False, method="first")
    best = df[df["__r"]==1].copy().drop(columns="__r").rename(columns={"label":"best_label","aff_party":"best_score"})
    second = df[df.groupby("cbg_id")["aff_party"].rank(ascending=False, method="first")==2][["cbg_id","label","aff_party"]]
    second = second.rename(columns={"label":"second_label","aff_party":"second_score"})
    best = best.merge(second, on="cbg_id", how="left")
    best["margin"] = best["best_score"] - best["second_score"]
    keep = ["period","cbg_id","best_label","best_score","second_label","second_score","margin"]
    for g in ("state","zcta"):
        if g in best.columns: keep.append(g)
    return best[keep]


def topic_timeseries_stream(party: str, topics_map: pd.DataFrame):
    periods = list_periods()
    out = []
    for p in periods:
        d = read_featuredot_period(p, party, with_geo=False, labels=True)[["period","label","label_key","aff_party"]]
        if d.empty: 
            continue
        if not topics_map.empty:
            m = d.merge(topics_map, on="label_key", how="left")
        else:
            m = d.copy()
            m["standardized_topic_names"] = ""
        m["standardized_topic_names"] = m["standardized_topic_names"].fillna("").astype(str)
        m = m.assign(std_topic=m["standardized_topic_names"].str.split(r"\s*;\s*")).explode("std_topic")
        m["std_topic"] = m["std_topic"].astype(str).str.strip()
        m.loc[m["std_topic"]=="", "std_topic"] = m["label"]  # fallback to label text
        ts = (m.groupby(["period","std_topic"], as_index=False)["aff_party"].sum()
                .rename(columns={"aff_party":"affinity_sum"}))
        out.append(ts)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["period","std_topic","affinity_sum"])

def subgroup_ranks(period: str, party: str, demo_col: str,
                   feat_small: pd.DataFrame, topics_map: pd.DataFrame):
    """Return subgroup-weighted ranks for stories and standardized topics.

    Path A (preferred): use detailed SURFACES for the selected period.
    Path B (fallback):  use best_per_cbg_<party>.parquet winners for that period.
    """
    col = party_col(party)

    # --- weights: subgroup share × adults (if present)
    if demo_col not in feat_small.columns:
        return pd.DataFrame(), pd.DataFrame()
    w = normalize_share_like(feat_small[demo_col])
    if "adults" in feat_small.columns:
        w = w * pd.to_numeric(feat_small["adults"], errors="coerce").fillna(0.0)
    weights = feat_small[["cbg_id"]].copy()
    weights["cbg_id"] = weights["cbg_id"].astype(str)
    weights["__w"] = w.fillna(0.0)

    # --- PATH A: try detailed surfaces for that period
    df = read_featuredot_period(period, party, with_geo=False, labels=True)
    if not df.empty:
        df = df[["period","label","label_key","cbg_id","aff_party"]].copy()
        df["cbg_id"] = df["cbg_id"].astype(str)

        d = df.merge(weights, on="cbg_id", how="left")
        # d has columns: period,label,label_key,cbg_id,col,__w
        # Cap work to avoid runaway memory
        MAX_ROWS = 100000
        if len(d) > MAX_ROWS:
            d = d.head(MAX_ROWS)

        # Vectorized weighted sum by story label
        vals = pd.to_numeric(d["aff_party"], errors="coerce").fillna(0.0)
        d["__wt"] = vals * d["__w"]
        story_scores = (
            d.groupby(["period","label"], as_index=False)["__wt"]
            .sum().rename(columns={"__wt":"subgroup_score"})
            .sort_values("subgroup_score", ascending=False)
        )

        # Topics via normalized label_key → std_topic mapping (uses your 'topics_map')
        if topics_map.empty:
            topic_scores = pd.DataFrame(columns=["period","std_topic","subgroup_score"])
        else:
            d2 = d.merge(topics_map, on="label_key", how="left")
            d2 = _explode_and_fill_std_topics(d2)
            d2["__wt"] = pd.to_numeric(d2[col], errors="coerce").fillna(0.0) * d2["__w"]
            topic_scores = (
                d2.groupby(["period","std_topic"], as_index=False)["__wt"]
                .sum().rename(columns={"__wt":"subgroup_score"})
                .sort_values("subgroup_score", ascending=False)
        )
        return story_scores, topic_scores

    # --- PATH B: fallback to winners for that period (fast + robust)
    best = load_best_for_period(party, period, columns=["period","cbg_id","best_label"])
    if best is None or best.empty:
        # last resort: use "all"
        best = load_best_for_period(party, "all", columns=["period","cbg_id","best_label"])
        if best is None or best.empty:
            return pd.DataFrame(), pd.DataFrame()

    resolved_period = str(best["period"].iloc[0]) if "period" in best.columns else str(period)

    best = best.rename(columns={"best_label":"label"})[["cbg_id","label"]].copy()
    best["cbg_id"] = best["cbg_id"].astype(str)
    best["label_key"] = best["label"].apply(_norm_key)

    # Each CBG contributes its subgroup weight to its winning story
    d = best.merge(weights, on="cbg_id", how="left")
    d["__w"] = d["__w"].fillna(0.0)

    MAX_ROWS = 250000
    if len(d) > MAX_ROWS:
        d = d.head(MAX_ROWS)

    story_scores = (
        d.groupby("label", as_index=False)["__w"]
        .sum().rename(columns={"__w":"subgroup_score"})
        .sort_values("subgroup_score", ascending=False)
    )
    story_scores.insert(0, "period", resolved_period)

    if topics_map.empty:
        topic_scores = pd.DataFrame(columns=["period","std_topic","subgroup_score"])
    else:
        d2 = d.merge(topics_map, on="label_key", how="left")
        d2 = _explode_and_fill_std_topics(d2)
        topic_scores = (
            d2.groupby("std_topic", as_index=False)["__w"]
            .sum().rename(columns={"__w":"subgroup_score"})
            .sort_values("subgroup_score", ascending=False)
        )
        topic_scores.insert(0, "period", resolved_period)

    return story_scores, topic_scores




# -------------------- UI --------------------
st.set_page_config(page_title="CBG Story Suite (lazy)", layout="wide")
st.title("CBG Story Suite — Party-oriented (lazy loading)")
# --- Data coverage banner ---
def _coverage():
    cov = {}
    # topics_enriched.csv
    try:
        dfe = pd.read_csv(ENRICHED_CSV, usecols=["period"])
        dfe["period_dt"] = pd.to_datetime(dfe["period"], errors="coerce", infer_datetime_format=True)
        cov["enriched"] = dict(
            rows=len(dfe),
            n_periods=dfe["period"].astype(str).nunique(),
            min=str(dfe["period_dt"].min().date()) if pd.notna(dfe["period_dt"].min()) else "—",
            max=str(dfe["period_dt"].max().date()) if pd.notna(dfe["period_dt"].max()) else "—",
        )
    except Exception as e:
        cov["enriched"] = {"error": str(e)}

    # timeline parquet (optional)
    if TIMELINE_PRE.exists():
        try:
            t = pd.read_parquet(TIMELINE_PRE, columns=[c for c in ["period","party"] if c in pd.read_parquet(TIMELINE_PRE, nrows=0).columns])
            t["period"] = t["period"].astype(str)
            cov["timeline"] = {}
            for pty in (t["party"].unique().tolist() if "party" in t.columns else ["(NA)"]):
                tt = t if pty=="(NA)" else t[t["party"]==pty]
                if tt.empty: continue
                cats = sorted(tt["period"].unique().tolist())
                cov["timeline"][pty] = {"n_periods": len(cats), "min": cats[0], "max": cats[-1]}
        except Exception as e:
            cov["timeline"] = {"error": str(e)}
    else:
        cov["timeline"] = {"missing": True}

    # best_per_cbg parquet (Dem/GOP)
    cov["best"] = {}
    for pty in ["Dem","GOP"]:
        pth = BEST_PRE[pty]
        if not pth.exists():
            cov["best"][pty] = {"missing": True}
            continue
        try:
            b = pd.read_parquet(pth, columns=["period"])
            b["period"] = b["period"].astype(str)
            cats = sorted(b["period"].unique().tolist())
            cov["best"][pty] = {"n_periods": len(cats), "min": cats[0], "max": cats[-1], "has_all": "all" in cats}
        except Exception as e:
            cov["best"][pty] = {"error": str(e)}
    return cov

cov = _coverage()
st.info(
    f"**Data coverage**  \n"
    f"- `topics_enriched.csv`: rows={cov.get('enriched',{}).get('rows','?')}, "
    f"periods={cov.get('enriched',{}).get('n_periods','?')} "
    f"({cov.get('enriched',{}).get('min','—')} → {cov.get('enriched',{}).get('max','—')})  \n"
    f"- `topic_timeline.parquet`: "
    + (
        ", ".join([f"{k}: {v.get('n_periods','?')} ({v.get('min','—')} → {v.get('max','—')})"
                   for k,v in cov.get('timeline',{}).items() if isinstance(v, dict) and 'error' not in v])
        if isinstance(cov.get('timeline'), dict) and 'missing' not in cov.get('timeline',{}) else
        ("missing" if cov.get('timeline',{}).get('missing') else cov.get('timeline',{}).get('error','?'))
      ) + "\n"
    f"- `best_per_cbg`: "
    + ", ".join([f"{pty}: {info.get('n_periods','?')} ({info.get('min','—')} → {info.get('max','—')})"
                 + (" +all" if info.get("has_all") else "")
                 if isinstance(info, dict) and 'missing' not in info and 'error' not in info
                 else f"{pty}: " + ("missing" if info.get('missing') else info.get('error','?'))
                 for pty, info in cov.get('best',{}).items()])
)

feat_small, has_points = load_features_small()
enriched = load_enriched()

# party & period
c1, c2 = st.columns([1,1])
party = c1.selectbox("Party", ["Dem","GOP"], index=0)
periods = list_periods()
if not periods:
    st.error("No periods found in surfaces parquet.")
    st.stop()
period = c2.selectbox("Period", periods, index=max(0, len(periods)-1))
period_str = str(period)
try:
    period_dt = pd.to_datetime(period_str).date()
except Exception:
    period_dt = None

# after period is selected
states_avail = sorted(feat_small["state"].dropna().astype(str).str.upper().unique().tolist()) \
               if "state" in feat_small.columns else []

# ensure first render defaults to ALL
if "state_sel" not in st.session_state:
    st.session_state["state_sel"] = states_avail

state_sel = st.multiselect(
    "States (optional)",
    states_avail,
    default=st.session_state["state_sel"],
    key="state_sel",
)


tabs = st.tabs(["📈 Topic Timeline", "📊 Story Landscape", "📊 Story Scorecards", "🏆 Story Leaderboard", "🗺️ Map", "👥 Subgroup Rankings"])

# -------------------- TAB 1 --------------------
with tabs[0]:
    st.subheader("Topic Timeline (sum of affinities per standardized topic)")

    # Fast path: precomputed timeline per party
    if TIMELINE_PRE.exists():
        ts = pd.read_parquet(TIMELINE_PRE)
        if "party" in ts.columns:
            ts = ts[ts["party"] == party].copy()
        ts["period"] = ts["period"].astype(str)
    else:
        with st.spinner("Computing topic timeline (streamed by period)…"):
            ts = topic_timeseries_stream(party, enriched)  # uses label_key mapping
            ts["period"] = ts["period"].astype(str)

    if ts.empty:
        st.info("No data for timeline.")
    else:
        # 1) explode multi-topic strings -> single topic
        ts["std_topic"] = ts["std_topic"].fillna("").astype(str)
        ts_exp = ts.assign(topic=ts["std_topic"].str.split(r"\s*;\s*")).explode("topic")
        ts_exp["topic"] = ts_exp["topic"].astype(str).str.strip()
        ts_exp = ts_exp[ts_exp["topic"] != ""]

        # 2) force numeric (Arrow sometimes returns strings)
        ts_exp["affinity_sum"] = pd.to_numeric(ts_exp["affinity_sum"], errors="coerce").fillna(0.0)

        # 3) aggregate to (period, topic)
        ts_agg = (ts_exp.groupby(["period", "topic"], as_index=False)["affinity_sum"]
                         .sum())

        if ts_agg.empty:
            st.info("No data after exploding standardized topics.")
        else:
            # Default selection: top-N topics by total
            totals = ts_agg.groupby("topic", as_index=False)["affinity_sum"].sum() \
                           .sort_values("affinity_sum", ascending=False)
            TOP_N = 20  # change to 40 if you prefer
            default_topics = totals.head(min(TOP_N, len(totals)))["topic"].tolist()
            all_topics = totals["topic"].tolist()

            sel = st.multiselect("Select standardized topics", all_topics, default=default_topics)
            view = ts_agg[ts_agg["topic"].isin(sel)].copy()
            periods_sorted = sorted(view["period"].unique().tolist())

            if not view.empty:
                if len(periods_sorted) <= 1:
                    st.caption(f"Single period detected: {periods_sorted[0] if periods_sorted else '(none)'} — using bar chart")
                    bar = (
                        alt.Chart(view)
                           .mark_bar()
                           .encode(
                               x=alt.X("topic:N", sort="-y", title="Topic"),
                               y=alt.Y("affinity_sum:Q", title=f"Sum of {party_col(party)}",
                                       scale=alt.Scale(zero=True)),
                               color=alt.Color("topic:N", legend=alt.Legend(title="Topic")),
                               tooltip=[
                                   alt.Tooltip("topic:N",         title="Topic"),
                                   alt.Tooltip("affinity_sum:Q",  title="Sum", format=",.0f"),
                               ]
                           )
                           .properties(height=460)
                    )
                    st.altair_chart(bar, use_container_width=True)
                else:
                    line = (
                        alt.Chart(view)
                           .mark_line()
                           .encode(
                               x=alt.X("period:N", sort=periods_sorted, title="Date"),
                               y=alt.Y("affinity_sum:Q", title=f"Sum of {party_col(party)}",
                                       scale=alt.Scale(zero=True)),
                               color=alt.Color("topic:N", legend=alt.Legend(title="Topic")),
                               tooltip=[
                                   alt.Tooltip("period:N",        title="Date"),
                                   alt.Tooltip("topic:N",         title="Topic"),
                                   alt.Tooltip("affinity_sum:Q",  title="Sum", format=",.0f"),
                               ]
                           )
                           .properties(height=460)
                    )
                    st.altair_chart(line, use_container_width=True)

                # Wide table for inspection/export
                st.dataframe(
                    view.pivot(index="period", columns="topic", values="affinity_sum").fillna(0.0),
                    use_container_width=True
                )
            else:
                st.info("No topics selected.")

# -------------------- TAB 2 --------------------
with tabs[1]:
    st.subheader("Story Landscape")

    with st.expander("What am I looking at?"):
        st.markdown("""(same explainer text you had)""")

    # Hardened loader
    # ---- discover emotions (prefers separate anger/outrage, falls back to merged) ----
    def discover_emotions(df: pd.DataFrame) -> dict[str, str]:
        cols = set(df.columns)
        out = []
        if "emo_anger" in cols:   out.append(("Anger",   "emo_anger"))
        if "emo_outrage" in cols: out.append(("Outrage", "emo_outrage"))
        if not any(k in cols for k in ("emo_anger","emo_outrage")) and "emo_anger_outrage" in cols:
            out.append(("Anger / Outrage", "emo_anger_outrage"))
        for label, key in [
            ("Anxiety","emo_anxiety"), ("Disgust","emo_disgust"), ("Fear","emo_fear"),
            ("Hope & Optimism","emo_hope_optimism"), ("Pride","emo_pride"), ("Sadness","emo_sadness")
        ]:
            if key in cols: out.append((label, key))
        return dict(out)

    # ---- load & date column ----
    df_en = load_enriched_metrics()
    def _resolve_fund_col(df, party: str) -> str:
        exact = "dem_fundraising_potential" if party=="Dem" else "gop_fundraising_potential"
        if exact in df.columns: return exact
        # fallback: any column mentioning party + fundraising + potential
        patt = r'(dem|gop).*(fundraising).*(potential)'
        cands = [c for c in df.columns if re.search(patt, c, flags=re.I)]
        return cands[0] if cands else exact  # returns exact even if missing (we’ll handle NaNs downstream)

    fund_col = _resolve_fund_col(df_en, party)
    pr = df_en.get("period", df_en.get("date"))
    df_en["period_dt"] = pd.to_datetime(pr, errors="coerce")
    if df_en.empty or df_en["period_dt"].dropna().empty:
        st.info("No parsable dates in topics_enriched.csv."); st.stop()

    EMOTION_COLS = discover_emotions(df_en)

    # ---- build bubble options once (Urgency + CTA Strength + emotions + morals) ----
    BUBBLE_OPTIONS = {"Urgency": "urgency_score"}
    if "cta_strength" in df_en.columns or "cta_ask_strength" in df_en.columns:
        BUBBLE_OPTIONS["CTA Strength"] = "cta_strength" if "cta_strength" in df_en.columns else "cta_ask_strength"
    BUBBLE_OPTIONS.update(EMOTION_COLS)
    BUBBLE_OPTIONS.update(MORAL_COLS)  # your static dict of mf_* → friendly names

    # ---- controls ----
    min_d, max_d = df_en["period_dt"].min().date(), df_en["period_dt"].max().date()
    d1, d2 = st.columns([1,1])  # <-- define BEFORE using d2
    date_range = d1.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_d, max_d))

    default_label = (
        "Anger / Outrage" if "Anger / Outrage" in BUBBLE_OPTIONS
        else ("CTA Strength" if "CTA Strength" in BUBBLE_OPTIONS else "Urgency")
    )
    bubble_friendly = d2.selectbox(
        "Bubble size metric", list(BUBBLE_OPTIONS.keys()),
        index=list(BUBBLE_OPTIONS.keys()).index(default_label)
    )
    bubble_col = BUBBLE_OPTIONS[bubble_friendly]

    bubble_contrast = 1.6
    min_bubble      = 80.0
    max_bubble      = 2200.0




    j1, j2 = st.columns([1,1])
    story_jitter = j1.slider("Jitter urgency (story labels)", 0.0, 2.0, 0.5, 0.1)
    topic_jitter = j2.slider("Jitter urgency (standardized topics)", 0.0, 2.0, 0.0, 0.1)

    st.caption(f"Using **{fund_col}** for Y-axis (party = {party}).")

    # Date filter
    msk = (df_en["period_dt"] >= pd.to_datetime(start_date)) & (df_en["period_dt"] <= pd.to_datetime(end_date))
    df_win = df_en[msk].copy()
    if df_win.empty:
        st.info("No rows in the selected date range.")
        st.stop()

    # ---------- TOP SCATTER: standardized topics ----------
    top_df = _explode_topics(df_win)  # expects standardized_topic_names as list

    for c in [fund_col, "urgency_score"]:
        if c in top_df.columns:
            top_df[c] = pd.to_numeric(top_df[c], errors="coerce")
    if top_df.empty:
        st.info("No standardized topics found in the selected range.")
    else:
        # guarantee bubble column exists for the chart
        if bubble_col not in top_df.columns:
            top_df[bubble_col] = np.nan
        # Guarantee bubble_col exists (fallback to 'urgency_score')
        topic_chart_df = _topic_agg(top_df, fund_col, bubble_col).dropna(subset=["urgency_mean","fund_mean"])
        if topic_chart_df.empty:
            st.info("No numeric rows for topic chart.")
        else:
            # clamp bubble metric so outliers don’t dwarf others
            p95 = float(topic_chart_df["bubble_mean"].quantile(0.95))
            topic_chart_df["bubble_mean"] = topic_chart_df["bubble_mean"].clip(upper=p95)

            topic_chart_df["bubble_size"] = _make_bubble_size(
                topic_chart_df["bubble_mean"],
                contrast=bubble_contrast,
                min_size=float(min_bubble),
                max_size=float(max_bubble),
            )
            x_col = "urgency_mean"
            if topic_jitter and topic_jitter > 0:
                topic_chart_df["x_jitter"] = topic_chart_df["urgency_mean"] + _jitter_by_key(topic_chart_df["std_topic"], amplitude=topic_jitter)
                x_col = "x_jitter"

            topic_chart = (
                alt.Chart(topic_chart_df)
                .mark_circle()
                .encode(
                    x=alt.X(f"{x_col}:Q", title="Avg Urgency"),
                    y=alt.Y("fund_mean:Q", title=f"Avg {fund_col}"),
                    size=alt.Size("bubble_size:Q", legend=None),
                    color=alt.Color("std_topic:N", legend=alt.Legend(title="Standardized Topic")),
                    tooltip=[
                        alt.Tooltip("std_topic:N", title="Topic"),
                        alt.Tooltip("urgency_mean:Q", title="Avg Urgency", format=",.1f"),
                        alt.Tooltip("fund_mean:Q", title=f"Avg {fund_col}", format=",.1f"),
                        alt.Tooltip("bubble_mean:Q", title=bubble_friendly, format=",.1f"),
                        alt.Tooltip("n_stories:Q", title="# Stories"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(topic_chart, use_container_width=True)
            st.download_button(
                "Download standardized topic metrics (CSV)",
                data=topic_chart_df.drop(columns=["bubble_size"], errors="ignore").to_csv(index=False),
                file_name=f"story_metrics_topics_{party}_{start_date}_to_{end_date}.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # ---------- BOTTOM SCATTER: story labels ----------
    topics_available = sorted(_explode_topics(df_win)["std_topic"].unique().tolist())
    topic_filter = st.selectbox("Filter labels by standardized topic", ["(All topics)"] + topics_available, index=0)

    if topic_filter != "(All topics)":
        # Keep rows whose list contains the chosen topic
        df_story = df_win[df_win["standardized_topic_names"].apply(lambda xs: topic_filter in (xs or []))].copy()
    else:
        df_story = df_win.copy()

    # Keep rows if ANY key metric is present (prevents over-filtering)
    df_story = df_story[(df_story.get("urgency_score", 0) > 0) |
                        (df_story.get("dem_fundraising_potential", 0) > 0) |
                        (df_story.get("gop_fundraising_potential", 0) > 0)]

    for c in [fund_col, "urgency_score"]:
        if c in df_story.columns:
            df_story[c] = pd.to_numeric(df_story[c], errors="coerce")

    # guarantee bubble column exists for the chart
    if bubble_col not in df_story.columns:
        df_story[bubble_col] = np.nan

    story_chart_df = _story_agg(df_story, fund_col, bubble_col).dropna(subset=["urgency_mean","fund_mean"])
    if story_chart_df.empty:
        st.info("No numeric rows for story-label chart.")
    else:
        # clamp + bubble sizing
        p95 = float(story_chart_df["bubble_mean"].quantile(0.95))
        story_chart_df["bubble_mean"] = story_chart_df["bubble_mean"].clip(upper=p95)
        story_chart_df["bubble_size"] = _make_bubble_size(
            story_chart_df["bubble_mean"],
            contrast=bubble_contrast,
            min_size=float(min_bubble),
            max_size=float(max_bubble),
        )
        x_col = "urgency_mean"
        if story_jitter and story_jitter > 0:
            story_chart_df["x_jitter"] = story_chart_df["urgency_mean"] + _jitter_by_key(story_chart_df["story_label"], amplitude=story_jitter)
            x_col = "x_jitter"

        story_chart = (
            alt.Chart(story_chart_df)
            .mark_circle()
            .encode(
                x=alt.X(f"{x_col}:Q", title="Avg Urgency"),
                y=alt.Y("fund_mean:Q", title=f"Avg {fund_col}"),
                size=alt.Size("bubble_size:Q", legend=None),
                color=alt.Color("story_label:N", legend=alt.Legend(title="Story Label")),
                tooltip=[
                    alt.Tooltip("story_label:N", title="Story"),
                    alt.Tooltip("urgency_mean:Q", title="Avg Urgency", format=",.1f"),
                    alt.Tooltip("fund_mean:Q", title=f"Avg {fund_col}", format=",.1f"),
                    alt.Tooltip("bubble_mean:Q", title=bubble_friendly, format=",.1f"),
                    alt.Tooltip("n_rows:Q", title="# Rows"),
                ],
            )
            .properties(height=460)
        )
        st.altair_chart(story_chart, use_container_width=True)
        st.download_button(
            "Download story-label metrics (CSV)",
            data=story_chart_df.drop(columns=["bubble_size"], errors="ignore").to_csv(index=False),
            file_name=f"story_metrics_labels_{party}_{start_date}_to_{end_date}.csv",
            mime="text/csv",
        )



# -------------------- TAB 3 --------------------
with tabs[2]:
    st.subheader("Story Scorecards")

    with st.expander("What is this?"):
        st.markdown("""(same explainer text you had)""")

    # Hardened loader: parses standardized_topic_names & numerics
    df_en = load_enriched_metrics()
    # Use dynamic column discovery (same logic as Tab 2)
    EMOTION_COLS_SC = discover_emotions(df_en)  # returns a dict label -> actual emo_* column names present
    MORAL_COLS_SC   = {k: v for k, v in MORAL_COLS.items() if v in df_en.columns}

    # pick whichever CTA column exists
    STRENGTH_COL_SC = "cta_strength" if "cta_strength" in df_en.columns else \
                    ("cta_ask_strength" if "cta_ask_strength" in df_en.columns else None)
    # Ensure a normalized lookup key for labels
    df_en["story_key"] = df_en["story_label"].astype(str).str.lower().str.replace(r"[^a-z0-9]+"," ", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()

    if df_en.empty:
        st.info("No data in topics_enriched.csv")
        st.stop()

    # Build topic → labels mapping from parsed list column
    et = _explode_topics(df_en[["story_label","standardized_topic_names"]])  # returns columns: story_label, std_topic
    topic_to_labels = (
        et.groupby("std_topic")["story_label"]
          .apply(lambda s: sorted(set(s.dropna().astype(str))))
          .to_dict()
    )
    topics_available = sorted(topic_to_labels.keys())

    # Numeric aggregates (safe: only present columns)
    agg_num_cols = ["urgency_score", "dem_fundraising_potential", "gop_fundraising_potential"]
    if STRENGTH_COL_SC: agg_num_cols.append(STRENGTH_COL_SC)
    agg_num_cols += list(EMOTION_COLS_SC.values()) + list(MORAL_COLS_SC.values())

    for c in agg_num_cols:
        if c in df_en.columns:
            df_en[c] = pd.to_numeric(df_en[c], errors="coerce")

    story_num = df_en.groupby("story_label", as_index=False)[agg_num_cols].mean(numeric_only=True)


    # Modes
    mode = st.radio("Select mode", ["🔎 Search", "🧭 Topic → Story", "📚 Full list (sortable)"], horizontal=True)
    selected_label = None

    if mode == "🔎 Search":
        q = st.text_input("Search story label")
        options = sorted([s for s in story_num["story_label"].astype(str) if q.lower() in s.lower()])[:300]
        if options:
            selected_label = st.selectbox("Pick a story", options)
        else:
            st.info("No matches.")

    elif mode == "🧭 Topic → Story":
        t_sel = st.selectbox("Standardized topic", ["(Choose)"] + topics_available, index=0)
        if t_sel != "(Choose)":
            lbls = topic_to_labels.get(t_sel, [])
            selected_label = st.selectbox("Story label", lbls) if lbls else st.info("No stories under this topic.")

    else:  # Full list (sortable)
        SORT_MAP = {
            "Urgency": "urgency_score",
            "Dem Fundraising Potential": "dem_fundraising_potential",
            "GOP Fundraising Potential": "gop_fundraising_potential",
            **{f"Emotion — {k}": v for k, v in EMOTION_COLS.items()},
            **{f"Moral — {k}": v for k, v in MORAL_COLS.items()},
        }
        c1, c2 = st.columns([2,1])
        sort_key_friendly = c1.selectbox("Sort by", list(SORT_MAP.keys()), index=0)
        sort_dir = c2.selectbox("Order", ["Descending", "Ascending"], index=0)
        col = SORT_MAP[sort_key_friendly]
        view = story_num[["story_label", col]].copy()
        view[col] = pd.to_numeric(view[col], errors="coerce")
        view = view.sort_values(col, ascending=(sort_dir == "Ascending"), na_position="last")
        ordered = view["story_label"].tolist()
        selected_label = st.selectbox("Pick a story", ordered)

    # ---------- SCORECARD ----------
    if selected_label:
        rows = _select_rows_for_label(df_en, selected_label)
        if rows.empty:
            st.warning("No rows for that story label.")
            st.stop()

        rows = _scorecard_enrich(rows)
        rows = _inject_spider_features(rows)   # ensure emo_/mf_ exist for spiders

        # ---- backfill dem/gop/cta from threads means when missing in the selected slice ----
        per_means, all_means = _threads_means_dem_gop_cta()
        sel_label  = rows["story_label"].iloc[0] if "story_label" in rows.columns and not rows.empty else None
        periods_in = rows["period"].astype(str).unique().tolist() if "period" in rows.columns and not rows.empty else []

        if sel_label:
            # per-period first
            if per_means is not None and not per_means.empty:
                donor = per_means[ (per_means["story_label"] == sel_label) &
                                (per_means["period"].astype(str).isin(periods_in)) ]
                # If there are multiple periods in the slice, take their mean
                if not donor.empty:
                    for c in ["dem_fundraising_potential","gop_fundraising_potential","urgency_score","cta_ask_strength"]:
                        if c in rows.columns:
                            rows[c] = pd.to_numeric(rows[c], errors="coerce")

            # if still missing, fill from all-period mean
            if all_means is not None and not all_means.empty:
                donor2 = all_means[ all_means["story_label"] == sel_label ]
                if not donor2.empty:
                    for c in ["dem_fundraising_potential","gop_fundraising_potential","cta_ask_strength"]:
                        if c in rows.columns and (rows[c].isna().all() or not rows[c].notna().any()):
                            rows[c] = float(pd.to_numeric(donor2[c], errors="coerce").mean())

        # TEMP PROBE — remove after confirming
        emo_nn = int(rows.filter(regex=r'^emo_').notna().sum().sum())
        mf_nn  = int(rows.filter(regex=r'^mf_').notna().sum().sum())
        st.caption(f"DEBUG[spiders]: emo_nonnull={emo_nn}, mf_nonnull={mf_nn}")

        # --- ensure expected numeric columns exist and are numeric in the selected slice ---

        # a) simple coercions for the standard columns if present
        for c in ["urgency_score", "dem_fundraising_potential", "gop_fundraising_potential"]:
            if c in rows.columns:
                rows[c] = pd.to_numeric(rows[c], errors="coerce")

        # b) CTA strength alias (already handled by _scorecard_enrich, but coerce again just in case)
        if STRENGTH_COL_SC and STRENGTH_COL_SC in rows.columns:
            rows[STRENGTH_COL_SC] = pd.to_numeric(rows[STRENGTH_COL_SC], errors="coerce")

        # c) last-ditch alias resolver if dem/gop fundraising got a different header in this export
        def _resolve_one(df: pd.DataFrame, target: str, patt: str) -> str:
            if target in df.columns:
                return target
            cands = [c for c in df.columns if re.search(patt, c, flags=re.I)]
            return cands[0] if cands else target  # return target even if missing; we'll handle NaN later

        dem_col = _resolve_one(rows, "dem_fundraising_potential", r"dem.*fundrais.*potential")
        gop_col = _resolve_one(rows, "gop_fundraising_potential", r"gop.*fundrais.*potential")

        # if the resolver found alternates, bring them under the expected names for mean_vals
        if dem_col != "dem_fundraising_potential" and dem_col in rows.columns:
            rows["dem_fundraising_potential"] = pd.to_numeric(rows[dem_col], errors="coerce")
        if gop_col != "gop_fundraising_potential" and gop_col in rows.columns:
            rows["gop_fundraising_potential"] = pd.to_numeric(rows[gop_col], errors="coerce")


        num_for_mean = ["urgency_score", "dem_fundraising_potential", "gop_fundraising_potential"] \
                    + list(EMOTION_COLS_SC.values()) + list(MORAL_COLS_SC.values())
        if STRENGTH_COL_SC:
            num_for_mean.append(STRENGTH_COL_SC)

        mean_vals = rows[num_for_mean].mean(numeric_only=True)

        # If everything is still NaN/0, pull direct means from tf for the selected label
        if (emo_nn + mf_nn) == 0:
            tf = _load_topic_features_sc()
            if not tf.empty and "story_label" in rows.columns:
                lab_candidates = ("story_label","label","topic_name","topic_label","topic")
                labcol_f = next((c for c in lab_candidates if c in tf.columns), None)
                if labcol_f:
                    sel = rows["story_label"].iloc[0]
                    tf["label_key"] = tf[labcol_f].astype(str).str.lower().str.replace(r"[^a-z0-9]+","", regex=True)
                    sel_key = re.sub(r"[^a-z0-9]+","", str(sel).lower())
                    sub = tf[tf["label_key"] == sel_key]
                    if sub.empty:
                        sub = tf[tf[labcol_f].astype(str).str.contains(sel, case=False, na=False)]
                    if not sub.empty:
                        # only for plotting; don't mutate rows
                        emo_means = sub.filter(regex=r"^emo_").mean(numeric_only=True)
                        mf_means  = sub.filter(regex=r"^mf_").mean(numeric_only=True)
                        mean_vals = pd.concat([mean_vals, emo_means, mf_means])


        classification = _mode_str(rows.get("classification"))
        gop_angle     = _mode_str(rows.get("gop_angle"))
        dem_angle     = _mode_str(rows.get("dem_angle"))


        # metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Dem fundraising potential", _fmt_num(mean_vals.get("dem_fundraising_potential")))
        m2.metric("GOP fundraising potential", _fmt_num(mean_vals.get("gop_fundraising_potential")))
        m3.metric("Urgency score",             _fmt_num(mean_vals.get("urgency_score")))

        # derive CTA fields (handle either column name)
        cta_type = _mode_str(rows.get("cta_ask_type") if "cta_ask_type" in rows.columns else rows.get("cta_type"))
        cta_copy = _mode_str(rows.get("cta_copy"))


        # CTA Strength (only if present)
        c1, c2, c3 = st.columns([1,2,1])
        c1.markdown("**CTA Type**");   c1.write(cta_type or "—")
        c2.markdown("**CTA Copy**");   c2.write(cta_copy or "—")
        c3.markdown("**CTA Strength**"); c3.write(_fmt_num(mean_vals.get(STRENGTH_COL_SC)) if STRENGTH_COL_SC else "—")


        # ---- replace the two radar lines with this block ----
        def _safe_scalar(d: pd.Series, key: str) -> float:
            if key in d.index:
                v = d[key]
                try:
                    return float(v) if pd.notna(v) else 0.0
                except Exception:
                    try:
                        # handle 0-dim numpy / pandas objects
                        return float(pd.to_numeric(v, errors="coerce"))
                    except Exception:
                        return 0.0
            return 0.0

        emo_series = pd.Series({lbl: _safe_scalar(mean_vals, EMOTION_COLS_SC[lbl])
                                for lbl in EMOTION_COLS_SC.keys()})
        mf_series  = pd.Series({lbl: _safe_scalar(mean_vals, MORAL_COLS_SC[lbl])
                                for lbl in MORAL_COLS_SC.keys()})



        rc1, rc2 = st.columns(2)
        rc1.plotly_chart(_radar_fig(emo_series, "Emotions (0–100)"), use_container_width=True)
        rc2.plotly_chart(_radar_fig(mf_series,  "Moral Foundations (0–100)"), use_container_width=True)

        def _join_list_col(series):
            items = []
            for x in series.fillna("").astype(str).tolist():
                xs = _to_list(x)
                items.extend(xs)
            s = ", ".join(sorted(set(items)))
            return s if s.strip() else "—"

        roles_df = pd.DataFrame({
            "Role": ["Heroes","Villains","Victims","Anti-heroes"],
            "Entities": [
                _join_list_col(rows.get("heroes",    pd.Series([], dtype="object"))),
                _join_list_col(rows.get("villains",  pd.Series([], dtype="object"))),
                _join_list_col(rows.get("victims",   pd.Series([], dtype="object"))),
                _join_list_col(rows.get("antiheroes",pd.Series([], dtype="object"))),
            ],
        })
        st.markdown("#### Narrative Roles")
        st.dataframe(roles_df, use_container_width=True)




# -------------------- TAB 4 --------------------
with tabs[3]:
    st.subheader("Story Leaderboard (CBGs where story is best)")
    if USE_PRECOMPUTED_ONLY:
        path = BEST_PRE[party]
        if not path.exists():
            st.warning("Precomputed best-per-CBG not found. Run s7g_precompute_artifacts.py.")
            st.stop()
        best = load_best_for_period(
            party,
            period,
            columns=["period","cbg_id","state","zcta","best_label","best_score","second_label","second_score","margin"]
        )
        if state_sel and "state" in best.columns:
            best = best[best["state"].isin(state_sel)]
        if state_sel and "state" in best.columns:
            best = best[best["state"].isin(state_sel)]
    else:
        # your on-the-fly path can stay here
        ...

    if best.empty:
        st.info("No data for this period.")
    else:
        best["__order"] = best[["margin","best_score"]].fillna(0).sum(axis=1)
        def top_cbgs(g):
            gg = g.sort_values(["margin","best_score"], ascending=False).head(10)
            return "; ".join([f"{r.cbg_id} ({r.state})" if "state" in gg.columns else r.cbg_id
                              for _, r in gg.iterrows()])
        counts = best.groupby("best_label").size().rename("cbgs").reset_index()
        tops = best.groupby("best_label").apply(top_cbgs).rename("top_cbgs").reset_index()
        board = counts.merge(tops, on="best_label", how="left").sort_values("cbgs", ascending=False)
        st.dataframe(board, use_container_width=True)
        st.download_button("Download leaderboard (CSV)",
                           data=board.to_csv(index=False),
                           file_name=f"story_leaderboard_{party}_{period}.csv",
                           mime="text/csv")



# -------------------- TAB 5 --------------------
with tabs[4]:
    st.subheader("Best Story by CBG (map)")

    path = BEST_PRE[party]
    if not path.exists():
        st.warning("Precomputed best-per-CBG not found. Run s7h_precompute_artifacts.py.")
        st.stop()

    # Read + harden best-per-CBG (drop 'all', coerce types)
    _best = pd.read_parquet(path)
    pr = _best.get("period", _best.get("date"))
    _best = _best[pr.astype(str).str.lower() != "all"].copy()
    _best["period"] = pd.to_datetime(pr, errors="coerce").dt.date
    _best = _best.dropna(subset=["period"])
    _best["cbg_id"] = _best.get("cbg_id", _best.get("cbg")).astype(str)
    _best["margin"] = pd.to_numeric(_best.get("margin"), errors="coerce").fillna(0.0)

    # Filter to selected period (your picker variable is `period`)
    best = _best[_best["period"] == period_dt].copy() if period_dt is not None else _best.iloc[0:0].copy()

    # if there are no rows for this date, fallback to the latest available period
    if best.empty:
        avail = _best["period"].dropna().sort_values().unique().tolist()
        if avail:
            best = _best[_best["period"] == avail[-1]].copy()
            st.caption(f"(No entries for {period_dt}; showing {avail[-1]})")

    # Optional state filter (works whether state is str or missing)
    if state_sel and "state" in best.columns:
        best = best[best["state"].isin(state_sel)]

    if best.empty:
        st.info("No data for map.")
        st.stop()

    # Margin filter
    margin_min = st.slider("Min margin vs #2 (confidence)", 0.0, 0.5, 0.05, 0.01)
    best = best[(best["margin"] >= float(margin_min))].copy()

    # Join tiny coord frame
    coords = (feat_small[["cbg_id","lat","lon"]]
              if {"lat","lon"}.issubset(feat_small.columns)
              else feat_small[["cbg_id"]])
    best = best.merge(coords, on="cbg_id", how="left").dropna(subset=["lat","lon"]).copy()

    # Optional focus on one story
    labels_sorted = ["(all)"] + sorted(best["best_label"].dropna().unique().tolist())
    focus_label = st.selectbox("Focus on story", labels_sorted, index=0)
    if focus_label != "(all)":
        best = best[best["best_label"] == focus_label]

    # Colors + tooltips
    best["color"] = best["best_label"].fillna("NA").apply(color_for_label)
    best["best_score_str"]   = pd.to_numeric(best.get("best_score"),   errors="coerce").round(3).astype("string")
    best["margin_str"]       = pd.to_numeric(best.get("margin"),       errors="coerce").round(3).astype("string")
    best["second_label_str"] = best.get("second_label").fillna("(none)").astype("string")
    best["state_str"]        = best.apply(_derive_state_from_any, axis=1)

    if "zcta" in best.columns:
        z = pd.to_numeric(best["zcta"], errors="coerce")
        best["zcta_str"] = z.round().astype("Int64").astype("string").replace("<NA>", "", regex=False)
    else:
        best["zcta_str"] = ""

    # Aggregated national (no state filter) vs. scatter (state-selected)
    if not state_sel:
        st.info("No states selected — showing aggregated national view.")
        df_map = best[["lat", "lon", "best_score"]].copy()
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_elevation_weight="best_score",
            elevation_scale=20,
            elevation_range=[0, 3000],
            extruded=True,
            radius=10000, coverage=0.9, pickable=False,
        )
        view = pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.6)
        st.pydeck_chart(pdk.Deck(layers=[hex_layer], initial_view_state=view,
                                 map_style="mapbox://styles/mapbox/dark-v9"))
    else:
        best = best[best["state_str"].isin(state_sel)].copy()
        if len(best) > 60000:
            best = best.sample(60000, random_state=7)

        tooltip = {
            "html": (
                "<b>CBG:</b> {cbg_id}<br/>"
                "<b>State:</b> {state_str}<br/>"
                "<b>ZIP:</b> {zcta_str}<br/>"
                "<b>Story:</b> {best_label}<br/>"
                "<b>Score:</b> {best_score_str}<br/>"
                "<b>Margin:</b> {margin_str}<br/>"
                "<b>#2:</b> {second_label_str}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "12px"},
        }
        # Keep only JSON-serializable columns for the layer
        cols_for_layer = [
            "cbg_id","lat","lon","best_label","best_score_str","margin_str",
            "second_label_str","state_str","zcta_str","color"
        ]
        df_layer = best[[c for c in cols_for_layer if c in best.columns]].copy()

        # Ensure native Python types (avoid numpy/extension types)
        for c in ["best_score_str","margin_str","second_label_str","state_str","zcta_str","cbg_id","best_label"]:
            if c in df_layer.columns:
                df_layer[c] = df_layer[c].astype(str)
        if "lat" in df_layer.columns:
            df_layer["lat"] = pd.to_numeric(df_layer["lat"], errors="coerce").astype(float)
        if "lon" in df_layer.columns:
            df_layer["lon"] = pd.to_numeric(df_layer["lon"], errors="coerce").astype(float)
        if "color" in df_layer.columns:
            df_layer["color"] = df_layer["color"].apply(lambda v: [int(x) for x in (v if isinstance(v, (list, tuple)) else [120,120,120,160])])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_layer,
            get_position='[lon, lat]',
            get_radius=500,
            get_fill_color="color",
            pickable=True,
            radius_min_pixels=2,
            radius_max_pixels=10,
            auto_highlight=True,
        )

        view = pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.6)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                 tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))

    st.download_button(
        "Download current CBG selections (CSV)",
        data=best.to_csv(index=False),
        file_name=f"cbg_best_{party}_{period}.csv",
        mime="text/csv",
    )


# -------------------- TAB 6 --------------------
with tabs[5]:
    st.subheader("Subgroup Rankings")

    # Offer only subgroup cols that exist in features
    avail_subgroups = [c for c in SUBGROUP_CANDIDATES if c in feat_small.columns]
    if not avail_subgroups:
        st.info("No subgroup columns found in CBG features.")
    else:
        demo_col = st.selectbox("Subgroup column", avail_subgroups, index=0)

        # Compute weighted ranks (uses normalized label_key mapping inside)
        with st.spinner("Computing subgroup rankings…"):
            stories, topics = subgroup_ranks(period_str, party, demo_col, feat_small, enriched)
            st.caption(f"DEBUG[subgroup]: stories={len(stories)}, topics={len(topics)}")
            resolved = stories["period"].iloc[0] if not stories.empty else (topics["period"].iloc[0] if not topics.empty else period_str)
            st.caption(f"Subgroup rankings computed for period: **{resolved}**")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stories (descending)**")
            st.dataframe(stories.head(1000), use_container_width=True)
        with c2:
            st.markdown("**Standardized Topics (descending)**")
            st.dataframe(topics.head(1000), use_container_width=True)

        st.download_button(
            "Download story rankings (CSV)",
            data=stories.to_csv(index=False),
            file_name=f"subgroup_story_rank_{demo_col}_{party}_{period}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download topic rankings (CSV)",
            data=topics.to_csv(index=False),
            file_name=f"subgroup_topic_rank_{demo_col}_{party}_{period}.csv",
            mime="text/csv",
        )
# FINGERPRINT: SPIDER_FIX_20251003

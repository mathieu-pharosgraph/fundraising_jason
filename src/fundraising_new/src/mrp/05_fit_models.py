#!/usr/bin/env python3
import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import pymc as pm
 
# Quiet the PPV NaN-rescale spam + legacy 'pps' deprecation chatter during AME
warnings.filterwarnings("ignore", message="`p` parameters sum to", category=UserWarning)
warnings.filterwarnings("ignore", message="'pps' has been replaced by 'response'", category=FutureWarning)

# --- Keep JAX memory in check (must be set before JAX/PyMC import paths are used)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

CCES = Path("outputs/mrp/cces_2020_prepped.parquet")
OUTD = Path("outputs/mrp/models")
OUTD.mkdir(parents=True, exist_ok=True)

FAST_DEBUG = os.getenv("FAST_DEBUG", "0") == "1"
RANDOM_SEED = 42
TOP_PRINT = 20
AME_SAMPLE_N = 5000
if FAST_DEBUG:
    TOP_PRINT = 15
    AME_SAMPLE_N = 2000

# Full run settings (unchanged)
FIT_KW = dict(draws=1500, tune=3000, target_accept=0.95, chains=4)
if FAST_DEBUG:
    FIT_KW.update(dict(draws=200, tune=1200, chains=2, target_accept=0.995))
CORES_NUTS = 1

print("Versions — bambi", bmb.__version__, "| pymc", pm.__version__)

# ---------- IO ----------
df = pd.read_parquet(CCES)

# Merge continuous covars
state_inc = pd.read_parquet("outputs/mrp/state_income_acs.parquet")[["state_postal","state_income_z"]]
state_urb = pd.read_parquet("outputs/mrp/state_urban_index_acs.parquet")[["state_postal","state_urban_z"]]
# --- State two-party GOP share (z-scored) ---
state_gop = pd.read_parquet("outputs/mrp/state_gop_2p.parquet")[["state_postal","state_gop_2p_z"]]
df = df.merge(state_gop, on="state_postal", how="left")
df["state_gop_2p_z"] = df["state_gop_2p_z"].fillna(0.0)  # conservative fallback

df = df.merge(state_inc, on="state_postal", how="left").merge(state_urb, on="state_postal", how="left")
df["state_income_z"] = df["state_income_z"].fillna(0.0)
df["state_urban_z"] = df["state_urban_z"].fillna(0.0)

# Required columns
req_common = ["state_postal","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
req_ideo   = req_common + ["ideology7"]
req_pid7   = req_common + ["pid7"]
req_reg    = req_common + ["registration_3"]

def keep_nonmissing(data, cols):
    return data.dropna(subset=[c for c in cols if c in data.columns])

def _ensure_seven_levels(series):
    """Force ordered categories to be exactly [1..7] even if some levels are unobserved."""
    s = pd.to_numeric(series, errors="coerce").astype("Int64")
    s = s.where(s.between(1, 7))
    return pd.Categorical(s, categories=[1,2,3,4,5,6,7], ordered=True)

def to_ordered_contiguous(series):
    """Make strictly ordered, contiguous categories [1..K] from any 7-point input."""
    s = pd.to_numeric(series, errors="coerce")
    s = s[~s.isna()]
    if s.empty:
        return pd.Categorical([], ordered=True)
    # Keep only actually observed levels; then remap to 1..K to avoid empty categories
    vals = np.sort(s.unique())
    mapping = {old:i+1 for i,old in enumerate(vals)}
    s2 = s.map(mapping).astype(int)
    cats = list(range(1, len(vals)+1))
    return pd.Categorical(s2, categories=cats, ordered=True)

print("Raw shape:", df.shape)
print("NA rates (top 12 cols):")
print(df.isna().mean().sort_values(ascending=False).head(12))

# Coerce predictors to category
for c in ["state_postal","age_bin","sex","race_eth","edu_bin","income_bin","married","owner","registration_3"]:
    if c in df.columns:
        df[c] = df[c].astype("category")

# Build per-model frames
df_ideo = keep_nonmissing(df, req_ideo).copy()
df_pid  = keep_nonmissing(df, req_pid7).copy()
df_reg  = keep_nonmissing(df, req_reg).copy()
df_ideo = df_ideo.iloc[0:0]
df_reg  = df_reg.iloc[0:0]

if FAST_DEBUG:
    df_ideo = df_ideo.sample(min(len(df_ideo), 10000), random_state=1)
    df_reg  = df_reg.sample(min(len(df_reg), 20000), random_state=2)
    df_pid  = df_pid.sample(min(len(df_pid), 15000), random_state=3)

    for _d, _y in [(df_ideo, "ideology7"), (df_pid, "pid7")]:
        if len(_d) and hasattr(_d[_y], "cat"):
            _d[_y] = _d[_y].cat.remove_unused_categories()

def _rebuild_ordered(series):
    s = pd.to_numeric(series, errors="coerce")
    vals = np.sort(pd.unique(s.dropna()))
    mapping = {old: i+1 for i, old in enumerate(vals)}  # contiguous 1..K
    s2 = s.map(mapping).astype("Int64")
    cats = list(range(1, len(vals)+1))
    return pd.Categorical(s2, categories=cats, ordered=True)

# Rebuild AFTER subsampling, then force 7-level categories to match model coords
if len(df_ideo):
    df_ideo["ideology7"] = _rebuild_ordered(df_ideo["ideology7"])
    df_ideo["ideology7"] = _ensure_seven_levels(df_ideo["ideology7"])
if len(df_pid):
    df_pid["pid7"] = _rebuild_ordered(df_pid["pid7"])
    df_pid["pid7"] = _ensure_seven_levels(df_pid["pid7"])

def _make_initvals_list(mdl, chains: int, seed: int = 42):
    ivals, base = [], mdl.initial_point()
    thr = None
    if "threshold" in base:
        k1 = int(np.array(base["threshold"]).shape[-1])
        thr = np.linspace(-0.75, 0.75, k1).astype(float)  # narrow, safe
    for _ in range(chains):
        v = {}
        for name, arr in base.items():
            arr = np.array(arr)
            if name == "threshold" and thr is not None:
                v[name] = thr.copy()
            elif name.endswith("_log__"):
                v[name] = np.full_like(arr, np.log(0.2), dtype=float)
            else:
                v[name] = np.zeros_like(arr, dtype=float)
        ivals.append(v)
    return ivals





# Outcomes to ordered *contiguous* categoricals (prevents empty threshold issues)
if len(df_ideo):
    df_ideo = df_ideo[df_ideo["ideology7"].notna()].copy()

if len(df_pid):
    df_pid = df_pid[df_pid["pid7"].notna()].copy()

if len(df_reg):
    # Keep only 3 main labels, drop others/NA cleanly and remove unused cats
    df_reg = df_reg[df_reg["registration_3"].isin(["Dem","Rep","Ind"])].copy()
    df_reg["registration_3"] = df_reg["registration_3"].cat.remove_unused_categories()

# Drop any rows that picked up NA after category ops
df_ideo = df_ideo.dropna(subset=req_ideo, how="any")
df_pid  = df_pid.dropna(subset=req_pid7, how="any")
df_reg  = df_reg.dropna(subset=req_reg, how="any")

if FAST_DEBUG:
    # Optional extra cap to prevent huge design in debug runs
    for name, d in [("ideo", df_ideo), ("pid", df_pid), ("reg", df_reg)]:
        if len(d) > 50_000:
            d = d.sample(50_000, random_state=123)
        if name == "ideo": df_ideo = d
        elif name == "pid": df_pid = d
        else: df_reg = d

def cat_levels(d, cols):
    out = {}
    for c in cols:
        if c in d.columns and hasattr(d[c], "cat"):
            out[c] = list(d[c].cat.categories)
    return out

print("\nAfter filtering:")
print("  ideology rows:", len(df_ideo))
print("  pid7     rows:", len(df_pid))
print("  reg      rows:", len(df_reg))

print("\nLevels snapshot (ideo):", cat_levels(df_ideo, ["state_postal","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]))
print("Levels snapshot (reg): ", cat_levels(df_reg,  ["registration_3"]))

if len(df_ideo) == 0 and len(df_pid) == 0 and len(df_reg) == 0:
    raise SystemExit("No observations remain after filtering. Check inputs.")

# ---------- Common bits ----------
def top_effects(nc_path, top=20):
    idata = az.from_netcdf(nc_path)
    summ = az.summary(idata).reset_index().rename(columns={"index":"term"})
    drop_re = re.compile(r"^p\[|\||threshold|_sigma|_offset|_log__", re.ASCII)
    summ = summ[~summ["term"].str.contains(drop_re)]
    summ["abs_mean"] = summ["mean"].abs()
    out = summ.sort_values("abs_mean", ascending=False).head(top)
    cols = ["term","mean","sd","hdi_3%","hdi_97%","r_hat","ess_bulk","ess_tail"]
    print(f"\nTop effects in {nc_path}:")
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    return out




def _prune_idata(idata):
    try:
        if hasattr(idata, "posterior") and "p" in idata.posterior:
            idata.posterior = idata.posterior.drop_vars("p")
    except Exception:
        pass
    for grp in ("log_likelihood","posterior_predictive","prior","prior_predictive"):
        if hasattr(idata, grp):
            try: setattr(idata, grp, None)
            except Exception: pass
    return idata

def _try_sample_pymc(model, *, label, init="adapt_diag", target_accept=0.95):
    """Robust sampler that avoids PyMC initvals string bugs by supplying
    an explicit list of per-chain init dicts.
    """
    mdl = model.backend.model

    def _sample(chains, init_kind=init):
        initvals_list = _make_initvals_list(mdl, chains, seed=RANDOM_SEED)
        with mdl:
            return pm.sample(
                draws=FIT_KW["draws"], tune=FIT_KW["tune"],
                chains=chains, cores=1,                        # sequential on macOS
                target_accept=target_accept, init=init_kind,   # 'adapt_diag' is fine (we supply initvals)
                initvals=initvals_list, random_seed=RANDOM_SEED,
                progressbar=True,
            )

    try:
        return _sample(FIT_KW["chains"], init_kind=init)
    except Exception as e1:
        print(f"[{label}] First attempt failed ({type(e1).__name__}: {e1}). Retrying with adapt_diag …")
        return _sample(FIT_KW["chains"], init_kind="adapt_diag")






# ---------- Models ----------
# Priors
priors_re_common = {
    "1|state_postal_sigma":       bmb.Prior("HalfNormal", sigma=0.5),
    "edu_bin|state_postal_sigma": bmb.Prior("HalfNormal", sigma=0.3),
}
prior_thresh = bmb.Prior("Normal", mu=0, sigma=2.0)
prior_race   = bmb.Prior("Normal", mu=0, sigma=1.0)

# Formulas
FORM_COMMON = (
    "  age_bin + sex + race_eth + edu_bin + income_bin + married + owner"
    " + state_income_z + income_bin:state_income_z"
    " + state_urban_z"
    " + race_eth:edu_bin + age_bin:edu_bin"
)

# ===== STABLE SETTINGS FOR ORDERED MODELS =====
# Main effects only (drop interactions for first pass)
FORM_COMMON_MAIN = (
    "  age_bin + sex + race_eth + edu_bin + income_bin + married + owner"
    " + state_income_z + state_urban_z + state_gop_2p_z"
)

RE_STATE = " + (1 | state_postal)"

# Random effects: intercept by state only (no varying slopes on first pass)
RE_SIMPLE = " + (1 | state_postal)"

# Tighter priors to avoid wild initial geometry
prior_coef   = bmb.Prior("Normal", mu=0, sigma=0.5)   # all fixed effects
prior_thresh = bmb.Prior("Normal", mu=0, sigma=1.0)   # ordered cutpoints, tighter than 2.0
prior_re_sig = bmb.Prior("HalfNormal", sigma=0.2)     # random-intercept sd (shrink)

# Full varying intercept + edu_bin slopes
RE_FULL   = " + (1 + edu_bin | state_postal)"
# Simpler fallback: varying intercept only
RE_SIMPLE = " + (1 | state_postal)"

# --- IDEOLOGY (ordered) ---
if len(df_ideo) > 0:
    form_ideo = "ideology7 ~ " + FORM_COMMON_MAIN
    priors_ideo = {
        "threshold": prior_thresh,
        "age_bin":   prior_coef,
        "sex":       prior_coef,
        "race_eth":  prior_coef,
        "edu_bin":   prior_coef,
        "income_bin":prior_coef,
        "married":   prior_coef,
        "owner":     prior_coef,
        "state_income_z": prior_coef,
        "state_urban_z":  prior_coef,
    }
    m1 = bmb.Model(
        form_ideo, df_ideo,
        family="cumulative", link="probit",
        priors=priors_ideo
    )

    m1.build()
    k_ideo = len(df_ideo["ideology7"].cat.categories)
    with m1.backend.model:
        m1.backend.model.coords["ideology7_dim"] = np.arange(6)  # K-1 thresholds
    idata_ideo = _try_sample_pymc(
        m1, label="IDEO + stable",
        init="adapt_diag", target_accept=0.99,
)




    idata_ideo = _prune_idata(idata_ideo)
    az.to_netcdf(idata_ideo, OUTD / "ideology_idata.nc", engine="h5netcdf")
    top_effects(str(OUTD / "ideology_idata.nc")).to_csv(OUTD / "coef_top_ideology.csv", index=False)
    print("✓ saved", OUTD / "ideology_idata.nc")

# --- REGISTRATION (multinomial/categorical) ---
if len(df_reg) > 0:
    form_reg = "registration_3 ~ " + FORM_COMMON + RE_SIMPLE  # intercept RE is stable here
    pri_reg = {"race_eth": prior_race, "1|state_postal_sigma": bmb.Prior("HalfNormal", sigma=0.5)}
    m2 = bmb.Model(form_reg, df_reg, family="categorical", priors=pri_reg)
    idata_reg = m2.fit(inference="nuts", chain_method="sequential", cores=CORES_NUTS, **FIT_KW,
                       idata_kwargs=dict(log_likelihood=False, save_warmup=False))
    idata_reg = _prune_idata(idata_reg)
    az.to_netcdf(idata_reg, OUTD / "registration_idata.nc", engine="h5netcdf")
    top_effects(str(OUTD / "registration_idata.nc")).to_csv(OUTD / "coef_top_registration.csv", index=False)
    print("✓ saved", OUTD / "registration_idata.nc")

# --- PID7 (ordered) ---
if len(df_pid) > 0:
    form_pid = "pid7 ~ " + FORM_COMMON_MAIN + RE_STATE
    priors_pid = {
        "threshold": prior_thresh,
        "age_bin":   prior_coef,
        "sex":       prior_coef,
        "race_eth":  prior_coef,
        "edu_bin":   prior_coef,
        "income_bin":prior_coef,
        "married":   prior_coef,
        "owner":     prior_coef,
        "state_income_z": prior_coef,
        "state_urban_z":  prior_coef,
        "state_gop_2p_z": bmb.Prior("Normal", mu=0, sigma=0.5),
        "1|state_postal_sigma": bmb.Prior("HalfNormal", sigma=0.5),  # RE sd prior
    }
    m3 = bmb.Model(form_pid, df_pid, family="cumulative", priors=priors_pid)
    m3.build()
    k_pid = len(df_pid["pid7"].cat.categories)
    with m3.backend.model:
        m3.backend.model.coords["pid7_dim"] = np.arange(6)       # K-1 thresholds



    idata_pid = _try_sample_pymc(
        m3, label="PID + stable",
        init="adapt_diag", target_accept=0.99
    )



    idata_pid = _prune_idata(idata_pid)
    az.to_netcdf(idata_pid, OUTD / "pid7_idata.nc", engine="h5netcdf")
    top_effects(str(OUTD / "pid7_idata.nc")).to_csv(OUTD / "coef_top_pid7.csv", index=False)
    print("✓ saved", OUTD / "pid7_idata.nc")

# =======================
# AME blocks (unchanged)
# =======================
def _ensure_category(series):
    return series if hasattr(series, "cat") else series.astype("category")

def _ref_level_of(series):
    series = _ensure_category(series)
    cats = list(series.cat.categories)
    return cats[0] if cats else None

def _align_cats(df_new: pd.DataFrame, df_ref: pd.DataFrame, cat_cols):
    """Make df_new categorical columns use the SAME categories as df_ref."""
    for c in cat_cols:
        if c in df_new.columns and c in df_ref.columns and hasattr(df_ref[c], "cat"):
            ref_cats = list(df_ref[c].cat.categories)
            df_new[c] = _ensure_category(df_new[c])
            df_new[c] = df_new[c].cat.set_categories(ref_cats)
    return df_new


def _pps_prob(model, idata, data, target):
    """
    Robust P(Y==target) using bambi 0.15 predict(kind="response").

    - Drops the response column from 'data' (Bambi expects only predictors).
    - Handles probability tensors with a class axis anywhere (K∈{3,5,6,7}).
    - Renormalizes per observation when NaNs appear, to avoid all-zero AMEs.
    - Falls back to integer labels or rounded floats if needed.
    """
    # 0) Ensure the response (y) column is not in 'data'
    try:
        yname = model.response.name
    except Exception:
        yname = None
    X = data.copy()
    if yname and yname in X.columns:
        X = X.drop(columns=[yname], errors="ignore")

    # 1) Predict on response scale (no categorical draws)
    pred = model.predict(data=X, kind="response", idata=idata)

    # 2) To numpy
    arr = getattr(pred, "values", pred)
    arr = np.asarray(arr)

    def _mean_bool(x):
        x = np.asarray(x, dtype=float)
        return 0.0 if x.size == 0 else float(np.nanmean(x))

    # 3) Handle object dtype (strings / None)
    if arr.dtype == object:
        flat = arr.reshape(-1).astype(str)
        mask = (flat != "None") & (flat != "nan")
        if not mask.any():
            return 0.0
        num = pd.to_numeric(flat[mask], errors="coerce")
        if np.isfinite(num).any() and np.allclose(num.dropna(), np.round(num.dropna())):
            return _mean_bool(num.values == int(target))
        return 0.0

    # 4) Probability tensor: find a class axis of size K (3/5/6/7) anywhere
    shape = arr.shape
    class_axes = [i for i, s in enumerate(shape) if s in (3, 5, 6, 7)]
    if class_axes:
        k_axis = class_axes[-1]  # prefer the last matching axis
        # move class axis to last
        if k_axis != arr.ndim - 1:
            axes = list(range(arr.ndim))
            axes.append(axes.pop(k_axis))
            arr = np.transpose(arr, axes)  # now arr[..., K]
        k = int(target) - 1
        probs = arr[..., k]

        # Sanitize NaNs/Infs
        probs = np.where(np.isfinite(probs), probs, 0.0)

        # If upstream gave NaNs in other classes, row sums may be < 1 — renormalize per obs if we can
        # Compute row sums across K to detect zeros; use the full tensor if available
        if arr.ndim >= 1:
            # sum across K on the sanitized tensor
            sums = np.sum(np.where(np.isfinite(arr), arr, 0.0), axis=-1)
            # Avoid division by zero; only renorm where sum>0
            need_norm = (sums > 0)
            # If many rows have zero sum, average the sanitized probs as-is
            if np.any(need_norm):
                # Renormalize our selected class prob only on valid rows
                probs_norm = np.zeros_like(probs, dtype=float)
                probs_norm[need_norm] = probs[need_norm] / sums[need_norm]
                probs = probs_norm

        return float(np.nanmean(probs))

    # 5) Integer labels (unlikely on 'response', but safe)
    if np.issubdtype(arr.dtype, np.integer):
        return _mean_bool(arr == int(target))

    # 6) Float responses: round to nearest category
    if np.issubdtype(arr.dtype, np.floating):
        rounded = np.rint(arr)
        rounded[~np.isfinite(rounded)] = np.nan
        return _mean_bool(rounded == float(int(target)))

    return 0.0


def _ame_for_var(model, idata, df, var, target, fit_df, cat_cols, sample_n=5000, ref=None):
    if var not in df.columns: return None
    work = df[df[var].notna()].copy()
    if len(work) == 0: return None
    if len(work) > sample_n: work = work.sample(sample_n, random_state=42).copy()
    work[var] = _ensure_category(work[var])
    levels = list(work[var].cat.categories)
    if ref is None or ref not in levels: ref = _ref_level_of(work[var]) or levels[0]
    base = work.copy()
    base[var] = base[var].cat.set_categories(levels)
    base[var] = ref
    base = _align_cats(base, fit_df, cat_cols)
    if CONT_COLS:
        base[CONT_COLS] = base[CONT_COLS].fillna(0.0)
    p_base = _pps_prob(model, idata, base, target)
    rows = []
    for lv in levels:
        if lv == ref:
            rows.append((var, lv, ref, 0.0, p_base, p_base)); continue
        alt = work.copy()
        alt[var] = alt[var].cat.set_categories(levels)
        alt[var] = lv
        alt = _align_cats(alt, fit_df, cat_cols)
        if CONT_COLS:
            alt[CONT_COLS] = alt[CONT_COLS].fillna(0.0)
        p_alt = _pps_prob(model, idata, alt, target)
        rows.append((var, lv, ref, float(p_alt - p_base), p_alt, p_base))
    out = pd.DataFrame(rows, columns=["variable","level","ref","AME(target prob)","P(level)","P(ref)"])
    out["abs_AME"] = out["AME(target prob)"].abs()
    return out.sort_values("abs_AME", ascending=False)

def print_top_ames(model, idata, df, target, predictors, title, fit_df, cat_cols, sample_n=AME_SAMPLE_N, top=TOP_PRINT):
    tables = []
    for var in predictors:
        try:
            tab = _ame_for_var(model, idata, df, var, target, fit_df, cat_cols, sample_n=sample_n)
            if tab is not None: tables.append(tab)
        except Exception as e:
            warnings.warn(f"AME failed for {var}: {e}")
    if not tables:
        print(f"\n[AME] No results for {title}")
        return
    all_tab = pd.concat(tables, ignore_index=True).sort_values("abs_AME", ascending=False)
    print(f"\n=== AME — {title} (Δ probability of target) ===")
    print(all_tab.head(top).to_string(index=False))
    return all_tab

PREDICTORS = ["age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
CATS_ORD = ["state_postal","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
CATS_REG = ["state_postal","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
CONT_COLS = ["state_income_z","state_urban_z"]

# Ideology AME: target = highest observed category in contiguous mapping
if len(df_ideo):
    tgt_ideo = int(max(df_ideo["ideology7"].cat.categories)) if len(df_ideo["ideology7"].cat.categories) else 7
    try:
        ame_ideo = print_top_ames(
            m1, idata_ideo, df_ideo, target=tgt_ideo,
            predictors=PREDICTORS, title=f"Ideology (P[Y={tgt_ideo}])",
            fit_df=df_ideo, cat_cols=CATS_ORD, sample_n=AME_SAMPLE_N, top=TOP_PRINT
        )
        if ame_ideo is not None:
            ame_ideo.to_csv(OUTD / f"ame_ideology_p{tgt_ideo}.csv", index=False)
    except Exception as e:
        warnings.warn(f"AME ideology failed: {e}")

# Registration AME: ΔP(Dem)
if len(df_reg):
    dem_label = "Dem" if "Dem" in list(df_reg["registration_3"].cat.categories) else list(df_reg["registration_3"].cat.categories)[0]
    try:
        ame_reg = print_top_ames(
            m2, idata_reg, df_reg, target=dem_label,
            predictors=PREDICTORS, title=f"Registration (P[{dem_label}])",
            fit_df=df_reg, cat_cols=CATS_REG, sample_n=AME_SAMPLE_N, top=TOP_PRINT
        )
        if ame_reg is not None:
            ame_reg.to_csv(OUTD / f"ame_registration_{dem_label}.csv", index=False)
    except Exception as e:
        warnings.warn(f"AME registration failed: {e}")

# PID7 AME: target = highest observed category
if len(df_pid):
    tgt_pid = int(max(df_pid["pid7"].cat.categories)) if len(df_pid["pid7"].cat.categories) else 7
    try:
        ame_pid = print_top_ames(
            m3, idata_pid, df_pid, target=tgt_pid,
            predictors=PREDICTORS, title=f"PID7 (P[Y={tgt_pid}])",
            fit_df=df_pid, cat_cols=CATS_ORD, sample_n=AME_SAMPLE_N, top=TOP_PRINT
        )
        if ame_pid is not None:
            ame_pid.to_csv(OUTD / f"ame_pid7_p{tgt_pid}.csv", index=False)
    except Exception as e:
        warnings.warn(f"AME PID7 failed: {e}")

print("\nAll done. Models in:", OUTD)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from scipy.stats import norm

# -------- Paths --------
RAKED = Path("outputs/mrp/cbg_raked_cells.parquet")
MODELD = Path("outputs/mrp/models")
OUTD   = Path("outputs/mrp/cbg_estimates")
OUTD.mkdir(parents=True, exist_ok=True)

# -------- Load raked micro-cells --------
raked = pd.read_parquet(RAKED)

# --- ensure raked has state_postal ---
if "state_postal" not in raked.columns:
    # try to pull from a cbg→state mapping if you saved it with marginals
    try:
        cbg_map = pd.read_parquet("outputs/mrp/cbg_marginals.parquet")[["cbg_id","state_postal"]].drop_duplicates("cbg_id")
        raked = raked.merge(cbg_map, on="cbg_id", how="left")
    except Exception:
        # last-resort: derive from CBG FIPS (first 2 digits) with a static map
        FIPS2POSTAL = {
            "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
            "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
            "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
            "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
            "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
            "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
            "54":"WV","55":"WI","56":"WY","72":"PR"
        }
        if "cbg_id" not in raked.columns:
            raise ValueError("raked is missing both state_postal and cbg_id, can’t derive state.")
        raked["state_postal"] = raked["cbg_id"].astype(str).str[:2].map(FIPS2POSTAL)

# --- bring in state covariates used in the fit ---
state_inc = pd.read_parquet("outputs/mrp/state_income_acs.parquet")[["state_postal","state_income_z"]]
state_urb = pd.read_parquet("outputs/mrp/state_urban_index_acs.parquet")[["state_postal","state_urban_z"]]
raked = (
    raked.merge(state_inc, on="state_postal", how="left")
         .merge(state_urb, on="state_postal", how="left")
)
# bring state two-party GOP z into raked cells
state_gop = pd.read_parquet("outputs/mrp/state_gop_2p.parquet")[["state_postal","state_gop_2p_z"]]
raked = raked.merge(state_gop, on="state_postal", how="left")
raked["state_gop_2p_z"] = raked["state_gop_2p_z"].fillna(0.0)

# fill any holes (should be rare)
raked["state_income_z"] = raked["state_income_z"].fillna(0.0)
raked["state_urban_z"]  = raked["state_urban_z"].fillna(0.0)

# Expected columns:
#   cbg_id, w_raked,
#   age_bin, sex, race_eth, edu_bin, income_bin, married, owner,
#   state_income_z, state_urban_z
need_cols = [
    "cbg_id", "w_raked",
    "age_bin","sex","race_eth","edu_bin","income_bin","married","owner",
    "state_income_z","state_urban_z","state_postal","state_gop_2p_z"  # add these two
]
missing = [c for c in need_cols if c not in raked.columns]
if missing:
    raise ValueError(f"Raked cells missing columns: {missing}")

# Enforce factor levels (must match training)
LEVELS = {
    "age_bin":    ["18-29","30-44","45-64","65+"],  # baseline: 18-29
    "sex":        ["Female","Male"],                 # baseline: Female
    "race_eth":   ["Asian","Black","Latino","Other","White"],  # baseline: Asian
    "edu_bin":    ["BAplus","HS_or_less","SomeCollege"],       # baseline: BAplus
    "income_bin": ["high","low","mid"],                          # baseline: high
    "married":    ["married","other"],                           # baseline: married
    "owner":      ["owner","renter"],                            # baseline: owner
}

for col, levels in LEVELS.items():
    raked[col] = pd.Categorical(raked[col].astype(str), categories=levels, ordered=(col=="age_bin"))

# -------- Load PID7 posterior (cumulative-probit) --------
idata_pid = az.from_netcdf(MODELD / "pid7_idata.nc")
post = idata_pid.posterior  # xarray Dataset

# Helper to get posterior-mean coefficients
def mean_term(name):
    if name not in post:
        return None
    da = post[name].mean(dim=("chain","draw"))
    return da

# -------- Extract posterior means (fixed) + build eta + add RE --------

# Posterior means for fixed effects
b_age   = mean_term("age_bin")         # vector (levels != baseline) or None
b_sex   = mean_term("sex")
b_race  = mean_term("race_eth")
b_edu   = mean_term("edu_bin")
b_inc   = mean_term("income_bin")
b_mar   = mean_term("married")         # scalar (other vs married) or None
b_own   = mean_term("owner")           # scalar (renter vs owner) or None
b_sinc  = mean_term("state_income_z")  # scalar or None
b_surb  = mean_term("state_urban_z")   # scalar or None
b_gop   = mean_term("state_gop_2p_z")  # scalar or None
taus    = mean_term("threshold")       # (6,)

if taus is None or np.asarray(taus).shape[-1] != 6:
    raise ValueError("Expected 'threshold' with length 6 in posterior means (cumulative ordered model).")

def to_np(x):
    return None if x is None else np.asarray(x)

def _to_float_scalar(x, default=0.0):
    if x is None:
        return float(default)
    try:
        # works for DataArray/np scalar/Python float
        return float(np.asarray(x).item())
    except Exception:
        return float(x)

# Cast scalars ONCE
b_sinc_f = _to_float_scalar(b_sinc, 0.0)
b_surb_f = _to_float_scalar(b_surb, 0.0)
b_gop_f  = _to_float_scalar(b_gop,  0.0)

# Vectors (may be None if factor absent)
b_age_np  = to_np(b_age)
b_sex_np  = to_np(b_sex)
b_race_np = to_np(b_race)
b_edu_np  = to_np(b_edu)
b_inc_np  = to_np(b_inc)
b_mar_np  = to_np(b_mar)
b_own_np  = to_np(b_own)
taus_np   = to_np(taus)  # (6,)

# Build level-index maps (non-baseline levels only)
def idx_map(term_name, levels_full):
    da = mean_term(term_name)
    if da is None or da.ndim == 0:
        return {}, []
    lvl_coord = [d for d in da.dims if d not in ("chain","draw")][0]
    levels_in_beta = list(da.coords[lvl_coord].values)
    mapping = {L: (levels_in_beta.index(L) if L in levels_in_beta else None) for L in levels_full}
    return mapping, levels_in_beta

idx_age_map,  _ = idx_map("age_bin",    LEVELS["age_bin"])
idx_sex_map,  _ = idx_map("sex",        LEVELS["sex"])
idx_race_map, _ = idx_map("race_eth",   LEVELS["race_eth"])
idx_edu_map,  _ = idx_map("edu_bin",    LEVELS["edu_bin"])
idx_inc_map,  _ = idx_map("income_bin", LEVELS["income_bin"])

# Linear predictor
N = len(raked)
eta = np.zeros(N, dtype=float)

def add_factor(eta, series, idx_map, beta_vec):
    if beta_vec is None:
        return eta
    vals = series.astype(str).to_numpy()
    idxs = np.array([idx_map.get(v, None) for v in vals], dtype=object)
    mask = np.array([i is not None for i in idxs])
    if mask.any():
        eta[mask] += beta_vec[idxs[mask]]
    return eta

eta = add_factor(eta, raked["age_bin"],    idx_age_map,  b_age_np)
eta = add_factor(eta, raked["sex"],        idx_sex_map,  b_sex_np)
eta = add_factor(eta, raked["race_eth"],   idx_race_map, b_race_np)
eta = add_factor(eta, raked["edu_bin"],    idx_edu_map,  b_edu_np)
eta = add_factor(eta, raked["income_bin"], idx_inc_map,  b_inc_np)

# Binary scalars
if b_mar_np is not None:
    eta += float(np.asarray(b_mar_np).item()) * (raked["married"].astype(str).to_numpy() == "other").astype(float)
if b_own_np is not None:
    eta += float(np.asarray(b_own_np).item()) * (raked["owner"].astype(str).to_numpy() == "renter").astype(float)

# Numeric scalars
eta += b_sinc_f * raked["state_income_z"].to_numpy(float)
eta += b_surb_f * raked["state_urban_z"].to_numpy(float)
eta += b_gop_f  * raked["state_gop_2p_z"].to_numpy(float)

# --- random intercept: (1 | state_postal) ---
re_sig = mean_term("1|state_postal_sigma")
re_sig_f = _to_float_scalar(re_sig, 0.0)

re_off = idata_pid.posterior["1|state_postal_offset"]         # dims: chain, draw, <lvl_dim>
re_off_mean = re_off.mean(dim=("chain","draw"))               # dims: <lvl_dim>

lvl_dims = [d for d in re_off_mean.dims if d not in ("chain","draw")]
if not lvl_dims:
    raise RuntimeError("Could not find a level dimension for state random intercept.")
lvl_dim = lvl_dims[0]

def _norm(v):
    try:
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    except Exception:
        return str(v)

re_map = {_norm(st): float(re_off_mean.sel({lvl_dim: st}).values)
          for st in list(re_off_mean.coords[lvl_dim].values)}

# add state RE
state_vec = raked["state_postal"].astype(str).map(re_map).fillna(0.0).to_numpy()
eta += re_sig_f * state_vec


# -------- Compute category probabilities (7 categories) --------
# Thresholds: tau0=-inf, tau1..tau6=taus_np, tau7=+inf
tau_full = np.empty(8)
tau_full[0]   = -np.inf
tau_full[1:7] = taus_np
tau_full[7]   =  np.inf

# p_k = Phi(tau_k - eta) - Phi(tau_{k-1} - eta)
cdf_hi = norm.cdf(tau_full[1:, None] - eta[None, :])  # shape (7, N)
cdf_lo = norm.cdf(tau_full[:-1, None] - eta[None, :]) # shape (7, N)
probs = np.clip(cdf_hi - cdf_lo, 1e-12, 1.0)          # (7, N)
probs /= probs.sum(axis=0, keepdims=True)             # normalize columns
probs = probs.T  # (N, 7) so each row aligns to raked rows

# -------- Aggregate to CBG by raked weights --------
prob_cols = [f"p{k}" for k in range(1,8)]
for k in range(7):
    raked[prob_cols[k]] = probs[:, k]

def agg_cbg(g: pd.DataFrame) -> pd.Series:
    w = g["w_raked"].to_numpy()
    W = w.sum() if w.sum() > 0 else 1.0
    out = {c: float(np.dot(g[c].to_numpy(), w) / W) for c in prob_cols}
    # Expected scale: sum_k k*p_k
    pvec = np.array([out[f"p{k}"] for k in range(1,8)])
    out["pid7_mean"] = float(np.dot(np.arange(1,8), pvec))
    out["pid7_mode"] = int(1 + np.argmax(pvec))
    return pd.Series(out)

cbg_out = raked.groupby("cbg_id", sort=False).apply(agg_cbg).reset_index()

# Optional: attach RUCA or other attrs if available
try:
    ruca = pd.read_parquet("outputs/mrp/cbg_marginals.parquet")[["cbg_id","ruca"]].drop_duplicates("cbg_id")
    cbg_out = cbg_out.merge(ruca, on="cbg_id", how="left")
except Exception:
    pass
cbg_out = cbg_out.merge(raked[["cbg_id","state_postal"]].drop_duplicates("cbg_id"), on="cbg_id", how="left")

# -------- Save --------
OUTF = OUTD / "pid7_shares.parquet"
cbg_out.to_parquet(OUTF, index=False)
print(f"✓ PID7 CBG shares saved → {OUTF}")

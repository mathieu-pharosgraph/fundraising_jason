#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

MICRO = Path("outputs/mrp/microcells_pums.parquet")     # has ST + dims + n (from step 02)
MARGS = Path("outputs/mrp/cbg_marginals.parquet")       # long marginals
OUT   = Path("outputs/mrp/cbg_raked_cells.parquet")

# -------------------- knobs (tune here) --------------------
BATCH_SIZE = 2000        # number of CBGs per write batch
MAX_ITERS  = 12          # IPF passes over all dims
TOL        = 1e-3        # early-stop tolerance on max rel change
EPS        = 1e-12       # numerical epsilon
DAMP       = 0.7         # 0<damp<=1, 1=no damping; <1 is more stable
USE_FLOAT32 = True       # speed + memory
N_PROCS    = 1           # set to min(4, cpu_count()) for parallel per-batch
# -----------------------------------------------------------

dims = ["age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]

def _prepare_state_bases(micro: pd.DataFrame):
    """
    Build and cache state-specific bases with factorized category codes per dim.
    Returns: dict[state_fips -> dict(base=<df>, codes=<dict dim->code array>, n_cats=<dict>) ]
    """
    bases = {}
    for st, g in micro.groupby("ST"):
        base = g[dims + ["n"]].copy()
        base["w"] = base["n"].astype(np.float32 if USE_FLOAT32 else np.float64)
        base.drop(columns=["n"], inplace=True)

        # factorize categories to int codes per dim
        codes = {}
        n_cats = {}
        for d in dims:
            code, uniques = pd.factorize(base[d], sort=False)
            base[d] = code  # store codes to avoid repeated mapping
            codes[d] = np.asarray(code, dtype=np.int32)
            n_cats[d] = int(code.max()) + 1
        # Keep as numpy for speed
        bases[st] = {"base": base.reset_index(drop=True), "n_cats": n_cats}
    return bases

def _targets_for_cbg(tg_cbg: pd.DataFrame, n_cats_map: dict, cats_order: dict):
    """
    Convert long targets (dim, category, n) to aligned vectors per dim
    (length = number of categories in the state's base for that dim).
    Unknown categories -> 0; missing categories filled with 0.
    cats_order maps dim -> dict(category_string -> code_int) for that state base.
    """
    per_dim = {}
    for d in dims:
        vec = np.zeros(n_cats_map[d], dtype=np.float32 if USE_FLOAT32 else np.float64)
        sub = tg_cbg[tg_cbg["dim"] == d]
        if len(sub):
            order = cats_order[d]
            for _, row in sub.iterrows():
                cat = row["category"]
                if cat in order:
                    vec[order[cat]] = float(row["n"] or 0.0)
                # else: category not in base -> ignored (0)
        per_dim[d] = vec
    return per_dim

def _build_cats_order(state_base: pd.DataFrame):
    """
    Build category->code dict per dim using the state's base (which already stores codes).
    Need the mapping from string category to integer code.
    We'll reconstruct mapping by reading uniques from original categories per dim.
    """
    mapping = {}
    # We lost original strings in base; rebuild from codes by carrying an example row.
    # Better: keep uniques at prepare time. We'll do that: infer from a re-factorization.
    # To reconstruct reliably, we refactorize on a Series cast from codes; this loses strings.
    # So instead, we pass in an additional tiny frame with the original uniques per dim.
    # Simpler: we’ll store uniques alongside base at prepare time. (Implement now.)
    raise RuntimeError("cats_order should be supplied from prepare step.")

# ---------- fast IPF core using bincount ----------
def _ipf_fast(base_codes: dict, n_cats: dict, targets: dict, w0: np.ndarray):
    """
    Do IPF on a state's base (codes per dim) for one CBG:
      - base_codes: dim -> int codes (len = n_rows)
      - n_cats:     dim -> number of categories
      - targets:    dim -> target vector length n_cats[dim]
      - w0: initial weights (np array)
    Returns new weights.
    """
    w = w0.copy()
    if USE_FLOAT32:
        w = w.astype(np.float32, copy=False)

    for _ in range(MAX_ITERS):
        max_rel = 0.0
        for d in dims:
            codes_d = base_codes[d]
            K = n_cats[d]
            cur = np.bincount(codes_d, weights=w, minlength=K)
            tgt = targets[d]
            # guard: if all-zero target, zero out those cats
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (tgt + EPS) / (cur + EPS)
            # track change
            rel = np.abs(ratio - 1.0)
            if rel.size:
                max_rel = max(max_rel, float(np.nanmax(rel)))
            # multiplicative update with damping
            mult = ratio[codes_d]
            if DAMP < 1.0:
                mult = 1.0 + DAMP * (mult - 1.0)
            w *= mult
        if max_rel < TOL:
            break

    return w

# ---------- write helper ----------
def _append_parquet(writer, df):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema, compression="snappy")
    writer.write_table(tbl)
    return writer

# ---------- batch worker (optionally parallel) ----------
def _process_batch(args):
    (batch_cbgs, targets, state_bases, cats_orders, normalize_to_rows) = args
    out_frames = []
    for cbg in batch_cbgs:
        tg = targets[targets["cbg_id"] == cbg]
        if tg.empty:
            continue
        state_fips = cbg[:2]  # first two digits
        if state_fips not in state_bases:
            # fall back to a small generic base (rare)
            # choose any state base (first key)
            state_fips = next(iter(state_bases.keys()))

        st_base = state_bases[state_fips]["base"]
        n_cats  = state_bases[state_fips]["n_cats"]
        codes   = {d: st_base[d].to_numpy(dtype=np.int32, copy=False) for d in dims}
        cats_order = cats_orders[state_fips]

        tgt_vecs = _targets_for_cbg(tg, n_cats, cats_order)

        w0 = st_base["w"].to_numpy()
        w_new = _ipf_fast(codes, n_cats, tgt_vecs, w0)

        g = st_base[dims].copy()
        g.insert(0, "cbg_id", cbg)
        g["w_raked"] = w_new

        if normalize_to_rows:
            s = float(g["w_raked"].sum())
            if s > 0:
                g["w_raked"] = g["w_raked"] / s * len(g)

        out_frames.append(g)
    if out_frames:
        return pd.concat(out_frames, ignore_index=True)
    return pd.DataFrame(columns=["cbg_id"] + dims + ["w_raked"])

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    # Load data
    micro = pd.read_parquet(MICRO)      # has ST + dims + n
    marg  = pd.read_parquet(MARGS)      # cbg_id, dim, category, n, (ruca)...

    # Ensure dtypes
    micro["ST"] = micro["ST"].astype(str).str.zfill(2)
    for d in dims:
        micro[d] = micro[d].astype("category")

    # Build per-state bases and keep uniques for category→code mapping
    # We need cat→code dicts per state for each dim
    state_bases = {}
    cats_orders = {}
    for st, g in micro.groupby("ST"):
        base = g[dims + ["n"]].copy()
        base["w"] = base["n"].astype(np.float32 if USE_FLOAT32 else np.float64)
        base.drop(columns=["n"], inplace=True)

        order = {}
        n_cats = {}
        # factorize using *consistent* order from category dtype
        for d in dims:
            cats = g[d].cat.categories.tolist()
            order[d] = {c: i for i, c in enumerate(cats)}
            code = g[d].cat.codes.to_numpy()
            base[d] = code  # store codes
            n_cats[d] = len(cats)

        state_bases[st] = {"base": base.reset_index(drop=True), "n_cats": n_cats}
        cats_orders[st] = order

    # Targets aggregated (ensure numeric)
    targets = (
        marg[["cbg_id","dim","category","n"]]
        .assign(n=lambda x: pd.to_numeric(x["n"], errors="coerce").fillna(0.0))
        .groupby(["cbg_id","dim","category"], as_index=False)["n"].sum()
    )

    cbgs = targets["cbg_id"].unique().tolist()

    # Process in batches (optionally parallel)
    writer = None
    normalize_to_rows = True

    if N_PROCS > 1:
        with Pool(processes=min(N_PROCS, cpu_count())) as pool:
            for i in tqdm(range(0, len(cbgs), BATCH_SIZE), desc="Raking CBGs (parallel)"):
                batch = cbgs[i:i+BATCH_SIZE]
                res = pool.map(
                    _process_batch,
                    [(batch, targets, state_bases, cats_orders, normalize_to_rows)]
                )
                batch_df = pd.concat(res, ignore_index=True) if len(res) else None
                if batch_df is not None and len(batch_df):
                    writer = _append_parquet(writer, batch_df)
    else:
        for i in tqdm(range(0, len(cbgs), BATCH_SIZE), desc="Raking CBGs"):
            batch = cbgs[i:i+BATCH_SIZE]
            batch_df = _process_batch((batch, targets, state_bases, cats_orders, normalize_to_rows))
            if len(batch_df):
                writer = _append_parquet(writer, batch_df)

    if writer is not None:
        writer.close()

    meta = pq.ParquetFile(OUT)
    print("✓ raked ->", OUT, "row_groups:", meta.num_row_groups)

if __name__ == "__main__":
    main()

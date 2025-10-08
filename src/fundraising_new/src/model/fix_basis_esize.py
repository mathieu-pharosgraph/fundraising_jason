#!/usr/bin/env python3
import sys, numpy as np, pandas as pd
path = sys.argv[1] if len(sys.argv)>1 else "data/donors/feature_basis.parquet"
B = pd.read_parquet(path)

# coalesce e_size variants into a single 'e_size'
cands = [c for c in B.columns if c.startswith("e_size")]
if cands:
    es = None
    # prefer canonical names in this order
    for name in ("e_size","e_size_y","e_size_x"):
        if name in B.columns:
            es = B[name].astype(float)
            break
    if es is None:
        es = pd.Series(0.0, index=B.index)

    # normalize L1 if there is signal
    s = float(np.nansum(np.abs(es.values)))
    if s > 0:
        es = es.fillna(0.0) / s

    # drop extras, keep only 'e_size'
    drop = [c for c in cands if c != "e_size"]
    B = B.drop(columns=drop, errors="ignore")
    B["e_size"] = es.fillna(0.0)

B.to_parquet(path, index=False)
print(f"✓ cleaned e_size columns; saved {path}")

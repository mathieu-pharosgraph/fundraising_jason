#!/usr/bin/env python3
import argparse, joblib, json, re
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def norm_key(c:str)->str:
    c = re.sub(r"\s+","_", str(c).strip().lower())
    return re.sub(r"[^a-z0-9_]+","", c)

def _get_pre_and_clf(pipe: Pipeline):
    # works for Pipeline([("pre", ...), ("clf", ...)])
    pre = pipe.named_steps.get("pre")
    clf = pipe.named_steps.get("clf")
    if pre is None or clf is None:
        raise ValueError("Expected pipeline with steps 'pre' and 'clf'.")
    return pre, clf

def _column_groups(pre: ColumnTransformer, raw_cols: list[str]) -> dict:
    """
    Return mapping:
      - 'num': (raw_numeric_cols, scaler)
      - 'cat': (raw_categorical_cols, ohe)
    """
    num_cols, cat_cols, scaler, ohe = [], [], None, None
    for name, trans, cols in pre.transformers_:
        if trans == "drop": continue
        if hasattr(trans, "named_steps"):
            # Pipeline inside
            st = trans.named_steps
            if "ohe" in st:
                ohe = st["ohe"]; cat_cols = list(cols)
            if "sc" in st:
                scaler = st["sc"]; num_cols = list(cols)
        else:
            # Could be passthrough; ignore
            pass
    return {"num": (num_cols, scaler), "cat": (cat_cols, ohe)}

def _coef_table_from_logit(pipe: Pipeline, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-original-feature signed weights for binary logit.
    - Numeric: coef / std (back to raw scale)
    - Categorical: sum(|coef_k|)*sign(weighted_mean_of_coef_k) across levels, weighted by level prevalence
    """
    pre, clf = _get_pre_and_clf(pipe)
    groups = _column_groups(pre, sample_df.columns.tolist())

    # Pull class-1 coef for binary
    coef = np.ravel(getattr(clf, "coef_", None))
    if coef is None or coef.size == 0:
        raise ValueError("Classifier has no coef_.")
    # Expanded feature names (post-transform)
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.arange(coef.size).astype(str)

    # Build prevalence for categories from sample_df (post-imp, pre-OHE)
    num_cols, scaler = groups["num"]
    cat_cols, ohe    = groups["cat"]
    prev = {}
    # Only estimate prevalence from sample_df for cat columns that actually exist there.
    # Otherwise we'll assume uniform prevalence across levels.
    if (sample_df is not None) and (ohe is not None) and cat_cols:
        present = [c for c in cat_cols if c in sample_df.columns]
        if present:
            Xcat = sample_df[present].copy()
            for c in cat_cols:
                if c in Xcat.columns:
                    vc = Xcat[c].astype(str).value_counts(normalize=True, dropna=False)
                    prev[c] = vc.to_dict()

    rows = []

    # numeric back-transform
    if num_cols and scaler is not None:
        # StandardScaler: z = (x - mean) / scale_
        scale = getattr(scaler, "scale_", np.ones(len(num_cols)))
        # map expanded name "num__<col>" to col
        for j, col in enumerate(num_cols):
            # find position of this numeric column in names
            name = f"num__{col}"
            idx = np.where(np.array(names) == name)[0]
            if idx.size:
                w_std = coef[idx[0]]  # weight on standardized var
                w_raw = float(w_std / (scale[j] if scale[j] else 1.0))
                rows.append({"feature_key": norm_key(col), "weight": w_raw})

    # categorical aggregate: sum over levels
    if cat_cols and ohe is not None:
        ohe_cats = ohe.categories_
        offset = 0
        # find indices for categories in names: they look like "cat__<col>_<level>"
        # scikit guarantees order of categories
        for c, cats in zip(cat_cols, ohe_cats):
            # collect all coefficients for this cat's levels
            c_key = norm_key(c)
            lvl_weights = []
            lvl_prev = prev.get(c, {})
            for lvl in cats:
                name = f"cat__{c}_{lvl}"
                idx = np.where(np.array(names) == name)[0]
                if idx.size:
                    w = float(coef[idx[0]])
                    # use sample prevalence if available, else uniform 1/len(cats)
                    p = prev.get(c, {}).get(str(lvl), 1.0 / max(1, len(cats)))
                    lvl_weights.append((w, p))
            if lvl_weights:
                # signed aggregate: direction from prevalence-weighted mean, magnitude from L1
                total_l1 = sum(abs(w) for w, _ in lvl_weights)
                mean_w   = sum(w * (p if p>0 else 0.0) for w, p in lvl_weights)
                signed   = total_l1 if mean_w >= 0 else -total_l1
                rows.append({"feature_key": c_key, "weight": signed})

    return pd.DataFrame(rows).groupby("feature_key", as_index=False)["weight"].sum()

def _load_pipe(joblib_path: str) -> Pipeline:
    return joblib.load(joblib_path)

def _normalize_within_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    s = df["weight"].abs().sum()
    if s > 0:
        df["weight"] = df["weight"] / s
    df["target"] = target
    return df

def build_feature_basis(
    # adjust these paths if you saved elsewhere
    p_any_model="data/outputs/donated_any_2020/model.joblib",
    party_model="data/outputs/party_tilt/model_party_tilt.joblib",
    type_model_stageA="data/outputs/type_hier/stageA_bucket_model.joblib",
    sample_csv="data/ces/processed/ces_2020_harmonized.csv",
):
    # sample is optional; used only to estimate categorical level prevalence
    sample = None
    if sample_csv and Path(sample_csv).exists():
        try:
            sample = pd.read_csv(sample_csv, nrows=50_000, low_memory=False)
        except Exception as e:
            print(f"[info] sample_csv read failed ({e}); falling back to uniform category prevalence")
    else:
        print("[info] no sample_csv provided; using uniform category prevalence for missing cats")
    out_rows = []

    # 1) P(any) donate
    try:
        pipe_any = _load_pipe(p_any_model)
        tbl_any  = _coef_table_from_logit(pipe_any, sample)
        out_rows.append(_normalize_within_target(tbl_any.copy(), "p_any"))
    except Exception as e:
        print("[warn] p_any export:", e)

    # 2) Party tilt (Dem vs Rep)
    try:
        pipe_party = _load_pipe(party_model)
        tbl_dem    = _coef_table_from_logit(pipe_party, sample)
        dem = _normalize_within_target(tbl_dem.copy(), "p_dem_given")
        gop = dem.copy(); gop["weight"] = -gop["weight"]; gop["target"] = "p_gop_given"
        out_rows += [dem, _normalize_within_target(gop, "p_gop_given")]
    except Exception as e:
        print("[warn] party export:", e)

    # 3) Type bucket (candidate vs org) — softmax → OvR by treating class logit as one-vs-rest
    # ---- TYPE BUCKET (candidate vs org) ----
    try:
        pipeA = _load_pipe(type_model_stageA)
        preA, clfA = _get_pre_and_clf(pipeA)

        # unwrap common wrappers (harmless if not wrapped)
        def _unwrap_classifier(clf):
            for attr in ("base_estimator","estimator","classifier","model"):
                if hasattr(clf, attr):
                    inner = getattr(clf, attr)
                    return inner if hasattr(inner, "fit") else clf
            return clf

        core = _unwrap_classifier(clfA)
        coefs = getattr(core, "coef_", None)
        classes = list(getattr(core, "classes_", []))
        if coefs is None or not classes:
            raise AttributeError("Classifier has no coef_ or classes_")

        import numpy as _np
        from sklearn.pipeline import Pipeline as _Pipe

        class _FakeBin:
            def __init__(self, row):
                self.coef_ = _np.atleast_2d(row)

        def _emit_for_row(row_vec, target_name):
            fake = _FakeBin(row_vec)
            class_pipe = _Pipe([("pre", preA), ("clf", fake)])
            tbl = _coef_table_from_logit(class_pipe, sample)  # 'sample' may be None; we guarded prevalence
            out_rows.append(_normalize_within_target(tbl.copy(), target_name))

        # map outputs to the expected names
        WANT = [("candidate","p_type_candidate"), ("org","p_type_org")]

        if coefs.shape[0] == 1 and len(classes) == 2:
            # Binary case: single row corresponds to positive class classes_[1]
            pos_cls = classes[1]
            neg_cls = classes[0]
            row = coefs[0]

            # Emit for whichever of candidate/org matches the positive class
            if pos_cls == "candidate":
                _emit_for_row(row, "p_type_candidate")
                _emit_for_row(-row, "p_type_org")
            elif pos_cls == "org":
                _emit_for_row(row, "p_type_org")
                _emit_for_row(-row, "p_type_candidate")
            else:
                # unexpected labels; just emit “pos/neg” with a warning
                print(f"[warn] StageA binary classes are {classes}; mapping positive→{pos_cls}")
                _emit_for_row(row,  f"p_type_{pos_cls}")
                _emit_for_row(-row, f"p_type_{neg_cls}")

        else:
            # Multinomial (n_classes x n_features): take the appropriate row per class
            for cls_name, tgt_name in WANT:
                if cls_name in classes:
                    k = classes.index(cls_name)
                    _emit_for_row(coefs[k], tgt_name)
                else:
                    print(f"[warn] StageA classes missing '{cls_name}' → skipping {tgt_name}")

    except Exception as e:
        print("[warn] type bucket export:", repr(e))




    if not out_rows:
        raise SystemExit("No feature tables exported; check model paths.")

    basis = pd.concat(out_rows, ignore_index=True)
    # pivot wide
    wide = basis.pivot_table(index="feature_key", columns="target", values="weight", aggfunc="sum").fillna(0.0).reset_index()
    # keep strongest features across targets
    score = wide.drop(columns=["feature_key"]).abs().sum(axis=1)
    wide  = wide.loc[score.sort_values(ascending=False).index].head(40)
    wide["feature_pretty"] = wide["feature_key"].str.replace("_"," ").str.title()
    wide["transform"] = "zscore"; wide["clip_lo"] = -3.0; wide["clip_hi"] = 3.0
    out = Path("data/donors/feature_basis.parquet"); out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out, index=False)
    print("✓ wrote", out, "rows=", len(wide), "cols=", len(wide.columns))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-any-model", default="data/outputs/donated_any_2020/model.joblib")
    ap.add_argument("--party-model", default="data/outputs/party_tilt/model_party_tilt.joblib")
    ap.add_argument("--type-stageA-model", default="data/outputs/type_hier/stageA_bucket_model.joblib")
    ap.add_argument("--sample-csv", default="data/ces/processed/ces_2020_harmonized.csv")
    args = ap.parse_args()
    build_feature_basis(
        p_any_model=args.p_any_model,
        party_model=args.party_model,
        type_model_stageA=args.type_stageA_model,
        sample_csv=args.sample_csv,
    )

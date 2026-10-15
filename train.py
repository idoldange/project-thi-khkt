import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_features

ROOT = Path(__file__).resolve().parent
MAX_CATS = 200  # HistGradientBoosting gioi han 255 gia tri / cot categorical


def collapse_rare(series, keep):
    return series.where(series.isin(keep), "other").where(series.notna())


def make_pipeline(seed):
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
    ])
    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    model = HistGradientBoostingRegressor(
        categorical_features=cat_mask, learning_rate=0.06, max_iter=400,
        max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    return Pipeline([("pre", pre), ("model", model)])


def fmt_ty(v):
    return f"{v / 1e9:.2f} ty"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "vn_houses_enriched.csv"))
    ap.add_argument("--out", default=str(ROOT / "models" / "house_price.joblib"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--time-split", action="store_true", help="test = tin moi nhat theo ngay dang")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw = pd.read_csv(a.data)
    raw = raw.dropna(subset=[TARGET, "area_m2"]).reset_index(drop=True)
    X = build_features(raw)
    y = np.log1p(raw[TARGET].astype(float))
    print(f"Du lieu: {len(raw)} dong tu {a.data}")
    if len(raw) < 50:
        print("  (!) It du lieu - hay crawl them truoc khi tin vao ket qua.")

    # Gom dia danh hiem de tranh vuot gioi han categorical
    for c in CATEGORICAL:
        keep = set(X[c].value_counts().head(MAX_CATS).index)
        X[c] = collapse_rare(X[c], keep)

    if a.time_split and X["year"].notna().any():
        order = pd.to_datetime(raw["date"], errors="coerce").fillna(pd.Timestamp("1970-01-01")).argsort().values
        cut = int(len(order) * (1 - a.test_size))
        tr_idx, te_idx = order[:cut], order[cut:]
        X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
        print("Chia theo thoi gian: train = tin cu, test = tin moi.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=a.test_size, random_state=a.seed)

    pipe = make_pipeline(a.seed)
    pipe.fit(X_tr, y_tr)

    pred_log = pipe.predict(X_te)
    pred, true = np.expm1(pred_log), np.expm1(y_te.values)
    ape = np.abs(pred - true) / true
    #gia/m2 trung vi cua tap train x dien tich
    ppm2 = np.median(np.expm1(y_tr.values) / X_tr["area_m2"].values)
    base = ppm2 * X_te["area_m2"].values
    base_ape = np.abs(base - true) / true

    print("\n=== Ket qua tren tap test ===")
    print(f"  So mau train/test : {len(X_tr)} / {len(X_te)}")
    print(f"  MAE               : {fmt_ty(mean_absolute_error(true, pred))}")
    print(f"  Median APE        : {np.median(ape) * 100:.1f}%   (baseline gia/m2 trung vi: {np.median(base_ape) * 100:.1f}%)")
    print(f"  Mean APE          : {ape.mean() * 100:.1f}%")
    print(f"  R2 (log gia)      : {r2_score(y_te, pred_log):.3f}")

    if len(X_te) >= 30:
        imp = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=a.seed)
        print("\n  Do quan trong feature:")
        for i in np.argsort(-imp.importances_mean):
            print(f"    {FEATURES[i]:<10} {imp.importances_mean[i]:.4f}")

    resid = y_te.values - pred_log
    meta = {
        "features": FEATURES,
        "data": a.data,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "median_ape": float(np.median(ape)),
        "resid_q10": float(np.quantile(resid, 0.10)),
        "resid_q90": float(np.quantile(resid, 0.90)),
    }

    #train lai voi toan bo tap du lieu
    final = make_pipeline(a.seed)
    final.fit(X, y)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "meta": meta}, out)
    print(f"\nDa luu model -> {out}")


if __name__ == "__main__":
    main()

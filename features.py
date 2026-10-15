"""Chuan hoa dia danh + tao feature dung chung cho train.py va infer.py."""
import re
import unicodedata

import numpy as np
import pandas as pd

BASE_NUMERIC = ["area_m2", "bedrooms", "toilets", "floors", "year", "month"]
BASE_CATEGORICAL = ["province", "district"]

# Feature backfill tu enrich_dataset.py (thieu cot nao thi de NaN - model xu ly duoc)
EXTRA_NUMERIC = [
    "frontage", "car_access", "planning_issue", "dist_center_km",
    "nha_no_hau", "chua_hoan_cong", "nha_nat",
    "amen_school_1km", "amen_hospital_1km", "amen_market_1km", "amen_supermarket_500m",
    "amen_park_1km", "amen_bus_stop_500m", "amen_metro_nearest_m", "amen_bank_1km",
    "text_near_school", "text_near_market", "text_near_hospital", "text_near_park", "text_near_transport",
]
EXTRA_CATEGORICAL = ["area_type", "land_type", "legal_status"]

NUMERIC = BASE_NUMERIC + EXTRA_NUMERIC
CATEGORICAL = BASE_CATEGORICAL + EXTRA_CATEGORICAL
FEATURES = NUMERIC + CATEGORICAL
TARGET = "price"

PROVINCE_ALIAS = {
    "tphcm": "ho chi minh", "tp hcm": "ho chi minh", "hcm": "ho chi minh", "sai gon": "ho chi minh",
    "tp ho chi minh": "ho chi minh", "hn": "ha noi", "tp ha noi": "ha noi", "hue": "thua thien hue",
}
PREFIX = re.compile(r"^(thanh pho|tp\.?|tinh|quan|huyen|thi xa|thi tran|q\.?|h\.?)\s+")


def norm_place(x):
    """'Quận Gò Vấp' -> 'go vap' ; 'TPHCM' -> 'ho chi minh' ; 'Quận 9 (TP. Thủ Đức)' -> '9'"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    s = PREFIX.sub("", s)
    s = PROVINCE_ALIAS.get(s, s)
    return s or np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nhan DataFrame tho (cot nhu vn_houses_enriched.csv) -> DataFrame feature."""
    out = pd.DataFrame(index=df.index)
    for c in ("area_m2", "bedrooms", "toilets", "floors"):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    date = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.Series(pd.NaT, index=df.index)
    out["year"] = date.dt.year
    out["month"] = date.dt.month
    for c in BASE_CATEGORICAL:
        out[c] = (df[c].map(norm_place) if c in df else pd.Series(np.nan, index=df.index)).astype(object)
    for c in EXTRA_NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in EXTRA_CATEGORICAL:
        out[c] = (df[c] if c in df else pd.Series(np.nan, index=df.index)).astype(object)
    return out[FEATURES]

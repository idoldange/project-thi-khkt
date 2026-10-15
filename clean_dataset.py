import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from features import norm_place

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"


def strip_accents(s):
    s = str(s).lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def title_key(t):
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(t)).strip()


COMMERCIAL_RE = re.compile(
    r"\b(toa nha|building|cao oc|khach san|hotel|van phong|can ho dich vu|chdv|nha tro|day tro|"
    r"phong tro|kho xuong|nha xuong|mat bang|tru so|resort|homestay|khu nghi duong|nghi duong)\b")
LAND_RE = re.compile(r"\b(ban dat|dat nen|dat tho cu|lo dat|nen dat|ban lo|quy dat)\b")
HOUSE_RE = re.compile(r"\bnha\b")
FLOORS_RE = re.compile(r"(\d{1,2})\s*(?:tang|lau|tam)\b")
BEDROOMS_RE = re.compile(r"(\d{1,2})\s*(?:pn|phong ngu)\b")


def floors_from_title(k):
    m = FLOORS_RE.search(k)
    if not m:
        return np.nan
    n = int(m.group(1))
    if "tret" in k and re.search(r"\d\s*lau\b", k):  # "1 tret 2 lau" = 3 tang
        n += 1
    return float(n) if 1 <= n <= 12 else np.nan


def bedrooms_from_title(k):
    m = BEDROOMS_RE.search(k)
    return float(m.group(1)) if m and 1 <= int(m.group(1)) <= 12 else np.nan


def regional_outliers(df, z_thr):
    lp = np.log(df["price_per_m2"])
    keys_d = [df["province_norm"], df["district_norm"]]
    keys_p = [df["province_norm"]]

    def stats(keys):
        med = lp.groupby(keys).transform("median")
        mad = (lp - med).abs().groupby(keys).transform("median")
        n = lp.groupby(keys).transform("size")
        return med, mad, n

    med_d, mad_d, n_d = stats(keys_d)
    med_p, mad_p, n_p = stats(keys_p)
    use_d = n_d >= 30
    med = med_d.where(use_d, med_p).fillna(lp.median())
    mad = mad_d.where(use_d, mad_p).fillna((lp - lp.median()).abs().median())
    mad = mad.clip(lower=0.15)
    z = 0.6745 * (lp - med) / mad
    return z.abs() > z_thr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default=str(PROC / "vn_all_sources.csv"))
    ap.add_argument("--out", default=str(PROC / "vn_houses_clean.csv"))
    ap.add_argument("--keep-commercial", action="store_true")
    ap.add_argument("--z", type=float, default=3.5, help="nguong robust z-score cho gia/m2")
    a = ap.parse_args()

    df = pd.read_csv(a.inp, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n0 = len(df)
    print(f"Doc {n0} dong tu {a.inp}")
    excluded = []

    def drop(mask, reason):
        nonlocal df
        mask = mask.fillna(False)
        if mask.any():
            excluded.append(df[mask].assign(reason=reason))
        print(f"  - {reason:<28} -{int(mask.sum()):>6}  (con {int((~mask).sum()):>6})")
        df = df[~mask].copy()

    # 1. Chuan hoa dia danh
    df["province_norm"] = df["province"].map(norm_place)
    df["district_norm"] = df["district"].map(norm_place)
    drop(df["province_norm"].isna() | df["district_norm"].isna(), "thieu tinh/quan")

    # 1b. Dong bo tinh theo quan: quan nao >=90% tin thuoc 1 tinh thi gan ve tinh do
    #     (loai bo lech giua ten tinh cu/moi sau sap nhap hanh chinh 2025)
    dp = df.groupby(["district", "province_norm"]).size().rename("n").reset_index()
    tot = dp.groupby("district")["n"].transform("sum")
    dom = dp[(dp["n"] / tot >= 0.9) & (tot >= 20)].set_index("district")["province_norm"]
    new_prov = df["district"].map(dom)
    chg = new_prov.notna() & (new_prov != df["province_norm"])
    print(f"  ~ dong bo tinh theo quan: {int(chg.sum())} dong")
    df.loc[chg, "province_norm"] = new_prov[chg]
    df["district_key"] = df["province_norm"] + "|" + df["district_norm"]

    # 1c. Gia "gia" (123456000, 111111000...)
    pstr = df["price"].round().astype("int64").astype(str)
    drop(pstr.str.match(r"^123456") | pstr.str.match(r"^(\d){5,}0*$"), "gia gia (123456..., 111111...)")

    # 2. Gia tri phi thuc te -> NaN
    for c, lo, hi in (("bedrooms", 1, 12), ("toilets", 1, 12), ("floors", 1, 12)):
        bad = ~df[c].between(lo, hi) & df[c].notna()
        print(f"  ~ {c}: {int(bad.sum())} gia tri ngoai [{lo},{hi}] -> NaN")
        df.loc[bad, c] = np.nan
    bad_geo = ~(df["latitude"].between(8, 24) & df["longitude"].between(102, 110)) & df["latitude"].notna()
    print(f"  ~ toa do ngoai Viet Nam: {int(bad_geo.sum())} -> NaN")
    df.loc[bad_geo, ["latitude", "longitude"]] = np.nan

    # 3. Dien thieu tu tieu de
    df["title_key"] = df["title"].fillna("").map(title_key)
    f_t = df["title_key"].map(floors_from_title)
    b_t = df["title_key"].map(bedrooms_from_title)
    df["floors_from_title"] = df["floors"].isna() & f_t.notna()
    df["bedrooms_from_title"] = df["bedrooms"].isna() & b_t.notna()
    df["floors"] = df["floors"].fillna(f_t)
    df["bedrooms"] = df["bedrooms"].fillna(b_t)
    print(f"  + dien so tang tu tieu de: {int(df['floors_from_title'].sum())}, "
          f"so phong ngu: {int(df['bedrooms_from_title'].sum())}")

    # 4. Loai tin khong phai nha o thong thuong
    df["is_commercial"] = df["title_key"].str.contains(COMMERCIAL_RE)
    df["is_land_only"] = df["title_key"].str.contains(LAND_RE) & ~df["title_key"].str.contains(HOUSE_RE)
    if not a.keep_commercial:
        drop(df["is_commercial"], "toa nha/khach san/nha tro")
        drop(df["is_land_only"], "dat nen (khong co nha)")

    # 5. Dien tich nghi sai: rong x dai >> dien tich
    df["dim_ratio"] = df["width_m"] * df["length_m"] / df["area_m2"]
    drop(df["dim_ratio"] > 2.2, "dien tich nho hon rong x dai")

    # 6. Loai tin trung (uu tien dong day du thong tin, moi nhat)
    df["_complete"] = df.notna().sum(axis=1)
    df["_d"] = df["date"].fillna(pd.Timestamp("1970-01-01"))
    df = df.sort_values(["_complete", "_d"], ascending=False)
    dup_title = df.duplicated("title_key") & (df["title_key"].str.len() >= 20)
    drop(dup_title, "trung tieu de")
    key2 = ["price", "area_m2", "province_norm", "district_norm", "ward", "street", "bedrooms"]
    dup_attr = df.duplicated(key2) & df["street"].notna() & df["ward"].notna()
    drop(dup_attr, "trung gia/dien tich/duong")
    df = df.drop(columns=["_complete", "_d"]).sort_index()

    # 7. Outlier gia/m2 theo khu vuc
    df["price_per_m2"] = df["price"] / df["area_m2"]
    drop(regional_outliers(df, a.z), f"outlier gia/m2 (|z|>{a.z})")

    df = df.drop(columns=["title_key"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    if excluded:
        pd.concat(excluded).drop(columns=["title_key"], errors="ignore").to_csv(
            PROC / "vn_houses_excluded.csv", index=False, encoding="utf-8-sig")

    print(f"\nKet qua: {len(df)}/{n0} dong ({len(df) / n0:.1%}) -> {a.out}")
    print(df.groupby("source").size().to_string())
    print("\nThong ke gia (ty):")
    print((df[["price", "price_per_m2"]] / 1e9).describe().round(3).to_string())
    print("\nTop tinh:", df["province_norm"].value_counts().head(8).to_dict())
    print("Ty le thieu:", df[["bedrooms", "toilets", "floors", "ward", "street", "latitude", "date"]]
          .isna().mean().round(2).to_dict())


if __name__ == "__main__":
    main()

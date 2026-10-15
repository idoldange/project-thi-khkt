import argparse
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_features

ROOT = Path(__file__).resolve().parent


def predict(bundle, raw: pd.DataFrame) -> pd.DataFrame:
    """raw DataFrame ->+ column price_pred, price_low, price_high (VND)."""
    pipe, meta = bundle["pipeline"], bundle["meta"]
    X = build_features(raw)
    log_pred = pipe.predict(X)
    out = raw.copy()
    out["price_pred"] = np.expm1(log_pred)
    # Khoang 80% dua tren phan phoi sai so (log) cua tap test luc train
    out["price_low"] = np.expm1(log_pred + meta["resid_q10"])
    out["price_high"] = np.expm1(log_pred + meta["resid_q90"])
    if "area_m2" in out:
        out["price_per_m2_pred"] = out["price_pred"] / pd.to_numeric(out["area_m2"], errors="coerce")
    return out


def fmt_ty(v):
    return f"{v / 1e9:.2f} ty"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "house_price.joblib"))
    ap.add_argument("--area", type=float, help="dien tich (m2)")
    ap.add_argument("--bedrooms", type=float)
    ap.add_argument("--toilets", type=float)
    ap.add_argument("--floors", type=float)
    ap.add_argument("--frontage", type=float, choices=[0, 1], help="1 = mat tien/mat pho, 0 = hem/ngo")
    ap.add_argument("--car-access", type=float, choices=[0, 1], help="1 = hem xe hoi / o to vao duoc")
    ap.add_argument("--province", help='vd: "Hồ Chí Minh", "Hà Nội"')
    ap.add_argument("--district", help='vd: "Quận Gò Vấp", "Thủ Đức", "Huyện Hóc Môn"')
    ap.add_argument("--date", default=date.today().isoformat(), help="ngay dinh gia (mac dinh: hom nay)")
    ap.add_argument("--csv", help="file CSV input de du doan hang loat")
    ap.add_argument("--out", help="file CSV output (voi --csv)")
    a = ap.parse_args()

    bundle = joblib.load(a.model)

    if a.csv:
        raw = pd.read_csv(a.csv)
        if "date" not in raw:
            raw["date"] = a.date
        res = predict(bundle, raw)
        out = a.out or str(Path(a.csv).with_suffix(".pred.csv"))
        res.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Da du doan {len(res)} can -> {out}")
        return

    if a.area is None or not a.province or not a.district:
        ap.error("can --area, --province, --district (hoac dung --csv)")

    raw = pd.DataFrame([{
        "area_m2": a.area, "bedrooms": a.bedrooms, "toilets": a.toilets, "floors": a.floors,
        "frontage": a.frontage, "car_access": a.car_access,
        "province": a.province, "district": a.district, "date": a.date,
    }])
    r = predict(bundle, raw).iloc[0]
    print(f"Gia du doan : {fmt_ty(r.price_pred)}  ({r.price_pred / 1e6:,.0f} trieu)")
    print(f"Khoang 80%  : {fmt_ty(r.price_low)} - {fmt_ty(r.price_high)}")
    print(f"Gia / m2    : {r.price_per_m2_pred / 1e6:,.1f} trieu/m2")
    print(f"(sai so trung vi cua model tren tap test: ~{bundle['meta']['median_ape'] * 100:.0f}%)")


if __name__ == "__main__":
    main()

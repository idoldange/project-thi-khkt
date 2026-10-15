# code backfill
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import requests

from clean_dataset import strip_accents

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
OSM_CACHE = EXT / "osm_cache"
IN_FILE = PROC / "vn_houses_clean.csv"
OUT_FILE = PROC / "vn_houses_enriched.csv"

# ------------------------------------------------------------------ text features
HOUSE_TYPE_NAME = {1: "Nhà mặt phố, mặt tiền", 2: "Nhà biệt thự", 3: "Nhà ngõ, hẻm", 4: "Nhà phố liền kề"}
LEGAL_NAME = {1: "Đã có sổ", 2: "Đang chờ sổ", 4: "Không có sổ", 5: "Sổ chung / công chứng vi bằng",
              6: "Giấy tờ viết tay"}

FRONT_RE = re.compile(r"\b(mat tien|mat pho|mat duong|mt duong)\b")
ALLEY_RE = re.compile(r"\b(hem|ngo|kiet|hxh)\b")
NOT_FRONT_RE = re.compile(r"\b(cach|gan|ra|sat|tuc|cap|di bo ra)\s+(mat tien|mat pho|mat duong)\b|"
                          r"\b(mat tien hem|mat hem)\b")
CAR_RE = re.compile(r"\b(hem xe hoi|hxh|hem oto|oto|o to|xe hoi|hem xe tai|xe tai)\b")
THO_CU_RE = re.compile(r"\b(tho cu|full tho|100 tho|dat o lau dai)\b|100% tho")
AGRI_RE = re.compile(r"\b(chua (len|chuyen) tho|dat nong nghiep|dat vuon|cln|dat trong cay|dat lua)\b")
PLAN_RE = re.compile(r"\b(quy hoach|le gioi|giai toa)\b")
NEG_PLAN_RE = re.compile(r"\b(khong|k|ko|chua|het|no|0)\s+(dinh\s+|vuong\s+|bi\s+)?(quy hoach|le gioi|giai toa)\b")
SO_RE = re.compile(r"\b(so hong|so do|shr|so rieng|so hong rieng)\b")

AMEN_TEXT = {
    "school": "trường|mầm non|đại học|tiểu học|thcs|thpt",
    "market": "chợ|siêu thị|coopmart|aeon|vincom|big c|winmart",
    "hospital": "bệnh viện|phòng khám",
    "park": "công viên|hồ bơi|ven sông|sông",
    "transport": "metro|bến xe|xe buýt|ga |sân bay|vành đai",
}


def amen_regex(kw):
    return re.compile(r"(gần|sát|cạnh|kế|ngay|đối diện|xung quanh|cách)(?:\W+\w+){0,4}?\W+(?:" + kw + ")")


AMEN_RE = {k: amen_regex(v) for k, v in AMEN_TEXT.items()}


def load_raw_text():
    """chuyen raw jsonl thanh DataFrame (source, listing_id, text, + truong rieng cua tung nguon)."""
    rows = []
    p = RAW / "chotot_listings.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            d = r.get("detail") or {}
            chars = [c.strip() for c in (d.get("pty_characteristics") or "").split(",") if c.strip()]
            ht = r.get("house_type")
            rows.append({
                "source": "chotot", "listing_id": str(r["list_id"]),
                "text": f"{r.get('subject') or ''}. {(r.get('body') or '')[:2500]}",
                "house_type_code": ht,
                "house_type_name": d.get("house_type") or HOUSE_TYPE_NAME.get(ht),
                "is_main_street": r.get("is_main_street"),
                "chars": chars,
                "legal_status": d.get("property_legal_document") or LEGAL_NAME.get(r.get("property_legal_document")),
                "direction": d.get("direction"),
            })
    for src in ("alonhadat", "mogi"):
        p = RAW / f"{src}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"source": src, "listing_id": str(r["id"]), "text": r.get("title") or "",
                         "street_width_m": r.get("street_width_m"), "chars": []})
    return pd.DataFrame(rows)


def text_features(raw: pd.DataFrame) -> pd.DataFrame:
    low = raw["text"].fillna("").map(lambda s: unicodedata.normalize("NFC", s.lower()))
    asc = low.map(strip_accents)
    out = raw[["source", "listing_id"]].copy()

    # mat tien
    front, alley = asc.str.contains(FRONT_RE), asc.str.contains(ALLEY_RE)
    notfront = asc.str.contains(NOT_FRONT_RE)
    ft = pd.Series(np.nan, index=raw.index)
    ft[front & ~alley & ~notfront] = 1.0
    ft[(alley & ~front) | notfront | (front & alley & notfront)] = 0.0
    code = pd.to_numeric(raw.get("house_type_code"), errors="coerce")
    frontage = pd.Series(np.nan, index=raw.index)
    frontage[code == 1] = 1.0
    frontage[code == 3] = 0.0
    out["frontage"] = frontage.fillna(ft)
    out["frontage_src"] = np.where(frontage.notna(), "house_type", np.where(ft.notna(), "text", None))
    out["house_type_name"] = raw.get("house_type_name")
    out["is_main_street"] = raw.get("is_main_street")
    out["street_width_m"] = pd.to_numeric(raw.get("street_width_m"), errors="coerce")

    chars = raw["chars"]
    has = lambda c: chars.map(lambda l: c in l)
    car = has("Hẻm xe hơi") | asc.str.contains(CAR_RE)
    out["car_access"] = np.where(car | (raw["source"] == "chotot"), car.astype(float), np.nan)

    # dat / phap ly / quy hoach
    tho = asc.str.contains(THO_CU_RE)
    agri = asc.str.contains(AGRI_RE) | has("Đất chưa chuyển thổ")
    out["land_type"] = np.where(agri, "nong_nghiep_chua_len_tho", np.where(tho, "tho_cu", None))
    pos = asc.str.contains(PLAN_RE) & ~asc.str.contains(NEG_PLAN_RE)
    pl = pd.Series(np.nan, index=raw.index)
    pl[asc.str.contains(NEG_PLAN_RE)] = 0.0
    pl[pos | has("Nhà dính quy hoạch / lộ giới")] = 1.0
    out["planning_issue"] = pl
    legal = raw.get("legal_status")
    out["legal_status"] = legal.where(legal.notna(), None) if legal is not None else None
    out.loc[out["legal_status"].isna() & asc.str.contains(SO_RE), "legal_status"] = "Đã có sổ"
    out["direction"] = raw.get("direction")
    for flag, name in (("Nhà nở hậu", "nha_no_hau"), ("Nhà tóp hậu", "nha_top_hau"),
                       ("Nhà chưa hoàn công", "chua_hoan_cong"), ("Nhà nát", "nha_nat")):
        out[name] = np.where(raw["source"] == "chotot", has(flag).astype(float), np.nan)

    for k, rx in AMEN_RE.items():
        out[f"text_near_{k}"] = low.str.contains(rx).astype(float)
    return out


#khu vuc
NORTH = "hanoi haiphong quangninh bacninh haiduong hungyen thaibinh namdinh hanam ninhbinh vinhphuc phutho thainguyen bacgiang langson caobang backan tuyenquang hagiang laocai yenbai laichau dienbien sonla hoabinh".split()
CENTRAL = "thanhhoa nghean hatinh quangbinh quangtri thuathienhue hue danang quangnam quangngai binhdinh phuyen khanhhoa ninhthuan binhthuan kontum gialai daklak daknong lamdong".split()
SOUTH = "hochiminh dongnai binhduong binhphuoc tayninh bariavungtau longan tiengiang bentre travinh vinhlong dongthap angiang kiengiang cantho haugiang soctrang baclieu camau".split()
CENTERS = {"ho chi minh": (10.7725, 106.6980), "ha noi": (21.0285, 105.8542), "da nang": (16.0678, 108.2208),
           "hai phong": (20.8449, 106.6881), "can tho": (10.0333, 105.7872)}


def region_vn(p):
    if not isinstance(p, str):
        return None
    k = re.sub(r"[^a-z]", "", p)
    return "bac" if k in NORTH else "trung" if k in CENTRAL else "nam" if k in SOUTH else None


def area_type(ward, district):
    w, d = strip_accents(ward or ""), strip_accents(district or "")
    if w.startswith("phuong"):
        return "urban"
    if w.startswith("thi tran"):
        return "town"
    if w.startswith("xa"):
        return "rural"
    if d.startswith("quan") or d.startswith("thanh pho") or d.startswith("tp"):
        return "urban"
    if d.startswith("thi xa"):
        return "town"
    if d.startswith("huyen"):
        return "rural"
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 +
         np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def region_features(df):
    out = pd.DataFrame(index=df.index)
    out["area_type"] = [area_type(w if isinstance(w, str) else "", d if isinstance(d, str) else "")
                        for w, d in zip(df["ward"], df["district"])]
    out["region_vn"] = df["province_norm"].map(region_vn)
    dist = pd.Series(np.nan, index=df.index)
    for prov, (la, lo) in CENTERS.items():
        m = (df["province_norm"] == prov) & df["latitude"].notna()
        dist[m] = haversine_m(df.loc[m, "latitude"], df.loc[m, "longitude"], la, lo) / 1000
    out["dist_center_km"] = dist.where(dist < 100)
    return out


#lai
def loan_rate(df):
    p = EXT / "loan_rates.csv"
    if not p.exists():
        return pd.DataFrame(index=df.index)
    r = pd.read_csv(p, comment="#")
    r["m"] = pd.to_datetime(r["month"] + "-01")
    r = r.sort_values("m")
    left = pd.DataFrame({"m": pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()},
                        index=df.index).dropna()
    merged = pd.merge_asof(left.sort_values("m"), r[["m", "home_loan_rate_pct"]], on="m", direction="backward")
    merged.index = left.sort_values("m").index
    return merged[["home_loan_rate_pct"]].reindex(df.index)


# ------------------------------------------------------------------ OSM (Geofabrik .pbf, xu ly offline)
PBF = EXT / "vietnam.osm.pbf"
PBF_URL = "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf"
POI_FILE = EXT / "osm_pois.csv.gz"
CATS = ["school", "hospital", "market", "supermarket", "park", "bus_stop", "metro", "bank"]
TAG_FILTERS = (
    [("amenity", v) for v in ("school", "kindergarten", "college", "university", "hospital", "clinic",
                              "marketplace", "bank", "atm")]
    + [("shop", v) for v in ("supermarket", "mall", "department_store", "convenience")]
    + [("leisure", "park"), ("highway", "bus_stop"), ("railway", "station"), ("railway", "subway_entrance")]
)


def classify(tags):
    a, s = tags.get("amenity"), tags.get("shop")
    if a in ("school", "kindergarten", "college", "university"):
        return "school"
    if a in ("hospital", "clinic"):
        return "hospital"
    if a == "marketplace":
        return "market"
    if s in ("supermarket", "mall", "department_store", "convenience"):
        return "supermarket"
    if tags.get("leisure") == "park":
        return "park"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if tags.get("railway") == "subway_entrance" or (
            tags.get("railway") == "station" and tags.get("station") in ("subway", "light_rail")):
        return "metro"
    if a in ("bank", "atm"):
        return "bank"
    return None


def cmd_osm():
    """Tai vietnam.osm.pbf (Geofabrik, ~180MB) roi trich cac dia diem tien ich -> osm_pois.csv.gz."""
    import osmium
    EXT.mkdir(parents=True, exist_ok=True)
    if not PBF.exists():
        print(f"Tai {PBF_URL} ...", flush=True)
        with requests.get(PBF_URL, stream=True, timeout=60, headers={"User-Agent": "pj1-research/1.0"}) as r:
            r.raise_for_status()
            with open(PBF, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    print("Doc file pbf (1-3 phut)...", flush=True)
    fp = osmium.FileProcessor(str(PBF)).with_locations().with_filter(osmium.filter.TagFilter(*TAG_FILTERS))
    rows = []
    for o in fp:
        if o.is_node():
            lat, lon = o.lat, o.lon
        elif o.is_way():
            pts = [(n.lat, n.lon) for n in o.nodes if n.location.valid()]
            if not pts:
                continue
            lat, lon = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        else:
            continue
        cat = classify(o.tags)
        if cat:
            rows.append((cat, f"{'n' if o.is_node() else 'w'}{o.id}", lat, lon))
    pois = pd.DataFrame(rows, columns=["cat", "id", "lat", "lon"]).drop_duplicates(["cat", "id"])
    pois.to_csv(POI_FILE, index=False, compression="gzip")
    print(f"{len(pois)} dia diem -> {POI_FILE}")
    print(pois["cat"].value_counts().to_string())


def amenity_features(df):
    from sklearn.neighbors import BallTree
    cols = [f"amen_{c}_{s}" for c in CATS for s in ("500m", "1km", "nearest_m")]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if not POI_FILE.exists():
        print("  (chua co osm_pois - chay 'osm' truoc, bo qua tien ich)")
        return out
    pool = pd.read_csv(POI_FILE)
    ok = df["latitude"].notna() & df["longitude"].notna()
    pts = np.radians(df.loc[ok, ["latitude", "longitude"]].values)
    print(f"  tien ich: {len(pool)} dia diem OSM, {int(ok.sum())} tin co toa do", flush=True)
    R = 6371000.0
    for c in CATS:
        sub = pool[pool["cat"] == c]
        if sub.empty:
            continue
        tree = BallTree(np.radians(sub[["lat", "lon"]].values), metric="haversine")
        out.loc[ok, f"amen_{c}_500m"] = tree.query_radius(pts, 500 / R, count_only=True)
        out.loc[ok, f"amen_{c}_1km"] = tree.query_radius(pts, 1000 / R, count_only=True)
        out.loc[ok, f"amen_{c}_nearest_m"] = tree.query(pts, k=1)[0][:, 0] * R
    return out


# ------------------------------------------------------------------ build
def cmd_build():
    df = pd.read_csv(IN_FILE, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)
    n = len(df)
    print(f"Doc {n} dong tu {IN_FILE}")
    raw = load_raw_text()
    tf = text_features(raw)
    df = df.merge(tf.drop_duplicates(["source", "listing_id"]), on=["source", "listing_id"], how="left")
    assert len(df) == n
    df = pd.concat([df, region_features(df), loan_rate(df), amenity_features(df)], axis=1)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    new_cols = [c for c in df.columns if c in tf.columns or c.startswith(("amen_", "text_near_"))] + \
               ["area_type", "region_vn", "dist_center_km", "home_loan_rate_pct"]
    print(f"\n-> {OUT_FILE}  ({df.shape[0]} dong x {df.shape[1]} cot)")
    fill = df[[c for c in dict.fromkeys(new_cols) if c in df.columns and c not in ("source", "listing_id")]].notna().mean()
    print("Ty le co du lieu cua cot moi:")
    print(fill.round(2).to_string())
    if df["frontage"].notna().any():
        g = df.groupby("frontage")["price_per_m2"].median() / 1e6
        print("\nGia/m2 trung vi (trieu) - hem(0) vs mat tien(1):", g.round(1).to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["osm", "build", "all"])
    a = ap.parse_args()
    if a.command in ("osm", "all"):
        cmd_osm()
    if a.command in ("build", "all"):
        cmd_build()


if __name__ == "__main__":
    main()

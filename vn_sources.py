import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5",
}
session = requests.Session()
session.headers.update(HEADERS)


#helpers
def vi_num(s):
    """'24,99' -> 24.99 ; '1.200' -> 1200 ; '8,1' -> 8.1"""
    s = s.strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_price(text):
    """'5 tỷ 600 triệu' -> 5.6e9 ; '24,99 tỷ' -> 2.499e10 ; '850 triệu' -> 8.5e8 ; 'Thỏa thuận' -> None"""
    t = (text or "").lower().replace("\xa0", " ")
    total, found = 0.0, False
    m = re.search(r"([\d.,]+)\s*tỷ", t)
    if m and vi_num(m.group(1)) is not None:
        total += vi_num(m.group(1)) * 1e9
        found = True
    m = re.search(r"([\d.,]+)\s*triệu(?!/)", t)
    if m and vi_num(m.group(1)) is not None:
        total += vi_num(m.group(1)) * 1e6
        found = True
    return int(total) if found else None


def txt(node, sel):
    el = node.select_one(sel)
    return el.get_text(" ", strip=True) if el else None


def to_num(v):
    """'4,5 m' -> 4.5 ; '131,1 m2' -> 131.1 ; 4.5 -> 4.5 ; None -> None"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d.,]+", str(v))
    return vi_num(m.group()) if m else None


def parse_dims(s):
    """'8,1x25m' -> (8.1, 25.0)  kich thuoc nha: mat tien x chieu dai"""
    m = re.match(r"\s*([\d.,]+)\s*[xX×]\s*([\d.,]+)", str(s or ""))
    if not m:
        return (None, None)
    return (vi_num(m.group(1)), vi_num(m.group(2)))


def fetch(url, retries=4):
    """Tra ve HTML, hoac None neu loi. 403/404 stop"""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 404):
                print(f"  HTTP {r.status_code} cho {url} - dung.", flush=True)
                return None
            wait = 5 * (attempt + 1)
            print(f"  HTTP {r.status_code}, doi {wait}s...", flush=True)
            time.sleep(wait)
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  Loi mang: {e} - doi {wait}s", flush=True)
            time.sleep(wait)
    return None


#parsers
def parse_alonhadat(html):
    s = BeautifulSoup(html, "lxml")
    out = []
    for a in s.select("article.property-item"):
        link = a.select_one("a.link")
        if not link or not link.get("href"):
            continue
        m = re.search(r"-(\d+)\.html$", link["href"])
        if not m:
            continue
        price_el = a.select_one('[itemprop="price"]')
        price = None
        if price_el is not None:
            price = int(price_el["content"]) if price_el.get("content", "").isdigit() \
                else parse_price(price_el.get_text())
        area = txt(a, '.area [itemprop="value"]')
        floors = re.search(r"\d+", txt(a, ".floors") or "")
        width = re.search(r"[\d.,]+", txt(a, ".street-width") or "")
        old = txt(a, ".old-address span") or ""
        old_parts = [p.strip() for p in old.split(",")]
        date_el = a.select_one("time.created-date")
        out.append({
            "id": m.group(1),
            "url": "https://alonhadat.com.vn" + link["href"],
            "title": txt(a, ".property-title"),
            "date": date_el.get("datetime") if date_el else None,
            "price": price,
            "area_m2": vi_num(area) if area else None,
            "bedrooms": int(txt(a, '.bedroom [itemprop="value"]')) if (txt(a, '.bedroom [itemprop="value"]') or "").isdigit() else None,
            "floors": int(floors.group()) if floors else None,
            "street_width_m": vi_num(width.group()) if width else None,
            "dimensions": txt(a, ".size span"),
            "street": txt(a, '[itemprop="streetAddress"]'),
            "ward": txt(a, '[itemprop="addressLocality"]'),
            "province": txt(a, '[itemprop="addressRegion"]'),
            "district_old": old_parts[-2] if len(old_parts) >= 2 else None,
            "address_old": old,
        })
    return out


def parse_mogi(html):
    s = BeautifulSoup(html, "lxml")
    out = []
    for li in s.select("ul.props > li"):
        info = li.select_one(".prop-info")
        link = li.select_one("a.link-overlay")
        if not info or not link:
            continue
        m = re.search(r"id(\d+)$", link["href"])
        if not m:
            continue
        area = bedrooms = toilets = None
        for attr in li.select(".prop-attr li"):
            t = attr.get_text(" ", strip=True)
            if re.search(r"m\s*2?$", t) or "m²" in t:
                n = re.search(r"[\d.,]+", t)
                area = vi_num(n.group()) if n else None
            elif t.upper().endswith("PN"):
                bedrooms = int(re.search(r"\d+", t).group()) if re.search(r"\d+", t) else None
            elif t.upper().endswith("WC"):
                toilets = int(re.search(r"\d+", t).group()) if re.search(r"\d+", t) else None
        addr = txt(li, ".prop-addr") or ""
        parts = [p.strip() for p in addr.split(",")]
        img = li.select_one(".prop-img img")
        img_src = " ".join(filter(None, [img.get("src"), img.get("data-src"), img.get("srcset")])) if img else ""
        d = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", img_src)
        out.append({
            "id": m.group(1),
            "url": link["href"],
            "title": txt(li, ".prop-title"),
            "date": f"{d.group(1)}-{d.group(2)}-{d.group(3)}" if d else None,  # xap xi: ngay upload anh
            "created_text": txt(li, ".prop-created"),
            "price": parse_price(txt(li, ".price")),
            "price_text": txt(li, ".price"),
            "area_m2": area,
            "bedrooms": bedrooms,
            "toilets": toilets,
            "district": parts[0] if parts else None,
            "province": parts[-1] if len(parts) > 1 else None,
        })
    return out


SOURCES = {
    "alonhadat": {
        "url": lambda p: "https://alonhadat.com.vn/can-ban-nha" + ("" if p == 1 else f"/trang-{p}"),
        "parse": parse_alonhadat,
    },
    "mogi": {
        "url": lambda p: "https://mogi.vn/mua-nha" + ("" if p == 1 else f"?cp={p}"),
        "parse": parse_mogi,
    },
}


#crawl
def crawl_site(name, max_pages, max_ads, delay, start_page=1):
    cfg = SOURCES[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"{name}.jsonl"
    seen = set()
    if raw.exists():
        with open(raw, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    print(f"[{name}] da co {len(seen)} tin. Crawl trang {start_page}..{start_page + max_pages - 1}, "
          f"toi da {max_ads} tin moi.", flush=True)
    crawled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = 0
    with open(raw, "a", encoding="utf-8") as out:
        for page in range(start_page, start_page + max_pages):
            html = fetch(cfg["url"](page))
            if html is None:
                break
            items = cfg["parse"](html)
            if not items:
                print(f"[{name}] trang {page} khong co tin - dung.", flush=True)
                break
            for it in items:
                if it["id"] in seen:
                    continue
                it["crawled_at"] = crawled_at
                out.write(json.dumps(it, ensure_ascii=False) + "\n")
                seen.add(it["id"])
                new += 1
            out.flush()
            if page % 5 == 0 or page == start_page:
                print(f"[{name}] trang {page}: +{new} tin moi", flush=True)
            if new >= max_ads:
                break
            time.sleep(delay + random.uniform(0, 0.7))
    print(f"[{name}] xong: +{new} tin moi, tong {len(seen)} -> {raw}", flush=True)


#
def merge_all():
    """Gop dlieu tu cac trang"""
    frames = []

    ct = OUT_DIR / "chotot_houses.csv"
    if ct.exists():
        d = pd.read_csv(ct)
        frames.append(pd.DataFrame({
            "source": "chotot", "listing_id": d["list_id"].astype(str), "title": d.get("subject"),
            "date": pd.to_datetime(d.get("list_date"), errors="coerce"),
            "price": d["price"], "area_m2": d["size"], "bedrooms": d.get("rooms"),
            "toilets": d.get("toilets"), "floors": d.get("floors"),
            "province": d.get("region_name"), "district": d.get("area_name"),
            "ward": d.get("ward_name"), "street": d.get("street_name"),
            "latitude": d.get("latitude"), "longitude": d.get("longitude"),
            "width_m": d.get("detail_width"), "length_m": d.get("detail_length"),
            "road_width_m": None,
            "living_size": d.get("detail_living_size"),
            "direction": d.get("detail_direction"),
            "furnishing": d.get("detail_furnishing_sell"),
            "characteristics": d.get("detail_pty_characteristics"),
        }))

    for name in SOURCES:
        raw = RAW_DIR / f"{name}.jsonl"
        if not raw.exists():
            continue
        rows = []
        with open(raw, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        g = lambda c: d[c] if c in d.columns else None
        district = g("district")
        if district is None:
            district = g("district_old")
        dims = g("dimensions")
        pdims = dims.map(parse_dims) if dims is not None else None
        frames.append(pd.DataFrame({
            "source": name, "listing_id": d["id"].astype(str), "title": g("title"),
            "date": pd.to_datetime(g("date"), errors="coerce"),
            "price": g("price"), "area_m2": g("area_m2"), "bedrooms": g("bedrooms"),
            "toilets": g("toilets"), "floors": g("floors"),
            "province": g("province"), "district": district,
            "ward": g("ward"), "street": g("street"),
            "width_m": pdims.map(lambda t: t[0]) if pdims is not None else None,
            "length_m": pdims.map(lambda t: t[1]) if pdims is not None else None,
            "road_width_m": g("street_width_m"),
            "living_size": None,
            "direction": None, "furnishing": None, "characteristics": None,
        }))

    if not frames:
        print("Khong co du lieu de gop.")
        return
    df = pd.concat(frames, ignore_index=True)
    for c in ("price", "area_m2", "bedrooms", "toilets", "floors"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("width_m", "length_m", "living_size", "road_width_m"):
        df[c] = df[c].map(to_num).astype(float)
    df.loc[~df["width_m"].between(0.5, 50), "width_m"] = None      # mat tien nha
    df.loc[~df["length_m"].between(1, 200), "length_m"] = None
    df.loc[~df["road_width_m"].between(0.5, 60), "road_width_m"] = None
    df.loc[~df["living_size"].between(5, 3000), "living_size"] = None
    df = df.dropna(subset=["price", "area_m2"])
    df = df[(df["price"] >= 1e8) & df["area_m2"].between(5, 2000)]
    df["price_per_m2"] = df["price"] / df["area_m2"]
    df = df[df["price_per_m2"].between(1e6, 1.5e9)]
    df = df.drop_duplicates(subset=["source", "listing_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "vn_all_sources.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(df)} dong -> {path}")
    print(df.groupby("source").agg(tin=("price", "size"), gia_trung_vi=("price", "median"),
                                   ppm2_trung_vi=("price_per_m2", "median")).round(0).to_string())

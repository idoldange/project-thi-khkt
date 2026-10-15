#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import vn_sources

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "chotot_listings.jsonl"

API = "https://gateway.chotot.com/v1/public/ad-listing"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}
CATEGORY_HOUSE = 1020      # Nha o, dell phai toa nha
PAGE_SIZE = 50
RESULT_CAP = 10000         # theo t nho la 10k
PRICE_EDGES = [0, 5e8, 1e9, 1.5e9, 2e9, 3e9, 4e9, 5e9, 7e9, 1e10, 2e10, 5e10, 1e12]
DROP_KEYS = {"images", "image", "image_thumbnails", "thumbnail_image", "webp_image", "avatar",
             "videos", "special_display_images", "inspection_images", "seller_info", "pty_map",
             "pty_map_v2", "ad_features", "ad_labels", "cta_buttons", "label_campaigns",
             "feature_params", "fee_type", "business_days"}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (attempt + 1)
                print(f"  HTTP {r.status_code}, wait {wait}s r retry...", flush=True)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code} cho {url} - skip", flush=True)
            return None
        except (requests.RequestException, ValueError) as e:
            wait = 5 * (attempt + 1)
            print(f"  VNPT-ed: {e} - wait {wait}s", flush=True)
            time.sleep(wait)
    return None


def load_seen():
    seen = set()
    if RAW_FILE.exists():
        with open(RAW_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["list_id"])
                except (ValueError, KeyError):
                    pass
    return seen


def crawl_chotot(max_ads, delay, details):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    print(f"Da co {len(seen)} tin. Muc tieu + max {max_ads} tin.", flush=True)
    new_count = 0
    crawled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Khoang gia 
    stack = [(int(PRICE_EDGES[i]), int(PRICE_EDGES[i + 1])) for i in range(len(PRICE_EDGES) - 1)][::-1]

    with open(RAW_FILE, "a", encoding="utf-8") as out:
        while stack and new_count < max_ads:
            lo, hi = stack.pop()
            params = {"cg": CATEGORY_HOUSE, "st": "s", "limit": PAGE_SIZE, "o": 0,
                      "price": f"{lo}-{hi}"}
            first = get_json(API, params)
            time.sleep(delay)
            if not first:
                continue
            total = first.get("total", 0)
            if total >= RESULT_CAP and hi - lo > 1:
                mid = (lo + hi) // 2
                stack.append((mid, hi))
                stack.append((lo, mid))
                continue
            print(f"Khoang gia {lo:,} - {hi:,}: {total} tin", flush=True)

            offset = 0
            page = first
            while page and new_count < max_ads:
                ads = page.get("ads") or []
                if not ads:
                    break
                for ad in ads:
                    lid = ad.get("list_id")
                    if lid is None or lid in seen:
                        continue
                    rec = {k: v for k, v in ad.items() if k not in DROP_KEYS}
                    rec["crawled_at"] = crawled_at
                    if details:
                        d = get_json(f"{API}/{lid}")
                        time.sleep(delay)
                        if d and d.get("ad_params"):
                            rec["detail"] = {k: v.get("value") for k, v in d["ad_params"].items()}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(lid)
                    new_count += 1
                    if new_count >= max_ads:
                        break
                out.flush()
                offset += PAGE_SIZE
                if offset >= min(total, RESULT_CAP):
                    break
                print(f"  ... {new_count} tin moi (offset {offset}/{total})", flush=True)
                time.sleep(delay + random.uniform(0, 0.5))
                params["o"] = offset
                page = get_json(API, params)
    print(f"Done. Them {new_count} tin moi, tong {len(seen)} tin -> {RAW_FILE}", flush=True)


def backfill_details(delay, max_ads):
    
    if not RAW_FILE.exists():
        sys.exit("Chua co du lieu raw. Chay 'chotot' truoc.")
    with open(RAW_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    recs, idxs = [], []
    total = 0
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        total += 1
        if r.get("list_id") and not r.get("detail"):
            recs.append(r)
            idxs.append(i)
    if max_ads:
        recs, idxs = recs[:max_ads], idxs[:max_ads]
    print(f"{total} tin, {len(recs)} tin chua co detail can bo sung.", flush=True)

    def checkpoint():
        tmp = RAW_FILE.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as out:
            out.writelines(lines)
        tmp.replace(RAW_FILE)

    done = 0
    for k, r in enumerate(recs):
        d = get_json(f"{API}/{r['list_id']}")
        time.sleep(delay)
        if d and d.get("ad_params"):
            r["detail"] = {kw: v.get("value") for kw, v in d["ad_params"].items()}
            lines[idxs[k]] = json.dumps(r, ensure_ascii=False) + "\n"
            done += 1
        if done and done % 100 == 0:
            checkpoint()
            print(f"  ... {done}/{len(recs)} co detail", flush=True)
    checkpoint()
    print(f"Xong. Bo sung detail cho {done}/{len(recs)} tin -> {RAW_FILE}", flush=True)


def build_clean():
    if not RAW_FILE.exists():
        sys.exit("Chua co du lieu raw. Chay 'chotot' truoc.")
    rows = []
    with open(RAW_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    df = pd.json_normalize(rows, sep="_")
    keep = ["list_id", "subject", "price", "original_price", "price_diff", "size", "rooms", "toilets",
            "floors", "house_type", "property_legal_document", "is_main_street", "region_name",
            "area_name", "ward_name", "street_name", "latitude", "longitude",
            "list_time", "orig_list_time", "crawled_at"]
    detail_cols = [c for c in df.columns if c.startswith("detail_")]
    df = df[[c for c in keep if c in df.columns] + detail_cols].copy()

    for c in ("price", "original_price", "size", "rooms", "toilets", "floors", "latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("list_time", "orig_list_time"):
        if c in df.columns:
            df[c.replace("_time", "_date")] = pd.to_datetime(df[c], unit="ms", errors="coerce")
            df = df.drop(columns=c)

    df = df.dropna(subset=["price", "size"])
    df = df[(df["price"] >= 1e8) & (df["size"].between(5, 2000))]
    df["price_per_m2"] = df["price"] / df["size"]
    df = df[df["price_per_m2"].between(1e6, 1.5e9)]
    df = df.drop_duplicates(subset="list_id")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "chotot_houses.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(df)} dong sach -> {path}")
    print(df[["price", "size", "price_per_m2"]].describe().round(0).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["chotot", "alonhadat", "mogi", "details", "clean", "merge", "all"])
    ap.add_argument("--max-ads", type=int, default=None,
                    help="so tin moi toi da moi nguon (mac dinh 1000; details: mac dinh 100000 = tat ca)")
    ap.add_argument("--max-pages", type=int, default=50, help="so trang toi da (alonhadat/mogi, mac dinh 50)")
    ap.add_argument("--start-page", type=int, default=1, help="trang bat dau (alonhadat/mogi) - de crawl tin cu hon")
    ap.add_argument("--delay", type=float, default=1.0, help="giay nghi giua cac request (mac dinh 1.0)")
    ap.add_argument("--details", action="store_true", help="chotot: goi them API chi tiet tung tin (cham hon)")
    a = ap.parse_args()
    max_ads = a.max_ads if a.max_ads is not None else (100000 if a.command == "details" else 1000)

    if a.command in ("chotot", "all"):
        crawl_chotot(max_ads, a.delay, a.details)
    for site in ("alonhadat", "mogi"):
        if a.command in (site, "all"):
            vn_sources.crawl_site(site, a.max_pages, max_ads, max(a.delay, 1.5), a.start_page)
    if a.command == "details":
        backfill_details(max(a.delay, 1.0), max_ads)
    if a.command in ("clean", "all"):
        build_clean()
    if a.command in ("merge", "clean", "all"):
        vn_sources.merge_all()


if __name__ == "__main__":
    main()

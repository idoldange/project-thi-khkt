# HƯỚNG DẪN DỰ ÁN `pj1` — Crawl dữ liệu giá nhà để train model

> Tài liệu này viết cho người **biết sơ Python** (biết biến, vòng lặp, function, có thể chưa
> từng viết crawler). Đọc xong bạn sẽ hiểu từng dòng code trong dự án và tự làm được
> một dự án tương tự cho bất kỳ trang web nào.

---

## 0. Dự án này làm gì?

Mục tiêu: thu thập dữ liệu rao bán nhà để train model **dự đoán giá nhà**.

Mỗi "tin" (listing) gồm: giá, diện tích, số phòng, vị trí (tỉnh/quận/phường/đường),
tọa độ, ngày đăng, + các chi tiết như mặt tiền, hướng nhà, nội thất.

**Kết quả cuối cùng:** 1 file CSV `data/processed/vn_all_sources.csv` — 24.379 dòng,
mỗi dòng 1 bất động sản, đã làm sạch, sẵn sàng cho bước train model.

### Luồng dữ liệu (pipeline)

```
                          ┌──────────────────────────────┐
   INTERNET               │       RAW (nguyên trạng)     │        PROCESSED (sạch)
                          │                              │
  Chotot API ──────────►  │  chotot_listings.jsonl       │──┐
  alonhadat (HTML) ────►  │  alonhadat.jsonl             │  │   clean  ──►  chotot_houses.csv
  mogi (HTML) ─────────►  │  mogi.jsonl                  │  │
                          │                              │  └─► merge ──►  vn_all_sources.csv
                          └──────────────────────────────┘       (bảng chung 3 nguồn)
```

- **RAW**: ghi thêm (append), 1 dòng = 1 tin dạng JSON. Crawl dở dừng được, chạy lại không trùng.
- **PROCESSED**: sinh lại từ RAW bất cứ lúc nào bằng lệnh `clean` — RAW là "kho báu", CSV chỉ là phái sinh.

### 2 file code

| File | Vai trò |
|---|---|
| `crawl_house_prices.py` (263 dòng) | CLI chính + crawler Chotot (API JSON) + làm sạch + gộp |
| `vn_sources.py` (326 dòng) | Crawler 2 trang HTML (alonhadat, mogi) + hàm gộp `merge_all()` |

---

## 1. Chuẩn bị

```bash
pip install -r requirements.txt      # requests, pandas, scikit-learn, beautifulsoup4, lxml

# Cấu trúc thư mục (code tự tạo nếu thiếu, trừ file)
pj1/
├── crawl_house_prices.py      # CLI chính
├── vn_sources.py              # crawler site HTML
├── requirements.txt
├── HUONG_DAN.md               # tài liệu này
└── data/
    ├── raw/                   # JSONL nguyên trạng — KHÔNG sửa tay
    │   ├── chotot_listings.jsonl
    │   ├── alonhadat.jsonl
    │   └── mogi.jsonl
    └── processed/             # CSV đã làm sạch
        ├── chotot_houses.csv
        └── vn_all_sources.csv
```

Các lệnh hay dùng:

```bash
python crawl_house_prices.py chotot --max-ads 2000      # crawl 2000 tin Chotot
python crawl_house_prices.py alonhadat --max-pages 200  # 200 trang alonhadat
python crawl_house_prices.py mogi --max-pages 200
python crawl_house_prices.py details                    # bổ sung chi tiết cho tin thiếu
python crawl_house_prices.py clean                      # làm sạch + gộp 3 nguồn
python crawl_house_prices.py all --max-ads 2000         # chạy hết mọi bước
```

---

## 2. Kiến thức nền (đọc trước khi xem code)

### 2.1. Hai kiểu thu thập dữ liệu web

**a) API trả JSON** — máy tính nói chuyện với máy tính:

```
GET https://gateway.chotot.com/v1/public/ad-listing?cg=1020&limit=50&o=0
→ {"total": 245774, "ads": [{"list_id": 123, "price": 5600000000, ...}, ...]}
```

→ Gọi `requests.get()`, đổi `.json()` ra `dict` là xong. **ổn định, nhanh, dễ.**

**b) Trang HTML** — phải "đọc" giao diện:

```python
from bs4 import BeautifulSoup
s = BeautifulSoup(html, "lxml")
for a in s.select("article.property-item"):   # CSS selector, như trong DevTools
    print(a.select_one(".property-title").get_text())
```

→ **Dễ vỡ** khi website đổi giao diện. CSS selector học qua F12 → Elements → chuột phải → Copy → Copy selector.

**Dự án này dùng cả hai**: Chotot có API JSON công khai; alonhadat/mogi không có nên parse HTML.

### 2.2. JSONL — vì sao không dùng CSV để lưu raw?

`chotot_listings.jsonl` = mỗi dòng một JSON hoàn chỉnh, `\n` phân cách:

```json
{"list_id": 123, "price": 5600000000, "size": 72, "detail": {"width": "4,5 m"}}
{"list_id": 124, "price": 3200000000, "size": 40}
```

Lý do:
1. **Ghi thêm từng dòng** (`open(..., "a")`) — crawl 1000 tiếng không cần đọc lại file.
2. **Không cột cố định** — tin A có `detail`, tin B không → CSV ép 1 cột, JSONL thì tùy.
3. **Tự vá** — đọc từng dòng, dòng hỏng (do dừng giữa chừng) thì `try/except` bỏ qua là được.

### 2.3. Dedup + resume — nguyên tắc sống còn

Crawl 20.000 tin chắc chắn sẽ bị ngắt (mất điện, Ctrl+C, mạng). Giải pháp:

```python
def load_seen():                     # crawl_house_prices.py:80
    seen = set()
    if RAW_FILE.exists():
        for line in open(RAW_FILE, encoding="utf-8"):
            try:
                seen.add(json.loads(line)["list_id"])   # ID đã có → bỏ qua
            except (ValueError, KeyError):
                pass                                     # dòng hỏng → bỏ qua
    return seen
```

Và khi crawl: `if lid in seen: continue`.

→ **Chạy lại bao nhiêu lần cũng được, không bao giờ trùng.** Đây là pattern quan trọng
nhất của mọi crawler nghiêm túc.

### 2.4. Lịch sự với server: delay, retry, User-Agent

```python
HEADERS = {"User-Agent": "Mozilla/5.0 ... Chrome/124.0 ..."}   # giả làm trình duyệt
time.sleep(delay + random.uniform(0, 0.5))                     # nghỉ >= 1s + ngẫu nhiên
```

- **delay**: server sẽ block IP nếu bạn bắn hàng nghìn request/giây. Mặc định 1–1.5s là tử tế.
- **retry với backoff tăng dần**: lỗi tạm thời (429/5xx) → đợi `5s, 10s, 15s...` rồi thử lại.
- **403/404 → dừng ngay** (xem hàm `fetch`, vn_sources.py:86): 403 nghĩa là họ chặn mình,
  retry chỉ khiến tồi tệ thêm. **Không viết code vượt chặn (anti-bot) — đây là ranh giới đạo đức.**
- Trước khi crawl bất kỳ site nào: đọc `https://site.com/robots.txt` xem họ có cấm URL mình cần không.

### 2.5. `requests.Session` — dùng chung kết nối

```python
session = requests.Session()
session.headers.update(HEADERS)   # mọi request sau đều tự gắn User-Agent
```

Nhanh hơn việc `requests.get()` lặp lại, và header thống nhất 1 chỗ.

---

## 3. Nguồn dữ liệu & cách chúng được khai thác

### 3.1. Chotot — API JSON (crawler chính, ~90% dữ liệu)

API: `https://gateway.chotot.com/v1/public/ad-listing`

```
?cg=1020     category "Nhà ở"
&st=s        loại tin: bán
&limit=50    50 tin/trang
&o=450       offset (phân trang: 0, 50, 100, ...)
&price=0-500000000   lọc khoảng giá
```

Hai điểm thú vị của Chotot mà code phải giải quyết:

**(1) Hạn mức 10.000 kết quả.** Dù `total` báo 245.000 tin, API chỉ cho lấy ~10.000
kết quả đầu với 1 bộ lọc. >10.000 thì trang sau trả rỗng → **lấy hụt**.

**Giải pháp: chia nhỏ theo khoảng giá** (`crawl_chotot`, dòng 99–116):

```python
stack = [(0, 5e8), (5e8, 1e9), (1e9, 1.5e9), ...]   # các khoảng giá
while stack:
    lo, hi = stack.pop()
    total = query(lo, hi)["total"]
    if total >= 10000:                # khoảng này quá đông → chia đôi
        mid = (lo + hi) // 2
        stack.append((mid, hi))       # đẩy 2 nửa lại vào stack
        stack.append((lo, mid))
        continue
    crawl_toàn_bộ(lo, hi)             # thoáng thì lấy hết
```

Ví dụ chạy thật: khoảng "0–500 triệu" có 103 tin → lấy hết. Khoảng nào total ≥ 10.000
tự động chia đôi **đệ quy** (qua `stack`) cho tới khi mỗi mẻ < 10.000. Thuật toán đơn giản
nhưng giải quyết triệt để vấn đề phân trang có hạn mức.

**(2) `--details`: mỗi tin thêm 1 request riêng** `GET /ad-listing/{id}` trả về `ad_params`:
hướng nhà, mặt tiền, nội thất, chiều dài... → lưu nhúng vào record:

```python
rec["detail"] = {"width": "4,5 m", "direction": "Đông Bắc", ...}
```

**Bài học của dự án này**: gộp luôn `--details` vào lần crawl đầu → 20.000 tin × 2 request
≈ 8 tiếng. Thế nên có lệnh `details` (xem §5.3) để **crawl nhanh trước, bổ sung chi tiết sau** —
vì dedup sẽ chặn nếu muốn quay lại thêm detail.

**(3) `DROP_KEYS`** (dòng 51): API trả về map ảnh, video, seller... vài KB/tin mà model không
dùng → vớt ra khỏi record ngay khi ghi để file raw nhẹ 5–10 lần.

### 3.2. alonhadat.com.vn — HTML, may mắn có schema.org

URL trang danh sách: `/can-ban-nha`, `/can-ban-nha/trang-2`, `trang-3`...
(~20 tin/trang, 245.000 tin toàn quốc).

Điểm may: mỗi tin là `<article class="property-item" itemscope itemtype="...RealEstateListing">`
với các thẻ `itemprop="price"`, `itemprop="floorSize"`... → **giá và diện tích nằm sẵn trong
HTML đúng chuẩn, không phải parse chuỗi giá "5 tỷ 6"**.

Nhận thấy thêm:
- `time.created-date datetime="2026-10-07"` → **ngày đăng thật** (rare, rất có giá trị)
- `.street-width` → đường rộng; `.size span` "8,1x25m" → kích thước nhà (mặt tiền × dài)
- `old-address` → địa chỉ hành chính cũ, tách ra lấy quận (`district_old`)

### 3.3. mogi.vn — HTML, thiếu thốn hơn

URL: `/mua-nha`, `/mua-nha?cp=2` (`cp` = page). ~15 tin/trang.

- `ul.props > li` là 1 tin; giá dạng text `"5 tỷ 600 triệu"` → phải regex
- **Không có ngày đăng thật** → suy ra từ đường dẫn ảnh `cloud.mogi.vn/images/.../2026/01/07/...`
  (ngày upload ảnh ≈ ngày đăng, xấp xỉ — ghi rõ trong code là `# xap xi`)
- Không có mặt tiền/hướng → các cột đó để `null` ở bước merge

### 3.4. Các site KHÔNG dùng (và vì sao)

| Site | Lý do |
|---|---|
| batdongsan.com.vn, nhatot.com | **Cloudflare chặn 403** — không tự viết code vượt chặn |
| homedy.com | Render bằng JavaScript → cần trình duyệt thật (Playwright) |

**Bài học**: trước khi viết parser, cứ `requests.get()` thử + đọc `robots.txt` 5 phút.
Script từ chối sớm tiết kiệm hàng giờ code vô ích.

---

## 4. Giải thích code — `crawl_house_prices.py`

### 4.1. `get_json()` — HTTP đáng tin cậy (dòng 60)

```python
def get_json(url, params=None, retries=5):
    for attempt in range(retries):                    # tối đa 5 lần
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (attempt + 1)              # 5s, 10s, 15s... (backoff)
                time.sleep(wait); continue            # lỗi tạm → thử lại
            print(f"HTTP {r.status_code} - bo qua")   # 400/403/404 → bỏ qua ngay
            return None
        except (requests.RequestException, ValueError) as e:   # lỗi mạng / JSON hỏng
            time.sleep(5 * (attempt + 1))
    return None                                        # hết lần thử → báo thất bại
```

Nguyên tắc: **mọi return đều hợp lệ** — hàm trả `None` khi thất bại, người gọi luôn
`if not first: continue` nên crawl không bao giờ crash vì 1 request lỗi.

### 4.2. `crawl_chotot()` — vòng lặp chính (dòng 92)

Đọc từ ngoài vào trong:

1. `load_seen()` → tập ID đã có (resume).
2. Tạo `stack` các khoảng giá (xem §3.1).
3. `while stack and new_count < max_ads:` — mỗi vòng xử lý 1 khoảng giá:
   - Query thử → `total` ≥ 10.000? → chia đôi, `continue`.
   - Ngược lại: lùi trang (`offset += PAGE_SIZE`) tới khi hết hoặc đủ `max_ads`.
4. Với mỗi tin mới: lọc `DROP_KEYS`, gắn `crawled_at` (thời điểm crawl — để về sau
   biết dữ liệu cũ mới thế nào), nếu `--details` thì gọi thêm API chi tiết, rồi
   `out.write(json.dumps(rec) + "\n")`.
5. `out.flush()` **sau mỗi trang** — nếu process chết, mọi trang đã qua không mất.

Chú ý 2 dòng kín kẽ:
```python
out.flush()                                   # dòng 141 — đảm bảo dữ liệu vào file
if offset >= min(total, RESULT_CAP): break    # dòng 143 — ngừng ở hạn mức 10.000
```

### 4.3. `backfill_details()` — sửa dữ liệu khi crawl đã xong (dòng 152)

Bài toán: 20.000 tin đã crawl **không có** `detail` (vì lần đầu chạy không bật
`--details`), mà dedup chặn mọi lần crawl lại. Giải pháp — **đi ngược lại**:

```python
lines = open(RAW_FILE).readlines()          # đọc toàn bộ file vào RAM
for i, ln in enumerate(lines):
    r = json.loads(ln)
    if r.get("list_id") and not r.get("detail"):
        recs.append(r); idxs.append(i)       # nhớ CẢ VỊ TRÍ DÒNG

for k, r in enumerate(recs):                # gọi API detail từng tin
    r["detail"] = {...}
    lines[idxs[k]] = json.dumps(r) + "\n"    # ghi đè đúng dòng đó trong list
    if done % 100 == 0:
        checkpoint()                         # cứ 100 tin mới ghi ra file
```

**`checkpoint()` ghi file tạm rồi `replace()`** (dòng 177–181):

```python
tmp = RAW_FILE.with_suffix(".jsonl.tmp")
tmp.write_text(...)      # ghi hết vào .tmp
tmp.replace(RAW_FILE)    # rename nguyên tử — file gốc KHÔNG BAO GIỜ nằm giữa trạng thái hỏng
```

> **Nguyên tắc**: bao giờ cũng ghi file tạm rồi `os.replace()`/`Path.replace()` khi
> overwrite file quan trọng. Giữa 2 lệnh đó process chết thì file gốc vẫn lành lặn.

### 4.4. `build_clean()` — làm sạch (dòng 198)

```python
df = pd.json_normalize(rows, sep="_")       # {detail: {width: ...}} → cột "detail_width"

# 1. Giữ cột cần, bỏ cột thừa (giữ luôn mọi cột detail_*)
# 2. Ép kiểu:
df[c] = pd.to_numeric(df[c], errors="coerce")       # chuỗi số hỏng → NaN thay vì crash
df["list_date"] = pd.to_datetime(df["list_time"], unit="ms")   # timestamp ms → 2026-10-07

# 3. Lọc outliers (bước QUAN TRỌNG nhất cho ML):
df = df.dropna(subset=["price", "size"])                    # thiếu giá/diện tích → bỏ
df = df[(df["price"] >= 1e8) & df["size"].between(5, 2000)] # giá >= 100 triệu, 5–2000 m²
df["price_per_m2"] = df["price"] / df["size"]
df = df[df["price_per_m2"].between(1e6, 1.5e9)]             # 1 triệu – 1,5 tỷ /m²
df = df.drop_duplicates(subset="list_id")
```

Mỗi điều kiện lọc đều có lý do: tin rao "thỏa thuận" (không ghi giá), tin 0 đồng quảng cáo,
tin sai diện tích... đều lọt qua nếu không lọc → model học rác.

### 4.5. `main()` — argparse (dòng 237)

```python
ap.add_argument("command", choices=["chotot", "alonhadat", "mogi", "details", "clean", "merge", "all"])
ap.add_argument("--max-ads", type=int, default=None, ...)
max_ads = a.max_ads if a.max_ads is not None else (100000 if a.command == "details" else 1000)
```

Pattern hay: `--max-ads` **mặc định None** rồi tự resolves theo lệnh, vì "max-ads" với
lệnh `details` (bổ sung detail) phải mặc định = tất cả, còn lệnh khác = 1000.
Nếu đặt `default=1000` thì không phân biệt được "user không truyền" vs "user truyền 1000".

Lệnh `all` chạy tuần tự: chotot → alonhadat → mogi → clean → merge.

---

## 5. Giải thích code — `vn_sources.py`

### 5.1. Parser số kiểu Việt Nam — `vi_num`, `parse_price`

Đây là phần "văn hóa dữ liệu" hay gặp khi crawl web Việt Nam:

```python
def vi_num(s):
    # '24,99'  → 24.99     (dấu phẩy = thập phân)
    # '1.200'  → 1200      (dấu chấm = phân cách hàng nghìn)
    # '8,1'    → 8.1
```

```python
def parse_price(text):
    # '5 tỷ 600 triệu'  → 5_600_000_000
    # '24,99 tỷ'        → 24_990_000_000
    # 'Thỏa thuận'       → None        (không ghi giá)
    t = text.lower()
    m = re.search(r"([\d.,]+)\s*tỷ", t)        # tìm "X tỷ"
    if m: total += vi_num(m.group(1)) * 1e9
    m = re.search(r"([\d.,]+)\s*triệu(?!/)", t) # (?!/) để không khớp "triệu/m²"
    if m: total += vi_num(m.group(1)) * 1e6
```

Hàm `to_num` (dòng 68) xử lý chuỗi kèm đơn vị từ API detail: `'4,5 m' → 4.5`.
`parse_dims` (dòng 78): `'8,1x25m' → (8.1, 25.0)`.

> **Bài học**: dữ liệu thô luôn bẩn. Chuẩn bị hàm parse cho TỪNG kiểu định dạng
> xuất hiện trong dữ liệu, và test với mọi sample thật (`'5 tỷ'`, `'850 triệu'`,
> `'Thỏa thuận'`, `'24,99 tỷ'`, `'1.200 m²'`...).

### 5.2. `parse_alonhadat(html)` / `parse_mogi(html)` — 2 parser HTML

Cùng công thức chung:

```python
def parse_xxx(html):
    s = BeautifulSoup(html, "lxml")
    out = []
    for node in s.select("CSS_CHỌN_TẤT_CẢ_TIN"):        # mỗi node = 1 tin
        link = node.select_one("a.ten_link")
        if not link: continue                            # không phải tin → bỏ
        m = re.search(r"id(\d+)", link["href"])          # TRÍCH ID từ URL
        if not m: continue                               # không có ID → không dedup được → bỏ
        out.append({
            "id": m.group(1),                            # BẮT BUỘC: khóa dedup
            "price": ..., "area_m2": ...,                # các field đã chuẩn hóa sẵn
            ...
        })
    return out
```

**Thiết kế quan trọng: parse 1 trang → list dict đã chuẩn hóa**, còn việc crawl lặp/trễ
dành cho `crawl_site()`. Tách bạch = test được: chạy parse trên 1 HTML mẫu là biết parser có hỏng không.

Cách tìm CSS selector (quy trình 5 phút):
1. F12 → mở tab Elements, Ctrl+F tìm tiêu đề 1 tin rao bán quen thuộc.
2. Lên trên tìm div bao bọc đúng 1 tin (quan sát các anh em cùng lớp).
3. Chuột phải → Copy → Copy selector → dán vào code, sửa cho ngắn gọn.
4. Xác minh: selector đó trả về đúng **20 node** cho 20 tin trên trang là đúng.

Mỗi parser có vài "mưu":
- alonhadat: `int(content) if content.isdigit() else parse_price(text)` — ưu tiên giá trị
  số trong attribute `content` (chuẩn schema.org), fallback parse text.
- mogi: lặp `.prop-attr li`, phân biệt `"50 m²"` / `"4 PN"` / `"3 WC"` bằng regex kết thúc chuỗi.
- mogi: ngày từ `srcset` ảnh — chỉ là ước lượng, code ghi rõ `# xap xi`.

### 5.3. `SOURCES` — cấu hình hóa (dòng 191)

```python
SOURCES = {
    "alonhadat": {"url": lambda p: f"https://alonhadat.com.vn/can-ban-nha" + ..., "parse": parse_alonhadat},
    "mogi":      {"url": lambda p: f"https://mogi.vn/mua-nha?cp={p}",          "parse": parse_mogi},
}
```

→ `crawl_site()` viết **1 lần duy nhất**, chạy cho mọi site. Thêm site thứ 3 chỉ cần
thêm 1 entry + 1 hàm parse. **Đừng copy-paste vòng lặp crawl.**

### 5.4. `crawl_site()` — vòng lặp cho site HTML (dòng 204)

```python
for page in range(start_page, start_page + max_pages):
    html = fetch(cfg["url"](page))
    if html is None: break                      # lỗi mạng/403 → dừng hẳn
    items = cfg["parse"](html)
    if not items: break                         # trang trắng → hết phân trang → dừng
    for it in items:
        if it["id"] in seen: continue           # dedup
        out.write(json.dumps(it) + "\n")
    out.flush()
    time.sleep(delay + random.uniform(0, 0.7))  # nghỉ, thêm random để không đều như robot
```

2 điều kiện `break` thay cho số trang tối đa: **trang rỗng = hết dữ liệu** — tự dừng
thay vì bắn tiếp 200 request 404.

### 5.5. `merge_all()` — hợp nhất 3 nguồn (dòng 246)

Bài toán: 3 nguồn có schema KHÁC NHAU (chotot: `size/rooms/region_name`; mogi:
`area_m2/bedrooms/province`) → phải map về **1 schema chung**:

```python
frames.append(pd.DataFrame({
    "source": "chotot", "listing_id": ..., "price": ..., "area_m2": d["size"], ...
}))
frames.append(pd.DataFrame({ "source": name, ... g("area_m2") ... }))   # site HTML
df = pd.concat(frames, ignore_index=True)     # xếp chồng
```

`g = lambda c: d[c] if c in d.columns else None` — cột site nào không có (mogi không có
`floors`) → truyền `None` (pandas broadcast thành cột toàn None) → NaN → model tự xử lý.

Tiếp theo là **làm sạch lần cuối** (dòng 307–319): ép kiểu số, parse chuỗi `'4,5 m'`,
lọc giá trị vô lý (`width_m` > 50 m → None — phát hiện 3300m khi xem `describe()`!),
lọc lại giá/điện tích, dedup theo `(source, listing_id)`.

**Không bao giờ tin data vừa crawl**: luôn `describe()` + nhìn min/max trước khi tin.

---

## 6. Những lỗi THẬT SỰ gặp trong dự án này & cách sửa

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec` | Windows console cp1252, in tiếng Việt | `PYTHONIOENCODING=utf-8` hoặc `encoding="utf-8"` khi đọc/ghi file |
| `ValueError: The truth value of a Series is ambiguous` | Truyền Series pandas vào `str(s or "")` — `or` không biết Series True/False | `s.map(hàm)` từng phần tử thay vì gọi hàm scalar trên Series |
| Ghi file raw bị mất dòng khi kill process | Ghi trực tiếp file quan trọng | Ghi `.tmp` rồi `Path.replace()` (atomic) |
| Crawl lại không được detail cho tin cũ | Dedup bỏ qua tin đã có | Lệnh `details` backfill riêng (đọc file, tìm tin thiếu, sửa tại chỗ) |
| API trả 0 kết quả ở trang >200 | Hạn mức 10.000 kết quả/bo loc | Chia nhỏ khoảng giá (stack) |
| `width_m` max = 3300m | API trả bừa / đơn vị lạ | Lọc `between(0.5, 50)` → None |
| Giá `'5 tỷ 600 triệu'` thành NaN | `pd.to_numeric` không hiểu tiếng Việt | `parse_price()` riêng trước khi vào pandas |
| File CSV trộn kiểu dữ liệu (DtypeWarning) | Cột detail có NaN + chuỗi số | Không crash, vẫn chạy được; muốn sạch thì `dtype=str` khi đọc |

---

## 7. Checklist làm dự án tương tự (site khác)

**Bước 1 — Thăm dò (30 phút, chưa code gì):**
- [ ] `GET site.com/robots.txt` — URL mình cần có bị cấm không?
- [ ] `GET` thử trang danh sách — 200 hay 403? (403 = Cloudflare → dừng, chọn nguồn khác)
- [ ] Tìm API ẩn: F12 → tab Network → tải lại trang → xem request nào trả `XHR/Fetch JSON`
      (rất nhiều site bán hàng có API JSON — chiếm 80% các trang thương mại điện tử/rao vặt)
- [ ] Quyết: dùng API hay parse HTML?

**Bước 2 — Hiểu dữ liệu (30 phút):**
- [ ] Liệt kê field CẦN cho mục tiêu dự án (vd model giá nhà: giá, diện tích, vị trí, phòng...)
- [ ] Với mỗi field: nằm ở đâu? (attr `content`? text? attribute? trang chi tiết?)
- [ ] Tìm khóa DEDUP ổn định (id trong URL, id trong JSON...) — **bắt buộc có**
- [ ] Ghi lại 3–5 HTML/JSON mẫu thật để test parser offline

**Bước 3 — Viết parser trước, crawl sau:**
- [ ] Hàm `parse_xxx(html) -> list[dict]`, field chuẩn hóa sẵn (số là số, ngày là ISO)
- [ ] Chạy trên file mẫu, in ra, đối chiếu từng dòng
- [ ] Test edge case: "Thỏa thuận", thiếu phòng ngủ, title có `"` trong JSON...

**Bước 4 — Vòng lặp crawl (bảo thủ):**
- [ ] `load_seen()` + `if id in seen: continue` (resume)
- [ ] `sleep(delay + random)` sau MỌI request
- [ ] retry backoff cho 429/5xx, dừng cho 403/404
- [ ] ghi JSONL append + `flush()` mỗi trang
- [ ] log tiến độ (`print(..., flush=True)`)
- [ ] Chạy thử `--max-ads 50` → kiểm tra file → mới chạy quy mô lớn (chạy nền, có log)

**Bước 5 — Làm sạch & gộp:**
- [ ] Ép kiểu: `pd.to_numeric(..., errors="coerce")`, `pd.to_datetime(...)`
- [ ] Lọc outliers theo NGHIÊM CỐNG nghiệp vụ (giá > 0, diện tích hợp lý, ppm2 hợp lý)
- [ ] `describe()` + nhìn min/max từng cột
- [ ] Gộp nhiều nguồn: map schema chung trước `pd.concat`
- [ ] Dedup cuối: `drop_duplicates(subset=["source", "listing_id"])`

**Bước 6 — Train model (ngoài phạm vi tài liệu này):**
- [ ] `train_test_split`, đặc trưng hóa categorical (One-Hot/Target Encode cho quận/phường)
- [ ] Baseline đơn giản trước (LinearRegression/Ridge) rồi mới RandomForest/XGBoost
- [ ] Đánh giá MAE/RMSE trên **log(price)** nếu giá trải trên nhiều bậc
- [ ] ⚠️ Không dùng cột suy ra từ giá làm feature (`price_per_m2`, `price_m2` = leakage)

---

## 8. Tham chiếu nhanh các lệnh

```bash
# --- crawl ---
python crawl_house_prices.py chotot --max-ads 20000 --delay 1.0     # ~20 phút
python crawl_house_prices.py chotot --max-ads 50000 --details       # chậm ~20 lần (1 request/tin)
python crawl_house_prices.py alonhadat --max-pages 200 --start-page 1
python crawl_house_prices.py mogi --max-pages 200
python crawl_house_prices.py details                                # backfill detail, ~8 tiếng

# --- xử lý ---
python crawl_house_prices.py clean      # build_clean + merge_all (sinh 2 file CSV)
python crawl_house_prices.py merge      # chỉ merge

# --- chạy hết pipeline ---
python crawl_house_prices.py all --max-ads 2000 --max-pages 100

# --- theo dõi crawl chạy nền (PowerShell) ---
Get-Content crawl.log -Wait
```

Kết quả kiểm chứng cuối (10/2026):

```
24.379 dong -> data/processed/vn_all_sources.csv
source       tin    gia_trung_vi   ppm2_trung_vi
alonhadat     644      10.325.000.000    138.507.878
chotot     20.769       5.200.000.000     92.592.593
mogi        2.966      13.500.000.000    153.333.333

Cột chi tiết (chotot): width_m 75.6% | length_m 70.9% | furnishing 49.6% | direction 27%
```

---

## 9. Nguyên tắc rút ra (đọc 1 dòng mỗi cái, nhớ đời)

1. **RAW là kho báu** — CSV sinh lại được, raw thì không (mất là crawl lại 8 tiếng).
2. **Dedup theo ID + append + flush** → crawl dừng giữa chừng không bao giờ là vấn đề.
3. **Delay >= 1s, không vượt chặn** — tôn trọng robots.txt và server người ta.
4. **Retry backoff, dừng cho 403/404** — cố thêm chỉ bị block nặng hơn.
5. **Parse chuẩn hóa ngay lúc crawl** — đừng để chuỗi `'5 tỷ 6'` lẫn vào pandas.
6. **Không tin data** — luôn `describe()`, lọc outlier, đối chiếu sample thật.
7. **Ghi file quan trọng bằng `.tmp` + `replace()`** — atomic, không bao giờ hỏng dở.
8. **Cấu hình hóa** (dict `SOURCES`) thay vì copy-paste vòng lặp.
9. **Tách backfill khỏi crawl** — tính năng thêm sau không được phụ thuộc vào dedup lúc đầu.
10. **Feature suy ra từ target là leakage** — đừng đưa `price_per_m2` vào predict `price`.

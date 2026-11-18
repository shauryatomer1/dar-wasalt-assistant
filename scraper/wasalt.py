"""Polite scraper for publicly available Wasalt (wasalt.sa) content.

* Respects robots.txt (crawling is allowed for non-/search paths).
* Uses the public sitemaps to discover URLs.
* Reads the structured JSON (__NEXT_DATA__) already embedded in each public page.
* Deliberately does NOT collect agent / owner contact details (personal data).
* Rate limited and resumable.

Usage:  python scraper/wasalt.py --sale 450 --rent 150
"""
import argparse, html, json, random, re, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "wasalt"
UA = "Mozilla/5.0 (compatible; AssignmentResearchBot/1.0; educational RAG project)"
SITEMAP_STATIC = "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz"
SITEMAP_PRODUCT = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
DELAY = 1.0  # seconds between requests

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "en"})


def get(url, retries=3):
    for i in range(retries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 429):
                time.sleep(5 * (i + 1))
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def sitemap_urls(url):
    txt = get(url) or ""
    return re.findall(r"<loc>(.*?)</loc>", txt)


def next_data(page_html):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page_html, re.S)
    return json.loads(m.group(1))["props"]["pageProps"] if m else None


def strip_html(s):
    s = re.sub(r"</p>|<br\s*/?>", "\n", s or "")
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def parse_property(url, pp):
    d = pp["propertyDetailsV3"]
    info = d.get("propertyInfo", {})
    attrs = {a["key"]: a.get("value") for a in d.get("attributes", []) if "key" in a}
    extra = {a["key"]: a.get("value") for a in d.get("additionalAttributes", []) if "key" in a}
    kind = "rent" if "/property/rent/" in url else "sale"
    crumbs = [c.get("subtitle") for c in info.get("breadCrumbs", []) if c.get("subtitle")]
    price = info.get("salePrice") if kind == "sale" else info.get("expectedRent")
    return {
        "id": d.get("id"), "url": url, "listing_type": kind,
        "property_type": info.get("propertySubType"), "usage": info.get("propertyMainType"),
        "city": info.get("city"), "address": info.get("address"), "area_path": crumbs,
        "price": price or None, "currency": info.get("currencyType", "SAR"),
        "rent_frequency": info.get("rentFreq") or None,
        "bedrooms": attrs.get("noOfBedrooms"), "bathrooms": attrs.get("noOfBathrooms"),
        "built_up_area_sqm": attrs.get("builtUpArea") or attrs.get("plotArea"),
        "attributes": {k: v for k, v in {**attrs, **extra}.items() if v not in (None, "")},
        "description": strip_html(info.get("description")),
        "published_at": d.get("publishedAt"), "updated_at": d.get("updatedAt"),
    }


def scrape_static():
    out = {}
    for u in sitemap_urls(SITEMAP_STATIC):
        page = get(u)
        time.sleep(DELAY)
        if not page:
            continue
        t = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
        title = re.search(r"<title[^>]*>(.*?)</title>", page, re.S)
        out[u] = {"url": u, "title": strip_html(title.group(1)) if title else u,
                  "text": re.sub(r"\s+", " ", t).strip()}
        print("static", u, len(out[u]["text"]))
    (RAW / "static_pages.json").write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=1), "utf-8")


def scrape_properties(n_sale, n_rent, seed=7):
    urls = sitemap_urls(SITEMAP_PRODUCT)
    sale = [u for u in urls if "/property/sale/" in u]
    rent = [u for u in urls if "/property/rent/" in u]
    random.Random(seed).shuffle(sale); random.Random(seed).shuffle(rent)
    targets = sale[:n_sale] + rent[:n_rent]
    f = RAW / "properties.jsonl"
    done = set()
    if f.exists():
        done = {json.loads(l)["url"] for l in f.read_text("utf-8").splitlines() if l.strip()}
    with f.open("a", encoding="utf-8") as fh:
        for i, u in enumerate(targets):
            if u in done:
                continue
            page = get(u)
            time.sleep(DELAY)
            pp = next_data(page) if page else None
            if not pp or "propertyDetailsV3" not in pp:
                print("skip", u); continue
            try:
                fh.write(json.dumps(parse_property(u, pp), ensure_ascii=False) + "\n"); fh.flush()
            except Exception as e:
                print("parse error", u, e)
            if i % 25 == 0:
                print(f"{i}/{len(targets)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sale", type=int, default=450)
    ap.add_argument("--rent", type=int, default=150)
    ap.add_argument("--skip-static", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if not a.skip_static:
        scrape_static()
    scrape_properties(a.sale, a.rent)

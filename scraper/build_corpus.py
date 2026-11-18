"""Turn raw scraped / manually exported data into retrieval-ready chunks.

Inputs
  data/manual_darglobal/*.html   pages saved from a normal browser ("Webpage, Complete")
  data/manual_darglobal/*.txt    optional pasted text (first line: "URL: ...")
  data/raw/wasalt/*              output of scraper/wasalt.py
Output
  data/processed/chunks.jsonl    {id, source, kind, title, url, text, meta}
  data/processed/listings.json   structured Wasalt listings (used for filtered search)
"""
import html as htmllib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "data" / "manual_darglobal"
RAW = ROOT / "data" / "raw" / "wasalt"
OUT = ROOT / "data" / "processed"

CHUNK_CHARS = 1100
OVERLAP = 150


# ---------------------------------------------------------------- DarGlobal
def clean_html(raw):
    m = re.search(r"saved from url=\(\d+\)(\S+)", raw)
    url = m.group(1) if m else None
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "iframe", "form", "button", "img", "video"]):
        t.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n").split("\n")]
    return url, title, [x for x in lines if x]


def load_darglobal():
    pages = []
    for f in sorted(MANUAL.glob("*.htm*")):
        url, title, lines = clean_html(f.read_text("utf-8", errors="ignore"))
        if url and "darglobal" in url:
            pages.append({"url": url.rstrip("/") or url, "title": title, "lines": lines, "file": f.name})
    seen = {p["url"] for p in pages}
    for f in sorted(MANUAL.glob("*.txt")):
        if f.name.lower() == "readme.txt":
            continue
        txt = f.read_text("utf-8", errors="ignore").strip()
        m = re.match(r"URL:\s*(\S+)\s*\n", txt)
        if not m or m.group(1).rstrip("/") in seen:
            continue  # no URL => can't cite it; or already covered by an HTML export
        body = txt[m.end():]
        pages.append({"url": m.group(1).rstrip("/"), "title": f.stem,
                      "lines": [l.strip() for l in body.split("\n") if l.strip()], "file": f.name})
    for f in sorted(MANUAL.glob("*.pdf")):
        try:
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(f)).pages)
            # file name -> page URL, e.g. "rayana__trump-mansions.pdf" -> https://darglobal.co.uk/rayana/trump-mansions
            pages.append({"url": "https://darglobal.co.uk/" + f.stem.replace("__", "/").strip("/"),
                          "title": f.stem, "file": f.name,
                          "lines": [l.strip() for l in text.split("\n") if l.strip()]})
        except Exception as e:  # pragma: no cover
            print("pdf failed", f, e)
    # strip site-wide boilerplate: lines present on >35% of pages (menus, footers, cookie text)
    cnt = Counter(l for p in pages for l in set(p["lines"]))
    limit = max(4, int(0.6 * len(pages)))
    for p in pages:
        p["lines"] = [l for l in p["lines"] if cnt[l] <= limit and len(l) > 1]
        p["text"] = "\n".join(p["lines"])
    return [p for p in pages if len(p["text"]) > 80]


def nice_title(p):
    """Readable, specific citation title (page <title>s are often generic SEO strings)."""
    path = re.sub(r"^https?://[^/]+/?", "", p["url"]).strip("/")
    seg = path.split("/")
    if seg[0] == "projects" and len(seg) > 1:
        place = seg[1].replace("properties-in-", "").replace("-", " ").title().replace("Uae", "UAE").replace("Uk", "UK")
        kind = f" - {seg[2].replace('-', ' ')}" if len(seg) > 2 else ""
        return f"DarGlobal projects in {place}{kind}"
    if seg[0] in ("partners", "press", "insights", "one-of-one") and len(seg) > 1:
        return f"DarGlobal {seg[0].replace('-', ' ')}: " + seg[1].replace("-", " ").title()[:70]
    t = re.sub(r"\s*\|.*$", "", p["title"]).strip()
    return t or (seg[-1].replace("-", " ").title() if path else "DarGlobal home")


def mostly_ascii(t, thr=0.7):
    return sum(ch.isascii() for ch in t) / max(1, len(t)) >= thr


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP):
    """Paragraph-aware sliding window."""
    paras = [p for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + "\n" if overlap else ""
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ------------------------------------------------------------------- Wasalt
NOISE_ATTRS = {"street", "adSource", "planNumber", "landNumber", "obligations", "zipCode", "regaAdvLicDate",
               "additionalNumber", "buildingNumber", "advertiserId", "licenseNumber", "regaAdvLicNumber"}


def money(v, cur="SAR"):
    return f"{int(v):,} {cur}" if isinstance(v, (int, float)) else str(v)


def listing_text(l):
    kind = "for sale" if l["listing_type"] == "sale" else "for rent"
    bits = [f"{l.get('property_type') or 'Property'} {kind} in {l.get('address') or l.get('city')}."]
    if l.get("price"):
        per = f" per {l['rent_frequency']}" if l.get("rent_frequency") else ""
        bits.append(f"Price: {money(l['price'], l['currency'])}{per}.")
    if l.get("bedrooms") not in (None, ""):
        bits.append(f"Bedrooms: {l['bedrooms']}.")
    if l.get("bathrooms") not in (None, ""):
        bits.append(f"Bathrooms: {l['bathrooms']}.")
    if l.get("built_up_area_sqm"):
        bits.append(f"Area: {l['built_up_area_sqm']} sqm.")
    extras = {k: v for k, v in (l.get("attributes") or {}).items()
              if k not in NOISE_ATTRS and k not in ("noOfBedrooms", "builtUpArea", "noOfBathrooms")
              and isinstance(v, (str, int, float)) and str(v).isascii()}
    if extras:
        bits.append("Details: " + "; ".join(f"{k}={v}" for k, v in list(extras.items())[:8]) + ".")
    return " ".join(bits)


def market_summaries(listings):
    """Aggregate stats from the scraped sample (clearly labelled as such)."""
    out = []
    groups = defaultdict(list)
    for l in listings:
        if l.get("price"):
            groups[(l["listing_type"], l.get("city"), l.get("property_type"))].append(l)
    for (kind, city, ptype), ls in groups.items():
        if len(ls) < 3 or not city or not ptype:
            continue
        prices = sorted(x["price"] for x in ls)
        med = statistics.median(prices)
        beds = Counter(x["bedrooms"] for x in ls if x.get("bedrooms") not in (None, ""))
        out.append(
            f"Market snapshot (from {len(ls)} sampled Wasalt listings, not the full market): "
            f"{ptype} listings {'for sale' if kind == 'sale' else 'for rent'} in {city}. "
            f"Price range {money(prices[0])} to {money(prices[-1])}; median {money(med)}. "
            f"Most common bedroom counts: {', '.join(f'{b} ({c})' for b, c in beds.most_common(3)) or 'n/a'}.")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = []

    def add(source, kind, title, url, text, **meta):
        chunks.append({"id": len(chunks), "source": source, "kind": kind, "title": title,
                       "url": url, "text": text, "meta": meta})

    dar = load_darglobal()
    for p in dar:
        title = nice_title(p)
        for i, c in enumerate(chunk_text(p["text"])):
            add("DarGlobal", "page", title, p["url"], f"[{title}] {c}", part=i)
    print("DarGlobal pages:", len(dar), "chunks:", len(chunks))

    pp = ROOT / "data" / "raw" / "press" / "press.jsonl"
    npress = 0
    if pp.exists():
        seen_titles = set()
        for l in pp.read_text("utf-8").splitlines():
            a = json.loads(l)
            key = re.sub(r"\W+", "", a["title"].lower())[:60]
            text = a["text"].replace("�", "'")
            if key in seen_titles:
                continue  # same release syndicated on two newswires
            seen_titles.add(key)
            title = f"Press release: {a['title'].title() if a['title'].isupper() else a['title']} ({a['publisher']})"
            for i, c in enumerate(chunk_text(text)):
                add("DarGlobal", "press", title[:140], a["url"], f"[{title[:140]}] {c}", part=i, publisher=a["publisher"])
            npress += 1
    print("Press releases:", npress)

    n0 = len(chunks)
    sp = RAW / "static_pages.json"
    if sp.exists():
        seen = set()
        for p in json.loads(sp.read_text("utf-8")):
            sig = p["text"][:300]
            if sig in seen or len(p["text"]) < 200:
                continue
            seen.add(sig)
            for i, c in enumerate(chunk_text(p["text"].replace(". ", ".\n"))):
                add("Wasalt", "page", p["title"], p["url"], f"[{p['title']}] {c}", part=i)
    print("Wasalt static chunks:", len(chunks) - n0)

    listings = []
    lp = RAW / "properties.jsonl"
    if lp.exists():
        listings = list({(x := json.loads(l))["id"]: x for l in lp.read_text("utf-8").splitlines() if l.strip()}.values())
        for l in listings:
            title = f"{l.get('property_type') or 'Property'} {l['listing_type']} - {l.get('city')}"
            add("Wasalt", "listing", title, l["url"], listing_text(l), listing_id=l["id"])
        for s in market_summaries(listings):
            add("Wasalt", "market", "Wasalt market snapshot (sample)", "https://wasalt.sa/en", s)
    print("Wasalt listings:", len(listings), "total chunks:", len(chunks))

    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (OUT / "listings.json").write_text(json.dumps(listings, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()

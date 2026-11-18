"""Collect company-issued DarGlobal press releases republished on open newswires
(PR Newswire, Zawya "press-release"). These are public, unprotected pages and are
stored with full attribution; they supplement the official-site pages that cannot be
scraped automatically (bot protection).

Editorial news articles (MEED, Arabian Business, ...) are intentionally NOT copied.
Usage: python scraper/press.py
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "press" / "press.jsonl"
UA = "Mozilla/5.0 (compatible; AssignmentResearchBot/1.0; educational RAG project)"

URLS = [l.strip() for l in (ROOT / "scraper" / "press_urls.txt").read_text("utf-8").splitlines()
        if l.strip() and not l.startswith("#")]


def extract(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "aside", "form", "svg", "header"]):
        t.decompose()
    title = (soup.find("h1") or soup.title)
    title = title.get_text(" ", strip=True) if title else ""
    root = soup.find("article") or soup.find("section", class_=re.compile("release-body|article", re.I)) or soup.body
    paras = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in root.find_all(["p", "li", "h2", "h3"])]
    paras = [p for p in paras if len(p) > 40]
    date = ""
    m = re.search(r'(datePublished|article:published_time)"?[^>]*?["=:]+\s*"?(\d{4}-\d{2}-\d{2})', html)
    if m:
        date = m.group(2)
    return title, date, "\n".join(paras)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["url"] for l in OUT.read_text("utf-8").splitlines()} if OUT.exists() else set()
    with OUT.open("a", encoding="utf-8") as fh:
        for u in URLS:
            if u in done:
                continue
            try:
                r = requests.get(u, headers={"User-Agent": UA, "Accept-Language": "en"}, timeout=30)
            except requests.RequestException as e:
                print("ERR", u, e)
                continue
            time.sleep(1.5)
            if r.status_code != 200:
                print(r.status_code, u)
                continue
            r.encoding = "utf-8"
            title, date, text = extract(r.text)
            if len(text) < 400 or "dar global" not in text.lower():
                print("SKIP (thin/off-topic)", u, len(text))
                continue
            host = re.sub(r"^www\.", "", re.search(r"https?://([^/]+)", u).group(1))
            fh.write(json.dumps({"url": u, "title": title, "date": date, "publisher": host, "text": text},
                                ensure_ascii=False) + "\n")
            fh.flush()
            print("OK", len(text), title[:80])


if __name__ == "__main__":
    main()

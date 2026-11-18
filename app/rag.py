"""Hybrid retrieval: BM25 + dense embeddings (fused with Reciprocal Rank Fusion)
plus a structured filter over scraped Wasalt listings."""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
LISTINGS = ROOT / "data" / "processed" / "listings.json"
INDEX = ROOT / "data" / "index"
EMB_MODEL = "BAAI/bge-small-en-v1.5"

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in for and or is are was were be on at by with from as it this that what which who how do does can i you me my we our your about tell".split())


def tok(s):
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP]


class Retriever:
    def __init__(self, use_dense=True):
        self.chunks = ([json.loads(l) for l in CHUNKS.read_text("utf-8").splitlines() if l.strip()]
                       if CHUNKS.exists() else [])
        self.listings = json.loads(LISTINGS.read_text("utf-8")) if LISTINGS.exists() else []
        self.by_listing = {c["meta"].get("listing_id"): c for c in self.chunks if c["kind"] == "listing"}
        self.bm25 = BM25Okapi([tok(c["title"] + " " + c["text"]) for c in self.chunks]) if self.chunks else None
        self.cities = sorted({(l.get("city") or "").lower() for l in self.listings if l.get("city")})
        self.types = sorted({(l.get("property_type") or "").lower() for l in self.listings if l.get("property_type")})
        self.emb = None
        self.model = None
        if use_dense and self.chunks:
            self._load_dense()

    # ------------------------------------------------------------ dense
    def _load_dense(self):
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(EMB_MODEL, cache_dir=str(INDEX / "models"))
            f = INDEX / "embeddings.npy"
            if f.exists() and np.load(f).shape[0] == len(self.chunks):
                self.emb = np.load(f)
            else:
                self.emb = self.build_embeddings()
        except Exception as e:  # dense is an enhancement; BM25 alone still works
            print("dense retrieval disabled:", e)
            self.model = self.emb = None

    def build_embeddings(self):
        INDEX.mkdir(parents=True, exist_ok=True)
        vecs = np.array(list(self.model.embed([c["text"] for c in self.chunks], batch_size=32)), dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        np.save(INDEX / "embeddings.npy", vecs)
        return vecs

    # --------------------------------------------------------- search
    def _rank_bm25(self, q, mask=None):
        if self.bm25 is None:
            return []
        s = self.bm25.get_scores(tok(q))
        if mask is not None:
            s = np.where(mask, s, -1)
        return list(np.argsort(-s)[:40])

    def _rank_dense(self, q, mask=None):
        if self.emb is None:
            return []
        qv = np.array(list(self.model.query_embed([q])))[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        s = self.emb @ qv
        if mask is not None:
            s = np.where(mask, s, -1)
        return list(np.argsort(-s)[:40])

    def source_mask(self, q):
        ql = q.lower()
        wa = bool(re.search(r"wasalt|saudi real estate marketplace|listing|for rent|for sale", ql))
        dg = bool(re.search(r"dar ?global|trump|aston|lamborghini|pagani|fendi|missoni|elie|astera|aida|amaya|rayana", ql))
        if wa and not dg:
            return np.array([c["source"] == "Wasalt" for c in self.chunks])
        if dg and not wa:
            return np.array([c["source"] == "DarGlobal" for c in self.chunks])
        return None

    def search(self, q, k=6):
        mask = self.source_mask(q)
        fused = {}
        for ranking in (self._rank_bm25(q, mask), self._rank_dense(q, mask)):
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1.0 / (60 + r)
        top = sorted(fused, key=fused.get, reverse=True)
        out, per_url = [], {}
        for i in top:  # cap passages per URL so one page can't crowd out the rest
            c = self.chunks[i]
            if c["kind"] == "listing":
                continue  # listings come from the structured filter below
            if per_url.get(c["url"], 0) >= 2:
                continue
            per_url[c["url"]] = per_url.get(c["url"], 0) + 1
            out.append(c)
            if len(out) >= k:
                break
        return out

    # ------------------------------------------- structured listing filter
    def parse_filters(self, q):
        ql = q.lower().replace(",", "")
        f = {}
        if re.search(r"\b(rent|rental|renting|lease)\b", ql):
            f["listing_type"] = "rent"
        elif re.search(r"\b(buy|sale|purchase|buying|sell)\b", ql):
            f["listing_type"] = "sale"
        # user spelling -> substring of the city name used by Wasalt
        alias = {"mecca": "makkah", "makkah": "makkah", "medina": "madinah", "madinah": "madinah",
                 "dammam": "dammam", "khobar": "khobar"}
        for c in self.cities:
            names = {c} | {a for a, k in alias.items() if k in c}
            if c and any(re.search(rf"\b{re.escape(n)}\b", ql) for n in names):
                f["city"] = c
        for t in self.types:
            if t and re.search(rf"\b{re.escape(t)}s?\b", ql):
                f["property_type"] = t
        m = re.search(r"(\d+)\s*[- ]?(?:bed|bedroom|br|bhk)", ql)
        if m:
            f["bedrooms"] = int(m.group(1))

        def num(s, unit):
            v = float(s)
            return v * {"m": 1e6, "million": 1e6, "k": 1e3, "": 1}[unit or ""]

        m = re.search(r"(?:under|below|less than|up to|max|maximum|within)\s*(?:sar\s*)?([\d.]+)\s*(m|million|k)?\b", ql)
        if m:
            f["max_price"] = num(m.group(1), m.group(2))
        m = re.search(r"(?:over|above|more than|at least|min|minimum)\s*(?:sar\s*)?([\d.]+)\s*(m|million|k)?\b", ql)
        if m:
            f["min_price"] = num(m.group(1), m.group(2))
        return f

    def filter_listings(self, q, k=6):
        f = self.parse_filters(q)
        # Only trigger when the user is clearly asking about listings
        # A city alone is not listing intent ("Trump Tower Jeddah"); need a listing-type signal
        if not any(x in f for x in ("listing_type", "property_type", "bedrooms", "max_price", "min_price")):
            return f, []
        mask = self.source_mask(q)
        if mask is not None and all(c["source"] == "DarGlobal" for c, m in zip(self.chunks, mask) if m):
            return f, []  # question is about a DarGlobal brand/project, not marketplace listings
        res = []
        for l in self.listings:
            if f.get("listing_type") and l["listing_type"] != f["listing_type"]:
                continue
            if f.get("city") and (l.get("city") or "").lower() != f["city"]:
                continue
            if f.get("property_type") and (l.get("property_type") or "").lower() != f["property_type"]:
                continue
            if f.get("bedrooms") and l.get("bedrooms") != f["bedrooms"]:
                continue
            p = l.get("price")
            if (f.get("max_price") or f.get("min_price")) and not p:
                continue
            if f.get("max_price") and p > f["max_price"]:
                continue
            if f.get("min_price") and p < f["min_price"]:
                continue
            res.append(l)
        res.sort(key=lambda l: l.get("price") or 1e18)
        return f, [self.by_listing[l["id"]] for l in res[:k] if l["id"] in self.by_listing]

    def retrieve(self, q, history_q=None, k=6):
        query = q if len(tok(q)) >= 3 or not history_q else f"{history_q} {q}"
        filters, listings = self.filter_listings(query)
        passages = self.search(query, k=k if not listings else max(3, k - 2))
        seen = {c["id"] for c in listings}
        return {"filters": filters, "chunks": listings + [c for c in passages if c["id"] not in seen]}

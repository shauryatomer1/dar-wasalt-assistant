# DarGlobal & Wasalt Assistant

A retrieval-augmented chatbot that answers questions about **DarGlobal** (London-listed luxury real-estate developer) and **Wasalt** (Saudi property marketplace) **using only publicly available data collected for this project**. Every answer cites its sources; if the data does not contain the answer, the bot says so instead of guessing.

- **Live demo:** _<add the deployed URL here>_
- **LLM:** free models via [OpenRouter](https://openrouter.ai) (fallback chain, default `google/gemma-4-31b-it:free`)
- **Stack:** Python, FastAPI, BM25 + `bge-small` embeddings, Docker

## What makes it more than a plain "chat with a PDF"

| Feature | Why it matters |
|---|---|
| **Hybrid retrieval** (BM25 + dense embeddings, Reciprocal Rank Fusion) | Exact names (e.g. "DaVinci Tower") *and* paraphrased questions both work |
| **Structured listing search** | "3-bedroom apartments for sale in Riyadh under 1M" is parsed into filters (type, city, bedrooms, price) and run against the scraped listings, so results respect the numbers instead of relying on the LLM |
| **Intent guard** | Questions about a DarGlobal project never get polluted with marketplace listings |
| **Grounded answers with citations** | `[1]`-style markers link to the exact source page; chips under each answer |
| **Prompt-injection hardening** | Retrieved text is labelled as data; clients cannot inject a `system` role; input length and rate limits |
| **Graceful degradation** | If every free model is rate-limited, the app still returns the most relevant passages |
| **Model fallback chain** | Free models are flaky; the app tries several in order and tells you which one answered |
| **Offline test suite** | 12 tests for retrieval quality, filters, API validation, and privacy (no contact data in the corpus) |

## Data: local inputs and reuse

This public repository contains the application and ingestion code, **not** the collected source material. Local DarGlobal page exports, Wasalt listing records, press-release text, processed chunks, and vector indexes are excluded by `.gitignore`. This avoids redistributing site content and keeps local data out of the public repository.

Only add source material that you have permission to collect and reuse. Check the source's current terms; a public sitemap or permissive `robots.txt` is not itself a content-reuse license. Do not bypass bot challenges. Keep manually saved DarGlobal pages in `data/manual_darglobal/` and Wasalt source records in `data/raw/`. Do not collect or publish advertiser/owner contact details. After adding approved source files, run `python scraper/build_corpus.py` to create the ignored files in `data/processed/`.

The local development copy used a sample of Wasalt listings and manually exported DarGlobal pages. Those local records are not included here, so a fresh clone starts with an empty corpus. The app starts in that state and returns no-relevant-data responses until approved data is added.

| Source | Method | Volume |
|---|---|---|
| **Wasalt** (`wasalt.sa`) | Optional low-rate scraper (`scraper/wasalt.py`); use only where collection and reuse are permitted | Local sample is not included |
| **DarGlobal official site** (`darglobal.co.uk`) | Manual browser exports can be ingested by `scraper/build_corpus.py`; automated challenge bypass is not supported | Local exports are not included |
| **DarGlobal press releases** | Optional collection script for company-issued releases republished on listed newswires; verify source terms before reuse | Local release text is not included |

Privacy: agent/owner contact details on Wasalt listings are **not** collected (a test enforces this).

### Known limitations (honest list)
- **Wasalt figures are a sample** (~580 of ~60,000 listings), not the market. The bot says so when quoting statistics.
- **Individual DarGlobal project pages** were only partially available; project detail comes from country pages and press releases. Questions beyond that get "not in my data".
- Press-release figures are as published on the date of each release; the data was collected in **November 2026** and is not live.
- Free OpenRouter models can be slow or rate-limited at peak times.
- The bot gives facts, **not investment advice**.

## Architecture

```
scraper/wasalt.py ──┐                                   ┌─ BM25 ──┐
scraper/press.py  ──┼─► scraper/build_corpus.py ─► chunks ─┤          ├─ RRF ─► top passages ┐
manual DarGlobal  ──┘      (clean, de-boilerplate,        └─ dense ─┘                       │
   exports                  chunk, label sources)   listings.json ─► structured filter ──────┤
                                                                                              ▼
 browser ◄── SSE stream ◄── FastAPI (/api/chat) ◄── OpenRouter free-model chain ◄── grounded prompt
```

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add OPENROUTER_API_KEY
python scraper/build_corpus.py                        # after adding permitted source files locally
OPENROUTER_API_KEY=... uvicorn app.main:app --port 7860
pytest -q tests
```

### Docker
```bash
docker build -t dar-wasalt-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=your-key dar-wasalt-bot
# open http://localhost:7860
```
If local processed data is present when building, the image includes it and creates its embedding index. Without local data, the app still starts but has no source passages to answer from. It runs as a non-root user and has a health check (`/api/health`).

### Re-collecting the data
```bash
# Only run collection if the current source terms permit your intended use.
python scraper/wasalt.py --sale 450 --rent 150
python scraper/press.py
python scraper/build_corpus.py
```

## Deployment
The repository includes a Render blueprint (`render.yaml`) for deployment. Set `OPENROUTER_API_KEY` as a secret environment variable in the host dashboard (never commit it). Set `DENSE_RETRIEVAL=0` to disable the embedding model on very small hosts (keyword search only).

## Notes
All information shown comes from public sources and may be out of date. This is an independent demo project, not an official DarGlobal or Wasalt product. "Trump" and other brand names belong to their owners.

# DarGlobal & Wasalt Assistant

A retrieval-augmented (RAG) chatbot that answers questions about **DarGlobal** (London-listed luxury real-estate developer) and **Wasalt** (Saudi property marketplace) using only publicly available data collected for this project. Every claim in an answer carries a numbered citation that links to its source page. When the data does not contain the answer, the assistant says so instead of guessing.

**Live demo:** https://dar-wasalt-assistant-deploy.onrender.com
*(Free hosting tier: after about 15 minutes of inactivity the first request can take around a minute while the container wakes up.)*

Built for the DarGlobal technical assignment: scrape public data from DarGlobal and Wasalt, build an AI chatbot on it using a free OpenRouter model, containerise with Docker, and deploy to a working URL.

---

## Highlights

| Feature | Why it matters |
|---|---|
| **Hybrid retrieval**: BM25 keyword search + `bge-small` embeddings, fused with Reciprocal Rank Fusion | Exact names ("DaVinci Tower") and paraphrased questions both work |
| **Structured listing search** | "3-bedroom villas for sale in Riyadh under 3M" is parsed into filters (type, city, bedrooms, price) and run against scraped listings, so results respect the numbers instead of trusting the LLM |
| **True market totals** | "How many villas are for sale in Riyadh?" is answered from Wasalt's own reported totals (dated November 2026), not from the size of my sample |
| **Intent handling** | Questions about a DarGlobal project never get marketplace listings mixed in; informational questions ("average price", "best area") get guide/market text, not a listing dump |
| **Grounded, cited answers** | `[1]`-style markers link to the exact source; source chips under each reply |
| **Honest refusals** | Off-topic or unsupported questions get a clear "not in the collected data", never an invented figure |
| **Model fallback chain** | Free models are flaky, so the app tries several in order and shows which one answered |
| **Graceful degradation** | If every model is rate-limited, the most relevant passages are returned instead of an error |
| **Data-quality guards** | Implausible advertiser prices (e.g. 500 SAR/yr for a 655 sqm flat) are flagged and excluded from searches and statistics; rents are always shown per year plus the monthly equivalent |
| **Safety basics** | Prompt-injection hardening (retrieved text is labelled as data, clients cannot inject a `system` role), input length limits, per-IP rate limiting, non-root container, health check |
| **Test suite** | 18 tests covering retrieval quality, filters, intent handling, API validation and privacy |

## Try these questions

- *What is Trump Tower Jeddah and how much does it cost?*
- *Which projects involve Aston Martin?* / *Tell me about DaVinci Tower by Pagani*
- *How did DarGlobal perform in H1 2026?*
- *How many villas are for sale in Riyadh?* (true total from Wasalt)
- *3-bedroom villas for sale in Riyadh under 3M* (structured filter)
- *What is the average price of villas in Riyadh?* (Wasalt guide/FAQ text)
- *What is the capital of France?* (should decline: outside the data)

---

## Architecture

```
scraper/wasalt.py ──────┐                                      ┌── BM25 ───┐
scraper/press.py ───────┼─► scraper/build_corpus.py ─► chunks ─┤            ├─ RRF ─► top passages ─┐
manual DarGlobal        │   (clean, de-boilerplate,            └── dense ──┘                       │
browser exports ────────┘    chunk, label sources)                                                  ▼
                                   └─► listings.json ─► structured filter ─► grounded prompt ─► OpenRouter
                                                                                                     │
        browser ◄── SSE stream ◄── FastAPI  /api/chat ◄── model fallback chain ◄───────────────────────┘
```

**Request flow:** the question is parsed for listing intent (type, city, bedrooms, price) and count intent; hybrid retrieval returns the best passages (max two per page so one page cannot crowd out the rest); listing matches, market totals and passages are numbered and placed into a prompt that tells the model to answer only from the context and cite by number; the answer streams back token by token over Server-Sent Events.

### Project structure

```
app/
  main.py          FastAPI app: /api/chat (SSE), /api/health, /api/stats, static UI
  rag.py           Hybrid retriever + structured listing filter + intent logic
  llm.py           OpenRouter client, model fallback chain, system prompt
  static/index.html  Single-file chat UI (no build step)
scraper/
  wasalt.py        Polite Wasalt scraper (listings + category/market pages)
  press.py         Company press-release collector (open newswires)
  press_urls.txt   Curated list of press-release URLs
  build_corpus.py  Cleans, chunks and labels all sources -> data/processed/
tests/test_app.py  Offline test suite
Dockerfile         Non-root image with the embedding model baked in
requirements.txt   Pinned dependencies
```

## Data: what was collected and how

| Source | Method | What was gathered |
|---|---|---|
| **Wasalt** (`wasalt.sa`) | Python scraper. `robots.txt` allows crawling; URLs discovered from Wasalt's public sitemaps; reads the JSON embedded in each public page; 1 request/second with an identifying User-Agent; resumable | About 1,450 public listings (sale and rent, random sample); true totals plus Wasalt's own guide/FAQ text from 42 city and category pages |
| **DarGlobal official site** (`darglobal.co.uk`) | The site is protected by bot detection that blocks automated clients. **I did not circumvent it.** Pages were exported manually from a normal browser (save page / copy text / print to PDF) and ingested by `build_corpus.py` | 44 pages: home, About, FAQ, country pages, individual project pages (e.g. Trump Tower Jeddah, The Astera, DaVinci Tower by Pagani, Les Vagues), partners, press and insights |
| **DarGlobal press releases** | `press.py` collects company-issued releases republished on open newswires (PR Newswire, Zawya's press-release section). Editorial articles from other outlets are deliberately not copied | 26 releases: results, financing, launches, contract awards |

**Privacy:** agent and owner contact details on Wasalt listings are not collected (a test enforces this).

### About this repository and the data

This public repository contains the **application and ingestion code only**. The collected source material (saved DarGlobal pages, Wasalt listing records, press-release text, processed chunks and vector indexes) is **not** committed, to avoid redistributing third-party site content. The live demo is deployed from a private repository that bundles the processed corpus. A fresh clone of this repo starts with an empty corpus and the app returns "no relevant data" responses until you add data you are permitted to use.

To rebuild a corpus yourself (check each source's current terms first; a public sitemap or permissive `robots.txt` is not itself a licence to republish content):

```bash
python scraper/wasalt.py --sale 450 --rent 150      # listings (low rate, resumable)
python scraper/wasalt.py --categories-only          # market totals + guide text
python scraper/press.py                             # company press releases
# DarGlobal pages: save them manually from your browser into data/manual_darglobal/
#   .html  - "Webpage, Complete" (the source URL is read from the file)
#   .txt   - first line is the page address, then the page text
#   .pdf   - named after the page, with "__" for "/" (e.g. rayana__trump-mansions.pdf)
python scraper/build_corpus.py                      # writes data/processed/
```

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add OPENROUTER_API_KEY
uvicorn app.main:app --port 7860                      # http://localhost:7860
pytest -q tests                                       # needs a locally built corpus
```

### Docker

```bash
docker build -t dar-wasalt-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=your-key dar-wasalt-bot
```

The image bakes in the embedding model and the vector index, so start-up is fast and needs no network apart from the LLM call. It runs as a non-root user and includes a health check on `/api/health`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter key (set as a secret, never commit it) |
| `OPENROUTER_MODELS` | free-model chain in `llm.py` | Comma-separated fallback list of free models |
| `DENSE_RETRIEVAL` | `1` | Set `0` to disable embeddings and save memory (keyword search only) |
| `RATE_LIMIT_PER_MIN` | `12` | Per-IP request limit |
| `PORT` | `7860` | HTTP port (hosts such as Render inject their own) |

### API

| Endpoint | Description |
|---|---|
| `POST /api/chat` | Body `{"message": "...", "history": [...]}`. Streams SSE events: `sources`, `model`, `token`, `notice`, `done` |
| `GET /api/health` | Status, corpus size, dense on/off, whether an LLM key is configured, deployed commit |
| `GET /api/stats` | Chunk counts by source and kind |

## Deployment

The live demo runs as a Docker web service on [Render](https://render.com) (free tier), built from the private repository's `Dockerfile`. `OPENROUTER_API_KEY` is set as a secret environment variable in the host dashboard. I tried Hugging Face Spaces first, but Docker Spaces were not available on a free account. Memory use measured at about 240 MB, within the 512 MB free limit.

## Known limitations

- **Wasalt listing-level statistics come from a sample** (about 1,400 of roughly 60,000 listings). Market totals come from Wasalt's own category pages and are dated November 2026. The assistant states which it is using.
- **DarGlobal coverage is partial**: 44 official pages and 26 press releases. Some sub-projects are covered only by country pages or press releases, and the assistant says "not in my data" beyond that.
- **Snapshot, not live**: data was collected in November 2026 and is not refreshed automatically.
- Free OpenRouter models can be slow or rate-limited at peak times; the fallback chain and passage-only fallback mitigate this.
- A minority of Wasalt listings contain advertiser data-entry errors; obvious ones are excluded, but not every inconsistency can be detected.
- The assistant provides facts from sources, **not investment advice**.

## Tech stack

Python 3.11, FastAPI, Uvicorn, `rank-bm25`, `fastembed` (`BAAI/bge-small-en-v1.5`, ONNX), httpx, BeautifulSoup, Docker, OpenRouter (free models).

## Disclaimer

An independent demonstration project, not an official DarGlobal or Wasalt product. All information comes from public sources and may be out of date. Brand names belong to their respective owners.

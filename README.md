# Siddh Guide Chatbot

FastAPI + Chroma RAG chatbot for Siddhanta Knowledge, grounded strictly on the
foundation's own websites (KB-only, no general-knowledge answers).

## Architecture

```mermaid
flowchart TD
  U[User / WordPress Frontend] -->|HTTP POST /chat| API[FastAPI Backend (Uvicorn)]
  U -->|HTTP GET /health| API

  API --> G{Greeting?}
  G -->|Yes| L1[Local Greeting Response]
  G -->|No| F{FAQ Match?}
  F -->|Yes| L2[FAQ Answer from data/faq.json]
  F -->|No| RAG[RAG Retrieve Hits]

  RAG --> VDB[(Chroma Vector DB\nchroma_db/ persistent)]
  RAG --> EMB[Amazon Titan Embeddings\n(encode query + docs)]

  VDB --> VAL{Validated KB context\nwith citations?}
  VAL -->|No| REF[KB-only refusal]
  VAL -->|Yes| LLM[Amazon Bedrock Nova\n(generate grounded answer)]
  LLM --> OUT{Answer supported by KB?}
  OUT -->|No| REF
  OUT -->|Yes| API
  REF --> API
  API --> U
```

## Live auto-crawler (keeps the knowledge base fresh)

The bot's knowledge comes from `backend/data/*.json`, which are regenerated from
the **live** websites by `backend/crawler.py` and re-embedded into Chroma by
`backend/refresh.py`.

**What it crawls** (via WordPress sitemaps + REST):
- Every enrollable Siksha course (WooCommerce `product` type — currently **38**,
  excluding the non-course "EIE Quiz" and "Gift Coupon"). Drives the dynamic
  course count/list.
- All key pages: home, about, Siksha, Aajivan, Sandhaan (+ linguistics / jyotisha
  / yoga / kosha / shastra-maps), Shodha (+ siddhanta-prastuti / indic-thought-models
  / conscious-enterprise-management), Prakashan, Events, Contact, all policies.
- `siddhantavijnan.org` and the latest blog posts.

**Automatic — no manual steps.** `AUTO_CRAWL` defaults to **on**. On each deploy
the app crawls once in the background shortly after startup, then re-crawls every
`CRAWL_INTERVAL_HOURS` (default 24h). Crawling never blocks boot, and any failure
leaves the last-good data + index untouched.

### Configuration (set in the deploy environment — `.env` is not shipped)

| Env var | Default | Purpose |
|---|---|---|
| `AUTO_CRAWL` | `1` (on) | Master switch. **Set to `0` to fully revert** to the static committed JSON snapshot — no code change. |
| `CRAWL_ON_STARTUP` | `1` | Crawl once shortly after boot. |
| `CRAWL_INTERVAL_HOURS` | `24` | Re-crawl cadence; `0` disables the periodic loop. |
| `CRAWL_REBUILD_INDEX` | `1` | Re-embed into Chroma after each crawl (needs Bedrock/Titan access). |
| `CRAWL_ADMIN_KEY` | — | Secret for `POST /admin/refresh` (defaults to `CHAT_API_KEY`). |
| `NOVA_MODEL_ID` | — | **Required.** Bedrock Nova inference-profile ARN. The committed `.env` has a placeholder — set the real value in the deploy env. |

### Manual refresh (no redeploy)

```bash
# background (returns immediately)
curl -X POST https://<host>/admin/refresh -H "x-api-key: <CHAT_API_KEY>"
# wait for the summary
curl -X POST "https://<host>/admin/refresh?wait=1" -H "x-api-key: <CHAT_API_KEY>"
```

`GET /health` reports `courses_live`, `website_entries`, and `auto_crawl`.

### Run the crawler standalone

```bash
cd backend && python -m crawler   # rewrites data/website.json + data/courses_catalog.json
```

## How to revert the crawler

1. **Fastest / no deploy:** set `AUTO_CRAWL=0` in the environment and restart. The
   app serves the committed static JSON exactly as before.
2. **Full removal:** `git revert` the crawler commit. New files
   (`crawler.py`, `refresh.py`, `data/courses_catalog.json`) and the flag-gated
   hooks in `main.py`/`config.py` are self-contained.

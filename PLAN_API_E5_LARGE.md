# Implementation Plans

## Plan A: FastAPI Service (Current - BgeM3_Q2D_API branch)

### Status: DONE

Wrap existing RAG pipeline as FastAPI service. No model changes.

- **Branch**: `BgeM3_Q2D_API` (from `BgeM3_Q2D`)
- **Files created**: `api/__init__.py`, `api/main.py`, `api/models.py`
- **Files modified**: `requirements.txt` (added fastapi, uvicorn)
- **Endpoints**: `POST /generate`, `GET /health`
- **Run**: `conda activate env_rag && uvicorn api.main:app --host 0.0.0.0 --port 8000`

---

## Plan B: OpenRouter E5-Large + Jina Reranker (Future - api-e5-large branch)

### Status: PLANNED (not started)

Switch from local GPU models to all-API architecture for free cloud hosting.

**Architecture change:**
```
BGE-M3 (GPU) -> E5-Large (OpenRouter API, $0.01/1M tokens)
BGE-Reranker (GPU) -> Jina Reranker API (free 10M tokens/key)
```

**Hosting**: Railway ($5 free/mo) with Weaviate Docker + FastAPI

### Key decisions made:
- Embedding: OpenRouter `intfloat/multilingual-e5-large` (1024-dim, $0.01/1M tokens)
- Reranker: Jina `jina-reranker-v2-base-multilingual` (Thai 81.06 nDCG, free 10M tokens)
- Hosting: Railway (API + Weaviate Docker, $5 free credit/month)
- Documents must be re-indexed with new embedding model

### Reranker comparison:

| | Jina | Cohere | SiliconFlow | Cloudflare |
|---|---|---|---|---|
| Free Tier | 10M tokens/key | 1K calls/mo | unclear | ~100K neurons/day |
| Pricing | $0.02/1M | $2/1K searches | ~$0.01/1M | $0.003/1M |
| Thai (MIRACL) | 81.06 | Good | Good | Limited |
| Speed | 15x faster | ~600ms | 2.3x faster | Fast |

### Weaviate hosting:

| Option | Cost | Persistence | Duration |
|--------|------|-------------|----------|
| Railway Docker | $5 free/mo | Yes | Unlimited |
| Weaviate Cloud | Free | Yes | 14 days only |
| Render Docker | Free 750hr/mo | Loses on sleep | Unlimited |

---

*Last updated: 2026-02-26*

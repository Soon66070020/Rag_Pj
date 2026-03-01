# Project Documentation for Claude Code

## Python Environment Setup

**Environment:** `env_rag` (local venv at `./env_rag/`)

### Paths

- Python: `./env_rag/Scripts/python.exe`
- Pip: `./env_rag/Scripts/pip.exe`
- Activate (PowerShell): `./env_rag/Scripts/Activate.ps1`

**IMPORTANT:** Do NOT use conda. There is no conda on this machine. System `python` points to the Microsoft Store stub and does not work.

### How to Run Python Scripts

```powershell
# Activate environment first
./env_rag/Scripts/Activate.ps1
python <script_name>.py

# Or use the executable directly (no activation needed)
./env_rag/Scripts/python.exe <script_name>.py
```

### Running the API

```powershell
./env_rag/Scripts/Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
PJ_Rag/
├── api/                            # FastAPI service
│   ├── main.py                     # POST /generate, GET /health
│   └── models.py                   # Pydantic request/response schemas
├── src/
│   ├── core/                       # Core types and exceptions
│   ├── retrieval/                  # Query guard, hybrid search, reranking
│   │   ├── query_guard.py          # 3-LLM guard pipeline
│   │   ├── query_processor.py      # Q2D expansion, embedding
│   │   ├── hybrid_search.py        # Dense + sparse search
│   │   ├── reranker.py             # BGE cross-encoder
│   │   └── service.py              # Pipeline orchestration
│   ├── generation/                 # Prompt building, LLM client, response parsing
│   ├── knowledge_base/             # Document processing, embeddings, indexing
│   ├── database/                   # Weaviate integration
│   └── utils/                      # Thai text processing, taxonomy inference
├── config/
│   ├── settings.py                 # Configuration loader
│   ├── model_config.yaml           # Model parameters
│   ├── prompts.yaml                # System + guard + Q2D prompt templates
│   ├── taxonomy.yaml               # 2-level category taxonomy
│   └── weaviate_schema.json        # Database schema
├── scripts/                        # setup_database.py, ingest_documents.py
├── examples/
│   └── API_input_ex.json           # Example API request payload
├── docker-compose.yml              # Weaviate service
├── requirements.txt
└── .env.example
```

---

## Pipeline Overview

```
Query -> [Guard: Scope -> Clarify -> Classify] -> Hybrid Search -> Rerank -> Generate
```

### Query Guard Pipeline (3 LLMs)
1. **LLM1 Scope Check** - Validates query is about dental post-op care
2. **LLM2 Clarification** - Rewrites unclear queries using patient context
3. **LLM3 Classification** - Multi-category assignment (Emergency, Medication, Nutrition, Post-op Care)

### Retrieval
- Hybrid search: BGE-M3 dense + BM25 sparse (alpha=0.7)
- Category filtering from LLM3 classification
- Reranking: BGE-Reranker-v2-m3, top 5 results

### Generation
- DeepSeek Reasoner with Thai nurse persona
- Citations generated from retrieval results (not LLM output)
- Patient summary prioritized when available

---

## Troubleshooting

### Python not found
System `python` is the Microsoft Store stub. Always use:
```powershell
./env_rag/Scripts/python.exe
```

### Import errors
```powershell
./env_rag/Scripts/pip.exe install -r requirements.txt
```

---

Last Updated: 2026-03-01

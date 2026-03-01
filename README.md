# Post-Oral Surgery Intelligent Assessment & Advisory System

RAG-based Q&A system for dental post-operative care, designed for Thai language.

## Architecture

```
User Query (Thai)
    |
    v
[Query Guard Pipeline - 3 LLMs]
    LLM1: Scope Check (in-scope / out-of-domain)
    LLM2: Clarification + Query Rewrite (with patient context)
    LLM3: Multi-Category Classification (Emergency, Medication, Nutrition, Post-op Care)
    |
    v
[Hybrid Search - Weaviate]
    Dense: BGE-M3 embeddings
    Sparse: BM25 keyword matching
    Category filter: contains_any from LLM3 classification
    |
    v
[Reranker - BGE-Reranker-v2-m3]
    Cross-encoder scoring -> Top 5 results
    |
    v
[Generation - DeepSeek Reasoner]
    System prompt (Thai nurse persona)
    Patient summary + flow assessments (if available)
    Retrieved context (content only)
    |
    v
JSON Response: { answer, citations, guard_info, metrics }
```

## Prerequisites

- Python 3.10+
- Docker (for Weaviate)
- CUDA-capable GPU (recommended, CPU also works)
- DeepSeek API key

## Installation

### 1. Clone the repository

```bash
mkdir PJ_Rag

cd PJ_Rag

git clone https://github.com/Soon66070020/Rag_Pj.git
```

### 2. Create virtual environment

```bash
python -m venv env_rag

# Windows (Powershell)
./env_rag/Scripts/Activate.ps1

# Linux/Mac
source env_rag/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:
- `DEEPSEEK_API_KEY` - Your DeepSeek API key
- `WEAVIATE_URL` - Weaviate URL (default: `http://localhost:8080`)

### 5. Start Weaviate

```bash
docker-compose up -d
```

### 6. Setup database schema

```bash
python scripts/setup_database.py
```

### 7. Ingest documents

Place PDF documents in `data/raw/`, then run:

```bash
python scripts/ingest_documents.py --directory data/raw
```

## Running the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## API Endpoints

### POST /generate

Full RAG pipeline: query guard + hybrid search + rerank + generate answer.

**Request body**: Patient assessment JSON (see `examples/API_input_ex.json`)

Key fields:
- `patient.assessment_data.additional_questions` - The user's question
- `flows` - Symptom assessment results per topic
- `summary` - Overall risk assessment summary

**Example request**:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @examples/API_input_ex.json
```

**Response**:

```json
{
  "answer": "ควรประคบครั้งละ 15-20 นาที...",
  "citations": [
    "Source: postop_care.pdf, Page: 3",
    "Source: compression_guide.pdf, Page: 1"
  ],
  "query": "ควรประคบนานแค่ไหน? ใช้น้ำแข็งหรือเจลเย็นดีกว่ากัน?",
  "retrieval_time_ms": 245.3,
  "generation_time_ms": 1823.5,
  "documents_used": 5,
  "guard": {
    "is_in_scope": true,
    "clarification_needed": false,
    "original_query": "ควรประคบนานแค่ไหน?...",
    "effective_query": "ควรประคบนานแค่ไหน?...",
    "classification": [
      {"category": "Post-op Care", "confidence": 0.95}
    ],
    "guard_time_ms": 312.7
  }
}
```

### GET /health

```json
{
  "status": "ok",
  "models_loaded": true
}
```

## Project Structure

```
PJ_Rag/
├── api/                            # FastAPI service
│   ├── main.py                     # POST /generate, GET /health
│   └── models.py                   # Pydantic request/response schemas
│
├── src/
│   ├── core/                       # Core types and exceptions
│   │   ├── types.py                # Dataclasses (Query, Document, RetrievalResult, etc.)
│   │   └── exceptions.py           # Custom exception hierarchy
│   │
│   ├── retrieval/                  # Query processing and search
│   │   ├── query_guard.py          # 3-LLM guard pipeline (scope, clarify, classify)
│   │   ├── query_processor.py      # Q2D expansion, embedding generation
│   │   ├── hybrid_search.py        # Dense + sparse search with category filtering
│   │   ├── reranker.py             # BGE cross-encoder reranking
│   │   └── service.py              # Orchestrates retrieval pipeline
│   │
│   ├── generation/                 # LLM response generation
│   │   ├── client.py               # DeepSeek API client
│   │   ├── prompt_builder.py       # Context formatting and prompt construction
│   │   ├── response_parser.py      # Citation generation from retrieval results
│   │   └── engine.py               # Generation pipeline orchestration
│   │
│   ├── knowledge_base/             # Document ingestion
│   │   ├── loader.py               # PDF loading
│   │   ├── chunker.py              # Thai token-based chunking
│   │   ├── metadata_extractor.py   # Taxonomy-based multi-category extraction
│   │   ├── embedding_generator.py  # BGE-M3 embedding generation
│   │   └── indexer.py              # Weaviate indexing
│   │
│   ├── database/                   # Weaviate integration
│   │   ├── connector.py            # Connection management
│   │   └── manager.py              # Schema and CRUD operations
│   │
│   └── utils/
│       └── text_processing.py      # Thai text processing, taxonomy inference
│
├── config/
│   ├── settings.py                 # Configuration loader
│   ├── model_config.yaml           # Model and pipeline parameters
│   ├── prompts.yaml                # System prompts, guard prompts, Q2D templates
│   ├── taxonomy.yaml               # 2-level category taxonomy with Thai keywords
│   └── weaviate_schema.json        # Database collection schema
│
├── scripts/
│   ├── setup_database.py           # Create Weaviate schema
│   └── ingest_documents.py         # Process and index PDF documents
│
├── examples/
│   └── API_input_ex.json           # Example API request payload
│
├── data/
│   ├── raw/                        # Source PDF documents
│   └── processed/                  # Processed chunks
│
├── docker-compose.yml              # Weaviate service
├── requirements.txt
├── .env.example                    # Environment variable template
└── README.md
```

## Configuration

### Model Parameters (`config/model_config.yaml`)

| Component | Model | Purpose |
|-----------|-------|---------|
| Embedding | `BAAI/bge-m3` | Dense + sparse vectors for hybrid search |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder relevance scoring |
| Generation | `deepseek-reasoner` | Answer generation (128K context) |
| Q2D / Guard | `deepseek-chat` | Query expansion and guard pipeline |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | Hybrid search blending (0=sparse, 1=dense) |
| `top_k` | 20 | Initial candidates from hybrid search |
| `top_n` | 5 | Final results after reranking |
| `use_q2d` | true | Enable Q2D query expansion |

### Prompts (`config/prompts.yaml`)

- `system_prompt` - Thai nurse persona with response format templates
- `scope_check_prompt` - LLM1: dental post-op scope validation
- `clarification_check_prompt` - LLM2: query clarity + rewrite with patient context
- `classification_prompt` - LLM3: multi-category classification

### Taxonomy (`config/taxonomy.yaml`)

2-level category hierarchy with Thai keywords:
- **Emergency** - bleeding, severe pain, high fever, swelling, breathing difficulty
- **Medication** - painkillers, antibiotics, anti-inflammatory, mouthwash
- **Nutrition** - soft food, forbidden food, drinks, supplements
- **Post-op Care** - wound care, compression, oral hygiene, activity, sutures, numbness, sleep position, swelling

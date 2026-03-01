# Post-Oral Surgery Intelligent Assessment & Advisory System

> RAG-based Medical Q&A System with **Thai Language Support** 🇹🇭

A Retrieval-Augmented Generation (RAG) system designed specifically for post-oral surgery patients, providing accurate medical guidance in Thai language using DeepSeek LLM, Weaviate vector database, and BGE-M3 multilingual embeddings.

## 🌟 Features

- **Thai Language Primary Support**: Fully optimized for Thai queries and mixed Thai-English medical documents
- **Hybrid Search**: Combines dense (semantic) and sparse (lexical) retrieval for optimal accuracy
- **Advanced Reranking**: Uses BGE-Reranker-v2-m3 for precise relevance scoring
- **Token-Based Chunking**: Thai token counting using pythainlp for accurate semantic segmentation
- **DeepSeek Reasoner**: Advanced reasoning model with 128K context and thinking process
- **Strict Citation Enforcement**: All responses include mandatory source citations
- **Category-Based Filtering**: Automatic intent classification (Emergency, Medication, Nutrition, Post-op Care)
- **Medical Domain Optimized**: Conservative temperature settings and fact-based responses

## 🏗️ Architecture

```
User Query (Thai) → Query Processor (HyDE + BGE-M3) → Metadata Filter
                     ↓
          Hybrid Searcher (Weaviate) → BGE Reranker → Prompt Builder
                     ↓
          DeepSeek Reasoner (128K) → Citation Enforcer → Thai Response with Citations
```

## 📋 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for embeddings)
- Weaviate database instance
- DeepSeek API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd PJ_Rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# DEEPSEEK_API_KEY=your_key_here
# WEAVIATE_URL=http://localhost:8080
```

### 5. Start Weaviate (Docker)

```bash
docker-compose up -d weaviate
```

### 6. Setup Database Schema

```bash
python scripts/setup_database.py
```

### 7. Ingest Documents

```bash
# Place your Thai/English PDF documents in data/raw/
python scripts/ingest_documents.py
```

### 8. Run Interactive Demo

```bash
python scripts/interactive_demo.py
```

## 📁 Project Structure

```
PJ_Rag/
├── config/                 # Configuration files
│   ├── model_config.yaml   # Model parameters
│   ├── weaviate_schema.json # Database schema
│   ├── prompts.yaml        # Prompt templates
│   └── settings.py         # Configuration loader
│
├── src/
│   ├── core/              # Core data types and exceptions
│   │   ├── types.py       # Dataclasses
│   │   └── exceptions.py  # Custom exceptions
│   │
│   ├── knowledge_base/    # Module 1: Document Management
│   ├── retrieval/         # Module 2: Retrieval Engine ⭐
│   ├── generation/        # Module 3: Response Generation ⭐
│   ├── evaluation/        # Module 4: Evaluation & Logging
│   ├── database/          # Weaviate client
│   └── utils/             # Utilities (Thai text processing)
│
├── data/                  # Data directories
│   ├── raw/              # Source PDFs
│   ├── processed/        # Processed chunks
│   └── evaluation/       # Test datasets
│
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── notebooks/            # Jupyter notebooks
└── logs/                 # Application logs
```

## 🇹🇭 Thai Language Support

The system is specifically designed for Thai language:

### Text Processing
- **pythainlp** integration for Thai tokenization and normalization
- **Token-based chunking**: Uses `word_tokenize` for accurate chunk sizing
- Unicode NFC normalization to preserve Thai characters
- Mixed Thai-English document handling

### Embeddings
- **BGE-M3** multilingual model with native Thai support
- Dual vectors: dense (semantic) + sparse (lexical)
- No fine-tuning required

### Response Generation
- **DeepSeek Reasoner** with thinking process for accurate medical responses
- Proper medical terminology and polite tone (ครับ/ค่ะ)
- Thai filename support in citations
- Reasoning tokens logged for transparency

### Category Inference
Uses Thai medical keywords:
- ยา, แก้ปวด, ปฏิชีวนะ → Medication
- เลือดออก, ปวดมาก, ไข้สูง → Emergency
- อาหาร, กิน, ดื่ม → Nutrition

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/unit/test_query_processor.py
```

## 📊 Evaluation Metrics

### Retrieval Quality
- Precision@K, Recall@K
- MRR (Mean Reciprocal Rank)
- nDCG (Normalized Discounted Cumulative Gain)

### Generation Quality
- BERTScore, BLEU, METEOR, ROUGE
- Human evaluation (Correctness, Relevance, Helpfulness)
- LLM-as-a-Judge automated scoring

## 🔧 Configuration

### Model Parameters (config/model_config.yaml)

```yaml
retrieval:
  alpha: 0.7      # Hybrid search blending
  top_k: 20       # Initial candidates
  top_n: 5        # Final reranked results
  use_hyde: true  # Enable HyDE expansion

llm:
  model_name: "deepseek-reasoner"  # 128K context, 64K output
  temperature: 0.1  # Conservative for medical accuracy
  max_tokens: 16384
  timeout: 120  # Increased for reasoning model

chunking:
  chunk_size: 256       # Thai tokens (not characters)
  chunk_overlap: 30
  use_token_count: true # Count tokens instead of chars
```

### Pricing (DeepSeek Reasoner)
| Type | Price |
|------|-------|
| Input (cache hit) | $0.028/M tokens |
| Input (cache miss) | $0.28/M tokens |
| Output | $0.42/M tokens |

### Prompts (config/prompts.yaml)

Customize system prompts, HyDE templates, and citation formats.

## 📚 Example Usage

```python
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.response_pipeline import ResponsePipeline

# Initialize pipelines
retrieval_pipeline = RetrievalPipeline(...)
response_pipeline = ResponsePipeline(...)

# Process Thai query
query = "ฉันควรกินอะไรหลังผ่าตัดฟัน?"

# Retrieve relevant context
retrieval_result = retrieval_pipeline.retrieve(query)

# Generate response with citations
response = response_pipeline.generate_response(query, retrieval_result)

print(response.answer)
# Output: "หลังผ่าตัดฟันคุณควรกินอาหารอ่อนและเย็น เช่น โจ๊ก ซุป...
#          [Source: nutrition_guide.pdf, Page: 12]"
```

## 🐛 Troubleshooting

### Common Issues

1. **pythainlp not found**
   ```bash
   pip install pythainlp>=4.0.0
   ```

2. **Weaviate connection error**
   ```bash
   # Check Weaviate is running
   docker ps
   # Restart if needed
   docker-compose restart weaviate
   ```

3. **CUDA out of memory**
   - Reduce batch_size in model_config.yaml
   - Use CPU mode: set `device: "cpu"`

## 📖 Documentation

- [PLAN_API_E5_LARGE.md](PLAN_API_E5_LARGE.md) - API & E5-Large migration plan

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Run `black` and `mypy` for code quality
5. Submit a pull request

## 📄 License

[Specify your license here]

## 👥 Authors

- [Your Name/Team]

## 🙏 Acknowledgments

- **BGE-M3** by BAAI for multilingual embeddings
- **DeepSeek** for LLM API
- **Weaviate** for vector database
- **pythainlp** for Thai NLP support

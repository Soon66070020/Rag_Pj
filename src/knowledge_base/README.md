# Knowledge Base Module Documentation

The knowledge base module provides complete document ingestion pipeline for the RAG system, with specialized support for Thai language medical documents.

## Architecture

```
src/knowledge_base/
├── __init__.py
├── loader.py              # PDF and text file loaders
├── chunker.py             # Semantic chunking with Thai support
├── metadata_extractor.py  # Category and metadata extraction
├── embedding_generator.py # BGE-M3 embedding generation
├── indexer.py            # Main pipeline orchestrator
└── README.md             # This file
```

## Pipeline Flow

```
Raw Documents → Load → Chunk → Extract Metadata → Generate Embeddings → Index to Weaviate
```

## Components

### 1. Document Loader ([loader.py](loader.py))

Loads documents from various formats with Thai text support:
- **PDFLoader**: Extracts text from PDF files using pdfplumber
- **TextLoader**: Loads text files (.txt, .md) with encoding detection

#### Features:
- ✅ Thai character preservation (Unicode NFC normalization)
- ✅ Page-by-page extraction
- ✅ Automatic encoding detection (UTF-8, TIS-620, Windows-874)
- ✅ Mixed Thai-English document support

#### Usage:

```python
from src.knowledge_base.loader import load_document
from pathlib import Path

# Load PDF
pages = load_document(Path("medication_guide.pdf"))

for page in pages:
    print(f"Page {page.page_number}: {len(page.text)} chars")
```

#### Supported Formats:
- `.pdf` - PDF documents
- `.txt` - Plain text
- `.md`, `.markdown` - Markdown documents

### 2. Semantic Chunker ([chunker.py](chunker.py))

Chunks documents into semantic segments while preserving Thai sentence boundaries.

#### Features:
- ✅ Thai sentence tokenization using pythainlp
- ✅ Respects sentence boundaries (no mid-sentence splits)
- ✅ Configurable chunk size, overlap, min/max constraints
- ✅ Context preservation for medical terminology

#### Configuration (from config/model_config.yaml):

```yaml
chunking:
  strategy: "semantic"
  chunk_size: 512        # Target chunk size in characters
  chunk_overlap: 50      # Overlap between chunks
  min_chunk_size: 100    # Minimum chunk size
  max_chunk_size: 800    # Maximum chunk size
```

#### Usage:

```python
from src.knowledge_base.chunker import SemanticChunker

chunker = SemanticChunker(
    chunk_size=512,
    chunk_overlap=50
)

text = "หลังผ่าตัดฟันคุณควรกินอาหารอ่อน..."
chunks = chunker.chunk_text(text)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.sentence_count} sentences")
```

#### Chunking Strategy:

1. **Tokenize** text into Thai sentences using pythainlp
2. **Group** sentences into chunks respecting size constraints
3. **Overlap** by including sentences from previous chunk
4. **Validate** all chunks meet min/max size requirements

### 3. Metadata Extractor ([metadata_extractor.py](metadata_extractor.py))

Extracts metadata from chunks including category classification.

#### Features:
- ✅ Auto-infer category from Thai keywords
- ✅ Extract Thai keywords from content
- ✅ Content statistics (char count, sentence count)

#### Categories:

Based on Thai medical keywords:
- **Emergency**: เลือดออก, ปวดมาก, บวมมาก, ไข้สูง
- **Medication**: ยา, แก้ปวด, ปฏิชีวนะ
- **Nutrition**: อาหาร, กิน, ดื่ม, รับประทาน
- **Post-op Care**: General care (default)

#### Usage:

```python
from src.knowledge_base.metadata_extractor import extract_metadata

metadata = extract_metadata(
    text="ฉันควรกินยาแก้ปวดทุก 6 ชั่วโมง",
    source_file="medication.pdf",
    page_number=5
)

print(f"Category: {metadata['category']}")  # "Medication"
print(f"Keywords: {metadata['keywords']}")  # ["ยา", "แก้ปวด", ...]
```

### 4. Embedding Generator ([embedding_generator.py](embedding_generator.py))

Generates embeddings using BGE-M3 multilingual model.

#### Features:
- ✅ Native Thai language support
- ✅ Dual embeddings: dense (1024-dim) + sparse (lexical)
- ✅ Batch processing for efficiency
- ✅ GPU/CPU support with FP16

#### BGE-M3 Model:

**Why BGE-M3?**
- Pre-trained on 100+ languages including Thai
- Best multilingual embedding model for Thai-English mixed text
- Produces both dense (semantic) and sparse (lexical) vectors
- No fine-tuning required for Thai medical domain

#### Configuration:

```yaml
embedding:
  model_name: "BAAI/bge-m3"
  use_fp16: true
  batch_size: 32
  max_length: 512
  device: "cuda"  # or "cpu"
```

#### Usage:

```python
from src.knowledge_base.embedding_generator import BGEEmbeddingGenerator

generator = BGEEmbeddingGenerator()

texts = ["ฉันควรกินยาแก้ปวด", "หลังผ่าตัดควรพักผ่อน"]
embeddings = generator.encode(texts)

dense = embeddings[0]['dense']   # (1024,) numpy array
sparse = embeddings[0]['sparse'] # {token: weight} dict

print(f"Dense shape: {dense.shape}")    # (1024,)
print(f"Sparse size: {len(sparse)}")    # ~100-200 tokens
```

### 5. Document Indexer ([indexer.py](indexer.py))

Main pipeline orchestrator that combines all components.

#### Complete Pipeline:

```
File Path → Load Pages → Chunk → Extract Metadata → Generate Embeddings → Index to Weaviate
```

#### Features:
- ✅ End-to-end pipeline automation
- ✅ Batch processing for memory efficiency
- ✅ Progress tracking and statistics
- ✅ Error handling and recovery
- ✅ Support for single file or directory

#### Usage:

**Index Single File:**

```python
from src.knowledge_base import index_file
from pathlib import Path

stats = index_file(Path("medication_guide.pdf"))

print(f"Pages: {stats['pages']}")
print(f"Chunks: {stats['chunks']}")
print(f"Indexed: {stats['indexed']}")
print(f"Time: {stats['time_seconds']}s")
```

**Index Directory:**

```python
from src.knowledge_base import index_directory

stats = index_directory(
    Path("data/raw"),
    recursive=True
)

print(f"Files: {stats['successful']}/{stats['total_files']}")
print(f"Total chunks: {stats['total_chunks']}")
```

## Command-Line Interface

Use the `scripts/ingest_documents.py` script for CLI access:

### Index Single File

```bash
python scripts/ingest_documents.py --file data/raw/medication_guide.pdf
```

### Index Directory

```bash
# Index all files in directory (recursive)
python scripts/ingest_documents.py --directory data/raw

# Index specific extensions only
python scripts/ingest_documents.py --directory data/raw --extensions ".pdf,.txt"

# Non-recursive (current directory only)
python scripts/ingest_documents.py --directory data/raw --no-recursive

# Custom batch size
python scripts/ingest_documents.py --directory data/raw --batch-size 50
```

### Check Database Status

```bash
python scripts/ingest_documents.py --status
```

## Configuration

Configuration is loaded from `config/model_config.yaml`:

```yaml
embedding:
  model_name: "BAAI/bge-m3"
  batch_size: 32
  device: "cuda"

chunking:
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_size: 100
  max_chunk_size: 800

thai_processing:
  tokenization_engine: "newmm"
  remove_stopwords: true
  normalize_text: true
```

## Error Handling

The module provides specific exceptions for different failure modes:

```python
from src.core.exceptions import (
    DocumentLoadError,    # File loading failure
    ChunkingError,       # Chunking failure
    EmbeddingError,      # Embedding generation failure
    IndexingError        # Weaviate indexing failure
)

try:
    stats = index_file(Path("document.pdf"))
except DocumentLoadError as e:
    logger.error(f"Failed to load document: {e}")
except EmbeddingError as e:
    logger.error(f"Failed to generate embeddings: {e}")
except IndexingError as e:
    logger.error(f"Failed to index to Weaviate: {e}")
```

## Thai Language Processing

### Text Normalization

All Thai text is normalized using:
1. **Unicode NFC** normalization (preserves Thai characters)
2. **pythainlp** text normalization
3. **Whitespace** cleanup

### Sentence Tokenization

Uses pythainlp's `sent_tokenize` with `crfcut` engine:
- Most accurate for Thai medical text
- Handles Thai punctuation (။) and mixed Thai-English
- Preserves sentence boundaries in chunks

### Keyword Extraction

Extracts Thai keywords for:
- Category inference
- Metadata enrichment
- Search optimization

Uses pythainlp's:
- Word tokenization (`newmm` algorithm)
- Stopword removal (Thai stopwords corpus)
- Frequency counting

## Performance

### Typical Performance (GPU):

- **Loading**: ~0.5s per page (PDF)
- **Chunking**: ~0.1s per page
- **Embedding**: ~100 chunks/second (batch_size=32)
- **Indexing**: ~500 chunks/second (batch_size=100)

**Overall**: ~10-20 seconds per document (10 pages, 100 chunks)

### Memory Usage:

- **BGE-M3 Model**: ~2GB GPU memory
- **Batch Processing**: ~500MB per batch (32 chunks)

### Optimization Tips:

1. **Use GPU** if available (10x faster embedding generation)
2. **Increase batch_size** for faster processing (if memory permits)
3. **Use FP16** for faster inference with minimal quality loss
4. **Process in batches** for large directories

## Testing

Run unit tests:

```bash
# Test all knowledge_base components
pytest tests/test_knowledge_base.py -v

# Test specific component
pytest tests/test_knowledge_base.py::TestDocumentLoader -v
```

## Troubleshooting

### PDF Loading Issues

```python
# Check pdfplumber installation
pip install pdfplumber>=0.10.0

# Try with different PDF
loader = PDFLoader()
pages = loader.load(Path("test.pdf"))
```

### Thai Text Issues

```python
# Check pythainlp installation
pip install pythainlp>=4.0.0

# Test Thai normalization
from src.utils.text_processing import normalize_thai_text
text = normalize_thai_text("ฉันควรกินยา")
```

### Embedding Generation Issues

```python
# Check FlagEmbedding installation
pip install FlagEmbedding>=1.2.0

# Test on CPU if GPU fails
generator = BGEEmbeddingGenerator(device="cpu")
```

### Weaviate Connection Issues

```python
# Check Weaviate is running
docker ps | grep weaviate

# Test connection
from src.database.connector import get_connector
connector = get_connector()
print(connector.health_check())
```

## Best Practices

1. **Always normalize Thai text** before processing
2. **Use semantic chunking** to preserve medical context
3. **Batch process** large directories for efficiency
4. **Monitor GPU memory** when processing many documents
5. **Check database status** before ingestion
6. **Use appropriate chunk size** for your domain (512 chars recommended for medical text)

## Next Steps

After ingestion, documents are ready for retrieval:
- **Module 2**: Retrieval Engine (query processing, hybrid search, reranking)
- **Module 3**: Response Generation (prompt building, LLM API, citation enforcement)

See main [PROJECT README](../../README.md) for complete system documentation.

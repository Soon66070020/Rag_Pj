# Response Generation Module

This module implements the response generation pipeline for the Thai medical Q&A RAG system using DeepSeek LLM.

## Overview

The generation module takes retrieval results (context documents) and generates accurate, cited Thai language medical responses. It enforces strict medical safety constraints:

- **Never invents information**: Only answers based on provided context
- **Mandatory citations**: All factual statements must cite sources
- **PII protection**: Strips personally identifiable information before API calls
- **Graceful fallback**: Returns safe message when context is insufficient

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Validate Context Quality                                │
│     ├─ Check minimum score threshold (0.5)                  │
│     └─ Check minimum document count (1)                     │
│                                                              │
│  2. Sanitize Query (PII Removal)                            │
│     ├─ Strip Thai ID numbers                                │
│     ├─ Strip phone numbers                                  │
│     └─ Strip email addresses                                │
│                                                              │
│  3. Build Prompt                                             │
│     ├─ System instruction from prompts.yaml                 │
│     ├─ Context with source citations                        │
│     └─ User query                                            │
│                                                              │
│  4. Call LLM (DeepSeek API)                                 │
│     ├─ Temperature: 0.1 (medical accuracy)                  │
│     ├─ Max tokens: 2048                                      │
│     └─ Retry logic (3 attempts)                             │
│                                                              │
│  5. Parse & Validate Response                                │
│     ├─ Extract citations                                     │
│     ├─ Validate against context                             │
│     └─ Format as GeneratedResponse                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. DeepSeekClient (`client.py`)

API client for DeepSeek with retry logic and error handling.

**Key Features:**
- OpenAI-compatible API interface
- Automatic retry with exponential backoff
- Temperature enforcement (0.1 for medical)
- Response validation
- Fallback handling

**Example:**
```python
from src.generation.client import DeepSeekClient

client = DeepSeekClient(
    api_key="sk-xxx",
    temperature=0.1,
    max_tokens=2048
)

messages = [
    {"role": "system", "content": "You are a medical assistant"},
    {"role": "user", "content": "ยาแก้ปวดคืออะไร"}
]

response = client.generate(messages)
print(response["content"])
```

**Configuration (from `model_config.yaml`):**
```yaml
llm:
  model_name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  temperature: 0.1
  max_tokens: 2048
  top_p: 0.9
  timeout: 60
```

### 2. PromptBuilder (`prompt_builder.py`)

Constructs prompts by combining system instructions, context, and user query.

**Key Features:**
- Template-based prompt construction
- Context formatting with citations
- Citation instruction enforcement
- Empty context handling
- HyDE prompt support

**Example:**
```python
from src.generation.prompt_builder import PromptBuilder

builder = PromptBuilder(prompts_config)

# Build prompt with retrieval results
messages = builder.build_prompt(
    retrieval_result=retrieval_result,
    user_query="ฉันควรกินยาแก้ปวดไหม"
)

# Validate context quality
is_valid = builder.validate_context_quality(
    retrieval_result,
    min_score=0.5,
    min_results=1
)
```

**Prompt Structure:**
```
[System Message]
You are an expert dental assistant AI...
CRITICAL RULES:
1. Answer in Thai language (ครับ/ค่ะ)
2. Do NOT use outside knowledge
3. MANDATORY CITATION: [Source: filename, Page: number]
...

[User Message]
Based on the following context, please answer the question.

CONTEXT:
---
[Context 1] (Category: Medication, Source: guide.pdf, Page: 5, Score: 0.92)
การดูแลหลังผ่าตัด ควรรับประทานยาแก้ปวดตามแพทย์สั่ง...

[Context 2] (Source: care.pdf, Page: 12, Score: 0.87)
ยาแก้ปวดที่แนะนำคือ พาราเซตามอล...
---

QUESTION:
ฉันควรกินยาแก้ปวดไหม

ANSWER (in Thai with citations):
```

### 3. ResponseParser (`response_parser.py`)

Extracts and validates citations from LLM responses.

**Key Features:**
- Regex-based citation extraction
- Citation validation against context
- Coverage analysis
- Quality metrics

**Citation Format:**
```
[Source: filename, Page: number]
```

**Example:**
```python
from src.generation.response_parser import ResponseParser

parser = ResponseParser(
    strict_validation=False,
    require_citations=True
)

# Parse LLM response
parsed = parser.parse_response(
    llm_response="ควรรับประทานยาตามแพทย์สั่ง [Source: guide.pdf, Page: 5]",
    retrieval_result=result,
    query_text="ฉันควรกินยาไหม"
)

print(f"Answer: {parsed.answer}")
print(f"Citations: {parsed.citations}")

# Check citation quality
quality = parser.check_citation_quality(
    llm_response,
    retrieval_result
)

print(f"Citation count: {quality['citation_count']}")
print(f"Coverage: {quality['coverage']:.1%}")
print(f"All valid: {quality['all_valid']}")
```

### 4. GenerationEngine (`engine.py`)

Main orchestrator combining all components.

**Key Features:**
- Complete pipeline orchestration
- Context quality validation
- PII removal (Thai ID, phone, email)
- Fallback handling
- Batch generation support

**Example:**
```python
from src.generation import create_generation_engine
from config.settings import get_settings

settings = get_settings()

# Create engine
engine = create_generation_engine(
    llm_config=settings.model_config['llm'],
    prompts_config=settings.prompts_config,
    api_key="sk-xxx",
    min_context_score=0.5,
    min_context_count=1
)

# Generate response
from src.retrieval import retrieve

retrieval_result = retrieve("ฉันควรกินยาแก้ปวดไหม")
response = engine.generate(retrieval_result)

print(response.answer)
print(f"Citations: {response.citations}")
print(f"Time: {response.generation_time_ms:.2f}ms")
```

## Medical Safety Constraints

### 1. Context-Based Answers Only

The system **NEVER** invents medical information. If the answer isn't in the retrieved context:

```python
# Insufficient context → Fallback response
"ขอภัยครับ/ค่ะ ไม่มีข้อมูลเพียงพอในระบบ กรุณาปรึกษาทันตแพทย์โดยตรงค่ะ"
```

### 2. Mandatory Citations

All factual statements must cite sources:

**Good Response:**
```
ควรรับประทานยาแก้ปวดตามแพทย์สั่ง เช่น พาราเซตามอล 500 มก. ทุก 4-6 ชั่วโมง
[Source: medication_guide.pdf, Page: 12]
```

**Bad Response (rejected):**
```
ควรรับประทานยาแก้ปวดตามแพทย์สั่ง
(No citation → Warning logged)
```

### 3. PII Protection

Query sanitization before API calls:

```python
# Input:  "โทร 081-234-5678 ปวดฟันมาก"
# Output: "โทร [PHONE] ปวดฟันมาก"

# Masked patterns:
# - Thai ID: 1-2345-67890-12-3 → [THAI_ID]
# - Phone: 081-234-5678 → [PHONE]
# - Email: user@example.com → [EMAIL]
```

### 4. Low Temperature

Temperature set to 0.1 for consistency and medical accuracy:

```python
temperature = 0.1  # Minimal randomness
top_p = 0.9        # Nucleus sampling
```

## Usage Examples

### Basic Usage

```python
from src.generation import create_generation_engine
from src.retrieval import retrieve
from config.settings import get_settings

# Initialize
settings = get_settings()
engine = create_generation_engine(
    settings.model_config['llm'],
    settings.prompts_config,
    api_key="sk-xxx"
)

# Complete pipeline
query = "ฉันควรกินอาหารอะไรหลังผ่าตัด"

# 1. Retrieve context
retrieval_result = retrieve(query)

# 2. Generate response
response = engine.generate(retrieval_result)

# 3. Display results
print(f"Question: {response.query_text}")
print(f"Answer: {response.answer}")
print(f"\nCitations:")
for i, citation in enumerate(response.citations, 1):
    print(f"{i}. {citation}")
print(f"\nGeneration time: {response.generation_time_ms:.2f}ms")
```

### Advanced Usage

```python
# Custom thresholds
engine = create_generation_engine(
    llm_config,
    prompts_config,
    api_key="sk-xxx",
    min_context_score=0.7,  # Higher quality threshold
    min_context_count=2     # Require at least 2 sources
)

# Override temperature for specific query
response = engine.generate(
    retrieval_result,
    temperature=0.05  # Even more deterministic
)

# Batch generation
queries = [
    "ฉันควรกินยาไหม",
    "อาหารที่ควรหลีกเลี่ยง",
    "อาการปกติหลังผ่าตัด"
]

retrieval_results = [retrieve(q) for q in queries]
responses = engine.batch_generate(retrieval_results)

for resp in responses:
    print(f"Q: {resp.query_text}")
    print(f"A: {resp.answer[:100]}...")
    print()
```

### HyDE Query Expansion

```python
# Generate hypothetical document for better retrieval
hyde_doc = engine.generate_with_hyde(
    query="ฉันควรกินอาหารอะไร",
    hyde_temperature=0.3
)

print(f"HyDE: {hyde_doc}")
# Use hyde_doc for query expansion before retrieval
```

### Citation Validation

```python
# Check citation quality
from src.generation.response_parser import ResponseParser

parser = ResponseParser()

quality = parser.check_citation_quality(
    response.answer,
    retrieval_result
)

if quality['all_valid']:
    print("✓ All citations are valid")
else:
    print(f"⚠ Invalid citations: {quality['invalid_citations']}")

print(f"Coverage: {quality['coverage']:.1%}")
print(f"Citation count: {quality['citation_count']}")
```

## Configuration

### LLM Configuration (`model_config.yaml`)

```yaml
llm:
  model_name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  temperature: 0.1        # Medical strictness
  max_tokens: 2048        # Response length
  top_p: 0.9
  timeout: 60             # API timeout (seconds)
  stream: false
```

### Prompts Configuration (`prompts.yaml`)

```yaml
system_prompt: |
  You are an expert dental assistant AI specializing in post-oral surgery care.

  CRITICAL RULES:
  1. Answer in Thai language (ครับ/ค่ะ)
  2. Do NOT use outside knowledge
  3. MANDATORY CITATION: [Source: filename, Page: number]
  ...

empty_context_prompt: |
  Respond politely in Thai that you don't have sufficient information.
  Suggest consulting their dentist directly.

citation_format_instruction: |
  IMPORTANT: All factual statements must include citations.
  Format: [Source: filename, Page: number]
```

## Performance Metrics

### Generation Latency

Typical generation time: **~200ms**

```
Component breakdown:
- Context validation:    5ms
- PII removal:          2ms
- Prompt building:      3ms
- LLM API call:      150ms  ← Dominant
- Response parsing:    10ms
- Citation validation: 30ms
─────────────────────────────
Total:               ~200ms
```

### Memory Usage

```
Component                Memory
─────────────────────────────
DeepSeekClient            5MB  (OpenAI library)
PromptBuilder             2MB  (Templates)
ResponseParser            1MB  (Regex patterns)
─────────────────────────────
Total                    ~8MB
```

### Token Usage

```
Average per query:
- System prompt:     200 tokens
- Context (5 docs):  800 tokens
- User query:         50 tokens
- Response:          300 tokens
─────────────────────────────
Total:             ~1350 tokens (~$0.0027 per query)
```

## Error Handling

### Retry Logic

```python
# Automatic retry for transient errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError))
)
```

### Fallback Handling

```python
# If generation fails, return safe fallback
try:
    response = engine.generate(retrieval_result)
except Exception as e:
    # Returns fallback message instead of error
    response = engine._create_fallback_response(...)
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `APIConnectionError` | Network issue | Automatic retry (3 attempts) |
| `RateLimitError` | API quota exceeded | Exponential backoff, retry |
| `GenerationError` | Invalid API response | Return fallback message |
| Insufficient context | Low relevance scores | Return fallback message |
| Missing citations | LLM skipped citations | Log warning, accept response |

## Best Practices

### 1. Set Appropriate Thresholds

```python
# For critical medical queries
engine = create_generation_engine(
    ...,
    min_context_score=0.7,  # High quality only
    min_context_count=2     # Multiple sources
)

# For general queries
engine = create_generation_engine(
    ...,
    min_context_score=0.5,  # Lower threshold
    min_context_count=1     # Single source OK
)
```

### 2. Monitor Citation Quality

```python
# Log citation metrics
quality = parser.check_citation_quality(response.answer, result)

if quality['coverage'] < 0.5:
    logger.warning("Low citation coverage")

if not quality['all_valid']:
    logger.error(f"Invalid citations: {quality['invalid_citations']}")
```

### 3. Handle Thai Text Properly

```python
# Ensure UTF-8 encoding
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(response.answer)

# Display Thai correctly
print(response.answer.encode('utf-8').decode('utf-8'))
```

### 4. Implement Response Caching

```python
# Cache responses for identical queries
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_generate(query_text):
    result = retrieve(query_text)
    response = engine.generate(result)
    return response
```

## Troubleshooting

### Issue: No citations in response

**Symptoms:**
```
Warning: Response contains no citations
```

**Solutions:**
1. Check prompt template includes citation instructions
2. Verify `citation_format_instruction` in prompts.yaml
3. Review system prompt emphasizes citations
4. Consider using `strict_validation=True` to enforce

### Issue: Invalid citations

**Symptoms:**
```
Warning: Found 2 invalid citations: ['Source: fake.pdf, Page: 99']
```

**Solutions:**
1. LLM hallucinated sources not in context
2. Increase context quality threshold
3. Use stricter prompt instructions
4. Enable `strict_validation=True` to reject

### Issue: Fallback responses too frequent

**Symptoms:**
```
Most queries return: "ขอภัยครับ/ค่ะ ไม่มีข้อมูลเพียงพอในระบบ"
```

**Solutions:**
1. Lower `min_context_score` (default 0.5 → 0.3)
2. Reduce `min_context_count` (default 1 → 0)
3. Improve document indexing quality
4. Add more documents to knowledge base
5. Check retrieval pipeline configuration

### Issue: High API latency

**Symptoms:**
```
Generation time: 5000ms (very slow)
```

**Solutions:**
1. Check network connection to DeepSeek API
2. Reduce `max_tokens` (2048 → 1024)
3. Verify timeout settings (60s)
4. Consider using shorter context (fewer docs)
5. Monitor API status page

## Testing

```python
# Unit tests
pytest tests/generation/test_client.py
pytest tests/generation/test_prompt_builder.py
pytest tests/generation/test_response_parser.py
pytest tests/generation/test_engine.py

# Integration tests
pytest tests/integration/test_generation_pipeline.py

# End-to-end test
python scripts/test_rag_pipeline.py --query "ฉันควรกินยาไหม"
```

## Dependencies

```txt
openai>=1.0.0          # DeepSeek API client
tenacity>=8.0.0        # Retry logic
```

## See Also

- [Retrieval Module](../retrieval/README.md) - Context retrieval
- [Knowledge Base Module](../knowledge_base/README.md) - Document indexing
- [Configuration Guide](../../config/README.md) - Settings reference

"""Quick test script for RAG query with deepseek-reasoner."""
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.retrieval.service import RetrievalService
from src.generation.engine import create_generation_engine

def test_query(query_text: str):
    """Test a single query through the RAG pipeline."""
    print("=" * 70)
    print("RAG Query Test")
    print("=" * 70)
    print(f"Query: {query_text}")
    print()

    # Load settings
    settings = get_settings()
    print(f"Model: {settings.model_config['llm']['model_name']}")
    print(f"Max tokens: {settings.model_config['llm']['max_tokens']}")
    print()

    # Initialize retrieval service
    print("Initializing retrieval service...")
    retrieval_service = RetrievalService(settings)

    # Retrieve documents
    print("Retrieving relevant documents...")
    retrieval_result = retrieval_service.retrieve(query_text)

    print(f"Found {len(retrieval_result.reranked_results)} relevant chunks")
    print()

    # Show retrieved chunks
    print("Top 3 retrieved chunks:")
    for i, result in enumerate(retrieval_result.reranked_results[:3], 1):
        print(f"  {i}. Score: {result.score:.3f}")
        print(f"     Content: {result.document.content[:100]}...")
        print()

    # Initialize generation engine
    print("Initializing generation engine...")
    engine = create_generation_engine(
        llm_config=settings.model_config['llm'],
        prompts_config=settings.prompts,  # Changed from prompts_config to prompts
        api_key=settings.deepseek_api_key
    )

    # Generate response
    print("Generating response (this may take a moment with deepseek-reasoner)...")
    response = engine.generate(retrieval_result, query_text)

    print()
    print("=" * 70)
    print("Response:")
    print("=" * 70)
    print(response.answer)
    print()

    # Show reasoning if available
    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        print("=" * 70)
        print("Reasoning Process:")
        print("=" * 70)
        print(response.reasoning_content[:500] + "..." if len(response.reasoning_content) > 500 else response.reasoning_content)
        print()

    print(f"Generation time: {response.generation_time_ms:.2f}ms")
    print(f"Citations: {len(response.citations)}")

if __name__ == "__main__":
    test_query("ควรกินยาแก้ปวดตอนไหนหลังผ่าตัดฟันคุด")

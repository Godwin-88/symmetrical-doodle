"""
Test script for the embedding generation service.

This script tests the complete embedding pipeline including content classification,
model selection, embedding generation, and quality validation.
"""

import asyncio
import os
import sys
import logging
from typing import List, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import EmbeddingService
from services.content_classifier import ContentDomain, ContentType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_content_classification():
    """Test content classification functionality"""
    print("\n=== Testing Content Classification ===")
    
    service = EmbeddingService(enable_quality_validation=False)
    
    test_texts = [
        ("Machine learning algorithms for portfolio optimization", "ML/Finance"),
        ("Deep reinforcement learning in trading strategies", "DRL/Finance"),
        ("Natural language processing for financial sentiment analysis", "NLP/Finance"),
        ("Large language models and their applications", "LLMs"),
        ("Mathematical optimization using gradient descent", "Math/General"),
        ("General research methodology and analysis", "General"),
    ]
    
    for text, expected_category in test_texts:
        classification = await service.classifier.classify_content(text)
        print(f"Text: {text[:50]}...")
        print(f"  Domain: {classification.domain.value}")
        print(f"  Type: {classification.content_type.value}")
        print(f"  Confidence: {classification.confidence_score:.3f}")
        print(f"  Expected: {expected_category}")
        print()


async def test_embedding_generation():
    """Test embedding generation with mock OpenAI key"""
    print("\n=== Testing Embedding Generation ===")
    
    # Use a mock API key for testing (will fail but we can test the pipeline)
    service = EmbeddingService(
        openai_api_key="test-key",
        enable_quality_validation=True,
        enable_regeneration=False
    )
    
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Portfolio optimization using modern portfolio theory",
        "The gradient descent algorithm minimizes the loss function"
    ]
    
    for text in test_texts:
        print(f"Processing: {text[:50]}...")
        result = await service.generate_embedding(text)
        
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Model: {result.model_used.value}")
            print(f"  Dimension: {len(result.embedding)}")
            print(f"  Domain: {result.classification.domain.value}")
            print(f"  Processing time: {result.processing_time:.3f}s")
            if result.quality_validation:
                print(f"  Quality score: {result.quality_validation.quality_score:.3f}")
        else:
            print(f"  Error: {result.error_message}")
        print()


async def test_model_availability():
    """Test model availability checking"""
    print("\n=== Testing Model Availability ===")
    
    service = EmbeddingService()
    available_models = await service.get_available_models()
    
    print("Available models:")
    for model in available_models:
        info = service.get_model_info(model)
        print(f"  {model.value}:")
        print(f"    Dimension: {info['dimension']}")
        print(f"    Max tokens: {info['max_tokens']}")
        print(f"    API-based: {info['api_based']}")
        print(f"    Requires GPU: {info['requires_gpu']}")
    
    if not available_models:
        print("  No models available (expected without proper API keys/GPU)")


async def test_quality_validation():
    """Test embedding quality validation"""
    print("\n=== Testing Quality Validation ===")
    
    from services.embedding_quality_validator import EmbeddingQualityValidator
    
    validator = EmbeddingQualityValidator()
    
    # Test with different types of embeddings
    test_embeddings = [
        ([0.1, 0.2, 0.3, 0.4], "Normal embedding"),
        ([0.0, 0.0, 0.0, 0.0], "Zero vector"),
        ([], "Empty embedding"),
        ([float('nan'), 0.1, 0.2], "NaN values"),
        ([1000.0, 2000.0, 3000.0], "High magnitude"),
    ]
    
    for embedding, description in test_embeddings:
        result = await validator.validate_embedding(embedding)
        print(f"{description}:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Quality score: {result.quality_score:.3f}")
        print(f"  Issues: {[issue.value for issue in result.issues]}")
        print(f"  Should regenerate: {result.should_regenerate}")
        print()


async def main():
    """Run all tests"""
    print("Testing Embedding Generation Service")
    print("=" * 50)
    
    try:
        await test_content_classification()
        await test_model_availability()
        await test_quality_validation()
        await test_embedding_generation()
        
        print("\n=== Test Summary ===")
        print("All tests completed. Check output above for results.")
        print("Note: Embedding generation may fail without proper API keys.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
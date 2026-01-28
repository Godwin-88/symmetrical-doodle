#!/usr/bin/env python3
"""
Test script for semantic chunker functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.semantic_chunker import SemanticChunkingService, ChunkType
from services.pdf_parser import ParsedDocument, DocumentStructure, MathElement
from core.config import get_settings


async def test_semantic_chunker():
    """Test the semantic chunking service with sample content."""
    
    print("Testing Semantic Chunking Service...")
    
    # Create test document
    test_content = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

## Mathematical Foundations

The core of machine learning relies on mathematical concepts. For example, linear regression uses the formula:

$$y = mx + b$$

Where $y$ is the dependent variable, $x$ is the independent variable, $m$ is the slope, and $b$ is the y-intercept.

## Types of Learning

There are three main types of machine learning:

1. Supervised Learning
2. Unsupervised Learning  
3. Reinforcement Learning

### Supervised Learning

Supervised learning uses labeled training data to learn a mapping function from input to output.

### Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labeled examples.

## Conclusion

Machine learning continues to evolve and find applications across many domains.
"""
    
    # Create mock document structure
    document_structure = DocumentStructure(
        sections=[
            {"title": "Introduction to Machine Learning", "level": 1, "start_pos": 0, "end_pos": 100},
            {"title": "Mathematical Foundations", "level": 2, "start_pos": 100, "end_pos": 400},
            {"title": "Types of Learning", "level": 2, "start_pos": 400, "end_pos": 600},
            {"title": "Supervised Learning", "level": 3, "start_pos": 600, "end_pos": 700},
            {"title": "Unsupervised Learning", "level": 3, "start_pos": 700, "end_pos": 800},
            {"title": "Conclusion", "level": 2, "start_pos": 800, "end_pos": len(test_content)}
        ],
        tables=[],
        images=[],
        metadata={}
    )
    
    # Create test document
    test_document = ParsedDocument(
        source_file_id="test_doc_001",
        title="Machine Learning Basics",
        content=test_content,
        structure=document_structure,
        math_notation=[
            MathElement(
                content="$$y = mx + b$$",
                math_type="display",
                start_pos=test_content.find("$$y = mx + b$$"),
                end_pos=test_content.find("$$y = mx + b$$") + len("$$y = mx + b$$")
            )
        ],
        tables=[],
        images=[],
        parsing_method="test",
        quality_score=1.0
    )
    
    # Initialize chunking service
    chunker = SemanticChunkingService()
    
    # Chunk the document
    chunks = await chunker.chunk_document(test_document)
    
    # Display results
    print(f"\nDocument chunked into {len(chunks)} chunks:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Type: {chunk.chunk_type.value}")
        print(f"  Section: {chunk.section_header}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Has Math: {chunk.semantic_metadata.get('has_math', False)}")
        print(f"  Math Elements: {len(chunk.math_elements)}")
        print(f"  Content Preview: {chunk.content[:100]}...")
        
        if chunk.math_elements:
            print(f"  Math Content: {[elem.content for elem in chunk.math_elements]}")
    
    # Test validation
    print(f"\nValidation Results:")
    print(f"  All chunks valid: {all(chunker._validate_chunk(chunk) for chunk in chunks)}")
    print(f"  Total tokens: {sum(chunk.token_count for chunk in chunks)}")
    print(f"  Chunk types: {set(chunk.chunk_type.value for chunk in chunks)}")
    
    return chunks


async def test_math_preservation():
    """Test mathematical notation preservation."""
    
    print("\n" + "=" * 60)
    print("Testing Mathematical Notation Preservation...")
    
    math_content = """
# Mathematical Formulas

Here are some important formulas:

Inline math: The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

Display math:
$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

LaTeX environment:
\\begin{equation}
E = mc^2
\\end{equation}

Matrix notation:
$$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$
"""
    
    # Create test document with math
    math_document = ParsedDocument(
        source_file_id="math_doc_001",
        title="Mathematical Formulas",
        content=math_content,
        structure=DocumentStructure(sections=[], tables=[], images=[], metadata={}),
        math_notation=[],
        tables=[],
        images=[],
        parsing_method="test",
        quality_score=1.0
    )
    
    chunker = SemanticChunkingService()
    chunks = await chunker.chunk_document(math_document)
    
    print(f"\nMath document chunked into {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Has Math: {chunk.semantic_metadata.get('has_math', False)}")
        print(f"  Math Elements: {len(chunk.math_elements)}")
        if chunk.math_elements:
            for elem in chunk.math_elements:
                print(f"    - {elem.math_type}: {elem.content}")
        print(f"  Content: {chunk.content}")
    
    return chunks


async def main():
    """Main test function."""
    try:
        # Test basic chunking
        basic_chunks = await test_semantic_chunker()
        
        # Test math preservation
        math_chunks = await test_math_preservation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print(f"Basic chunking: {len(basic_chunks)} chunks")
        print(f"Math chunking: {len(math_chunks)} chunks")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
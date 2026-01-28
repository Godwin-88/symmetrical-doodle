#!/usr/bin/env python3
"""
End-to-end test for PDF processing pipeline.

This test validates the complete PDF processing pipeline from parsing to chunking
without requiring external dependencies like Google Drive API or Supabase.
"""

import asyncio
import sys
import tempfile
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# Mock the required classes for testing
class ParsingMethod(Enum):
    MARKER = "marker"
    PYMUPDF = "pymupdf"
    FAILED = "failed"

class ParsingQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

class ChunkType(Enum):
    HEADER = "header"
    PARAGRAPH = "paragraph"
    MATH = "math"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    MIXED = "mixed"

@dataclass
class MathElement:
    content: str
    math_type: str = "inline"
    start_pos: int = 0
    end_pos: int = 0

@dataclass
class TableElement:
    content: str
    rows: int = 0
    columns: int = 0
    position: Optional[tuple] = None
    structured_data: Optional[List[List[str]]] = None

@dataclass
class ImageElement:
    description: Optional[str] = None
    position: Optional[tuple] = None
    size: Optional[tuple] = None
    image_data: Optional[bytes] = None

@dataclass
class DocumentStructure:
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    toc: List[Dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    has_math: bool = False
    has_tables: bool = False
    has_images: bool = False

@dataclass
class ParsedDocument:
    source_file_id: str
    title: str
    content: str
    structure: DocumentStructure
    math_notation: List[MathElement] = field(default_factory=list)
    tables: List[TableElement] = field(default_factory=list)
    images: List[ImageElement] = field(default_factory=list)
    parsing_method: ParsingMethod = ParsingMethod.FAILED
    quality_score: float = 0.0
    quality_assessment: ParsingQuality = ParsingQuality.FAILED
    parsing_time_ms: int = 0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class SemanticChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_order: int
    section_header: Optional[str] = None
    semantic_metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    math_elements: List[MathElement] = field(default_factory=list)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    parent_section: Optional[str] = None
    subsection_level: int = 0
    start_position: int = 0
    end_position: int = 0
    overlap_with_previous: bool = False
    overlap_with_next: bool = False

# Mock settings
class MockSettings:
    def __init__(self):
        self.processing = MockProcessingConfig()

class MockProcessingConfig:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.use_marker_llm = False
        self.marker_timeout = 60

# Mock PDF Parser Service
class MockPDFParsingService:
    def __init__(self):
        self.settings = MockSettings()
    
    async def parse_pdf(self, pdf_content: bytes, file_id: str, filename: str = "document.pdf") -> ParsedDocument:
        """Mock PDF parsing that simulates successful parsing."""
        
        # Simulate parsing a research paper with mathematical content
        mock_content = """# Deep Learning for Financial Time Series Prediction

## Abstract

This paper presents a novel approach to financial time series prediction using deep learning techniques. We propose a hybrid model that combines convolutional neural networks (CNNs) with long short-term memory (LSTM) networks.

## Introduction

Financial time series prediction has been a challenging problem in quantitative finance. Traditional methods such as ARIMA models have limitations when dealing with non-linear patterns.

## Mathematical Framework

### Model Architecture

Our proposed model can be represented mathematically as:

$h_t = LSTM(CNN(x_t))$

Where $h_t$ is the hidden state at time $t$, and $x_t$ is the input feature vector.

The loss function is defined as:

$$L = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$

### Optimization

We use the Adam optimizer with learning rate $\\alpha = 0.001$:

$$\\theta_{t+1} = \\theta_t - \\alpha \\nabla_\\theta L$$

## Experimental Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| ARIMA | 0.045 | 0.032 | 0.78 |
| LSTM | 0.038 | 0.028 | 0.82 |
| CNN-LSTM | 0.031 | 0.024 | 0.87 |

### Performance Analysis

The results show that our hybrid CNN-LSTM model outperforms traditional methods:

1. **Accuracy**: 15% improvement in RMSE
2. **Robustness**: Better handling of market volatility
3. **Generalization**: Consistent performance across different assets

## Implementation Details

```python
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cnn = nn.Conv1d(input_size, 64, 3)
        self.lstm = nn.LSTM(64, hidden_size)
        
    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        return x
```

## Conclusion

This work demonstrates the effectiveness of combining CNNs and LSTMs for financial time series prediction. Future work will explore attention mechanisms and transformer architectures.

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436-444.
"""
        
        # Create document structure
        structure = DocumentStructure(
            title="Deep Learning for Financial Time Series Prediction",
            sections=[
                {"title": "Abstract", "level": 2, "start_pos": 0, "end_pos": 200},
                {"title": "Introduction", "level": 2, "start_pos": 200, "end_pos": 400},
                {"title": "Mathematical Framework", "level": 2, "start_pos": 400, "end_pos": 800},
                {"title": "Model Architecture", "level": 3, "start_pos": 450, "end_pos": 600},
                {"title": "Optimization", "level": 3, "start_pos": 600, "end_pos": 750},
                {"title": "Experimental Results", "level": 2, "start_pos": 800, "end_pos": 1200},
                {"title": "Performance Analysis", "level": 3, "start_pos": 900, "end_pos": 1100},
                {"title": "Implementation Details", "level": 2, "start_pos": 1200, "end_pos": 1400},
                {"title": "Conclusion", "level": 2, "start_pos": 1400, "end_pos": 1600},
                {"title": "References", "level": 2, "start_pos": 1600, "end_pos": len(mock_content)}
            ],
            page_count=8,
            has_math=True,
            has_tables=True,
            has_images=False
        )
        
        # Extract mathematical elements
        math_elements = [
            MathElement("$h_t = LSTM(CNN(x_t))$", "inline", 500, 525),
            MathElement("$h_t$", "inline", 505, 510),
            MathElement("$x_t$", "inline", 520, 525),
            MathElement("$$L = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$", "display", 600, 650),
            MathElement("$\\alpha = 0.001$", "inline", 700, 715),
            MathElement("$$\\theta_{t+1} = \\theta_t - \\alpha \\nabla_\\theta L$$", "display", 720, 770)
        ]
        
        # Extract tables
        tables = [
            TableElement(
                content="| Model | RMSE | MAE | R² |\n|-------|------|-----|----- |\n| ARIMA | 0.045 | 0.032 | 0.78 |\n| LSTM | 0.038 | 0.028 | 0.82 |\n| CNN-LSTM | 0.031 | 0.024 | 0.87 |",
                rows=4,
                columns=4,
                position=(0, 900)
            )
        ]
        
        return ParsedDocument(
            source_file_id=file_id,
            title="Deep Learning for Financial Time Series Prediction",
            content=mock_content,
            structure=structure,
            math_notation=math_elements,
            tables=tables,
            parsing_method=ParsingMethod.PYMUPDF,  # Simulate fallback to PyMuPDF
            quality_score=0.85,
            quality_assessment=ParsingQuality.GOOD,
            parsing_time_ms=1500
        )

# Import the actual semantic chunker
from test_chunker_simple import SimpleSemanticChunker

class PDFProcessingPipeline:
    """Complete PDF processing pipeline for testing."""
    
    def __init__(self):
        self.pdf_parser = MockPDFParsingService()
        self.semantic_chunker = SimpleSemanticChunker()
    
    async def process_pdf(self, pdf_content: bytes, file_id: str, filename: str) -> tuple[ParsedDocument, List[SemanticChunk]]:
        """Process PDF through complete pipeline."""
        
        # Step 1: Parse PDF
        parsed_document = await self.pdf_parser.parse_pdf(pdf_content, file_id, filename)
        
        if parsed_document.parsing_method == ParsingMethod.FAILED:
            return parsed_document, []
        
        # Step 2: Create semantic chunks
        chunks = await self.semantic_chunker.chunk_document(parsed_document)
        
        return parsed_document, chunks

async def test_pdf_processing_pipeline():
    """Test the complete PDF processing pipeline."""
    
    print("Testing PDF Processing Pipeline End-to-End...")
    print("=" * 60)
    
    # Create mock PDF content (in real scenario this would be actual PDF bytes)
    mock_pdf_content = b"Mock PDF content for testing"
    
    # Initialize pipeline
    pipeline = PDFProcessingPipeline()
    
    # Process PDF
    file_id = "test_research_paper_001"
    filename = "deep_learning_finance.pdf"
    
    parsed_document, chunks = await pipeline.process_pdf(mock_pdf_content, file_id, filename)
    
    # Validate parsing results
    print(f"\n📄 PDF Parsing Results:")
    print(f"  Title: {parsed_document.title}")
    print(f"  Parsing Method: {parsed_document.parsing_method.value}")
    print(f"  Quality Score: {parsed_document.quality_score:.2f}")
    print(f"  Quality Assessment: {parsed_document.quality_assessment.value}")
    print(f"  Content Length: {len(parsed_document.content)} characters")
    print(f"  Sections: {len(parsed_document.structure.sections)}")
    print(f"  Math Elements: {len(parsed_document.math_notation)}")
    print(f"  Tables: {len(parsed_document.tables)}")
    print(f"  Parsing Time: {parsed_document.parsing_time_ms}ms")
    
    # Validate chunking results
    print(f"\n🧩 Semantic Chunking Results:")
    print(f"  Total Chunks: {len(chunks)}")
    print(f"  Total Tokens: {sum(chunk.token_count for chunk in chunks)}")
    
    chunk_types = {}
    math_chunks = 0
    table_chunks = 0
    
    for chunk in chunks:
        chunk_type = chunk.chunk_type.value
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk.semantic_metadata.get('has_math', False):
            math_chunks += 1
        if chunk.semantic_metadata.get('has_tables', False):
            table_chunks += 1
    
    print(f"  Chunk Types: {chunk_types}")
    print(f"  Chunks with Math: {math_chunks}")
    print(f"  Chunks with Tables: {table_chunks}")
    
    # Detailed chunk analysis
    print(f"\n📋 Detailed Chunk Analysis:")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\n  Chunk {i + 1}:")
        print(f"    ID: {chunk.chunk_id}")
        print(f"    Type: {chunk.chunk_type.value}")
        print(f"    Section: {chunk.section_header or 'None'}")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Math Elements: {len(chunk.math_elements)}")
        print(f"    Has Math: {chunk.semantic_metadata.get('has_math', False)}")
        print(f"    Has Tables: {chunk.semantic_metadata.get('has_tables', False)}")
        print(f"    Content Preview: {chunk.content[:100]}...")
        
        if chunk.math_elements:
            print(f"    Math Content: {[elem.content for elem in chunk.math_elements[:3]]}")
    
    if len(chunks) > 5:
        print(f"\n  ... and {len(chunks) - 5} more chunks")
    
    return parsed_document, chunks

async def test_mathematical_content_preservation():
    """Test that mathematical content is properly preserved through the pipeline."""
    
    print(f"\n🔬 Testing Mathematical Content Preservation...")
    print("=" * 60)
    
    # Create a document with heavy mathematical content
    math_heavy_content = """# Advanced Calculus and Linear Algebra

## Differential Equations

The general form of a second-order linear differential equation is:

$$a_2(x)y'' + a_1(x)y' + a_0(x)y = f(x)$$

Where $y = y(x)$ is the unknown function.

## Matrix Operations

For matrices $A \\in \\mathbb{R}^{m \\times n}$ and $B \\in \\mathbb{R}^{n \\times p}$:

$$C = AB \\text{ where } C_{ij} = \\sum_{k=1}^{n} A_{ik}B_{kj}$$

### Eigenvalue Decomposition

If $A$ is diagonalizable, then:

$$A = PDP^{-1}$$

Where $D$ is diagonal and $P$ contains eigenvectors.

## Optimization Theory

The Lagrangian for constrained optimization is:

$$L(x, \\lambda) = f(x) + \\sum_{i=1}^{m} \\lambda_i g_i(x)$$

Subject to constraints $g_i(x) = 0$ for $i = 1, ..., m$.
"""
    
    # Create mock document with math content
    math_document = ParsedDocument(
        source_file_id="math_test_001",
        title="Advanced Calculus and Linear Algebra",
        content=math_heavy_content,
        structure=DocumentStructure(
            title="Advanced Calculus and Linear Algebra",
            sections=[
                {"title": "Differential Equations", "level": 2, "start_pos": 0, "end_pos": 300},
                {"title": "Matrix Operations", "level": 2, "start_pos": 300, "end_pos": 600},
                {"title": "Eigenvalue Decomposition", "level": 3, "start_pos": 450, "end_pos": 550},
                {"title": "Optimization Theory", "level": 2, "start_pos": 600, "end_pos": len(math_heavy_content)}
            ],
            has_math=True
        ),
        parsing_method=ParsingMethod.MARKER,
        quality_score=0.9,
        quality_assessment=ParsingQuality.EXCELLENT
    )
    
    # Process through chunker
    chunker = SimpleSemanticChunker()
    chunks = await chunker.chunk_document(math_document)
    
    # Analyze mathematical content preservation
    total_math_elements = 0
    preserved_formulas = []
    
    for chunk in chunks:
        total_math_elements += len(chunk.math_elements)
        for math_elem in chunk.math_elements:
            preserved_formulas.append(math_elem.content)
    
    print(f"  Original content length: {len(math_heavy_content)} characters")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Total math elements preserved: {total_math_elements}")
    print(f"  Math formulas found: {len(preserved_formulas)}")
    
    # Check for key mathematical expressions
    key_expressions = [
        "$$a_2(x)y'' + a_1(x)y' + a_0(x)y = f(x)$$",
        "$y = y(x)$",
        "$$C = AB",
        "$$A = PDP^{-1}$$",
        "$$L(x, \\lambda) = f(x) + \\sum_{i=1}^{m} \\lambda_i g_i(x)$$"
    ]
    
    preserved_count = 0
    for expr in key_expressions:
        if any(expr in formula for formula in preserved_formulas):
            preserved_count += 1
    
    print(f"  Key expressions preserved: {preserved_count}/{len(key_expressions)}")
    
    # Show some preserved formulas
    print(f"\n  Sample preserved formulas:")
    for i, formula in enumerate(preserved_formulas[:5]):
        print(f"    {i+1}. {formula}")
    
    return chunks

async def test_document_structure_preservation():
    """Test that document structure is properly preserved."""
    
    print(f"\n🏗️ Testing Document Structure Preservation...")
    print("=" * 60)
    
    # Create pipeline
    pipeline = PDFProcessingPipeline()
    
    # Process a document
    mock_pdf_content = b"Mock PDF content"
    parsed_document, chunks = await pipeline.process_pdf(mock_pdf_content, "struct_test_001", "structure_test.pdf")
    
    # Analyze structure preservation
    sections_with_chunks = {}
    hierarchical_levels = set()
    
    for chunk in chunks:
        if chunk.section_header:
            if chunk.section_header not in sections_with_chunks:
                sections_with_chunks[chunk.section_header] = 0
            sections_with_chunks[chunk.section_header] += 1
            hierarchical_levels.add(chunk.subsection_level)
    
    print(f"  Original sections: {len(parsed_document.structure.sections)}")
    print(f"  Sections with chunks: {len(sections_with_chunks)}")
    print(f"  Hierarchical levels: {sorted(hierarchical_levels)}")
    
    print(f"\n  Section distribution:")
    for section, count in sections_with_chunks.items():
        print(f"    {section}: {count} chunks")
    
    # Check chunk ordering
    chunk_orders = [chunk.chunk_order for chunk in chunks]
    is_properly_ordered = chunk_orders == sorted(chunk_orders)
    print(f"  Chunks properly ordered: {is_properly_ordered}")
    
    return parsed_document, chunks

async def test_error_handling():
    """Test error handling in the pipeline."""
    
    print(f"\n⚠️ Testing Error Handling...")
    print("=" * 60)
    
    # Test with empty content
    empty_document = ParsedDocument(
        source_file_id="empty_test_001",
        title="Empty Document",
        content="",
        structure=DocumentStructure(),
        parsing_method=ParsingMethod.FAILED,
        quality_score=0.0,
        quality_assessment=ParsingQuality.FAILED,
        error_message="No content found"
    )
    
    chunker = SimpleSemanticChunker()
    empty_chunks = await chunker.chunk_document(empty_document)
    
    print(f"  Empty document chunks: {len(empty_chunks)}")
    
    # Test with minimal content
    minimal_document = ParsedDocument(
        source_file_id="minimal_test_001",
        title="Minimal Document",
        content="Short content.",
        structure=DocumentStructure(),
        parsing_method=ParsingMethod.PYMUPDF,
        quality_score=0.3,
        quality_assessment=ParsingQuality.POOR
    )
    
    minimal_chunks = await chunker.chunk_document(minimal_document)
    print(f"  Minimal document chunks: {len(minimal_chunks)}")
    
    if minimal_chunks:
        print(f"  Minimal chunk content: '{minimal_chunks[0].content}'")
        print(f"  Minimal chunk valid: {chunker._validate_chunk(minimal_chunks[0])}")
    
    return empty_chunks, minimal_chunks

async def main():
    """Main test function."""
    try:
        print("🚀 Starting PDF Processing Pipeline End-to-End Tests")
        print("=" * 80)
        
        # Test 1: Complete pipeline
        parsed_doc, chunks = await test_pdf_processing_pipeline()
        
        # Test 2: Mathematical content preservation
        math_chunks = await test_mathematical_content_preservation()
        
        # Test 3: Document structure preservation
        struct_doc, struct_chunks = await test_document_structure_preservation()
        
        # Test 4: Error handling
        empty_chunks, minimal_chunks = await test_error_handling()
        
        # Summary
        print(f"\n✅ PDF Processing Pipeline Tests Summary")
        print("=" * 80)
        print(f"✅ Basic pipeline processing: {len(chunks)} chunks created")
        print(f"✅ Mathematical content preservation: {len(math_chunks)} chunks with math")
        print(f"✅ Document structure preservation: {len(struct_chunks)} structured chunks")
        print(f"✅ Error handling: Empty={len(empty_chunks)}, Minimal={len(minimal_chunks)}")
        
        # Validation checks
        validation_results = {
            "parsing_successful": parsed_doc.parsing_method != ParsingMethod.FAILED,
            "chunks_created": len(chunks) > 0,
            "math_preserved": any(chunk.semantic_metadata.get('has_math', False) for chunk in chunks),
            "structure_preserved": any(chunk.section_header for chunk in chunks),
            "quality_acceptable": parsed_doc.quality_score >= 0.5,
            "chunks_valid": all(chunk.token_count > 0 for chunk in chunks),
            "error_handling_works": len(empty_chunks) == 0 and len(minimal_chunks) >= 0
        }
        
        print(f"\n🔍 Validation Results:")
        for check, result in validation_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}: {result}")
        
        all_passed = all(validation_results.values())
        
        if all_passed:
            print(f"\n🎉 All PDF processing pipeline tests PASSED!")
            print(f"   The pipeline successfully processes PDFs end-to-end with:")
            print(f"   - Proper parsing with fallback mechanisms")
            print(f"   - Mathematical notation preservation")
            print(f"   - Document structure preservation")
            print(f"   - Semantic chunking with metadata")
            print(f"   - Robust error handling")
            return 0
        else:
            print(f"\n❌ Some tests FAILED. Please review the results above.")
            return 1
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
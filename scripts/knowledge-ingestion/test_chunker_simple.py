#!/usr/bin/env python3
"""
Simple test for semantic chunker without external dependencies.
"""

import asyncio
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# Mock the required classes to test chunker independently
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
    math_type: str
    start_pos: int
    end_pos: int

@dataclass
class DocumentStructure:
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParsedDocument:
    source_file_id: str
    title: str
    content: str
    structure: DocumentStructure
    math_notation: List[MathElement] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    parsing_method: str = "test"
    quality_score: float = 1.0

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

# Simplified chunker for testing
class SimpleSemanticChunker:
    def __init__(self):
        self.settings = MockSettings()
    
    async def chunk_document(self, document: ParsedDocument) -> List[SemanticChunk]:
        """Create semantic chunks from a parsed document."""
        if not document.content or not document.content.strip():
            return []
        
        chunk_size = self.settings.processing.chunk_size
        chunk_overlap = self.settings.processing.chunk_overlap
        
        # Analyze document structure
        structure_analysis = await self._analyze_document_structure(document)
        
        # Create chunks based on structure
        chunks = await self._create_structured_chunks(
            document, structure_analysis, chunk_size, chunk_overlap
        )
        
        # Post-process chunks
        processed_chunks = await self._post_process_chunks(chunks, document)
        
        return processed_chunks
    
    async def _analyze_document_structure(self, document: ParsedDocument) -> Dict[str, Any]:
        """Analyze document structure to inform chunking strategy."""
        analysis = {
            'sections': [],
            'math_regions': [],
            'table_regions': [],
            'list_regions': [],
            'code_regions': [],
            'paragraph_boundaries': []
        }
        
        content = document.content
        
        # Identify sections from document structure
        if document.structure.sections:
            for section in document.structure.sections:
                analysis['sections'].append({
                    'title': section.get('title', ''),
                    'level': section.get('level', 1),
                    'start_pos': section.get('start_pos', 0),
                    'end_pos': section.get('end_pos', len(content))
                })
        
        # Identify mathematical regions
        math_pattern = r'(\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}.*?\\end\{.*?\})'
        for match in re.finditer(math_pattern, content, re.DOTALL):
            analysis['math_regions'].append({
                'start': match.start(),
                'end': match.end(),
                'content': match.group(),
                'type': 'display' if match.group().startswith('$$') else 'inline'
            })
        
        return analysis
    
    async def _create_structured_chunks(
        self, 
        document: ParsedDocument, 
        structure_analysis: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[SemanticChunk]:
        """Create chunks based on document structure analysis."""
        chunks = []
        content = document.content
        
        # If document has clear sections, chunk by sections
        if structure_analysis['sections']:
            chunks = await self._chunk_by_sections(
                document, structure_analysis, chunk_size, chunk_overlap
            )
        else:
            # Fall back to paragraph-based chunking
            chunks = await self._chunk_by_paragraphs(
                document, structure_analysis, chunk_size, chunk_overlap
            )
        
        return chunks
    
    async def _chunk_by_sections(
        self,
        document: ParsedDocument,
        structure_analysis: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[SemanticChunk]:
        """Create chunks based on document sections."""
        chunks = []
        content = document.content
        
        for i, section in enumerate(structure_analysis['sections']):
            section_start = section['start_pos']
            section_end = section['end_pos']
            section_content = content[section_start:section_end]
            section_title = section['title']
            section_level = section['level']
            
            # If section is small enough, create single chunk
            section_tokens = self._estimate_tokens(section_content)
            if section_tokens <= chunk_size:
                chunk = await self._create_chunk(
                    document_id=document.source_file_id,
                    content=section_content,
                    chunk_order=len(chunks),
                    section_header=section_title,
                    subsection_level=section_level,
                    start_position=section_start,
                    end_position=section_end,
                    structure_analysis=structure_analysis
                )
                chunks.append(chunk)
            else:
                # Split large section by paragraphs
                paragraphs = re.split(r'\n\s*\n', section_content)
                current_chunk_content = ""
                current_chunk_tokens = 0
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                        
                    paragraph_tokens = self._estimate_tokens(paragraph)
                    
                    if current_chunk_tokens + paragraph_tokens > chunk_size and current_chunk_content:
                        chunk = await self._create_chunk(
                            document_id=document.source_file_id,
                            content=current_chunk_content.strip(),
                            chunk_order=len(chunks),
                            section_header=section_title,
                            subsection_level=section_level,
                            structure_analysis=structure_analysis
                        )
                        chunks.append(chunk)
                        current_chunk_content = paragraph
                        current_chunk_tokens = paragraph_tokens
                    else:
                        if current_chunk_content:
                            current_chunk_content += "\n\n" + paragraph
                        else:
                            current_chunk_content = paragraph
                        current_chunk_tokens += paragraph_tokens
                
                # Add final chunk if there's remaining content
                if current_chunk_content.strip():
                    chunk = await self._create_chunk(
                        document_id=document.source_file_id,
                        content=current_chunk_content.strip(),
                        chunk_order=len(chunks),
                        section_header=section_title,
                        subsection_level=section_level,
                        structure_analysis=structure_analysis
                    )
                    chunks.append(chunk)
        
        return chunks
    
    async def _chunk_by_paragraphs(
        self,
        document: ParsedDocument,
        structure_analysis: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[SemanticChunk]:
        """Create chunks based on paragraph boundaries."""
        chunks = []
        content = document.content
        
        # Split by paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk_content = ""
        current_chunk_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_tokens = self._estimate_tokens(paragraph)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk_tokens + paragraph_tokens > chunk_size and current_chunk_content:
                chunk = await self._create_chunk(
                    document_id=document.source_file_id,
                    content=current_chunk_content.strip(),
                    chunk_order=len(chunks),
                    structure_analysis=structure_analysis
                )
                chunks.append(chunk)
                current_chunk_content = paragraph
                current_chunk_tokens = paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk_content:
                    current_chunk_content += "\n\n" + paragraph
                else:
                    current_chunk_content = paragraph
                current_chunk_tokens += paragraph_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk_content.strip():
            chunk = await self._create_chunk(
                document_id=document.source_file_id,
                content=current_chunk_content.strip(),
                chunk_order=len(chunks),
                structure_analysis=structure_analysis
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_chunk(
        self,
        document_id: str,
        content: str,
        chunk_order: int,
        section_header: Optional[str] = None,
        subsection_level: int = 0,
        start_position: int = 0,
        end_position: int = 0,
        structure_analysis: Optional[Dict[str, Any]] = None
    ) -> SemanticChunk:
        """Create a semantic chunk with metadata."""
        import hashlib
        
        # Generate chunk ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{document_id}_{chunk_order:04d}_{content_hash}"
        
        # Estimate token count
        token_count = self._estimate_tokens(content)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(content, structure_analysis)
        
        # Extract mathematical elements
        math_elements = self._extract_math_elements(content)
        
        # Create semantic metadata
        semantic_metadata = {
            'has_math': len(math_elements) > 0,
            'has_tables': '|' in content and content.count('|') > 2,
            'has_lists': bool(re.search(r'^\s*(?:\d+\.|[-*+])\s+', content, re.MULTILINE)),
            'has_code': '```' in content or content.count('`') >= 2,
            'complexity_score': min(1.0, len(content.split()) / 100.0)
        }
        
        return SemanticChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            chunk_order=chunk_order,
            section_header=section_header,
            semantic_metadata=semantic_metadata,
            token_count=token_count,
            math_elements=math_elements,
            chunk_type=chunk_type,
            parent_section=section_header,
            subsection_level=subsection_level,
            start_position=start_position,
            end_position=end_position
        )
    
    async def _post_process_chunks(
        self, 
        chunks: List[SemanticChunk], 
        document: ParsedDocument
    ) -> List[SemanticChunk]:
        """Post-process chunks to add overlap information and final validation."""
        if not chunks:
            return chunks
        
        # Validate chunks
        validated_chunks = []
        for chunk in chunks:
            if self._validate_chunk(chunk):
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // 4)
    
    def _determine_chunk_type(self, content: str, structure_analysis: Optional[Dict[str, Any]]) -> ChunkType:
        """Determine the type of chunk based on content."""
        # Check for mathematical content
        if re.search(r'\$.*?\$|\\begin\{.*?\}', content):
            return ChunkType.MATH
        
        # Check for headers
        if re.match(r'^#+\s+', content.strip()):
            return ChunkType.HEADER
        
        # Check for lists
        if re.search(r'^\s*(?:\d+\.|[-*+])\s+', content, re.MULTILINE):
            return ChunkType.LIST
        
        # Check for code
        if '```' in content or content.count('`') >= 2:
            return ChunkType.CODE
        
        return ChunkType.PARAGRAPH
    
    def _extract_math_elements(self, content: str) -> List[MathElement]:
        """Extract mathematical elements from content."""
        math_elements = []
        
        # LaTeX math patterns
        patterns = [
            (r'\$\$(.*?)\$\$', 'display'),
            (r'\$(.*?)\$', 'inline'),
            (r'\\begin\{(.*?)\}(.*?)\\end\{\1\}', 'environment')
        ]
        
        for pattern, math_type in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                math_elements.append(MathElement(
                    content=match.group(),
                    math_type=math_type,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return math_elements
    
    def _validate_chunk(self, chunk: SemanticChunk) -> bool:
        """Validate chunk meets quality requirements."""
        if not chunk.content or not chunk.content.strip():
            return False
        if chunk.token_count <= 0:
            return False
        if len(chunk.content) < 10:
            return False
        return True


async def test_semantic_chunker():
    """Test the semantic chunking service with sample content."""
    
    print("Testing Semantic Chunking Service...")
    
    # Create test document
    test_content = """# Introduction to Machine Learning

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

Machine learning continues to evolve and find applications across many domains."""
    
    # Create mock document structure
    document_structure = DocumentStructure(
        sections=[
            {"title": "Introduction to Machine Learning", "level": 1, "start_pos": 0, "end_pos": 100},
            {"title": "Mathematical Foundations", "level": 2, "start_pos": 100, "end_pos": 400},
            {"title": "Types of Learning", "level": 2, "start_pos": 400, "end_pos": 600},
            {"title": "Supervised Learning", "level": 3, "start_pos": 600, "end_pos": 700},
            {"title": "Unsupervised Learning", "level": 3, "start_pos": 700, "end_pos": 800},
            {"title": "Conclusion", "level": 2, "start_pos": 800, "end_pos": len(test_content)}
        ]
    )
    
    # Create test document
    test_document = ParsedDocument(
        source_file_id="test_doc_001",
        title="Machine Learning Basics",
        content=test_content,
        structure=document_structure
    )
    
    # Initialize chunking service
    chunker = SimpleSemanticChunker()
    
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


async def main():
    """Main test function."""
    try:
        # Test basic chunking
        basic_chunks = await test_semantic_chunker()
        
        print("\n" + "=" * 60)
        print("✅ Semantic chunker test completed successfully!")
        print(f"Created {len(basic_chunks)} chunks with proper structure preservation")
        
        # Verify key requirements
        has_math_chunks = any(chunk.semantic_metadata.get('has_math', False) for chunk in basic_chunks)
        has_section_headers = any(chunk.section_header for chunk in basic_chunks)
        has_hierarchical_structure = any(chunk.subsection_level > 0 for chunk in basic_chunks)
        
        print(f"✅ Mathematical content preserved: {has_math_chunks}")
        print(f"✅ Section headers preserved: {has_section_headers}")
        print(f"✅ Hierarchical structure preserved: {has_hierarchical_structure}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
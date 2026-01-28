"""
PDF Parsing Service with Marker and PyMuPDF Fallback

Integrates marker parser as primary parsing method, implements pymupdf fallback
for marker failures, adds parsing quality assessment and logging, and preserves
document structure and hierarchy.

Requirements: 2.2, 2.5
"""

import asyncio
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import re

import fitz  # PyMuPDF
from PIL import Image
import io

from ..core.config import get_settings
from ..core.logging import get_logger


class ParsingMethod(Enum):
    """PDF parsing method used"""
    MARKER = "marker"
    PYMUPDF = "pymupdf"
    FAILED = "failed"


class ParsingQuality(Enum):
    """Quality assessment of parsing result"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class MathElement:
    """Mathematical notation element"""
    content: str
    latex: Optional[str] = None
    position: Optional[Tuple[int, int]] = None
    confidence: float = 0.0


@dataclass
class TableElement:
    """Table structure element"""
    content: str
    rows: int = 0
    columns: int = 0
    position: Optional[Tuple[int, int]] = None
    structured_data: Optional[List[List[str]]] = None


@dataclass
class ImageElement:
    """Image element in document"""
    description: Optional[str] = None
    position: Optional[Tuple[int, int]] = None
    size: Optional[Tuple[int, int]] = None
    image_data: Optional[bytes] = None


@dataclass
class DocumentStructure:
    """Document hierarchical structure"""
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    toc: List[Dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    has_math: bool = False
    has_tables: bool = False
    has_images: bool = False


@dataclass
class ParsedDocument:
    """Complete parsed document with metadata"""
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


class PDFParsingService:
    """
    PDF parsing service with marker primary parser and PyMuPDF fallback.
    Includes quality assessment and document structure preservation.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
    async def parse_pdf(self, pdf_content: bytes, file_id: str, filename: str = "document.pdf") -> ParsedDocument:
        """
        Parse PDF content using marker with PyMuPDF fallback.
        
        Args:
            pdf_content: PDF file content as bytes
            file_id: Unique identifier for the source file
            filename: Original filename for logging
            
        Returns:
            ParsedDocument with parsing results
        """
        start_time = time.time()
        
        self.logger.info(f"Starting PDF parsing for: {filename} (ID: {file_id})")
        
        # Create temporary file for parsing
        temp_file = None
        try:
            temp_file = await self._create_temp_file(pdf_content, filename)
            if not temp_file:
                return ParsedDocument(
                    source_file_id=file_id,
                    title=filename,
                    content="",
                    structure=DocumentStructure(),
                    error_message="Failed to create temporary file"
                )
            
            # Try marker parser first
            marker_result = await self._parse_with_marker(temp_file, file_id, filename)
            
            if marker_result.parsing_method == ParsingMethod.MARKER and marker_result.quality_score >= 0.5:
                # Marker parsing successful with acceptable quality
                parsing_time_ms = int((time.time() - start_time) * 1000)
                marker_result.parsing_time_ms = parsing_time_ms
                
                self.logger.info(
                    f"Successfully parsed with marker: {filename} "
                    f"(quality: {marker_result.quality_assessment.value}, time: {parsing_time_ms}ms)"
                )
                return marker_result
            
            # Marker failed or poor quality, try PyMuPDF fallback
            self.logger.warning(
                f"Marker parsing failed or poor quality for {filename}, "
                f"falling back to PyMuPDF. Marker error: {marker_result.error_message}"
            )
            
            pymupdf_result = await self._parse_with_pymupdf(temp_file, file_id, filename)
            
            # Choose the better result
            if (pymupdf_result.quality_score > marker_result.quality_score or 
                marker_result.parsing_method == ParsingMethod.FAILED):
                
                parsing_time_ms = int((time.time() - start_time) * 1000)
                pymupdf_result.parsing_time_ms = parsing_time_ms
                
                self.logger.info(
                    f"Successfully parsed with PyMuPDF fallback: {filename} "
                    f"(quality: {pymupdf_result.quality_assessment.value}, time: {parsing_time_ms}ms)"
                )
                return pymupdf_result
            else:
                # Return marker result even if quality is poor
                parsing_time_ms = int((time.time() - start_time) * 1000)
                marker_result.parsing_time_ms = parsing_time_ms
                return marker_result
                
        except Exception as e:
            error_msg = f"PDF parsing failed with exception: {str(e)}"
            self.logger.error(f"Error parsing {filename}: {error_msg}")
            
            return ParsedDocument(
                source_file_id=file_id,
                title=filename,
                content="",
                structure=DocumentStructure(),
                error_message=error_msg,
                parsing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file {temp_file}: {str(e)}")
    
    async def _create_temp_file(self, content: bytes, filename: str) -> Optional[Path]:
        """Create temporary file with PDF content"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_path = Path(temp_file.name)
            
            temp_file.write(content)
            temp_file.close()
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary file for {filename}: {str(e)}")
            return None
    
    async def _parse_with_marker(self, pdf_path: Path, file_id: str, filename: str) -> ParsedDocument:
        """
        Parse PDF using marker library.
        
        Args:
            pdf_path: Path to PDF file
            file_id: Source file ID
            filename: Original filename
            
        Returns:
            ParsedDocument with marker parsing results
        """
        try:
            # Prepare marker command
            cmd = ["marker", str(pdf_path)]
            
            # Add LLM flag if configured
            if self.settings.processing.use_marker_llm:
                cmd.append("--use_llm")
            
            # Set timeout
            timeout = self.settings.processing.marker_timeout
            
            self.logger.debug(f"Running marker command: {' '.join(cmd)}")
            
            # Run marker process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=pdf_path.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ParsedDocument(
                    source_file_id=file_id,
                    title=filename,
                    content="",
                    structure=DocumentStructure(),
                    error_message=f"Marker parsing timed out after {timeout} seconds"
                )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown marker error"
                return ParsedDocument(
                    source_file_id=file_id,
                    title=filename,
                    content="",
                    structure=DocumentStructure(),
                    error_message=f"Marker failed with return code {process.returncode}: {error_msg}"
                )
            
            # Parse marker output
            marker_output = stdout.decode('utf-8', errors='ignore')
            
            # Look for output files (marker typically creates .md files)
            output_files = list(pdf_path.parent.glob(f"{pdf_path.stem}*.md"))
            
            if output_files:
                # Read the markdown output
                with open(output_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Clean up output files
                for output_file in output_files:
                    try:
                        output_file.unlink()
                    except Exception:
                        pass
            else:
                # Use stdout if no output files
                content = marker_output
            
            if not content.strip():
                return ParsedDocument(
                    source_file_id=file_id,
                    title=filename,
                    content="",
                    structure=DocumentStructure(),
                    error_message="Marker produced empty output"
                )
            
            # Extract document structure and elements
            structure = await self._extract_structure_from_markdown(content)
            math_elements = await self._extract_math_elements(content)
            tables = await self._extract_tables_from_markdown(content)
            
            # Assess parsing quality
            quality_score, quality_assessment = await self._assess_parsing_quality(
                content, structure, math_elements, tables
            )
            
            return ParsedDocument(
                source_file_id=file_id,
                title=structure.title or filename,
                content=content,
                structure=structure,
                math_notation=math_elements,
                tables=tables,
                parsing_method=ParsingMethod.MARKER,
                quality_score=quality_score,
                quality_assessment=quality_assessment
            )
            
        except Exception as e:
            error_msg = f"Marker parsing error: {str(e)}"
            self.logger.error(f"Error in marker parsing for {filename}: {error_msg}")
            
            return ParsedDocument(
                source_file_id=file_id,
                title=filename,
                content="",
                structure=DocumentStructure(),
                error_message=error_msg
            )
    
    async def _parse_with_pymupdf(self, pdf_path: Path, file_id: str, filename: str) -> ParsedDocument:
        """
        Parse PDF using PyMuPDF as fallback.
        
        Args:
            pdf_path: Path to PDF file
            file_id: Source file ID
            filename: Original filename
            
        Returns:
            ParsedDocument with PyMuPDF parsing results
        """
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            
            if doc.page_count == 0:
                return ParsedDocument(
                    source_file_id=file_id,
                    title=filename,
                    content="",
                    structure=DocumentStructure(),
                    error_message="PDF has no pages"
                )
            
            # Extract text content
            content_parts = []
            structure = DocumentStructure(page_count=doc.page_count)
            math_elements = []
            tables = []
            images = []
            
            # Get document metadata
            metadata = doc.metadata
            title = metadata.get('title', filename) if metadata else filename
            
            # Extract table of contents
            toc = doc.get_toc()
            if toc:
                structure.toc = [
                    {
                        'level': level,
                        'title': title_text,
                        'page': page - 1  # Convert to 0-based
                    }
                    for level, title_text, page in toc
                ]
            
            # Process each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text_dict = page.get_text("dict")
                page_text = await self._extract_text_with_structure(text_dict, page_num)
                
                if page_text.strip():
                    content_parts.append(f"\n--- Page {page_num + 1} ---\n")
                    content_parts.append(page_text)
                
                # Extract tables
                page_tables = await self._extract_tables_from_page(page, page_num)
                tables.extend(page_tables)
                
                # Extract images
                page_images = await self._extract_images_from_page(page, page_num)
                images.extend(page_images)
                
                # Look for mathematical content
                page_math = await self._detect_math_in_text(page_text, page_num)
                math_elements.extend(page_math)
            
            doc.close()
            
            # Combine content
            full_content = "\n".join(content_parts)
            
            if not full_content.strip():
                return ParsedDocument(
                    source_file_id=file_id,
                    title=title,
                    content="",
                    structure=structure,
                    error_message="No text content extracted"
                )
            
            # Update structure flags
            structure.title = title
            structure.has_math = len(math_elements) > 0
            structure.has_tables = len(tables) > 0
            structure.has_images = len(images) > 0
            
            # Extract sections from content
            structure.sections = await self._extract_sections_from_text(full_content)
            
            # Assess parsing quality
            quality_score, quality_assessment = await self._assess_parsing_quality(
                full_content, structure, math_elements, tables
            )
            
            return ParsedDocument(
                source_file_id=file_id,
                title=title,
                content=full_content,
                structure=structure,
                math_notation=math_elements,
                tables=tables,
                images=images,
                parsing_method=ParsingMethod.PYMUPDF,
                quality_score=quality_score,
                quality_assessment=quality_assessment
            )
            
        except Exception as e:
            error_msg = f"PyMuPDF parsing error: {str(e)}"
            self.logger.error(f"Error in PyMuPDF parsing for {filename}: {error_msg}")
            
            return ParsedDocument(
                source_file_id=file_id,
                title=filename,
                content="",
                structure=DocumentStructure(),
                error_message=error_msg
            )
    
    async def _extract_text_with_structure(self, text_dict: Dict, page_num: int) -> str:
        """Extract text while preserving structure from PyMuPDF text dict"""
        text_parts = []
        
        try:
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_text = []
                    
                    for line in block["lines"]:
                        line_text = []
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                # Check for headers based on font size
                                font_size = span.get("size", 12)
                                flags = span.get("flags", 0)
                                
                                # Bold text (flag 16) or large font might be headers
                                if flags & 16 or font_size > 14:
                                    text = f"**{text}**"
                                
                                line_text.append(text)
                        
                        if line_text:
                            block_text.append(" ".join(line_text))
                    
                    if block_text:
                        text_parts.append("\n".join(block_text))
        
        except Exception as e:
            self.logger.warning(f"Error extracting structured text from page {page_num}: {str(e)}")
            # Fallback to simple text extraction
            return ""
        
        return "\n\n".join(text_parts)
    
    async def _extract_tables_from_page(self, page, page_num: int) -> List[TableElement]:
        """Extract tables from a PyMuPDF page"""
        tables = []
        
        try:
            # Try to find tables using PyMuPDF's table detection
            table_finder = page.find_tables()
            
            for table in table_finder:
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 1:  # At least header + 1 row
                        # Convert to text representation
                        table_text = []
                        for row in table_data:
                            row_text = " | ".join(str(cell) if cell else "" for cell in row)
                            table_text.append(row_text)
                        
                        content = "\n".join(table_text)
                        
                        tables.append(TableElement(
                            content=content,
                            rows=len(table_data),
                            columns=len(table_data[0]) if table_data else 0,
                            position=(page_num, 0),
                            structured_data=table_data
                        ))
                
                except Exception as e:
                    self.logger.debug(f"Error extracting table from page {page_num}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.debug(f"Error finding tables on page {page_num}: {str(e)}")
        
        return tables
    
    async def _extract_images_from_page(self, page, page_num: int) -> List[ImageElement]:
        """Extract images from a PyMuPDF page"""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        images.append(ImageElement(
                            description=f"Image {img_index + 1} on page {page_num + 1}",
                            position=(page_num, img_index),
                            size=(pix.width, pix.height),
                            image_data=img_data
                        ))
                    
                    pix = None  # Free memory
                
                except Exception as e:
                    self.logger.debug(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.debug(f"Error finding images on page {page_num}: {str(e)}")
        
        return images
    
    async def _detect_math_in_text(self, text: str, page_num: int) -> List[MathElement]:
        """Detect mathematical content in text"""
        math_elements = []
        
        # Patterns for mathematical notation
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\$\$[^$]+\$\$',  # LaTeX display math
            r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
            r'[∑∏∫∂∇∆∞±≤≥≠≈∈∉⊂⊃∪∩]',  # Mathematical symbols
            r'\b\d+\s*[+\-*/=]\s*\d+',  # Simple equations
            r'\b[a-zA-Z]\s*=\s*[^,\s]+',  # Variable assignments
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                content = match.group()
                
                # Skip very short matches that might be false positives
                if len(content.strip()) < 2:
                    continue
                
                math_elements.append(MathElement(
                    content=content,
                    latex=content if content.startswith('$') else None,
                    position=(page_num, match.start()),
                    confidence=0.7  # Medium confidence for pattern matching
                ))
        
        return math_elements
    
    async def _extract_structure_from_markdown(self, content: str) -> DocumentStructure:
        """Extract document structure from markdown content"""
        structure = DocumentStructure()
        
        # Extract title (first # header)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            structure.title = title_match.group(1).strip()
        
        # Extract sections
        sections = []
        section_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(section_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            sections.append({
                'level': level,
                'title': title,
                'position': match.start()
            })
        
        structure.sections = sections
        
        # Check for mathematical content
        structure.has_math = bool(re.search(r'\$[^$]+\$|\\\w+\{', content))
        
        # Check for tables
        structure.has_tables = bool(re.search(r'\|.*\|', content))
        
        return structure
    
    async def _extract_math_elements(self, content: str) -> List[MathElement]:
        """Extract mathematical elements from markdown content"""
        math_elements = []
        
        # LaTeX math patterns
        patterns = [
            (r'\$\$([^$]+)\$\$', True),  # Display math
            (r'\$([^$]+)\$', False),     # Inline math
        ]
        
        for pattern, is_display in patterns:
            for match in re.finditer(pattern, content):
                latex_content = match.group(1).strip()
                
                math_elements.append(MathElement(
                    content=match.group(0),
                    latex=latex_content,
                    position=(0, match.start()),
                    confidence=0.9  # High confidence for LaTeX
                ))
        
        return math_elements
    
    async def _extract_tables_from_markdown(self, content: str) -> List[TableElement]:
        """Extract tables from markdown content"""
        tables = []
        
        # Find markdown tables
        table_pattern = r'(\|.*\|(?:\n\|.*\|)*)'
        
        for match in re.finditer(table_pattern, content, re.MULTILINE):
            table_text = match.group(1)
            lines = table_text.strip().split('\n')
            
            # Skip separator lines
            data_lines = [line for line in lines if not re.match(r'^\|[\s\-:]+\|$', line)]
            
            if len(data_lines) >= 2:  # At least header + 1 row
                # Parse table structure
                rows = len(data_lines)
                columns = len(data_lines[0].split('|')) - 2  # Remove empty start/end
                
                tables.append(TableElement(
                    content=table_text,
                    rows=rows,
                    columns=columns,
                    position=(0, match.start())
                ))
        
        return tables
    
    async def _extract_sections_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from plain text content"""
        sections = []
        
        # Look for common section patterns
        patterns = [
            r'^([A-Z][A-Z\s]{2,})\s*$',  # ALL CAPS headers
            r'^\*\*([^*]+)\*\*\s*$',     # Bold headers
            r'^(\d+\.?\s+[A-Z][^.!?]*)\s*$',  # Numbered sections
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                title = match.group(1).strip()
                
                # Skip very short titles
                if len(title) < 3:
                    continue
                
                sections.append({
                    'level': 1,
                    'title': title,
                    'position': match.start()
                })
        
        return sections
    
    async def _assess_parsing_quality(self, content: str, structure: DocumentStructure, 
                                    math_elements: List[MathElement], 
                                    tables: List[TableElement]) -> Tuple[float, ParsingQuality]:
        """
        Assess the quality of parsing results.
        
        Returns:
            Tuple of (quality_score, quality_assessment)
        """
        if not content or not content.strip():
            return 0.0, ParsingQuality.FAILED
        
        score = 0.0
        max_score = 100.0
        
        # Content length score (0-20 points)
        content_length = len(content.strip())
        if content_length > 10000:
            score += 20
        elif content_length > 5000:
            score += 15
        elif content_length > 1000:
            score += 10
        elif content_length > 100:
            score += 5
        
        # Structure score (0-25 points)
        if structure.title:
            score += 5
        if structure.sections:
            score += min(len(structure.sections) * 2, 10)
        if structure.toc:
            score += 5
        if structure.page_count > 0:
            score += 5
        
        # Content richness score (0-25 points)
        if structure.has_math and math_elements:
            score += 10
        if structure.has_tables and tables:
            score += 10
        if structure.has_images:
            score += 5
        
        # Text quality score (0-30 points)
        # Check for common parsing artifacts
        artifacts = [
            r'\s{3,}',  # Excessive whitespace
            r'[^\w\s]{5,}',  # Long sequences of special characters
            r'\n{4,}',  # Excessive line breaks
        ]
        
        artifact_count = sum(len(re.findall(pattern, content)) for pattern in artifacts)
        artifact_penalty = min(artifact_count * 2, 15)
        
        # Word count and readability
        words = len(content.split())
        if words > 500:
            score += 15 - artifact_penalty
        elif words > 100:
            score += 10 - artifact_penalty
        elif words > 50:
            score += 5 - artifact_penalty
        
        # Normalize score
        final_score = min(score / max_score, 1.0)
        
        # Determine quality assessment
        if final_score >= 0.8:
            quality = ParsingQuality.EXCELLENT
        elif final_score >= 0.6:
            quality = ParsingQuality.GOOD
        elif final_score >= 0.4:
            quality = ParsingQuality.FAIR
        elif final_score >= 0.2:
            quality = ParsingQuality.POOR
        else:
            quality = ParsingQuality.FAILED
        
        return final_score, quality
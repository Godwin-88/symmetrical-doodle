"""
Simple Chunker Service

A lightweight chunker for testing purposes that works with SimplePDFParser.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SimpleChunk:
    """Simple chunk structure for testing"""
    content: str
    chunk_order: int
    section_header: Optional[str] = None
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(self.content.split())


class SimpleChunker:
    """
    Simple chunker for testing purposes.
    Splits content into chunks based on paragraphs or fixed size.
    """
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    async def chunk_document(self, document) -> List[SimpleChunk]:
        """
        Create simple chunks from a parsed document.
        
        Args:
            document: Parsed document with content
            
        Returns:
            List of simple chunks
        """
        if not document or not document.content:
            return []
        
        content = document.content
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = len(paragraph.split())
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                chunks.append(SimpleChunk(
                    content=current_chunk.strip(),
                    chunk_order=len(chunks),
                    token_count=current_tokens
                ))
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunks.append(SimpleChunk(
                content=current_chunk.strip(),
                chunk_order=len(chunks),
                token_count=current_tokens
            ))
        
        return chunks
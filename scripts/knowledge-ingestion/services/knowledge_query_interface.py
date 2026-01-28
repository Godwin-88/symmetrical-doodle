"""
Knowledge query interface for the ingestion system.

This module provides:
- Semantic search using vector similarity
- Keyword search functionality
- Domain-filtered query capabilities
- Consistent response formats across all search types
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.logging import get_logger
from .supabase_storage import SupabaseStorageService
from .embedding_service import EmbeddingService
from .concurrent_access_manager import get_access_manager, LockType, OperationType, Priority


class SearchType(Enum):
    """Search type enumeration"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    DOMAIN_FILTERED = "domain_filtered"


class SortOrder(Enum):
    """Sort order enumeration"""
    RELEVANCE = "relevance"
    DATE = "date"
    TITLE = "title"
    DOMAIN = "domain"


@dataclass
class SearchQuery:
    """Search query specification"""
    query_text: str
    search_type: SearchType = SearchType.SEMANTIC
    domains: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0
    similarity_threshold: float = 0.7
    sort_order: SortOrder = SortOrder.RELEVANCE
    include_metadata: bool = True
    include_content: bool = False
    filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Individual search result"""
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    similarity_score: Optional[float] = None
    keyword_matches: Optional[List[str]] = None
    domain_classification: Optional[str] = None
    section_header: Optional[str] = None
    chunk_order: int = 0
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResponse:
    """Search response with results and metadata"""
    query: str
    search_type: str
    results: List[SearchResult]
    total_results: int
    execution_time_ms: float
    filters_applied: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    facets: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeQueryInterface:
    """
    Knowledge query interface for searching ingested documents.
    
    Provides:
    - Semantic search using vector embeddings
    - Keyword search with full-text capabilities
    - Domain-filtered searches
    - Hybrid search combining multiple approaches
    - Consistent response formatting
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._storage_service: Optional[SupabaseStorageService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        
        # Search configuration
        self.default_similarity_threshold = 0.7
        self.max_results_per_query = 100
        self.keyword_search_boost = 1.2
        self.semantic_search_boost = 1.0
        
        # Domain mappings
        self.domain_keywords = {
            "ML": ["machine learning", "neural network", "deep learning", "artificial intelligence", "model", "training", "algorithm"],
            "DRL": ["reinforcement learning", "deep reinforcement", "q-learning", "policy gradient", "actor-critic", "markov decision"],
            "NLP": ["natural language", "text processing", "tokenization", "embedding", "transformer", "bert", "gpt"],
            "LLMs": ["large language model", "llm", "gpt", "bert", "transformer", "attention", "pre-trained"],
            "finance": ["trading", "portfolio", "risk", "market", "financial", "investment", "derivatives", "options"],
            "general": ["general", "overview", "introduction", "basic", "fundamental"]
        }
        
        # Cache for frequent queries
        self._query_cache: Dict[str, SearchResponse] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """
        Initialize the query interface.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing knowledge query interface")
            
            # Initialize storage service
            self._storage_service = SupabaseStorageService(self.settings.supabase)
            await self._storage_service.initialize_client()
            
            # Initialize embedding service
            self._embedding_service = EmbeddingService()
            await self._embedding_service.initialize()
            
            self.logger.info("Knowledge query interface initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge query interface: {e}")
            return False
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Execute a search query.
        
        Args:
            query: Search query specification
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self._query_cache:
                cached_response = self._query_cache[cache_key]
                if (datetime.now() - cached_response.timestamp).total_seconds() < self._cache_ttl_seconds:
                    self.logger.debug(f"Returning cached result for query: {query.query_text}")
                    return cached_response
            
            # Execute search based on type
            if query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query)
            elif query.search_type == SearchType.KEYWORD:
                results = await self._keyword_search(query)
            elif query.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(query)
            elif query.search_type == SearchType.DOMAIN_FILTERED:
                results = await self._domain_filtered_search(query)
            else:
                raise ValueError(f"Unsupported search type: {query.search_type}")
            
            # Apply post-processing
            results = await self._post_process_results(results, query)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = SearchResponse(
                query=query.query_text,
                search_type=query.search_type.value,
                results=results[:query.limit],
                total_results=len(results),
                execution_time_ms=execution_time,
                filters_applied=query.filters,
                suggestions=await self._generate_suggestions(query, results),
                facets=await self._generate_facets(results),
                timestamp=datetime.now()
            )
            
            # Cache the response
            self._query_cache[cache_key] = response
            
            self.logger.info(f"Search completed: {len(results)} results in {execution_time:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query.query_text}': {e}")
            
            # Return empty response on error
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                query=query.query_text,
                search_type=query.search_type.value,
                results=[],
                total_results=0,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )
    
    async def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute semantic search using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self._embedding_service.generate_embeddings([query.query_text])
            if not query_embedding:
                return []
            
            query_vector = query_embedding[0].embedding_vector
            
            # Search similar chunks
            similar_chunks = await self._storage_service.search_similar_chunks(
                query_vector=query_vector,
                limit=query.limit * 2,  # Get more results for filtering
                similarity_threshold=query.similarity_threshold
            )
            
            # Convert to SearchResult objects
            results = []
            for chunk_data in similar_chunks:
                result = SearchResult(
                    chunk_id=chunk_data['id'],
                    document_id=chunk_data['document_id'],
                    document_title=chunk_data.get('documents', {}).get('title', 'Unknown'),
                    content=chunk_data['content'],
                    similarity_score=chunk_data.get('similarity', 0.0),
                    domain_classification=chunk_data.get('documents', {}).get('domain_classification'),
                    section_header=chunk_data.get('section_header'),
                    chunk_order=chunk_data.get('chunk_order', 0),
                    created_at=datetime.fromisoformat(chunk_data['created_at']) if chunk_data.get('created_at') else None,
                    metadata=chunk_data.get('semantic_metadata')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute keyword search using full-text search"""
        try:
            # Use Supabase full-text search
            search_query = self._prepare_keyword_query(query.query_text)
            
            # Execute search
            result = self._storage_service.client.table('chunks').select(
                '*, documents!inner(title, domain_classification)'
            ).text_search(
                'content', search_query
            ).limit(query.limit * 2).execute()
            
            # Convert to SearchResult objects
            results = []
            for chunk_data in result.data:
                # Calculate keyword matches
                keyword_matches = self._find_keyword_matches(query.query_text, chunk_data['content'])
                
                result_obj = SearchResult(
                    chunk_id=chunk_data['id'],
                    document_id=chunk_data['document_id'],
                    document_title=chunk_data.get('documents', {}).get('title', 'Unknown'),
                    content=chunk_data['content'],
                    keyword_matches=keyword_matches,
                    domain_classification=chunk_data.get('documents', {}).get('domain_classification'),
                    section_header=chunk_data.get('section_header'),
                    chunk_order=chunk_data.get('chunk_order', 0),
                    created_at=datetime.fromisoformat(chunk_data['created_at']) if chunk_data.get('created_at') else None,
                    metadata=chunk_data.get('semantic_metadata')
                )
                results.append(result_obj)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute hybrid search combining semantic and keyword approaches"""
        try:
            # Execute both searches in parallel
            semantic_task = asyncio.create_task(self._semantic_search(query))
            keyword_task = asyncio.create_task(self._keyword_search(query))
            
            semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, 
                keyword_results, 
                semantic_weight=0.6, 
                keyword_weight=0.4
            )
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _domain_filtered_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute domain-filtered search"""
        try:
            # Start with semantic search
            base_results = await self._semantic_search(query)
            
            # Apply domain filters
            if query.domains:
                filtered_results = []
                for result in base_results:
                    if result.domain_classification in query.domains:
                        filtered_results.append(result)
                    elif self._matches_domain_keywords(result.content, query.domains):
                        filtered_results.append(result)
                
                return filtered_results
            
            return base_results
            
        except Exception as e:
            self.logger.error(f"Domain-filtered search failed: {e}")
            return []
    
    def _prepare_keyword_query(self, query_text: str) -> str:
        """Prepare query text for full-text search"""
        # Clean and prepare the query
        query_text = re.sub(r'[^\w\s]', ' ', query_text)
        words = query_text.split()
        
        # Create search query with OR logic
        return ' | '.join(words)
    
    def _find_keyword_matches(self, query_text: str, content: str) -> List[str]:
        """Find keyword matches in content"""
        query_words = set(query_text.lower().split())
        content_words = set(re.findall(r'\w+', content.lower()))
        
        matches = list(query_words.intersection(content_words))
        return matches
    
    def _matches_domain_keywords(self, content: str, domains: List[str]) -> bool:
        """Check if content matches domain keywords"""
        content_lower = content.lower()
        
        for domain in domains:
            if domain in self.domain_keywords:
                domain_words = self.domain_keywords[domain]
                if any(keyword in content_lower for keyword in domain_words):
                    return True
        
        return False
    
    def _combine_search_results(
        self, 
        semantic_results: List[SearchResult], 
        keyword_results: List[SearchResult],
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results with weighted scoring"""
        
        # Create a map of chunk_id to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            score = (result.similarity_score or 0.0) * semantic_weight
            result_map[result.chunk_id] = {
                'result': result,
                'score': score,
                'sources': ['semantic']
            }
        
        # Add keyword results
        for result in keyword_results:
            keyword_score = len(result.keyword_matches or []) / 10.0  # Normalize keyword score
            score = keyword_score * keyword_weight
            
            if result.chunk_id in result_map:
                # Combine scores
                result_map[result.chunk_id]['score'] += score
                result_map[result.chunk_id]['sources'].append('keyword')
                # Update result with keyword matches
                result_map[result.chunk_id]['result'].keyword_matches = result.keyword_matches
            else:
                result_map[result.chunk_id] = {
                    'result': result,
                    'score': score,
                    'sources': ['keyword']
                }
        
        # Sort by combined score
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['result'] for item in sorted_results]
    
    async def _post_process_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Post-process search results"""
        try:
            # Apply additional filters
            if query.filters:
                results = self._apply_filters(results, query.filters)
            
            # Sort results
            results = self._sort_results(results, query.sort_order)
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            results = results[start_idx:end_idx]
            
            # Enhance results with additional metadata if requested
            if query.include_metadata:
                results = await self._enhance_with_metadata(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            return results
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply additional filters to results"""
        filtered_results = []
        
        for result in results:
            include_result = True
            
            # Date range filter
            if 'date_from' in filters and result.created_at:
                date_from = datetime.fromisoformat(filters['date_from'])
                if result.created_at < date_from:
                    include_result = False
            
            if 'date_to' in filters and result.created_at:
                date_to = datetime.fromisoformat(filters['date_to'])
                if result.created_at > date_to:
                    include_result = False
            
            # Domain filter
            if 'domains' in filters:
                if result.domain_classification not in filters['domains']:
                    include_result = False
            
            # Minimum similarity filter
            if 'min_similarity' in filters and result.similarity_score:
                if result.similarity_score < filters['min_similarity']:
                    include_result = False
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _sort_results(self, results: List[SearchResult], sort_order: SortOrder) -> List[SearchResult]:
        """Sort results according to specified order"""
        if sort_order == SortOrder.RELEVANCE:
            # Sort by similarity score (descending) or keyword matches
            return sorted(
                results,
                key=lambda x: (x.similarity_score or 0.0) + (len(x.keyword_matches or []) * 0.1),
                reverse=True
            )
        elif sort_order == SortOrder.DATE:
            return sorted(
                results,
                key=lambda x: x.created_at or datetime.min,
                reverse=True
            )
        elif sort_order == SortOrder.TITLE:
            return sorted(results, key=lambda x: x.document_title)
        elif sort_order == SortOrder.DOMAIN:
            return sorted(results, key=lambda x: x.domain_classification or 'zzz')
        
        return results
    
    async def _enhance_with_metadata(self, results: List[SearchResult]) -> List[SearchResult]:
        """Enhance results with additional metadata"""
        # This could include additional database queries to get more document metadata
        # For now, we'll just return the results as-is
        return results
    
    async def _generate_suggestions(self, query: SearchQuery, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions based on query and results"""
        suggestions = []
        
        # If few results, suggest related terms
        if len(results) < 3:
            query_words = query.query_text.lower().split()
            
            # Suggest domain-specific terms
            for domain, keywords in self.domain_keywords.items():
                for keyword in keywords:
                    if any(word in keyword for word in query_words):
                        suggestions.extend([k for k in keywords if k != keyword][:2])
        
        return list(set(suggestions))[:5]  # Return unique suggestions, max 5
    
    async def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate facets for search results"""
        facets = {
            'domains': defaultdict(int),
            'documents': defaultdict(int),
            'sections': defaultdict(int)
        }
        
        for result in results:
            if result.domain_classification:
                facets['domains'][result.domain_classification] += 1
            
            facets['documents'][result.document_title] += 1
            
            if result.section_header:
                facets['sections'][result.section_header] += 1
        
        # Convert to regular dicts and limit entries
        return {
            'domains': dict(list(facets['domains'].items())[:10]),
            'documents': dict(list(facets['documents'].items())[:10]),
            'sections': dict(list(facets['sections'].items())[:10])
        }
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.query_text,
            query.search_type.value,
            str(query.domains),
            str(query.limit),
            str(query.offset),
            str(query.similarity_threshold),
            query.sort_order.value,
            str(query.filters)
        ]
        return '|'.join(key_parts)
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document data with all chunks, or None if not found
        """
        try:
            # Get document metadata
            doc_result = self._storage_service.client.table('documents').select('*').eq('id', document_id).execute()
            
            if not doc_result.data:
                return None
            
            document = doc_result.data[0]
            
            # Get all chunks for the document
            chunks = await self._storage_service.get_chunks_by_document_id(document_id)
            
            return {
                'document': document,
                'chunks': chunks,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def get_similar_documents(
        self, 
        document_id: str, 
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to the given document.
        
        Args:
            document_id: Reference document ID
            limit: Maximum number of similar documents to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Get chunks from the reference document
            chunks = await self._storage_service.get_chunks_by_document_id(document_id)
            
            if not chunks:
                return []
            
            # Use the first chunk's embedding as the query vector
            first_chunk = chunks[0]
            if 'embedding' not in first_chunk:
                return []
            
            query_vector = first_chunk['embedding']
            
            # Find similar chunks
            similar_chunks = await self._storage_service.search_similar_chunks(
                query_vector=query_vector,
                limit=limit * 5,  # Get more chunks to find diverse documents
                similarity_threshold=similarity_threshold
            )
            
            # Group by document and calculate average similarity
            doc_similarities = defaultdict(list)
            for chunk in similar_chunks:
                if chunk['document_id'] != document_id:  # Exclude the reference document
                    doc_similarities[chunk['document_id']].append(chunk.get('similarity', 0.0))
            
            # Calculate average similarity per document
            similar_docs = []
            for doc_id, similarities in doc_similarities.items():
                avg_similarity = sum(similarities) / len(similarities)
                
                # Get document metadata
                doc_result = self._storage_service.client.table('documents').select('*').eq('id', doc_id).execute()
                
                if doc_result.data:
                    doc_data = doc_result.data[0]
                    similar_docs.append({
                        'document_id': doc_id,
                        'title': doc_data.get('title', 'Unknown'),
                        'domain_classification': doc_data.get('domain_classification'),
                        'similarity_score': avg_similarity,
                        'chunk_count': len(similarities)
                    })
            
            # Sort by similarity and return top results
            similar_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_docs[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar documents for {document_id}: {e}")
            return []
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search and query statistics"""
        try:
            stats = await self._storage_service.get_storage_statistics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_documents': stats.get('total_documents', 0),
                'total_chunks': stats.get('total_chunks', 0),
                'cache_size': len(self._query_cache),
                'domain_distribution': stats.get('document_status_breakdown', {}),
                'embedding_models': stats.get('embedding_models', {}),
                'recent_activity': stats.get('recent_activity', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            return {}


# Global query interface instance
_query_interface: Optional[KnowledgeQueryInterface] = None


async def get_query_interface() -> KnowledgeQueryInterface:
    """Get or create global query interface instance"""
    global _query_interface
    
    if _query_interface is None:
        _query_interface = KnowledgeQueryInterface()
        await _query_interface.initialize()
    
    return _query_interface


async def initialize_query_interface() -> bool:
    """Initialize knowledge query interface"""
    try:
        interface = await get_query_interface()
        return interface is not None
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to initialize query interface: {e}")
        return False


# Convenience functions for common search operations

async def semantic_search(
    query_text: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    domains: Optional[List[str]] = None
) -> SearchResponse:
    """
    Perform semantic search.
    
    Args:
        query_text: Search query
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score
        domains: Optional domain filter
        
    Returns:
        SearchResponse with results
    """
    interface = await get_query_interface()
    
    query = SearchQuery(
        query_text=query_text,
        search_type=SearchType.SEMANTIC,
        limit=limit,
        similarity_threshold=similarity_threshold,
        domains=domains
    )
    
    return await interface.search(query)


async def keyword_search(
    query_text: str,
    limit: int = 10,
    domains: Optional[List[str]] = None
) -> SearchResponse:
    """
    Perform keyword search.
    
    Args:
        query_text: Search query
        limit: Maximum results to return
        domains: Optional domain filter
        
    Returns:
        SearchResponse with results
    """
    interface = await get_query_interface()
    
    query = SearchQuery(
        query_text=query_text,
        search_type=SearchType.KEYWORD,
        limit=limit,
        domains=domains
    )
    
    return await interface.search(query)


async def hybrid_search(
    query_text: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    domains: Optional[List[str]] = None
) -> SearchResponse:
    """
    Perform hybrid search combining semantic and keyword approaches.
    
    Args:
        query_text: Search query
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score for semantic component
        domains: Optional domain filter
        
    Returns:
        SearchResponse with results
    """
    interface = await get_query_interface()
    
    query = SearchQuery(
        query_text=query_text,
        search_type=SearchType.HYBRID,
        limit=limit,
        similarity_threshold=similarity_threshold,
        domains=domains
    )
    
    return await interface.search(query)
"""
Platform API endpoints for knowledge ingestion integration.

This module provides:
- RESTful API endpoints for knowledge retrieval
- Integration with existing intelligence layer
- Consistent response formats
- Error handling and logging
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from core.config import get_settings
from core.logging import get_logger
from .platform_integration import get_integration_service, PlatformIntegrationService
from .concurrent_access_manager import get_access_manager, ConcurrentAccessManager
from .knowledge_query_interface import (
    get_query_interface, 
    KnowledgeQueryInterface,
    SearchQuery,
    SearchType,
    SortOrder,
    semantic_search,
    keyword_search,
    hybrid_search
)


# Pydantic models for API requests/responses

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    search_type: str = Field(default="semantic", description="Search type: semantic, keyword, hybrid, domain_filtered")
    domains: Optional[List[str]] = Field(default=None, description="Domain filter list")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Results offset for pagination")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    sort_order: str = Field(default="relevance", description="Sort order: relevance, date, title, domain")
    include_metadata: bool = Field(default=True, description="Include result metadata")
    include_content: bool = Field(default=False, description="Include full content in results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class SearchResultModel(BaseModel):
    """Search result model"""
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


class SearchResponseModel(BaseModel):
    """Search response model"""
    query: str
    search_type: str
    results: List[SearchResultModel]
    total_results: int
    execution_time_ms: float
    filters_applied: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    facets: Optional[Dict[str, Any]] = None
    timestamp: datetime


class DocumentModel(BaseModel):
    """Document model"""
    document_id: str
    title: str
    content: Optional[str] = None
    domain_classification: Optional[str] = None
    processing_status: str
    created_at: datetime
    chunks: Optional[List[Dict[str, Any]]] = None
    total_chunks: int = 0


class SimilarDocumentModel(BaseModel):
    """Similar document model"""
    document_id: str
    title: str
    domain_classification: Optional[str] = None
    similarity_score: float
    chunk_count: int


class HealthCheckModel(BaseModel):
    """Health check response model"""
    status: str
    service: str
    timestamp: datetime
    connections: Dict[str, Any]
    statistics: Dict[str, Any]


class PlatformAPIService:
    """
    Platform API service for knowledge ingestion endpoints.
    
    Provides RESTful API endpoints that integrate with the existing
    algorithmic trading platform architecture.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._integration_service: Optional[PlatformIntegrationService] = None
        self._access_manager: Optional[ConcurrentAccessManager] = None
        self._query_interface: Optional[KnowledgeQueryInterface] = None
        
        # FastAPI app
        self.app = FastAPI(
            title="Knowledge Ingestion API",
            description="API endpoints for knowledge base search and retrieval",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
    
    async def initialize(self) -> bool:
        """Initialize the API service"""
        try:
            self.logger.info("Initializing platform API service")
            
            # Initialize service dependencies
            self._integration_service = await get_integration_service()
            self._access_manager = await get_access_manager()
            self._query_interface = await get_query_interface()
            
            self.logger.info("Platform API service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform API service: {e}")
            return False
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthCheckModel)
        async def health_check():
            """Health check endpoint"""
            try:
                # Get connection status
                connections = {}
                statistics = {}
                
                if self._integration_service:
                    connections = await self._integration_service.get_connection_status()
                
                if self._access_manager:
                    statistics['access_manager'] = self._access_manager.get_lock_statistics()
                
                if self._query_interface:
                    statistics['query_interface'] = await self._query_interface.get_search_statistics()
                
                return HealthCheckModel(
                    status="healthy",
                    service="knowledge-ingestion-api",
                    timestamp=datetime.now(),
                    connections=connections,
                    statistics=statistics
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
        
        @self.app.post("/search", response_model=SearchResponseModel)
        async def search_knowledge_base(request: SearchRequest):
            """Search the knowledge base"""
            try:
                if not self._query_interface:
                    raise HTTPException(status_code=503, detail="Query interface not available")
                
                # Convert request to SearchQuery
                search_query = SearchQuery(
                    query_text=request.query,
                    search_type=SearchType(request.search_type),
                    domains=request.domains,
                    limit=request.limit,
                    offset=request.offset,
                    similarity_threshold=request.similarity_threshold,
                    sort_order=SortOrder(request.sort_order),
                    include_metadata=request.include_metadata,
                    include_content=request.include_content,
                    filters=request.filters
                )
                
                # Execute search
                response = await self._query_interface.search(search_query)
                
                # Convert to API response model
                return SearchResponseModel(
                    query=response.query,
                    search_type=response.search_type,
                    results=[
                        SearchResultModel(
                            chunk_id=result.chunk_id,
                            document_id=result.document_id,
                            document_title=result.document_title,
                            content=result.content,
                            similarity_score=result.similarity_score,
                            keyword_matches=result.keyword_matches,
                            domain_classification=result.domain_classification,
                            section_header=result.section_header,
                            chunk_order=result.chunk_order,
                            created_at=result.created_at,
                            metadata=result.metadata
                        ) for result in response.results
                    ],
                    total_results=response.total_results,
                    execution_time_ms=response.execution_time_ms,
                    filters_applied=response.filters_applied,
                    suggestions=response.suggestions,
                    facets=response.facets,
                    timestamp=response.timestamp
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.get("/search/semantic")
        async def semantic_search_endpoint(
            q: str = Query(..., description="Search query"),
            limit: int = Query(10, ge=1, le=100, description="Maximum results"),
            threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
            domains: Optional[str] = Query(None, description="Comma-separated domain list")
        ):
            """Semantic search endpoint"""
            try:
                domain_list = domains.split(',') if domains else None
                response = await semantic_search(q, limit, threshold, domain_list)
                
                return {
                    "query": response.query,
                    "results": [
                        {
                            "chunk_id": result.chunk_id,
                            "document_title": result.document_title,
                            "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                            "similarity_score": result.similarity_score,
                            "domain": result.domain_classification
                        } for result in response.results
                    ],
                    "total": response.total_results,
                    "execution_time_ms": response.execution_time_ms
                }
                
            except Exception as e:
                self.logger.error(f"Semantic search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.get("/search/keyword")
        async def keyword_search_endpoint(
            q: str = Query(..., description="Search query"),
            limit: int = Query(10, ge=1, le=100, description="Maximum results"),
            domains: Optional[str] = Query(None, description="Comma-separated domain list")
        ):
            """Keyword search endpoint"""
            try:
                domain_list = domains.split(',') if domains else None
                response = await keyword_search(q, limit, domain_list)
                
                return {
                    "query": response.query,
                    "results": [
                        {
                            "chunk_id": result.chunk_id,
                            "document_title": result.document_title,
                            "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                            "keyword_matches": result.keyword_matches,
                            "domain": result.domain_classification
                        } for result in response.results
                    ],
                    "total": response.total_results,
                    "execution_time_ms": response.execution_time_ms
                }
                
            except Exception as e:
                self.logger.error(f"Keyword search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.get("/search/hybrid")
        async def hybrid_search_endpoint(
            q: str = Query(..., description="Search query"),
            limit: int = Query(10, ge=1, le=100, description="Maximum results"),
            threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
            domains: Optional[str] = Query(None, description="Comma-separated domain list")
        ):
            """Hybrid search endpoint"""
            try:
                domain_list = domains.split(',') if domains else None
                response = await hybrid_search(q, limit, threshold, domain_list)
                
                return {
                    "query": response.query,
                    "results": [
                        {
                            "chunk_id": result.chunk_id,
                            "document_title": result.document_title,
                            "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                            "similarity_score": result.similarity_score,
                            "keyword_matches": result.keyword_matches,
                            "domain": result.domain_classification
                        } for result in response.results
                    ],
                    "total": response.total_results,
                    "execution_time_ms": response.execution_time_ms
                }
                
            except Exception as e:
                self.logger.error(f"Hybrid search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.get("/documents/{document_id}", response_model=DocumentModel)
        async def get_document(document_id: str = Path(..., description="Document ID")):
            """Get document by ID"""
            try:
                if not self._query_interface:
                    raise HTTPException(status_code=503, detail="Query interface not available")
                
                document_data = await self._query_interface.get_document_by_id(document_id)
                
                if not document_data:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                doc = document_data['document']
                chunks = document_data['chunks']
                
                return DocumentModel(
                    document_id=doc['id'],
                    title=doc['title'],
                    content=doc.get('content'),
                    domain_classification=doc.get('domain_classification'),
                    processing_status=doc['processing_status'],
                    created_at=datetime.fromisoformat(doc['created_at']),
                    chunks=chunks,
                    total_chunks=len(chunks)
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")
        
        @self.app.get("/documents/{document_id}/similar", response_model=List[SimilarDocumentModel])
        async def get_similar_documents(
            document_id: str = Path(..., description="Document ID"),
            limit: int = Query(5, ge=1, le=20, description="Maximum similar documents"),
            threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold")
        ):
            """Get documents similar to the specified document"""
            try:
                if not self._query_interface:
                    raise HTTPException(status_code=503, detail="Query interface not available")
                
                similar_docs = await self._query_interface.get_similar_documents(
                    document_id, limit, threshold
                )
                
                return [
                    SimilarDocumentModel(
                        document_id=doc['document_id'],
                        title=doc['title'],
                        domain_classification=doc['domain_classification'],
                        similarity_score=doc['similarity_score'],
                        chunk_count=doc['chunk_count']
                    ) for doc in similar_docs
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to get similar documents for {document_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to find similar documents: {str(e)}")
        
        @self.app.get("/statistics")
        async def get_statistics():
            """Get knowledge base statistics"""
            try:
                stats = {}
                
                if self._query_interface:
                    stats['search'] = await self._query_interface.get_search_statistics()
                
                if self._access_manager:
                    stats['access_manager'] = self._access_manager.get_lock_statistics()
                
                if self._integration_service:
                    stats['connections'] = await self._integration_service.get_connection_status()
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "statistics": stats
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get statistics: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")
        
        @self.app.get("/domains")
        async def get_available_domains():
            """Get available knowledge domains"""
            try:
                if not self._query_interface:
                    raise HTTPException(status_code=503, detail="Query interface not available")
                
                # Get domain distribution from statistics
                stats = await self._query_interface.get_search_statistics()
                domain_distribution = stats.get('domain_distribution', {})
                
                domains = []
                for domain, count in domain_distribution.items():
                    domains.append({
                        "domain": domain,
                        "document_count": count,
                        "description": self._get_domain_description(domain)
                    })
                
                return {
                    "domains": domains,
                    "total_domains": len(domains)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get domains: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve domains: {str(e)}")
        
        @self.app.post("/test-connection")
        async def test_connections():
            """Test all platform connections"""
            try:
                if not self._integration_service:
                    raise HTTPException(status_code=503, detail="Integration service not available")
                
                test_results = await self._integration_service.test_connections()
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "test_results": test_results,
                    "overall_status": "healthy" if all(test_results.values()) else "degraded"
                }
                
            except Exception as e:
                self.logger.error(f"Connection test failed: {e}")
                raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")
    
    def _get_domain_description(self, domain: str) -> str:
        """Get description for a domain"""
        descriptions = {
            "ML": "Machine Learning and Artificial Intelligence",
            "DRL": "Deep Reinforcement Learning",
            "NLP": "Natural Language Processing",
            "LLMs": "Large Language Models",
            "finance": "Financial Markets and Trading",
            "general": "General Knowledge and Overviews"
        }
        return descriptions.get(domain, f"Domain: {domain}")
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the API server"""
        try:
            await self.initialize()
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            self.logger.info(f"Starting knowledge ingestion API server on {host}:{port}")
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise


# Global API service instance
_api_service: Optional[PlatformAPIService] = None


def get_api_service() -> PlatformAPIService:
    """Get or create global API service instance"""
    global _api_service
    
    if _api_service is None:
        _api_service = PlatformAPIService()
    
    return _api_service


async def start_knowledge_api(host: str = "0.0.0.0", port: int = 8080):
    """Start the knowledge ingestion API server"""
    service = get_api_service()
    await service.start_server(host, port)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_knowledge_api())
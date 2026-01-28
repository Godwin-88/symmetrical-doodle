# Platform Integration Layer

This document describes the platform integration layer for the Google Drive Knowledge Base Ingestion system, which enables seamless integration with the existing algorithmic trading platform.

## Overview

The platform integration layer provides:

1. **Connection Management** - Integration with existing Supabase and Neo4j instances
2. **Concurrent Access Management** - Support for multiple intelligence layer processes
3. **Search and Query Interface** - Consistent API endpoints for knowledge retrieval
4. **Conflict Resolution** - Automatic handling of data conflicts during ingestion

## Components

### 1. Platform Integration Service (`platform_integration.py`)

Manages connections to existing platform services and handles conflict detection/resolution.

**Key Features:**
- Automatic connection to existing Supabase instances
- Conflict detection for documents, chunks, and embeddings
- Multiple conflict resolution strategies (skip, overwrite, merge, version)
- Connection health monitoring and status reporting

**Usage:**
```python
from services.platform_integration import get_integration_service

# Initialize integration service
integration_service = await get_integration_service()

# Check connection status
status = await integration_service.get_connection_status()

# Detect conflicts before storing data
conflict_result = await integration_service.detect_conflicts(
    'documents', 
    document_data, 
    'file_id'
)

# Resolve conflicts using specified strategy
resolved_data = await integration_service.resolve_conflict(
    conflict_result,
    new_data,
    ConflictResolution.MERGE
)
```

### 2. Concurrent Access Manager (`concurrent_access_manager.py`)

Provides distributed locking and process management to ensure data consistency during concurrent operations.

**Key Features:**
- Priority-based locking (trading operations get highest priority)
- Multiple lock types (READ, WRITE, EXCLUSIVE)
- Process registration and heartbeat monitoring
- Deadlock prevention and automatic cleanup
- Lock statistics and monitoring

**Usage:**
```python
from services.concurrent_access_manager import (
    get_access_manager,
    with_trading_priority_lock,
    with_query_lock
)

# Register a process
access_manager = await get_access_manager()
await access_manager.register_process("my_process", "intelligence_layer")

# Use context manager for automatic lock management
async with access_manager.acquire_lock(
    resource_id="knowledge_base",
    lock_type=LockType.READ,
    operation_type=OperationType.QUERY,
    process_id="my_process",
    priority=Priority.NORMAL
):
    # Perform operations with lock held
    pass

# Convenience functions for common operations
result = await with_query_lock(
    resource_id="knowledge_base",
    process_id="my_process",
    operation_func=my_search_function
)
```

### 3. Knowledge Query Interface (`knowledge_query_interface.py`)

Provides unified search and query capabilities across the ingested knowledge base.

**Key Features:**
- Multiple search types (semantic, keyword, hybrid, domain-filtered)
- Vector similarity search using embeddings
- Full-text keyword search
- Domain-based filtering
- Result ranking and pagination
- Query caching for performance

**Usage:**
```python
from services.knowledge_query_interface import (
    semantic_search,
    keyword_search,
    hybrid_search,
    SearchQuery,
    SearchType
)

# Simple semantic search
response = await semantic_search(
    query_text="machine learning algorithms",
    limit=10,
    similarity_threshold=0.7,
    domains=["ML", "finance"]
)

# Advanced search with custom parameters
query = SearchQuery(
    query_text="portfolio optimization",
    search_type=SearchType.HYBRID,
    domains=["finance"],
    limit=20,
    sort_order=SortOrder.RELEVANCE,
    filters={"min_similarity": 0.8}
)

response = await query_interface.search(query)
```

### 4. Platform API Endpoints (`platform_api_endpoints.py`)

RESTful API endpoints that integrate with the existing platform architecture.

**Available Endpoints:**

- `GET /health` - Health check and service status
- `POST /search` - Advanced search with full parameters
- `GET /search/semantic` - Quick semantic search
- `GET /search/keyword` - Quick keyword search  
- `GET /search/hybrid` - Quick hybrid search
- `GET /documents/{id}` - Get document by ID
- `GET /documents/{id}/similar` - Find similar documents
- `GET /statistics` - System statistics
- `GET /domains` - Available knowledge domains
- `POST /test-connection` - Test platform connections

**Usage:**
```python
from services.platform_api_endpoints import get_api_service

# Initialize API service
api_service = get_api_service()
await api_service.initialize()

# Start server (in production)
await api_service.start_server(host="0.0.0.0", port=8080)
```

## Integration with Trading Platform

### Priority-Based Access Control

The system ensures trading operations always get priority access to shared resources:

```python
# Trading operations get CRITICAL priority
async with access_manager.acquire_lock(
    resource_id="market_data",
    lock_type=LockType.EXCLUSIVE,
    operation_type=OperationType.TRADING,
    process_id="trading_engine",
    priority=Priority.CRITICAL  # Highest priority
):
    # Trading operations execute immediately
    pass

# Ingestion operations use NORMAL priority
async with access_manager.acquire_lock(
    resource_id="knowledge_base",
    lock_type=LockType.WRITE,
    operation_type=OperationType.INGESTION,
    process_id="ingestion_worker",
    priority=Priority.NORMAL  # Lower priority
):
    # Ingestion waits for trading operations
    pass
```

### Connection Management

The system automatically connects to existing platform services:

- **Supabase**: Uses existing database for document and chunk storage
- **Intelligence Layer**: Integrates with existing FastAPI intelligence service
- **Neo4j**: Reserved for future graph analytics integration

### Conflict Resolution

When multiple processes attempt to modify the same data, the system provides several resolution strategies:

- **SKIP**: Skip conflicting operations (safe default)
- **OVERWRITE**: Replace existing data with new data
- **MERGE**: Combine existing and new data
- **VERSION**: Create new version while preserving old data

## Configuration

The platform integration uses the existing configuration system:

```yaml
# config/config.yaml
supabase:
  url: "your-supabase-url"
  service_role_key: "your-service-role-key"
  max_connections: 10
  timeout: 30

embeddings:
  openai_api_key: "your-openai-key"
  batch_size: 32
  use_gpu: true

processing:
  max_concurrent_operations: 10
  default_lock_timeout: 30.0
```

## Error Handling

The system provides comprehensive error handling:

- **Connection Failures**: Automatic retry with exponential backoff
- **Lock Timeouts**: Configurable timeouts with clear error messages
- **Conflict Resolution**: Graceful handling of data conflicts
- **Process Failures**: Automatic cleanup of stale processes and locks

## Monitoring and Statistics

The system provides detailed monitoring capabilities:

```python
# Get connection status
status = await integration_service.get_connection_status()

# Get lock statistics
stats = access_manager.get_lock_statistics()

# Get search statistics
search_stats = await query_interface.get_search_statistics()
```

## Testing

Run the platform integration tests:

```bash
cd scripts/knowledge-ingestion
python test_platform_integration.py
```

Run the integration example:

```bash
python platform_integration_example.py
```

## Production Deployment

For production deployment:

1. **Configure Environment Variables**:
   ```bash
   export SUPABASE_URL="your-production-url"
   export SUPABASE_SERVICE_ROLE_KEY="your-production-key"
   export OPENAI_API_KEY="your-openai-key"
   ```

2. **Start API Server**:
   ```bash
   python -m services.platform_api_endpoints
   ```

3. **Monitor Health**:
   ```bash
   curl http://localhost:8080/health
   ```

## Integration Examples

### Intelligence Layer Integration

```python
# In your intelligence layer service
from scripts.knowledge_ingestion.services.knowledge_query_interface import semantic_search

async def enhanced_research_query(query: str):
    # Use knowledge base for enhanced research
    knowledge_results = await semantic_search(
        query_text=query,
        domains=["finance", "ML"],
        limit=5
    )
    
    # Combine with existing intelligence capabilities
    return combine_knowledge_and_intelligence(knowledge_results, query)
```

### Trading Strategy Integration

```python
# In your trading strategy
from scripts.knowledge_ingestion.services.concurrent_access_manager import with_trading_priority_lock

async def execute_strategy_with_research():
    # Get priority access to knowledge base
    research_data = await with_trading_priority_lock(
        resource_id="knowledge_base",
        process_id="strategy_engine",
        operation_func=lambda: semantic_search("market regime detection")
    )
    
    # Use research data in trading decisions
    return make_trading_decision(research_data)
```

## Troubleshooting

### Common Issues

1. **Connection Failures**:
   - Check Supabase credentials and URL
   - Verify network connectivity
   - Check service status with `/health` endpoint

2. **Lock Timeouts**:
   - Increase timeout values in configuration
   - Check for deadlocks in lock statistics
   - Verify process heartbeats are working

3. **Search Performance**:
   - Check embedding service initialization
   - Verify vector indexes are created
   - Monitor query cache hit rates

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about:
- Connection attempts and failures
- Lock acquisition and release
- Search query execution
- Conflict detection and resolution

## Future Enhancements

Planned improvements:

1. **Neo4j Integration**: Full graph analytics capabilities
2. **Advanced Caching**: Redis-based distributed caching
3. **Metrics Export**: Prometheus metrics for monitoring
4. **Load Balancing**: Multiple API server instances
5. **Real-time Updates**: WebSocket support for live updates
"""
Platform integration example demonstrating how to use the knowledge ingestion system
with the existing algorithmic trading platform.

This example shows:
- How to initialize platform integration
- How to perform searches with concurrent access management
- How to integrate with the intelligence layer
- How to use the API endpoints
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from services.platform_integration import get_integration_service
from services.concurrent_access_manager import (
    get_access_manager, 
    LockType, 
    OperationType, 
    Priority,
    with_query_lock,
    with_ingestion_lock
)
from services.knowledge_query_interface import (
    get_query_interface,
    SearchQuery,
    SearchType,
    semantic_search,
    keyword_search,
    hybrid_search
)
from services.platform_api_endpoints import get_api_service
from core.logging import get_logger


async def demonstrate_platform_integration():
    """Demonstrate platform integration functionality"""
    logger = get_logger(__name__)
    logger.info("Starting platform integration demonstration")
    
    try:
        # 1. Initialize platform integration
        logger.info("Initializing platform integration services...")
        
        integration_service = await get_integration_service()
        access_manager = await get_access_manager()
        query_interface = await get_query_interface()
        
        logger.info("Platform integration services initialized successfully")
        
        # 2. Check connection status
        logger.info("Checking platform connections...")
        
        connection_status = await integration_service.get_connection_status()
        logger.info(f"Connection status: {connection_status['overall_status']}")
        
        for service_name, connection in connection_status['connections'].items():
            logger.info(f"  {service_name}: {connection['status']}")
        
        # 3. Register a process for concurrent access management
        process_id = "demo_process_001"
        logger.info(f"Registering process: {process_id}")
        
        await access_manager.register_process(
            process_id=process_id,
            process_type="intelligence_layer",
            metadata={"demo": True, "started_at": datetime.now().isoformat()}
        )
        
        # 4. Demonstrate search functionality with concurrent access
        logger.info("Demonstrating search functionality...")
        
        async def perform_search_with_lock():
            """Perform search with proper locking"""
            try:
                # Semantic search example
                response = await semantic_search(
                    query_text="machine learning algorithms",
                    limit=5,
                    similarity_threshold=0.7,
                    domains=["ML", "finance"]
                )
                
                logger.info(f"Semantic search found {response.total_results} results in {response.execution_time_ms:.2f}ms")
                
                for i, result in enumerate(response.results[:3]):
                    logger.info(f"  Result {i+1}: {result.document_title} (score: {result.similarity_score:.3f})")
                
                return response
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return None
        
        # Execute search with query lock
        search_response = await with_query_lock(
            resource_id="knowledge_base",
            process_id=process_id,
            operation_func=perform_search_with_lock,
            timeout_seconds=10.0
        )
        
        # 5. Demonstrate different search types
        logger.info("Demonstrating different search types...")
        
        search_queries = [
            ("reinforcement learning", SearchType.SEMANTIC),
            ("portfolio optimization", SearchType.KEYWORD),
            ("neural networks trading", SearchType.HYBRID)
        ]
        
        for query_text, search_type in search_queries:
            try:
                if search_type == SearchType.SEMANTIC:
                    response = await semantic_search(query_text, limit=3)
                elif search_type == SearchType.KEYWORD:
                    response = await keyword_search(query_text, limit=3)
                elif search_type == SearchType.HYBRID:
                    response = await hybrid_search(query_text, limit=3)
                
                logger.info(f"{search_type.value.title()} search for '{query_text}': {response.total_results} results")
                
            except Exception as e:
                logger.warning(f"Search failed for '{query_text}': {e}")
        
        # 6. Demonstrate concurrent access statistics
        logger.info("Getting access management statistics...")
        
        lock_stats = access_manager.get_lock_statistics()
        logger.info(f"Active locks: {lock_stats['active_locks']}")
        logger.info(f"Active processes: {lock_stats['active_processes']}")
        
        process_info = access_manager.get_process_info(process_id)
        if process_info:
            logger.info(f"Process info: {process_info['process_type']} started at {process_info['started_at']}")
        
        # 7. Demonstrate conflict detection (mock example)
        logger.info("Demonstrating conflict detection...")
        
        test_document = {
            'file_id': 'demo_file_123',
            'title': 'Demo Document',
            'content': 'This is a demonstration document for testing conflict detection.',
            'processing_status': 'completed'
        }
        
        conflict_result = await integration_service.detect_conflicts(
            'documents',
            test_document,
            'file_id'
        )
        
        logger.info(f"Conflict detection result: {conflict_result.conflict_type}")
        
        # 8. Test connection health
        logger.info("Testing connection health...")
        
        test_results = await integration_service.test_connections()
        for service, result in test_results.items():
            status = "✓" if result else "✗"
            logger.info(f"  {service}: {status}")
        
        # 9. Cleanup
        logger.info("Cleaning up...")
        
        await access_manager.unregister_process(process_id)
        
        logger.info("Platform integration demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Platform integration demonstration failed: {e}")
        raise


async def demonstrate_api_endpoints():
    """Demonstrate API endpoints functionality"""
    logger = get_logger(__name__)
    logger.info("Demonstrating API endpoints...")
    
    try:
        # Get API service
        api_service = get_api_service()
        await api_service.initialize()
        
        # The API service is now ready to handle requests
        # In a real scenario, you would start the server with:
        # await api_service.start_server(host="0.0.0.0", port=8080)
        
        logger.info("API service initialized successfully")
        logger.info("Available endpoints:")
        
        routes = [route.path for route in api_service.app.routes if hasattr(route, 'path')]
        for route in sorted(routes):
            if not route.startswith('/openapi') and not route.startswith('/docs'):
                logger.info(f"  {route}")
        
        logger.info("API endpoints demonstration completed")
        
    except Exception as e:
        logger.error(f"API endpoints demonstration failed: {e}")


async def demonstrate_trading_priority_access():
    """Demonstrate trading priority access management"""
    logger = get_logger(__name__)
    logger.info("Demonstrating trading priority access...")
    
    try:
        access_manager = await get_access_manager()
        
        # Register trading and ingestion processes
        trading_process = "trading_engine_001"
        ingestion_process = "ingestion_worker_001"
        
        await access_manager.register_process(trading_process, "trading")
        await access_manager.register_process(ingestion_process, "ingestion")
        
        # Simulate concurrent access where trading gets priority
        async def trading_operation():
            """Simulate a trading operation that needs priority access"""
            async with access_manager.acquire_lock(
                resource_id="market_data",
                lock_type=LockType.EXCLUSIVE,
                operation_type=OperationType.TRADING,
                process_id=trading_process,
                priority=Priority.CRITICAL,
                timeout_seconds=5.0
            ):
                logger.info("Trading operation acquired exclusive lock")
                await asyncio.sleep(0.5)  # Simulate work
                logger.info("Trading operation completed")
        
        async def ingestion_operation():
            """Simulate an ingestion operation with normal priority"""
            async with access_manager.acquire_lock(
                resource_id="market_data",
                lock_type=LockType.WRITE,
                operation_type=OperationType.INGESTION,
                process_id=ingestion_process,
                priority=Priority.NORMAL,
                timeout_seconds=10.0
            ):
                logger.info("Ingestion operation acquired write lock")
                await asyncio.sleep(0.2)  # Simulate work
                logger.info("Ingestion operation completed")
        
        # Start both operations concurrently
        # Trading should get priority due to CRITICAL priority
        await asyncio.gather(
            trading_operation(),
            ingestion_operation()
        )
        
        # Cleanup
        await access_manager.unregister_process(trading_process)
        await access_manager.unregister_process(ingestion_process)
        
        logger.info("Trading priority access demonstration completed")
        
    except Exception as e:
        logger.error(f"Trading priority access demonstration failed: {e}")


async def main():
    """Main demonstration function"""
    logger = get_logger(__name__)
    
    try:
        logger.info("=== Knowledge Ingestion Platform Integration Demo ===")
        
        # Run demonstrations
        await demonstrate_platform_integration()
        print()
        
        await demonstrate_api_endpoints()
        print()
        
        await demonstrate_trading_priority_access()
        print()
        
        logger.info("=== All demonstrations completed successfully ===")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise
    
    finally:
        # Cleanup any remaining resources
        try:
            access_manager = await get_access_manager()
            await access_manager.stop()
        except:
            pass


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    asyncio.run(main())
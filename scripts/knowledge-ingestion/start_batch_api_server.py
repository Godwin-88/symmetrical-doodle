#!/usr/bin/env python3
"""
Start the multi-source API server with batch ingestion management support.

This script starts the FastAPI server that provides:
- Multi-source authentication endpoints
- Unified source browsing API
- Batch ingestion management API
- WebSocket support for real-time progress tracking
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import load_config
from core.logging import setup_logging, get_logger
from services.multi_source_api_endpoints import get_api_service


class BatchAPIServer:
    """Batch ingestion API server manager"""
    
    def __init__(self):
        self.logger = None
        self.api_service = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the server"""
        try:
            # Load configuration
            config = load_config()
            
            # Setup logging
            setup_logging(config.logging)
            self.logger = get_logger(__name__)
            
            self.logger.info("Initializing Batch Ingestion API Server")
            self.logger.info(f"Environment: {config.environment}")
            self.logger.info(f"Debug mode: {config.debug}")
            
            # Initialize API service
            self.api_service = get_api_service()
            await self.api_service.initialize()
            
            self.logger.info("Server initialization completed")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize server: {e}")
            else:
                print(f"Failed to initialize server: {e}")
            return False
    
    async def start(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the API server"""
        try:
            if not await self.initialize():
                return False
            
            self.logger.info(f"Starting Batch Ingestion API Server on {host}:{port}")
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, initiating shutdown")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start the server
            await self.api_service.start_server(host, port)
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the server gracefully"""
        try:
            self.logger.info("Shutting down Batch Ingestion API Server")
            
            # Shutdown batch manager
            if self.api_service and self.api_service._batch_manager:
                await self.api_service._batch_manager.shutdown()
            
            # Set shutdown event
            self.shutdown_event.set()
            
            self.logger.info("Server shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Ingestion API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--environment", help="Environment to run in")
    
    args = parser.parse_args()
    
    # Override environment if specified
    if args.environment:
        import os
        os.environ["KNOWLEDGE_INGESTION_ENV"] = args.environment
    
    # Create and start server
    server = BatchAPIServer()
    
    try:
        await server.start(args.host, args.port)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await server.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
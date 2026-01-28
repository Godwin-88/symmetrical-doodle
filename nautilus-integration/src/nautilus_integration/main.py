"""
Main entry point for NautilusTrader integration service.

This module provides the FastAPI application and CLI interface for the
NautilusTrader integration service.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nautilus_integration.core.config import NautilusConfig, load_config
from nautilus_integration.core.logging import get_logger, setup_logging, with_correlation_id
from nautilus_integration.services.integration_service import (
    BacktestAnalysis,
    BacktestConfig,
    NautilusIntegrationService,
    SessionSummary,
    StrategyConfig,
    TradingSession,
)

# Global service instance
integration_service: Optional[NautilusIntegrationService] = None
logger = structlog.get_logger("nautilus_integration.main")


def create_app(config: NautilusConfig) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: NautilusTrader integration configuration
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="NautilusTrader Integration API",
        description="Integration service for NautilusTrader with existing trading platform",
        version="0.1.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        health_status = await integration_service.get_health_status()
        return JSONResponse(content=health_status)
    
    # Backtest endpoints
    @app.post("/api/v1/backtests", response_model=dict)
    async def create_backtest(
        backtest_config: BacktestConfig,
        strategies: list[StrategyConfig],
    ):
        """Create and execute a backtest."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        with with_correlation_id() as correlation_id:
            try:
                result = await integration_service.create_backtest(
                    backtest_config, strategies
                )
                return {
                    "backtest_id": backtest_config.backtest_id,
                    "status": "completed",
                    "correlation_id": correlation_id,
                    "result": result.model_dump() if hasattr(result, 'model_dump') else str(result)
                }
            except Exception as error:
                logger.error(
                    "Backtest creation failed",
                    backtest_id=backtest_config.backtest_id,
                    error=str(error),
                    correlation_id=correlation_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Backtest creation failed: {error}"
                )
    
    @app.get("/api/v1/backtests/{backtest_id}", response_model=BacktestAnalysis)
    async def get_backtest_results(backtest_id: str):
        """Get backtest results."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            analysis = await integration_service.get_backtest_results(backtest_id)
            return analysis
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error))
        except Exception as error:
            logger.error(
                "Failed to retrieve backtest results",
                backtest_id=backtest_id,
                error=str(error),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve backtest results: {error}"
            )
    
    # Live trading endpoints
    @app.post("/api/v1/trading/sessions", response_model=TradingSession)
    async def start_trading_session(
        strategies: list[StrategyConfig],
        risk_limits: dict = None,
    ):
        """Start a live trading session."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if risk_limits is None:
            risk_limits = {}
        
        with with_correlation_id() as correlation_id:
            try:
                session = await integration_service.start_live_trading(
                    strategies, risk_limits
                )
                return session
            except Exception as error:
                logger.error(
                    "Failed to start trading session",
                    error=str(error),
                    correlation_id=correlation_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start trading session: {error}"
                )
    
    @app.delete("/api/v1/trading/sessions/{session_id}", response_model=SessionSummary)
    async def stop_trading_session(session_id: str):
        """Stop a trading session."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        try:
            summary = await integration_service.stop_trading_session(session_id)
            return summary
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error))
        except Exception as error:
            logger.error(
                "Failed to stop trading session",
                session_id=session_id,
                error=str(error),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop trading session: {error}"
            )
    
    # Configuration endpoint
    @app.get("/api/v1/config")
    async def get_configuration():
        """Get current configuration (sanitized)."""
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Return sanitized configuration (remove sensitive data)
        config_dict = integration_service.config.to_dict()
        
        # Remove sensitive information
        if "database" in config_dict:
            for key in config_dict["database"]:
                if any(sensitive in key.lower() for sensitive in ["password", "key", "secret"]):
                    config_dict["database"][key] = "***REDACTED***"
        
        return config_dict
    
    return app


async def initialize_service(config: NautilusConfig) -> NautilusIntegrationService:
    """
    Initialize the NautilusTrader integration service.
    
    Args:
        config: NautilusTrader integration configuration
        
    Returns:
        Initialized integration service
    """
    global integration_service
    
    logger.info("Initializing NautilusTrader integration service")
    
    try:
        integration_service = NautilusIntegrationService(config)
        await integration_service.initialize()
        
        logger.info("NautilusTrader integration service initialized successfully")
        return integration_service
        
    except Exception as error:
        logger.error(
            "Failed to initialize integration service",
            error=str(error),
            exc_info=True,
        )
        raise


async def shutdown_service() -> None:
    """Shutdown the integration service gracefully."""
    global integration_service
    
    if integration_service is not None:
        logger.info("Shutting down NautilusTrader integration service")
        try:
            await integration_service.shutdown()
            logger.info("Integration service shutdown completed")
        except Exception as error:
            logger.error(
                "Error during service shutdown",
                error=str(error),
                exc_info=True,
            )
        finally:
            integration_service = None


def run_server(config: NautilusConfig) -> None:
    """
    Run the FastAPI server.
    
    Args:
        config: NautilusTrader integration configuration
    """
    app = create_app(config)
    
    # Add startup and shutdown event handlers
    @app.on_event("startup")
    async def startup_event():
        await initialize_service(config)
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown_service()
    
    # Run server
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.logging.level.lower(),
        access_log=config.debug,
    )


def run_cli_command(args: argparse.Namespace) -> None:
    """
    Run CLI command.
    
    Args:
        args: Parsed command line arguments
    """
    if args.command == "validate-config":
        try:
            config = load_config(args.config_file, validate=True)
            print("✅ Configuration validation passed")
            print(f"Environment: {config.environment}")
            print(f"API endpoint: http://{config.api_host}:{config.api_port}")
        except Exception as error:
            print(f"❌ Configuration validation failed: {error}")
            sys.exit(1)
    
    elif args.command == "test-connection":
        async def test_connection():
            try:
                config = load_config(args.config_file, validate=False)
                service = NautilusIntegrationService(config)
                await service.initialize()
                
                health = await service.get_health_status()
                print("✅ Connection test passed")
                print(f"Service status: {health['service_status']}")
                print(f"Components: {health['components']}")
                
                await service.shutdown()
            except Exception as error:
                print(f"❌ Connection test failed: {error}")
                sys.exit(1)
        
        asyncio.run(test_connection())
    
    elif args.command == "create-config":
        config_path = Path(args.output or "config/.env")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy from example if available
        example_path = config_path.parent / ".env.example"
        if example_path.exists():
            import shutil
            shutil.copy2(example_path, config_path)
            print(f"✅ Configuration created from example: {config_path}")
        else:
            # Create minimal configuration
            minimal_config = """# NautilusTrader Integration Configuration
NAUTILUS_INTEGRATION_ENV=development
NAUTILUS_DEBUG=true
NAUTILUS_DATABASE__POSTGRES_URL=postgresql://postgres:password@localhost:5432/trading_system
NAUTILUS_DATABASE__NEO4J_URL=bolt://localhost:7687
NAUTILUS_DATABASE__REDIS_URL=redis://localhost:6379
NAUTILUS_API_PORT=8002
"""
            config_path.write_text(minimal_config)
            print(f"✅ Minimal configuration created: {config_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NautilusTrader Integration Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the API server
  python -m nautilus_integration.main

  # Run with custom configuration
  python -m nautilus_integration.main --config config/.env

  # Validate configuration
  python -m nautilus_integration.main validate-config

  # Test database connections
  python -m nautilus_integration.main test-connection

  # Create configuration file
  python -m nautilus_integration.main create-config --output config/.env
        """
    )
    
    parser.add_argument(
        "--config",
        dest="config_file",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate configuration file"
    )
    
    # Test connection command
    test_parser = subparsers.add_parser(
        "test-connection",
        help="Test database and service connections"
    )
    
    # Create config command
    create_parser = subparsers.add_parser(
        "create-config",
        help="Create configuration file"
    )
    create_parser.add_argument(
        "--output",
        help="Output path for configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config_file, validate=False)
        
        # Override debug mode if specified
        if args.debug:
            config.debug = True
            config.logging.level = "DEBUG"
        
        # Setup logging
        setup_logging(config.logging)
        
        # Run command or server
        if args.command:
            run_cli_command(args)
        else:
            run_server(config)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    except Exception as error:
        logger.error(
            "Application failed to start",
            error=str(error),
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
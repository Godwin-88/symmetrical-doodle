#!/usr/bin/env python3
"""
Main entry point for Google Drive Knowledge Base Ingestion System.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

from core.config import ConfigManager, get_settings
from core.logging import configure_logging, get_logger, set_correlation_id


async def main():
    """Main application entry point"""
    
    # Set up correlation ID for this execution
    correlation_id = set_correlation_id()
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__, component="main")
    
    logger.info(
        "Starting Google Drive Knowledge Base Ingestion System",
        correlation_id=correlation_id,
        version="1.0.0"
    )
    
    try:
        # Load configuration
        settings = get_settings()
        logger.info(
            "Configuration loaded",
            environment=settings.environment,
            debug=settings.debug
        )
        
        # Validate configuration
        config_manager = ConfigManager()
        validation_results = config_manager.validate_config()
        
        if not validation_results["valid"]:
            logger.error(
                "Configuration validation failed",
                errors=validation_results["errors"]
            )
            return 1
            
        if validation_results["warnings"]:
            for warning in validation_results["warnings"]:
                logger.warning("Configuration warning", warning=warning)
        
        logger.info("Configuration validation passed")
        
        # TODO: Initialize and run the ingestion pipeline
        # This will be implemented in subsequent tasks
        logger.info("Ingestion pipeline initialization - TODO")
        
        logger.info("Google Drive Knowledge Base Ingestion System completed successfully")
        return 0
        
    except Exception as e:
        logger.error(
            "Application failed with unexpected error",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        return 1


def cli_main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Google Drive Knowledge Base Ingestion System"
    )
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production"],
        help="Environment to run in"
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=Path,
        help="Configuration directory path"
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Load configuration for specified environment
    if args.config_dir:
        config_manager = ConfigManager(args.config_dir)
    else:
        config_manager = ConfigManager()
        
    if args.environment:
        config_manager.load_config(args.environment)
    
    # If validate-only, just validate and exit
    if args.validate_only:
        configure_logging()
        logger = get_logger(__name__, component="validation")
        
        validation_results = config_manager.validate_config()
        
        if validation_results["valid"]:
            logger.info("Configuration validation passed")
            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    logger.warning("Configuration warning", warning=warning)
            return 0
        else:
            logger.error(
                "Configuration validation failed",
                errors=validation_results["errors"]
            )
            return 1
    
    # Run main application
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(cli_main())
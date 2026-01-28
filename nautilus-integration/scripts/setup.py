#!/usr/bin/env python3
"""
Idempotent setup script for NautilusTrader integration.

This script provides reproducible environment creation following the patterns
established in the knowledge-ingestion system.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import structlog

# Configure basic logging for setup
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("nautilus_integration.setup")


class SetupError(Exception):
    """Exception raised during setup process."""
    pass


class NautilusIntegrationSetup:
    """Idempotent setup manager for NautilusTrader integration."""
    
    def __init__(self, project_root: Path):
        """
        Initialize setup manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.integration_root = project_root / "nautilus-integration"
        self.venv_path = self.integration_root / ".venv"
        self.config_path = self.integration_root / "config"
        self.data_path = self.integration_root / "data"
        
        logger.info(
            "Setup manager initialized",
            project_root=str(project_root),
            integration_root=str(self.integration_root),
        )
    
    def run_setup(
        self,
        skip_venv: bool = False,
        skip_deps: bool = False,
        skip_config: bool = False,
        skip_data: bool = False,
        development: bool = False,
    ) -> None:
        """
        Run complete idempotent setup process.
        
        Args:
            skip_venv: Skip virtual environment creation
            skip_deps: Skip dependency installation
            skip_config: Skip configuration setup
            skip_data: Skip data directory creation
            development: Setup for development environment
        """
        logger.info("Starting NautilusTrader integration setup")
        
        try:
            # Validate prerequisites
            self._validate_prerequisites()
            
            # Create directory structure
            self._create_directory_structure()
            
            # Setup virtual environment
            if not skip_venv:
                self._setup_virtual_environment()
            
            # Install dependencies
            if not skip_deps:
                self._install_dependencies(development)
            
            # Setup configuration
            if not skip_config:
                self._setup_configuration()
            
            # Create data directories
            if not skip_data:
                self._create_data_directories()
            
            # Validate installation
            self._validate_installation()
            
            logger.info("NautilusTrader integration setup completed successfully")
            
        except Exception as error:
            logger.error(
                "Setup failed",
                error=str(error),
                error_type=type(error).__name__,
                exc_info=True,
            )
            raise SetupError(f"Setup failed: {error}")
    
    def _validate_prerequisites(self) -> None:
        """Validate system prerequisites."""
        logger.info("Validating prerequisites")
        
        # Check Python version
        if sys.version_info < (3, 11):
            raise SetupError("Python 3.11 or higher is required")
        
        # Check for required system tools
        required_tools = ["git", "curl"]
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            raise SetupError(f"Missing required tools: {', '.join(missing_tools)}")
        
        # Check for Rust (required for NautilusTrader)
        if not shutil.which("rustc"):
            logger.warning(
                "Rust not found - will attempt to install during dependency installation"
            )
        
        logger.info("Prerequisites validation completed")
    
    def _create_directory_structure(self) -> None:
        """Create necessary directory structure."""
        logger.info("Creating directory structure")
        
        directories = [
            self.integration_root,
            self.integration_root / "src",
            self.integration_root / "tests",
            self.integration_root / "scripts",
            self.config_path,
            self.data_path,
            self.data_path / "catalog",
            self.data_path / "logs",
            self.integration_root / "monitoring",
            self.integration_root / "database",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory", path=str(directory))
        
        logger.info("Directory structure created")
    
    def _setup_virtual_environment(self) -> None:
        """Setup Python virtual environment."""
        logger.info("Setting up virtual environment")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists, skipping creation")
            return
        
        # Create virtual environment
        subprocess.run([
            sys.executable, "-m", "venv", str(self.venv_path)
        ], check=True)
        
        # Upgrade pip in virtual environment
        pip_path = self._get_pip_path()
        subprocess.run([
            str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"
        ], check=True)
        
        logger.info("Virtual environment setup completed")
    
    def _install_dependencies(self, development: bool = False) -> None:
        """Install Python dependencies."""
        logger.info("Installing dependencies", development=development)
        
        pip_path = self._get_pip_path()
        requirements_file = self.integration_root / "requirements.txt"
        
        if not requirements_file.exists():
            raise SetupError(f"Requirements file not found: {requirements_file}")
        
        # Install main dependencies
        subprocess.run([
            str(pip_path), "install", "-r", str(requirements_file)
        ], check=True)
        
        # Install development dependencies if requested
        if development:
            subprocess.run([
                str(pip_path), "install", "-e", ".[dev]"
            ], cwd=str(self.integration_root), check=True)
        
        # Install the integration package in development mode
        if (self.integration_root / "pyproject.toml").exists():
            subprocess.run([
                str(pip_path), "install", "-e", "."
            ], cwd=str(self.integration_root), check=True)
        
        logger.info("Dependencies installation completed")
    
    def _setup_configuration(self) -> None:
        """Setup configuration files."""
        logger.info("Setting up configuration")
        
        env_example = self.config_path / ".env.example"
        env_file = self.config_path / ".env"
        
        # Copy example configuration if .env doesn't exist
        if env_example.exists() and not env_file.exists():
            shutil.copy2(env_example, env_file)
            logger.info("Created configuration file from example")
        elif env_file.exists():
            logger.info("Configuration file already exists, skipping")
        else:
            logger.warning("No configuration example found")
        
        # Create additional configuration files if needed
        self._create_monitoring_config()
        self._create_database_init_scripts()
        
        logger.info("Configuration setup completed")
    
    def _create_data_directories(self) -> None:
        """Create data directories with proper structure."""
        logger.info("Creating data directories")
        
        data_subdirs = [
            "catalog/bars",
            "catalog/ticks", 
            "catalog/order_book",
            "catalog/instruments",
            "logs",
            "backups",
            "temp",
        ]
        
        for subdir in data_subdirs:
            dir_path = self.data_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Created data directory", path=str(dir_path))
        
        # Create .gitkeep files to preserve empty directories
        for subdir in data_subdirs:
            gitkeep_path = self.data_path / subdir / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
        
        logger.info("Data directories created")
    
    def _create_monitoring_config(self) -> None:
        """Create monitoring configuration files."""
        monitoring_path = self.integration_root / "monitoring"
        
        # Create Prometheus configuration
        prometheus_config = monitoring_path / "prometheus.yml"
        if not prometheus_config.exists():
            prometheus_content = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nautilus-integration'
    static_configs:
      - targets: ['nautilus-integration:8002']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
            prometheus_config.write_text(prometheus_content.strip())
            logger.debug("Created Prometheus configuration")
        
        # Create Grafana directories
        grafana_dirs = [
            monitoring_path / "grafana" / "dashboards",
            monitoring_path / "grafana" / "datasources",
        ]
        
        for dir_path in grafana_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _create_database_init_scripts(self) -> None:
        """Create database initialization scripts."""
        database_path = self.integration_root / "database"
        init_path = database_path / "init"
        init_path.mkdir(parents=True, exist_ok=True)
        
        # Create PostgreSQL initialization script
        postgres_init = init_path / "01-nautilus-init.sql"
        if not postgres_init.exists():
            postgres_content = """
-- NautilusTrader Integration Database Initialization

-- Create extension for pgvector if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for Nautilus data
CREATE SCHEMA IF NOT EXISTS nautilus;

-- Create tables for Nautilus integration
CREATE TABLE IF NOT EXISTS nautilus.backtests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id VARCHAR(255) UNIQUE NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    results JSONB
);

CREATE TABLE IF NOT EXISTS nautilus.trading_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_backtests_status ON nautilus.backtests(status);
CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON nautilus.backtests(created_at);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_status ON nautilus.trading_sessions(status);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_created_at ON nautilus.trading_sessions(created_at);

-- Grant permissions
GRANT USAGE ON SCHEMA nautilus TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nautilus TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA nautilus TO postgres;
"""
            postgres_init.write_text(postgres_content.strip())
            logger.debug("Created PostgreSQL initialization script")
    
    def _validate_installation(self) -> None:
        """Validate the installation."""
        logger.info("Validating installation")
        
        python_path = self._get_python_path()
        
        # Test NautilusTrader import
        result = subprocess.run([
            str(python_path), "-c", 
            "import nautilus_trader; print(f'NautilusTrader {nautilus_trader.__version__}')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise SetupError(f"NautilusTrader validation failed: {result.stderr}")
        
        logger.info("NautilusTrader validation passed", version=result.stdout.strip())
        
        # Test integration package import
        result = subprocess.run([
            str(python_path), "-c",
            "import nautilus_integration; print('Integration package loaded')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise SetupError(f"Integration package validation failed: {result.stderr}")
        
        logger.info("Integration package validation passed")
        
        # Test configuration loading
        result = subprocess.run([
            str(python_path), "-c",
            "from nautilus_integration.core.config import load_config; "
            "config = load_config(validate=False); print('Configuration loaded')"
        ], capture_output=True, text=True, cwd=str(self.integration_root))
        
        if result.returncode != 0:
            logger.warning(
                "Configuration validation failed (non-critical)",
                error=result.stderr
            )
        else:
            logger.info("Configuration validation passed")
        
        logger.info("Installation validation completed")
    
    def _get_python_path(self) -> Path:
        """Get path to Python executable in virtual environment."""
        if os.name == "nt":  # Windows
            return self.venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            return self.venv_path / "bin" / "python"
    
    def _get_pip_path(self) -> Path:
        """Get path to pip executable in virtual environment."""
        if os.name == "nt":  # Windows
            return self.venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            return self.venv_path / "bin" / "pip"


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Idempotent setup for NautilusTrader integration"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd().parent,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip configuration setup"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data directory creation"
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Setup for development environment"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    try:
        setup = NautilusIntegrationSetup(args.project_root)
        setup.run_setup(
            skip_venv=args.skip_venv,
            skip_deps=args.skip_deps,
            skip_config=args.skip_config,
            skip_data=args.skip_data,
            development=args.development,
        )
        
        print("\n✅ NautilusTrader integration setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and customize config/.env")
        print("2. Run: docker-compose up -d --profile with-database")
        print("3. Test the integration: python -m nautilus_integration.main --help")
        
    except SetupError as error:
        print(f"\n❌ Setup failed: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
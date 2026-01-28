#!/usr/bin/env python3
"""
Deployment script for Google Drive Knowledge Base Ingestion.
Handles containerized and local deployments with validation.
"""

import asyncio
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager
from core.logging import setup_logging, get_logger
from validate_deployment import DeploymentValidator


class DeploymentManager:
    """Manages deployment of the knowledge ingestion system"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.settings = config_manager.settings
        self.logger = get_logger(__name__)
        self.base_dir = Path(__file__).parent
    
    async def deploy(self, deployment_type: str, validate_first: bool = True) -> bool:
        """Deploy the system with specified type"""
        
        self.logger.info(f"Starting {deployment_type} deployment")
        
        # Validate deployment first if requested
        if validate_first:
            self.logger.info("Running pre-deployment validation")
            validator = DeploymentValidator(self.config_manager)
            validation_passed = await validator.run_all_validations()
            
            if not validation_passed:
                self.logger.error("Pre-deployment validation failed")
                return False
            
            self.logger.info("Pre-deployment validation passed")
        
        # Execute deployment based on type
        if deployment_type == "local":
            return await self._deploy_local()
        elif deployment_type == "docker":
            return await self._deploy_docker()
        elif deployment_type == "docker-compose":
            return await self._deploy_docker_compose()
        else:
            self.logger.error(f"Unknown deployment type: {deployment_type}")
            return False
    
    async def _deploy_local(self) -> bool:
        """Deploy for local execution"""
        try:
            self.logger.info("Setting up local deployment")
            
            # Create necessary directories
            directories = [
                "data",
                "data/logs",
                "data/state",
                "data/extracted_pdfs",
                "credentials"
            ]
            
            for dir_name in directories:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            
            # Check Python dependencies
            if not await self._check_python_dependencies():
                return False
            
            # Create default configuration files if they don't exist
            self.config_manager.create_default_config_files()
            
            # Make scripts executable (on Unix systems)
            if sys.platform != "win32":
                script_files = ["run_ingestion.py", "validate_deployment.py"]
                for script_file in script_files:
                    script_path = self.base_dir / script_file
                    if script_path.exists():
                        script_path.chmod(0o755)
            
            self.logger.info("Local deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Local deployment failed: {e}")
            return False
    
    async def _deploy_docker(self) -> bool:
        """Deploy using Docker"""
        try:
            self.logger.info("Building Docker image")
            
            # Check if Docker is available
            if not shutil.which("docker"):
                self.logger.error("Docker is not installed or not in PATH")
                return False
            
            # Build Docker image
            build_cmd = [
                "docker", "build",
                "--target", "production",
                "--tag", "knowledge-ingestion:latest",
                "."
            ]
            
            result = subprocess.run(
                build_cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            self.logger.info("Docker image built successfully")
            
            # Validate the built image
            validate_cmd = [
                "docker", "run", "--rm",
                "knowledge-ingestion:latest",
                "python", "validate_deployment.py"
            ]
            
            result = subprocess.run(
                validate_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Container validation had issues: {result.stderr}")
            else:
                self.logger.info("Container validation passed")
            
            self.logger.info("Docker deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker deployment failed: {e}")
            return False
    
    async def _deploy_docker_compose(self) -> bool:
        """Deploy using Docker Compose"""
        try:
            self.logger.info("Deploying with Docker Compose")
            
            # Check if Docker Compose is available
            docker_compose_cmd = None
            for cmd in ["docker-compose", "docker compose"]:
                if shutil.which(cmd.split()[0]):
                    docker_compose_cmd = cmd.split()
                    break
            
            if not docker_compose_cmd:
                self.logger.error("Docker Compose is not installed or not in PATH")
                return False
            
            # Check if .env file exists
            env_file = self.base_dir / ".env"
            if not env_file.exists():
                self.logger.warning("No .env file found, creating from template")
                template_file = self.base_dir / ".env.docker"
                if template_file.exists():
                    shutil.copy(template_file, env_file)
                    self.logger.info("Created .env file from template - please configure it")
            
            # Build and start services
            build_cmd = docker_compose_cmd + ["build"]
            result = subprocess.run(
                build_cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Docker Compose build failed: {result.stderr}")
                return False
            
            self.logger.info("Docker Compose build completed successfully")
            
            # Validate the deployment
            validate_cmd = docker_compose_cmd + [
                "run", "--rm", "knowledge-ingestion",
                "python", "validate_deployment.py"
            ]
            
            result = subprocess.run(
                validate_cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Compose validation had issues: {result.stderr}")
            else:
                self.logger.info("Compose validation passed")
            
            self.logger.info("Docker Compose deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker Compose deployment failed: {e}")
            return False
    
    async def _check_python_dependencies(self) -> bool:
        """Check if Python dependencies are installed"""
        try:
            requirements_file = self.base_dir / "requirements.txt"
            if not requirements_file.exists():
                self.logger.warning("requirements.txt not found")
                return True
            
            # Try to import key packages
            import importlib
            
            key_packages = [
                "google.auth",
                "supabase",
                "openai",
                "pydantic",
                "yaml"
            ]
            
            missing_packages = []
            for package in key_packages:
                try:
                    importlib.import_module(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.logger.error(f"Missing Python packages: {missing_packages}")
                self.logger.info("Install with: pip install -r requirements.txt")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check Python dependencies: {e}")
            return False
    
    def generate_deployment_guide(self) -> str:
        """Generate deployment guide"""
        guide = f"""
# Google Drive Knowledge Base Ingestion - Deployment Guide

## Environment: {self.settings.environment}

### Local Deployment

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your settings
   ```

3. Add Google Drive credentials:
   ```bash
   # Place your service account JSON file in credentials/
   ```

4. Run deployment:
   ```bash
   python deploy.py local
   ```

5. Run ingestion:
   ```bash
   python run_ingestion.py
   ```

### Docker Deployment

1. Build image:
   ```bash
   python deploy.py docker
   # OR
   ./build.sh
   ```

2. Run container:
   ```bash
   docker run --rm -it \\
     -e SUPABASE_URL=your_url \\
     -e SUPABASE_KEY=your_key \\
     -e OPENAI_API_KEY=your_key \\
     -v ./credentials:/app/data/credentials:ro \\
     -v ./data/logs:/app/data/logs \\
     knowledge-ingestion:latest
   ```

### Docker Compose Deployment

1. Configure environment:
   ```bash
   cp .env.docker .env
   # Edit .env with your settings
   ```

2. Deploy:
   ```bash
   python deploy.py docker-compose
   ```

3. Run ingestion:
   ```bash
   docker-compose up knowledge-ingestion
   ```

### Validation

Run deployment validation:
```bash
python validate_deployment.py --verbose
```

### Configuration Files

- `config/config.yaml` - Base configuration
- `config/config.{self.settings.environment}.yaml` - Environment-specific config
- `config/.env` - Environment variables
- `credentials/` - Google Drive service account JSON

### Data Directories

- `data/logs/` - Application logs
- `data/state/` - Execution state for idempotent runs
- `data/extracted_pdfs/` - Cached PDF files

### Troubleshooting

1. Check logs in `data/logs/`
2. Run validation: `python validate_deployment.py`
3. Check configuration: `python -c "from core.config import get_settings; print(get_settings())"`
4. Test components individually using the test scripts

For more information, see the README.md file.
"""
        return guide


async def main():
    """Main entry point for deployment"""
    parser = argparse.ArgumentParser(
        description="Deploy Google Drive Knowledge Base Ingestion system"
    )
    
    parser.add_argument(
        "deployment_type",
        choices=["local", "docker", "docker-compose"],
        help="Type of deployment"
    )
    
    parser.add_argument(
        "--config-env",
        default=None,
        help="Configuration environment (development, production)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip pre-deployment validation"
    )
    
    parser.add_argument(
        "--generate-guide",
        action="store_true",
        help="Generate deployment guide and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        settings = config_manager.load_config(args.config_env)
        
        # Setup logging
        setup_logging(settings.logging)
        logger = get_logger(__name__)
        
        # Create deployment manager
        deployment_manager = DeploymentManager(config_manager)
        
        # Generate guide if requested
        if args.generate_guide:
            guide = deployment_manager.generate_deployment_guide()
            print(guide)
            
            # Save to file
            guide_file = Path(__file__).parent / "DEPLOYMENT_GUIDE.md"
            with open(guide_file, 'w') as f:
                f.write(guide)
            
            logger.info(f"Deployment guide saved to: {guide_file}")
            return 0
        
        # Run deployment
        success = await deployment_manager.deploy(
            args.deployment_type,
            validate_first=not args.no_validate
        )
        
        if success:
            logger.info("Deployment completed successfully")
            
            # Show next steps
            print("\n" + "="*50)
            print("DEPLOYMENT COMPLETED")
            print("="*50)
            print(f"Deployment type: {args.deployment_type}")
            print(f"Environment: {settings.environment}")
            print("\nNext steps:")
            
            if args.deployment_type == "local":
                print("1. Configure your .env file in config/")
                print("2. Add Google Drive credentials to credentials/")
                print("3. Run: python run_ingestion.py")
            elif args.deployment_type == "docker":
                print("1. Configure environment variables")
                print("2. Mount credentials directory")
                print("3. Run: docker run knowledge-ingestion:latest")
            elif args.deployment_type == "docker-compose":
                print("1. Configure .env file")
                print("2. Add credentials to credentials/")
                print("3. Run: docker-compose up knowledge-ingestion")
            
            print("\nFor detailed instructions, run:")
            print("python deploy.py --generate-guide")
            
            return 0
        else:
            logger.error("Deployment failed")
            return 1
            
    except Exception as e:
        print(f"Deployment failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
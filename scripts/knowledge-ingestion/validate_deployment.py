#!/usr/bin/env python3
"""
Deployment validation script for Google Drive Knowledge Base Ingestion.
Verifies all components are properly configured and functional.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager, get_settings
from core.logging import setup_logging, get_logger


class DeploymentValidator:
    """Validates deployment configuration and functionality"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.settings = config_manager.settings
        self.logger = get_logger(__name__)
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.settings.environment,
            "overall_status": "unknown",
            "checks": {}
        }
    
    async def run_all_validations(self) -> bool:
        """Run all deployment validations"""
        self.logger.info("Starting deployment validation")
        
        validation_methods = [
            ("configuration", self._validate_configuration),
            ("environment_variables", self._validate_environment_variables),
            ("file_permissions", self._validate_file_permissions),
            ("dependencies", self._validate_dependencies),
            ("google_drive_auth", self._validate_google_drive_auth),
            ("supabase_connection", self._validate_supabase_connection),
            ("embedding_services", self._validate_embedding_services),
            ("storage_directories", self._validate_storage_directories),
            ("containerization", self._validate_containerization),
        ]
        
        all_passed = True
        
        for check_name, method in validation_methods:
            self.logger.info(f"Running validation: {check_name}")
            try:
                result = await method()
                self.validation_results["checks"][check_name] = result
                
                if not result.get("passed", False):
                    all_passed = False
                    self.logger.error(f"Validation failed: {check_name}")
                    if result.get("error"):
                        self.logger.error(f"  Error: {result['error']}")
                else:
                    self.logger.info(f"Validation passed: {check_name}")
                    
            except Exception as e:
                all_passed = False
                self.validation_results["checks"][check_name] = {
                    "passed": False,
                    "error": str(e),
                    "details": "Validation method threw exception"
                }
                self.logger.error(f"Validation error: {check_name} - {e}")
        
        self.validation_results["overall_status"] = "passed" if all_passed else "failed"
        self.logger.info(f"Deployment validation {'passed' if all_passed else 'failed'}")
        
        return all_passed
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files and settings"""
        try:
            # Use the config manager's validation
            validation_results = self.config_manager.validate_config()
            
            return {
                "passed": validation_results["valid"],
                "error": "; ".join(validation_results["errors"]) if validation_results["errors"] else None,
                "warnings": validation_results["warnings"],
                "details": {
                    "config_file_exists": (Path(__file__).parent / "config" / "config.yaml").exists(),
                    "environment_config_exists": (Path(__file__).parent / "config" / f"config.{self.settings.environment}.yaml").exists(),
                    "env_file_exists": (Path(__file__).parent / "config" / ".env").exists(),
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Configuration validation failed"
            }
    
    async def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validate required environment variables"""
        try:
            required_vars = {
                "SUPABASE_URL": self.settings.supabase.url,
                "SUPABASE_KEY": self.settings.supabase.key,
            }
            
            optional_vars = {
                "OPENAI_API_KEY": self.settings.embeddings.openai_api_key,
                "HUGGINGFACE_API_TOKEN": getattr(self.settings.embeddings, 'huggingface_api_token', None),
                "GOOGLE_DRIVE_CREDENTIALS_PATH": self.settings.google_drive.credentials_path,
            }
            
            missing_required = []
            missing_optional = []
            
            for var_name, value in required_vars.items():
                if not value:
                    missing_required.append(var_name)
            
            for var_name, value in optional_vars.items():
                if not value:
                    missing_optional.append(var_name)
            
            return {
                "passed": len(missing_required) == 0,
                "error": f"Missing required variables: {missing_required}" if missing_required else None,
                "warnings": f"Missing optional variables: {missing_optional}" if missing_optional else None,
                "details": {
                    "required_vars_present": len(required_vars) - len(missing_required),
                    "required_vars_total": len(required_vars),
                    "optional_vars_present": len(optional_vars) - len(missing_optional),
                    "optional_vars_total": len(optional_vars),
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Environment variable validation failed"
            }
    
    async def _validate_file_permissions(self) -> Dict[str, Any]:
        """Validate file and directory permissions"""
        try:
            checks = {}
            all_passed = True
            
            # Check script directory permissions
            script_dir = Path(__file__).parent
            checks["script_directory_readable"] = script_dir.is_dir() and os.access(script_dir, os.R_OK)
            checks["script_directory_writable"] = os.access(script_dir, os.W_OK)
            
            # Check config directory
            config_dir = script_dir / "config"
            checks["config_directory_readable"] = config_dir.is_dir() and os.access(config_dir, os.R_OK)
            
            # Check credentials file if specified
            if self.settings.google_drive.credentials_path:
                cred_path = Path(self.settings.google_drive.credentials_path)
                if not cred_path.is_absolute():
                    cred_path = script_dir / cred_path
                checks["credentials_file_readable"] = cred_path.exists() and os.access(cred_path, os.R_OK)
            
            # Check data directories
            data_dirs = ["data", "data/logs", "data/state", "data/extracted_pdfs"]
            for dir_name in data_dirs:
                dir_path = script_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                checks[f"{dir_name.replace('/', '_')}_writable"] = os.access(dir_path, os.W_OK)
            
            # Check if any critical checks failed
            critical_checks = ["script_directory_readable", "config_directory_readable"]
            for check in critical_checks:
                if not checks.get(check, False):
                    all_passed = False
            
            return {
                "passed": all_passed,
                "error": "Critical file permission checks failed" if not all_passed else None,
                "details": checks
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "File permission validation failed"
            }
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate Python dependencies are installed"""
        try:
            import importlib
            
            required_packages = [
                "google.auth",
                "googleapiclient",
                "supabase",
                "openai",
                "transformers",
                "sentence_transformers",
                "marker",
                "pymupdf",
                "numpy",
                "pandas",
                "pydantic",
                "yaml",
                "structlog",
                "asyncio",
                "aiohttp",
            ]
            
            missing_packages = []
            installed_packages = []
            
            for package in required_packages:
                try:
                    importlib.import_module(package.replace("-", "_"))
                    installed_packages.append(package)
                except ImportError:
                    missing_packages.append(package)
            
            return {
                "passed": len(missing_packages) == 0,
                "error": f"Missing packages: {missing_packages}" if missing_packages else None,
                "details": {
                    "installed_packages": installed_packages,
                    "missing_packages": missing_packages,
                    "total_required": len(required_packages)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Dependency validation failed"
            }
    
    async def _validate_google_drive_auth(self) -> Dict[str, Any]:
        """Validate Google Drive authentication"""
        try:
            if not self.settings.google_drive.credentials_path:
                return {
                    "passed": False,
                    "error": "Google Drive credentials path not configured",
                    "details": "Skipping Google Drive authentication test"
                }
            
            # Check if credentials file exists
            cred_path = Path(self.settings.google_drive.credentials_path)
            if not cred_path.is_absolute():
                cred_path = Path(__file__).parent / cred_path
            
            if not cred_path.exists():
                return {
                    "passed": False,
                    "error": f"Credentials file not found: {cred_path}",
                    "details": "Cannot test Google Drive authentication"
                }
            
            # Try to load and validate credentials
            try:
                from services.google_drive_auth import GoogleDriveAuthService
                auth_service = GoogleDriveAuthService()
                
                # This would attempt authentication
                # For validation, we'll just check if the file is valid JSON
                with open(cred_path, 'r') as f:
                    cred_data = json.load(f)
                
                required_fields = ["type", "client_email", "private_key"]
                missing_fields = [field for field in required_fields if field not in cred_data]
                
                return {
                    "passed": len(missing_fields) == 0,
                    "error": f"Invalid credentials file, missing fields: {missing_fields}" if missing_fields else None,
                    "details": {
                        "credentials_file_exists": True,
                        "credentials_file_valid_json": True,
                        "credential_type": cred_data.get("type", "unknown")
                    }
                }
                
            except json.JSONDecodeError:
                return {
                    "passed": False,
                    "error": "Credentials file is not valid JSON",
                    "details": "Cannot parse credentials file"
                }
            except Exception as e:
                return {
                    "passed": False,
                    "error": f"Failed to validate credentials: {e}",
                    "details": "Credentials validation error"
                }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Google Drive authentication validation failed"
            }
    
    async def _validate_supabase_connection(self) -> Dict[str, Any]:
        """Validate Supabase connection"""
        try:
            if not self.settings.supabase.url or not self.settings.supabase.key:
                return {
                    "passed": False,
                    "error": "Supabase URL or key not configured",
                    "details": "Cannot test Supabase connection"
                }
            
            # Try to create Supabase client
            try:
                from supabase import create_client
                
                client = create_client(
                    self.settings.supabase.url,
                    self.settings.supabase.key
                )
                
                # Try a simple query to test connection
                # This is a basic connectivity test
                result = client.table("_supabase_migrations").select("*").limit(1).execute()
                
                return {
                    "passed": True,
                    "details": {
                        "supabase_client_created": True,
                        "connection_test_passed": True,
                        "supabase_url": self.settings.supabase.url[:50] + "..." if len(self.settings.supabase.url) > 50 else self.settings.supabase.url
                    }
                }
                
            except Exception as e:
                # Connection might fail for various reasons, but client creation should work
                return {
                    "passed": False,
                    "error": f"Supabase connection test failed: {e}",
                    "details": {
                        "supabase_client_created": True,
                        "connection_test_passed": False
                    }
                }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Supabase connection validation failed"
            }
    
    async def _validate_embedding_services(self) -> Dict[str, Any]:
        """Validate embedding service configurations"""
        try:
            checks = {}
            
            # Check OpenAI configuration
            if self.settings.embeddings.openai_api_key:
                checks["openai_api_key_configured"] = True
                # Could test API key validity here
            else:
                checks["openai_api_key_configured"] = False
            
            # Check HuggingFace configuration
            hf_token = getattr(self.settings.embeddings, 'huggingface_api_token', None)
            checks["huggingface_token_configured"] = bool(hf_token)
            
            # Check model configurations
            checks["embedding_models_configured"] = all([
                self.settings.embeddings.openai_model,
                self.settings.embeddings.huggingface_model_financial,
                self.settings.embeddings.huggingface_model_mathematical
            ])
            
            # At least one embedding service should be available
            has_embedding_service = (
                checks["openai_api_key_configured"] or 
                checks["huggingface_token_configured"]
            )
            
            return {
                "passed": has_embedding_service and checks["embedding_models_configured"],
                "error": "No embedding services configured or models missing" if not (has_embedding_service and checks["embedding_models_configured"]) else None,
                "details": checks
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Embedding service validation failed"
            }
    
    async def _validate_storage_directories(self) -> Dict[str, Any]:
        """Validate storage directories are accessible"""
        try:
            base_dir = Path(__file__).parent
            directories = [
                "data",
                "data/logs",
                "data/state", 
                "data/extracted_pdfs",
                "config",
                "credentials"
            ]
            
            checks = {}
            all_passed = True
            
            for dir_name in directories:
                dir_path = base_dir / dir_name
                
                # Create directory if it doesn't exist
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    checks[f"{dir_name.replace('/', '_')}_exists"] = dir_path.exists()
                    checks[f"{dir_name.replace('/', '_')}_writable"] = os.access(dir_path, os.W_OK)
                    
                    if not (dir_path.exists() and os.access(dir_path, os.W_OK)):
                        all_passed = False
                        
                except Exception as e:
                    checks[f"{dir_name.replace('/', '_')}_error"] = str(e)
                    all_passed = False
            
            return {
                "passed": all_passed,
                "error": "Some storage directories are not accessible" if not all_passed else None,
                "details": checks
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Storage directory validation failed"
            }
    
    async def _validate_containerization(self) -> Dict[str, Any]:
        """Validate containerization setup"""
        try:
            checks = {}
            
            # Check if running in container
            is_container = self.config_manager._is_running_in_container()
            checks["running_in_container"] = is_container
            
            # Check Docker files exist
            base_dir = Path(__file__).parent
            docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
            
            for file_name in docker_files:
                file_path = base_dir / file_name
                checks[f"{file_name.replace('.', '_').replace('-', '_')}_exists"] = file_path.exists()
            
            # Check build scripts
            build_scripts = ["build.sh", "build.ps1"]
            for script_name in build_scripts:
                script_path = base_dir / script_name
                checks[f"{script_name.replace('.', '_')}_exists"] = script_path.exists()
            
            # If in container, check container-specific paths
            if is_container:
                container_paths = [
                    "/app/scripts/knowledge-ingestion",
                    "/app/data/credentials",
                    "/app/data/logs"
                ]
                
                for path in container_paths:
                    path_obj = Path(path)
                    checks[f"container_path_{path.replace('/', '_')}_exists"] = path_obj.exists()
            
            # Basic validation - Docker files should exist
            required_files = ["Dockerfile", "docker-compose.yml"]
            missing_files = [f for f in required_files if not (base_dir / f).exists()]
            
            return {
                "passed": len(missing_files) == 0,
                "error": f"Missing Docker files: {missing_files}" if missing_files else None,
                "details": checks
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": "Containerization validation failed"
            }
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate validation report"""
        report = {
            "deployment_validation_report": self.validation_results,
            "summary": {
                "total_checks": len(self.validation_results["checks"]),
                "passed_checks": sum(1 for check in self.validation_results["checks"].values() if check.get("passed", False)),
                "failed_checks": sum(1 for check in self.validation_results["checks"].values() if not check.get("passed", False)),
                "overall_status": self.validation_results["overall_status"]
            }
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)
            self.logger.info(f"Validation report saved to: {output_file}")
        
        return report_json


async def main():
    """Main entry point for deployment validation"""
    parser = argparse.ArgumentParser(
        description="Validate Google Drive Knowledge Base Ingestion deployment"
    )
    
    parser.add_argument(
        "--config-env",
        default=None,
        help="Configuration environment (development, production)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for validation report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        settings = config_manager.load_config(args.config_env)
        
        # Setup logging
        if args.verbose:
            settings.logging.level = "DEBUG"
        
        setup_logging(settings.logging)
        logger = get_logger(__name__)
        
        logger.info("Starting deployment validation")
        
        # Run validation
        validator = DeploymentValidator(config_manager)
        success = await validator.run_all_validations()
        
        # Generate report
        report = validator.generate_report(args.output)
        
        if args.verbose or not args.output:
            print("\n" + "="*50)
            print("DEPLOYMENT VALIDATION REPORT")
            print("="*50)
            print(report)
        
        if success:
            logger.info("Deployment validation passed")
            return 0
        else:
            logger.error("Deployment validation failed")
            return 1
            
    except Exception as e:
        print(f"Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(asyncio.run(main()))
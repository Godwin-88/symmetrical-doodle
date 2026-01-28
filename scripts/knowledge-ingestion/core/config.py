"""
Configuration management system for Google Drive Knowledge Base Ingestion.
Supports environment-specific settings without hardcoded values.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
import yaml
from dotenv import load_dotenv


@dataclass
class GoogleDriveConfig:
    """Google Drive API configuration"""
    credentials_path: str = ""
    service_account_file: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    scopes: List[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/drive.readonly"])
    folder_ids: List[str] = field(default_factory=list)


@dataclass
class LocalZipConfig:
    """Local ZIP file processing configuration"""
    zip_path: str = ""
    extract_path: str = "./extracted_pdfs"
    use_local_zip: bool = False
    recursive_scan: bool = True


@dataclass
class SupabaseConfig:
    """Supabase database configuration"""
    url: str = ""
    key: str = ""
    service_role_key: Optional[str] = None
    database_url: Optional[str] = None
    max_connections: int = 10
    timeout: int = 30


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-large"
    huggingface_model_financial: str = "BAAI/bge-large-en-v1.5"
    huggingface_model_mathematical: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32
    max_tokens: int = 8192
    use_gpu: bool = True


@dataclass
class ProcessingConfig:
    """PDF processing configuration"""
    use_marker_llm: bool = False
    marker_timeout: int = 300
    pymupdf_fallback: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_math: bool = True
    max_file_size_mb: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    correlation_id_header: str = "X-Correlation-ID"
    log_file: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class MultiSourceAuthConfig:
    """Multi-source authentication configuration"""
    credentials_dir: str = "./credentials"
    upload_dir: str = "./uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    token_expiry_hours: int = 24
    encryption_key_file: str = ".encryption_key"
    connection_timeout: int = 30
    max_connections_per_user: int = 10
    
    # Google Drive OAuth2 settings
    google_oauth_redirect_uri: str = "http://localhost:8080/auth/callback"
    google_oauth_scopes: List[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.metadata.readonly"
    ])
    
    # Cloud storage settings
    aws_default_region: str = "us-east-1"
    azure_default_timeout: int = 30
    gcs_default_timeout: int = 30
    
    # Local source settings
    local_scan_recursive: bool = True
    local_max_depth: int = 10
    zip_extract_timeout: int = 300


class KnowledgeIngestionSettings(BaseModel):
    """Main configuration class using Pydantic for validation"""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Google Drive
    google_drive: GoogleDriveConfig = Field(default_factory=GoogleDriveConfig)
    
    # Local ZIP processing
    local_zip: LocalZipConfig = Field(default_factory=LocalZipConfig)
    
    # Supabase
    supabase: SupabaseConfig = Field(default_factory=SupabaseConfig)
    
    # Embeddings
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Processing
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Multi-source authentication
    multi_source_auth: MultiSourceAuthConfig = Field(default_factory=MultiSourceAuthConfig)
    
    # Concurrency
    max_concurrent_downloads: int = Field(default=5)
    max_concurrent_processing: int = Field(default=3)
    max_concurrent_jobs: int = Field(default=4)  # Maximum concurrent batch jobs
    
    # Retry configuration
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
        
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v


class ConfigManager:
    """Configuration manager with environment-specific loading"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        self._settings: Optional[KnowledgeIngestionSettings] = None
        self._environment = self._detect_environment()
    def _detect_environment(self) -> str:
        """Detect the current environment from various sources"""
        # Check environment variables in order of precedence
        env_vars = [
            "KNOWLEDGE_INGESTION_ENV",
            "ENVIRONMENT", 
            "ENV",
            "NODE_ENV"  # Common in containerized environments
        ]
        
        for var in env_vars:
            env_value = os.getenv(var)
            if env_value:
                # Normalize environment names
                env_value = env_value.lower()
                if env_value in ["dev", "develop"]:
                    return "development"
                elif env_value in ["prod", "production"]:
                    return "production"
                elif env_value in ["test", "testing"]:
                    return "testing"
                elif env_value in ["stage", "staging"]:
                    return "staging"
                else:
                    return env_value
        
        # Check if running in container
        if self._is_running_in_container():
            return "production"
        
        # Default to development
        return "development"
    
    def _is_running_in_container(self) -> bool:
        """Detect if running inside a Docker container"""
        # Check for Docker-specific files
        docker_indicators = [
            Path("/.dockerenv"),
            Path("/proc/1/cgroup")
        ]
        
        for indicator in docker_indicators:
            if indicator.exists():
                if indicator.name == "cgroup":
                    # Check if cgroup contains docker
                    try:
                        with open(indicator, 'r') as f:
                            content = f.read()
                            if "docker" in content or "containerd" in content:
                                return True
                    except:
                        pass
                else:
                    return True
        
        # Check environment variables that indicate containerization
        container_env_vars = [
            "DOCKER_CONTAINER",
            "KUBERNETES_SERVICE_HOST",
            "CONTAINER_NAME"
        ]
        
        return any(os.getenv(var) for var in container_env_vars)
        
    def load_config(self, environment: Optional[str] = None) -> KnowledgeIngestionSettings:
        """Load configuration for specified environment"""
        
        # Use detected environment if not specified
        if environment is None:
            environment = self._environment
        
        # Load environment variables from multiple sources
        self._load_environment_files(environment)
        
        # Load YAML configuration
        yaml_config = self._load_yaml_config(environment)
        
        # Override with environment variables
        env_overrides = self._load_environment_overrides()
        
        # Merge configurations
        yaml_config.update(env_overrides)
        
        # Create settings
        self._settings = KnowledgeIngestionSettings(**yaml_config)
            
        return self._settings
    
    def _load_environment_files(self, environment: str):
        """Load environment files in order of precedence"""
        env_files = [
            self.config_dir / ".env",
            self.config_dir / f".env.{environment}",
            self.config_dir / ".env.local",  # Local overrides
        ]
        
        # In containerized environments, also check standard locations
        if self._is_running_in_container():
            container_env_files = [
                Path("/app/config/.env"),
                Path("/app/config/.env.production"),
                Path("/run/secrets/env"),  # Docker secrets
            ]
            env_files.extend(container_env_files)
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file, override=True)
    
    def _load_yaml_config(self, environment: str) -> Dict[str, Any]:
        """Load YAML configuration files"""
        yaml_config = {}
        
        # Load base configuration
        config_files = [
            self.config_dir / "config.yaml",
            self.config_dir / f"config.{environment}.yaml"
        ]
        
        # In containerized environments, also check mounted config
        if self._is_running_in_container():
            container_config_files = [
                Path("/app/config/config.yaml"),
                Path("/app/config/config.production.yaml"),
                Path("/app/scripts/knowledge-ingestion/config/config.yaml"),
            ]
            config_files.extend(container_config_files)
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                        yaml_config.update(file_config)
                except Exception as e:
                    print(f"Warning: Failed to load config file {config_file}: {e}")
        
        return yaml_config
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        env_overrides = {}
        
        # Environment and debug settings
        if os.getenv("KNOWLEDGE_INGESTION_ENV"):
            env_overrides["environment"] = os.getenv("KNOWLEDGE_INGESTION_ENV")
        elif os.getenv("ENVIRONMENT"):
            env_overrides["environment"] = os.getenv("ENVIRONMENT")
        
        if os.getenv("DEBUG"):
            env_overrides["debug"] = os.getenv("DEBUG").lower() in ("true", "1", "yes")
        
        # Concurrency settings
        if os.getenv("MAX_CONCURRENT_DOWNLOADS"):
            env_overrides["max_concurrent_downloads"] = int(os.getenv("MAX_CONCURRENT_DOWNLOADS"))
        if os.getenv("MAX_CONCURRENT_PROCESSING"):
            env_overrides["max_concurrent_processing"] = int(os.getenv("MAX_CONCURRENT_PROCESSING"))
        if os.getenv("MAX_CONCURRENT_JOBS"):
            env_overrides["max_concurrent_jobs"] = int(os.getenv("MAX_CONCURRENT_JOBS"))
        if os.getenv("MAX_RETRIES"):
            env_overrides["max_retries"] = int(os.getenv("MAX_RETRIES"))
        if os.getenv("RETRY_DELAY"):
            env_overrides["retry_delay"] = float(os.getenv("RETRY_DELAY"))
            
        # Handle nested configurations
        self._load_google_drive_overrides(env_overrides)
        self._load_local_zip_overrides(env_overrides)
        self._load_supabase_overrides(env_overrides)
        self._load_embedding_overrides(env_overrides)
        self._load_processing_overrides(env_overrides)
        self._load_logging_overrides(env_overrides)
        self._load_multi_source_auth_overrides(env_overrides)
        
        return env_overrides
    
    def _load_google_drive_overrides(self, env_overrides: Dict[str, Any]):
        """Load Google Drive configuration from environment variables"""
        if not env_overrides.get("google_drive"):
            env_overrides["google_drive"] = {}
        
        # Support both container and local paths
        if os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH"):
            env_overrides["google_drive"]["credentials_path"] = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
        elif os.getenv("GOOGLE_CREDENTIALS_PATH"):
            env_overrides["google_drive"]["credentials_path"] = os.getenv("GOOGLE_CREDENTIALS_PATH")
        elif self._is_running_in_container():
            # Default container path
            container_cred_path = "/app/data/credentials/google-drive-credentials.json"
            if Path(container_cred_path).exists():
                env_overrides["google_drive"]["credentials_path"] = container_cred_path
        
        if os.getenv("GOOGLE_DRIVE_FOLDER_IDS"):
            folder_ids = os.getenv("GOOGLE_DRIVE_FOLDER_IDS").split(",")
            env_overrides["google_drive"]["folder_ids"] = [fid.strip() for fid in folder_ids if fid.strip()]
    
    def _load_local_zip_overrides(self, env_overrides: Dict[str, Any]):
        """Load local ZIP configuration from environment variables"""
        if not env_overrides.get("local_zip"):
            env_overrides["local_zip"] = {}
        
        if os.getenv("LOCAL_ZIP_PATH"):
            env_overrides["local_zip"]["zip_path"] = os.getenv("LOCAL_ZIP_PATH")
        if os.getenv("LOCAL_EXTRACT_PATH"):
            env_overrides["local_zip"]["extract_path"] = os.getenv("LOCAL_EXTRACT_PATH")
        if os.getenv("USE_LOCAL_ZIP"):
            env_overrides["local_zip"]["use_local_zip"] = os.getenv("USE_LOCAL_ZIP").lower() in ("true", "1", "yes")
    
    def _load_supabase_overrides(self, env_overrides: Dict[str, Any]):
        """Load Supabase configuration from environment variables"""
        if not env_overrides.get("supabase"):
            env_overrides["supabase"] = {}
        
        if os.getenv("SUPABASE_URL"):
            env_overrides["supabase"]["url"] = os.getenv("SUPABASE_URL")
        if os.getenv("SUPABASE_KEY"):
            env_overrides["supabase"]["key"] = os.getenv("SUPABASE_KEY")
        if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            env_overrides["supabase"]["service_role_key"] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if os.getenv("DATABASE_URL"):
            env_overrides["supabase"]["database_url"] = os.getenv("DATABASE_URL")
        if os.getenv("SUPABASE_MAX_CONNECTIONS"):
            env_overrides["supabase"]["max_connections"] = int(os.getenv("SUPABASE_MAX_CONNECTIONS"))
        if os.getenv("SUPABASE_TIMEOUT"):
            env_overrides["supabase"]["timeout"] = int(os.getenv("SUPABASE_TIMEOUT"))
    
    def _load_embedding_overrides(self, env_overrides: Dict[str, Any]):
        """Load embedding configuration from environment variables"""
        if not env_overrides.get("embeddings"):
            env_overrides["embeddings"] = {}
        
        if os.getenv("OPENAI_API_KEY"):
            env_overrides["embeddings"]["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("HUGGINGFACE_API_TOKEN"):
            env_overrides["embeddings"]["huggingface_api_token"] = os.getenv("HUGGINGFACE_API_TOKEN")
        if os.getenv("EMBEDDING_BATCH_SIZE"):
            env_overrides["embeddings"]["batch_size"] = int(os.getenv("EMBEDDING_BATCH_SIZE"))
        if os.getenv("EMBEDDING_USE_GPU"):
            env_overrides["embeddings"]["use_gpu"] = os.getenv("EMBEDDING_USE_GPU").lower() in ("true", "1", "yes")
    
    def _load_processing_overrides(self, env_overrides: Dict[str, Any]):
        """Load processing configuration from environment variables"""
        if not env_overrides.get("processing"):
            env_overrides["processing"] = {}
        
        if os.getenv("USE_MARKER_LLM"):
            env_overrides["processing"]["use_marker_llm"] = os.getenv("USE_MARKER_LLM").lower() in ("true", "1", "yes")
        if os.getenv("MARKER_TIMEOUT"):
            env_overrides["processing"]["marker_timeout"] = int(os.getenv("MARKER_TIMEOUT"))
        if os.getenv("CHUNK_SIZE"):
            env_overrides["processing"]["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            env_overrides["processing"]["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("MAX_FILE_SIZE_MB"):
            env_overrides["processing"]["max_file_size_mb"] = int(os.getenv("MAX_FILE_SIZE_MB"))
    
    def _load_logging_overrides(self, env_overrides: Dict[str, Any]):
        """Load logging configuration from environment variables"""
        if not env_overrides.get("logging"):
            env_overrides["logging"] = {}
        
        if os.getenv("LOG_LEVEL"):
            env_overrides["logging"]["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FORMAT"):
            env_overrides["logging"]["format"] = os.getenv("LOG_FORMAT")
        if os.getenv("LOG_FILE"):
            env_overrides["logging"]["log_file"] = os.getenv("LOG_FILE")
        elif self._is_running_in_container():
            # In containers, default to stdout
            env_overrides["logging"]["log_file"] = None
    
    def _load_multi_source_auth_overrides(self, env_overrides: Dict[str, Any]):
        """Load multi-source authentication configuration from environment variables"""
        if not env_overrides.get("multi_source_auth"):
            env_overrides["multi_source_auth"] = {}
        
        if os.getenv("CREDENTIALS_DIR"):
            env_overrides["multi_source_auth"]["credentials_dir"] = os.getenv("CREDENTIALS_DIR")
        if os.getenv("UPLOAD_DIR"):
            env_overrides["multi_source_auth"]["upload_dir"] = os.getenv("UPLOAD_DIR")
        if os.getenv("MAX_UPLOAD_SIZE"):
            env_overrides["multi_source_auth"]["max_upload_size"] = int(os.getenv("MAX_UPLOAD_SIZE"))
        if os.getenv("TOKEN_EXPIRY_HOURS"):
            env_overrides["multi_source_auth"]["token_expiry_hours"] = int(os.getenv("TOKEN_EXPIRY_HOURS"))
        if os.getenv("CONNECTION_TIMEOUT"):
            env_overrides["multi_source_auth"]["connection_timeout"] = int(os.getenv("CONNECTION_TIMEOUT"))
        if os.getenv("MAX_CONNECTIONS_PER_USER"):
            env_overrides["multi_source_auth"]["max_connections_per_user"] = int(os.getenv("MAX_CONNECTIONS_PER_USER"))
        
        # Google OAuth2 settings
        if os.getenv("GOOGLE_OAUTH_REDIRECT_URI"):
            env_overrides["multi_source_auth"]["google_oauth_redirect_uri"] = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
        
        # Cloud storage settings
        if os.getenv("AWS_DEFAULT_REGION"):
            env_overrides["multi_source_auth"]["aws_default_region"] = os.getenv("AWS_DEFAULT_REGION")
        if os.getenv("AZURE_DEFAULT_TIMEOUT"):
            env_overrides["multi_source_auth"]["azure_default_timeout"] = int(os.getenv("AZURE_DEFAULT_TIMEOUT"))
        if os.getenv("GCS_DEFAULT_TIMEOUT"):
            env_overrides["multi_source_auth"]["gcs_default_timeout"] = int(os.getenv("GCS_DEFAULT_TIMEOUT"))
        
        # Local source settings
        if os.getenv("LOCAL_SCAN_RECURSIVE"):
            env_overrides["multi_source_auth"]["local_scan_recursive"] = os.getenv("LOCAL_SCAN_RECURSIVE").lower() in ("true", "1", "yes")
        if os.getenv("LOCAL_MAX_DEPTH"):
            env_overrides["multi_source_auth"]["local_max_depth"] = int(os.getenv("LOCAL_MAX_DEPTH"))
        if os.getenv("ZIP_EXTRACT_TIMEOUT"):
            env_overrides["multi_source_auth"]["zip_extract_timeout"] = int(os.getenv("ZIP_EXTRACT_TIMEOUT"))
    
    @property
    def settings(self) -> KnowledgeIngestionSettings:
        """Get current settings, loading default if not loaded"""
        if self._settings is None:
            self._settings = self.load_config()
        return self._settings
    
    def create_default_config_files(self):
        """Create default configuration files"""
        
        # Create default .env file
        env_content = """# Google Drive Knowledge Base Ingestion Configuration

# Environment
ENVIRONMENT=development
DEBUG=true

# Google Drive API
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google-drive-credentials.json
GOOGLE_DRIVE_FOLDER_IDS=

# Supabase
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# OpenAI
OPENAI_API_KEY=

# Processing
MAX_CONCURRENT_DOWNLOADS=5
MAX_CONCURRENT_PROCESSING=3
MAX_RETRIES=3
RETRY_DELAY=1.0
"""
        
        env_file = self.config_dir / ".env.example"
        with open(env_file, 'w') as f:
            f.write(env_content)
            
        # Create default YAML config
        yaml_content = {
            "google_drive": {
                "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
                "folder_ids": []
            },
            "supabase": {
                "max_connections": 10,
                "timeout": 30
            },
            "embeddings": {
                "openai_model": "text-embedding-3-large",
                "huggingface_model_financial": "BAAI/bge-large-en-v1.5",
                "huggingface_model_mathematical": "sentence-transformers/all-mpnet-base-v2",
                "batch_size": 32,
                "max_tokens": 8192,
                "use_gpu": True
            },
            "processing": {
                "use_marker_llm": False,
                "marker_timeout": 300,
                "pymupdf_fallback": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "preserve_math": True,
                "max_file_size_mb": 100
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "correlation_id_header": "X-Correlation-ID",
                "max_file_size_mb": 100,
                "backup_count": 5
            }
        }
        
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, indent=2)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        settings = self.settings
        
        # Check required Google Drive settings
        if not settings.google_drive.credentials_path:
            validation_results["errors"].append("Google Drive credentials path not configured")
            validation_results["valid"] = False
            
        # Check Supabase settings
        if not settings.supabase.url:
            validation_results["errors"].append("Supabase URL not configured")
            validation_results["valid"] = False
            
        if not settings.supabase.key:
            validation_results["errors"].append("Supabase key not configured")
            validation_results["valid"] = False
            
        # Check OpenAI settings if using OpenAI embeddings
        if not settings.embeddings.openai_api_key:
            validation_results["warnings"].append("OpenAI API key not configured - will use HuggingFace models only")
            
        # Check file paths exist
        if settings.google_drive.credentials_path:
            if not os.path.isabs(settings.google_drive.credentials_path):
                # Make relative paths relative to config directory
                cred_path = self.config_dir.parent / settings.google_drive.credentials_path
            else:
                cred_path = Path(settings.google_drive.credentials_path)
            if not cred_path.exists():
                validation_results["errors"].append(f"Google Drive credentials file not found: {cred_path}")
                validation_results["valid"] = False
                
        return validation_results


# Global configuration manager instance
config_manager = ConfigManager()


def get_settings() -> KnowledgeIngestionSettings:
    """Get current configuration settings"""
    return config_manager.settings


def load_config(environment: Optional[str] = None) -> KnowledgeIngestionSettings:
    """Load configuration for specified environment"""
    return config_manager.load_config(environment)
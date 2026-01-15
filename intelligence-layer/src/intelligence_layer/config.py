"""Configuration management for intelligence layer."""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration."""
    postgres_url: str = Field(default="postgresql://postgres:password@localhost:5432/trading_system")
    neo4j_url: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")


class RedisConfig(BaseModel):
    """Redis configuration."""
    url: str = Field(default="redis://localhost:6379")
    max_connections: int = Field(default=10)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="json")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_type: str = Field(default="TCN")  # TCN, VAE, Transformer
    input_dim: int = Field(default=64)
    embedding_dim: int = Field(default=128)
    window_size: int = Field(default=64)
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)


class Config(BaseSettings):
    """Main configuration class."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # Database connections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Intelligence models
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    class Config:
        env_prefix = "INTELLIGENCE_"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config()
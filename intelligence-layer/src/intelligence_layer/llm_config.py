"""LLM and RAG configuration for intelligence layer."""

import os
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LocalLLMConfig(BaseModel):
    """Local LLM configuration."""
    enabled: bool = Field(default=True)
    model_path: str = Field(default="./models/llama-3.1-8b-instruct")
    model_type: Literal["llama", "mistral", "qwen", "phi"] = Field(default="llama")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cuda")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)
    context_length: int = Field(default=8192)
    gpu_memory_fraction: float = Field(default=0.8)
    quantization: Optional[Literal["4bit", "8bit"]] = Field(default="4bit")


class ExternalLLMConfig(BaseModel):
    """External LLM API configuration."""
    enabled: bool = Field(default=False)
    primary_provider: Literal["openai", "anthropic", "groq", "together"] = Field(default="openai")
    fallback_providers: List[str] = Field(default=["groq", "together"])
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    groq_api_key: Optional[str] = Field(default=None)
    together_api_key: Optional[str] = Field(default=None)
    
    # Model configurations
    openai_model: str = Field(default="gpt-4o-mini")
    anthropic_model: str = Field(default="claude-3-haiku-20240307")
    groq_model: str = Field(default="llama-3.1-8b-instant")
    together_model: str = Field(default="meta-llama/Llama-3-8b-chat-hf")
    
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    timeout: int = Field(default=30)


class RAGConfig(BaseModel):
    """RAG (Retrieval Augmented Generation) configuration."""
    enabled: bool = Field(default=True)
    
    # Vector database
    vector_db: Literal["chroma", "pinecone", "weaviate", "pgvector"] = Field(default="chroma")
    vector_db_path: str = Field(default="./data/vector_db")
    
    # Embedding model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(default="cuda")
    
    # Retrieval settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_retrieval: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    
    # Document processing
    supported_formats: List[str] = Field(default=["pdf", "txt", "md", "csv", "json", "html"])
    max_file_size_mb: int = Field(default=50)


class ResearchConfig(BaseModel):
    """Financial research configuration."""
    enabled: bool = Field(default=True)
    
    # Data sources
    financial_data_sources: List[str] = Field(default=[
        "sec_filings", "earnings_calls", "analyst_reports", 
        "news_articles", "research_papers", "market_data"
    ])
    
    # Research capabilities
    enable_web_search: bool = Field(default=True)
    enable_document_analysis: bool = Field(default=True)
    enable_sentiment_analysis: bool = Field(default=True)
    enable_technical_analysis: bool = Field(default=True)
    
    # Web search
    search_engines: List[str] = Field(default=["duckduckgo", "serper", "tavily"])
    max_search_results: int = Field(default=10)
    
    # Analysis settings
    sentiment_model: str = Field(default="finbert")
    technical_indicators: List[str] = Field(default=[
        "sma", "ema", "rsi", "macd", "bollinger_bands", "stochastic"
    ])


class LLMIntelligenceConfig(BaseSettings):
    """Main LLM Intelligence configuration."""
    
    # Core settings
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # LLM configurations
    local_llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    external_llm: ExternalLLMConfig = Field(default_factory=ExternalLLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    
    # System prompts
    system_prompt_financial: str = Field(default="""You are an expert financial analyst and quantitative researcher with deep knowledge of:
- Financial markets, instruments, and trading strategies
- Quantitative analysis, statistical modeling, and machine learning
- Risk management and portfolio optimization
- Market microstructure and algorithmic trading
- Regulatory frameworks and compliance requirements

Provide accurate, data-driven insights with proper risk disclaimers. Always cite sources when available.""")
    
    system_prompt_research: str = Field(default="""You are a financial research assistant specializing in:
- Market analysis and trend identification
- Company and sector research
- Economic indicator interpretation
- Technical and fundamental analysis
- Risk assessment and scenario planning

Provide comprehensive, well-structured analysis with clear reasoning and evidence.""")
    
    # Performance settings
    max_concurrent_requests: int = Field(default=10)
    request_timeout: int = Field(default=60)
    cache_ttl: int = Field(default=3600)  # 1 hour
    
    class Config:
        env_prefix = "LLM_"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_llm_config() -> LLMIntelligenceConfig:
    """Load LLM configuration from environment variables."""
    config = LLMIntelligenceConfig()
    
    # Load API keys from environment
    config.external_llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.external_llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    config.external_llm.groq_api_key = os.getenv("GROQ_API_KEY")
    config.external_llm.together_api_key = os.getenv("TOGETHER_API_KEY")
    
    return config
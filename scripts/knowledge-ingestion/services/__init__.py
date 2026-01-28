"""
Services module for Google Drive Knowledge Base Ingestion.

This module contains all service implementations for the ingestion pipeline.
"""

# Core embedding services (always available)
from .content_classifier import ContentClassifier, ContentDomain, ContentType, ClassificationResult
from .embedding_router import EmbeddingRouter, EmbeddingModel, EmbeddingResult, ModelConfig
from .embedding_quality_validator import EmbeddingQualityValidator, ValidationResult, QualityMetrics, QualityIssue
from .embedding_service import EmbeddingService, EmbeddingServiceResult, BatchProcessingStats

# Google Drive services (optional - require google-auth)
try:
    from .google_drive_auth import GoogleDriveAuthService, AuthMethod, AuthResult
    from .google_drive_discovery import GoogleDriveDiscoveryService, PDFMetadata, DiscoveryResult, AccessStatus
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    GoogleDriveAuthService = None
    AuthMethod = None
    AuthResult = None
    GoogleDriveDiscoveryService = None
    PDFMetadata = None
    DiscoveryResult = None
    AccessStatus = None

# Supabase services (optional - require supabase)
try:
    from .supabase_schema import SupabaseSchemaManager, SchemaValidationResult, MigrationResult
    from .supabase_storage import (
        SupabaseStorageService, TransactionManager,
        DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
        StorageResult, TransactionResult,
        ProcessingStatus, IngestionPhase, IngestionStatus
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    SupabaseSchemaManager = None
    SchemaValidationResult = None
    MigrationResult = None
    SupabaseStorageService = None
    TransactionManager = None
    DocumentMetadata = None
    ChunkData = None
    EmbeddedChunk = None
    IngestionLogEntry = None
    StorageResult = None
    TransactionResult = None
    ProcessingStatus = None
    IngestionPhase = None
    IngestionStatus = None

__all__ = [
    "ContentClassifier",
    "ContentDomain",
    "ContentType", 
    "ClassificationResult",
    "EmbeddingRouter",
    "EmbeddingModel",
    "EmbeddingResult",
    "ModelConfig",
    "EmbeddingQualityValidator",
    "ValidationResult",
    "QualityMetrics",
    "QualityIssue",
    "EmbeddingService",
    "EmbeddingServiceResult",
    "BatchProcessingStats",
    "GOOGLE_DRIVE_AVAILABLE",
    "SUPABASE_AVAILABLE"
]

# Add Google Drive services to __all__ if available
if GOOGLE_DRIVE_AVAILABLE:
    __all__.extend([
        "GoogleDriveAuthService",
        "AuthMethod", 
        "AuthResult",
        "GoogleDriveDiscoveryService",
        "PDFMetadata",
        "DiscoveryResult", 
        "AccessStatus"
    ])

# Add Supabase services to __all__ if available
if SUPABASE_AVAILABLE:
    __all__.extend([
        "SupabaseSchemaManager",
        "SchemaValidationResult",
        "MigrationResult",
        "SupabaseStorageService",
        "TransactionManager",
        "DocumentMetadata",
        "ChunkData",
        "EmbeddedChunk",
        "IngestionLogEntry",
        "StorageResult",
        "TransactionResult",
        "ProcessingStatus",
        "IngestionPhase",
        "IngestionStatus"
    ])
# Requirements Document

## Introduction

The Multi-Source Knowledge Base Ingestion system enables algorithmic trading platforms to build comprehensive, traceable, and queryable knowledge bases from diverse document sources including Google Drive, local ZIP archives, individual file uploads, local directories, and other cloud storage providers. The system executes a three-phase pipeline: discovery and inventory of documents, ingestion into Supabase with semantic processing, and post-ingestion audit and profiling.

This system provides a unified interface through the Intelligence tab (F5) Documents section, allowing users to seamlessly connect to multiple data sources, browse and select documents, and monitor processing progress in real-time. The architecture supports extensible data source connectors while maintaining consistent processing pipelines and user experience across all source types.

## Glossary

- **Knowledge_Base**: The complete searchable repository of ingested technical documents stored in Supabase
- **Multi_Source_Connector**: Extensible interface for connecting to different data sources (Google Drive, local files, cloud storage)
- **Data_Source_Type**: Enumeration of supported source types (google_drive, local_zip, local_directory, individual_upload, cloud_storage)
- **PDF_Inventory**: Comprehensive catalog of all PDF files discovered across all connected data sources
- **Source_Attribution**: Metadata tracking the original source and location of each ingested document
- **Unified_Browser**: Intelligence tab interface component for browsing files across multiple data sources
- **Batch_Selection**: Multi-source file selection interface supporting cross-source batch operations
- **Processing_Monitor**: Real-time status tracking for ingestion jobs across all data source types
- **Semantic_Chunk**: Text segment extracted from PDFs using semantic boundaries that respect document hierarchy
- **Embedding_Vector**: High-dimensional numerical representation of text chunks optimized for financial/ML domain similarity search
- **Math_Aware_Embedder**: Specialized embedding service that preserves mathematical notation and LaTeX formatting
- **Model_Router**: Intelligent service that selects optimal embedding models based on content classification
- **HNSW_Index**: Hierarchical Navigable Small World algorithm for efficient approximate nearest neighbor search in pgvector
- **Ingestion_Pipeline**: The ELT (Extract, Load, Transform) process that converts PDFs into searchable knowledge
- **Knowledge_Readiness_Memo**: Post-ingestion audit report assessing coverage, quality, and system integration
- **Supabase_Schema**: Database structure with documents and chunks tables for storing processed content
- **Idempotent_Script**: Script that produces the same result when run multiple times without side effects
- **Google_Drive_API**: Google's REST API v3 for programmatic access to Drive files and metadata
- **Local_ZIP_Processor**: Service for extracting and processing PDFs from ZIP archive files
- **Directory_Scanner**: Service for discovering PDFs in local file system directories
- **Cloud_Storage_Adapter**: Extensible interface for connecting to cloud storage providers (AWS S3, Azure Blob, etc.)
- **Marker_Parser**: High-fidelity PDF parsing library for extracting structured text content
- **Technical_Domain**: Subject area classification (ML, DRL, NLP, LLMs, finance) for content organization

## Requirements

### Requirement 1: Multi-Source Discovery and Inventory

**User Story:** As a research analyst, I want to discover and catalog all technical PDFs from multiple data sources including Google Drive folders, local ZIP archives, local directories, and individual file uploads, so that I can build a comprehensive inventory of available knowledge resources from all my document collections.

#### Acceptance Criteria

1. WHEN the system connects to Google Drive API v3, THE Google_Drive_Client SHALL authenticate using OAuth2 or service account credentials
2. WHEN processing local ZIP archives, THE ZIP_Processor SHALL extract PDF files and generate metadata including file_id, name, size, modified_time, and local_path
3. WHEN scanning local directories, THE Directory_Scanner SHALL recursively discover PDF files and extract comprehensive metadata
4. WHEN processing individual file uploads, THE Upload_Handler SHALL validate PDF files and generate consistent metadata format
5. WHEN generating inventory reports, THE Multi_Source_Report_Generator SHALL produce unified Knowledge Inventory Reports containing total PDF count across all sources, source distribution, estimated domain distribution, and flagged inaccessible files
6. WHEN encountering access restrictions or errors, THE Error_Handler SHALL flag inaccessible files per source type and continue processing remaining sources and files

### Requirement 2: Universal PDF Content Extraction and Processing

**User Story:** As a knowledge engineer, I want to extract high-fidelity text content from PDFs regardless of their source (Google Drive, local ZIP, directory, or upload), while preserving technical notation and document structure, so that the ingested content maintains its original meaning and context across all data sources.

#### Acceptance Criteria

1. WHEN accessing PDFs from Google Drive, THE Download_Service SHALL securely retrieve files using google-api-python-client with proper authentication
2. WHEN accessing PDFs from local ZIP archives, THE ZIP_Extractor SHALL extract files to temporary locations for processing
3. WHEN accessing PDFs from local directories, THE File_Reader SHALL read files directly from the file system with proper permission handling
4. WHEN accessing uploaded PDFs, THE Upload_Processor SHALL handle file streams and temporary storage securely
5. WHEN parsing PDF content from any source, THE Content_Parser SHALL use marker as the primary parser with pymupdf as fallback for extraction failures
6. WHEN chunking extracted text, THE Semantic_Chunker SHALL create segments using semantic boundaries that respect section headers and document hierarchy regardless of source
7. WHEN processing technical content, THE Content_Processor SHALL preserve mathematical notation, formulas, and technical symbols without truncation across all source types
8. WHEN handling parsing failures, THE Fallback_Handler SHALL attempt pymupdf parsing and log extraction quality metrics with source attribution

### Requirement 3: Embedding Generation and Vector Storage

**User Story:** As a system architect, I want to generate consistent high-quality embeddings optimized for financial and ML research content, so that the knowledge base supports accurate semantic search and retrieval with domain-specific performance.

#### Acceptance Criteria

1. WHEN generating embeddings, THE Embedding_Service SHALL use a tiered approach: OpenAI text-embedding-3-large for general content, with BAAI/bge-large-en-v1.5 or sentence-transformers/all-mpnet-base-v2 for specialized financial/technical content
2. WHEN processing mathematical content, THE Math_Aware_Embedder SHALL preserve LaTeX notation and mathematical symbols using specialized preprocessing before embedding generation
3. WHEN storing embeddings, THE Vector_Store SHALL use Supabase pgvector extension with HNSW indexing for efficient similarity search at scale
4. WHEN handling embedding model selection, THE Model_Router SHALL automatically select optimal embedding model based on content type classification (financial, ML research, general technical)
5. WHEN validating embeddings, THE Quality_Checker SHALL verify vector dimensions, detect null embeddings, and measure semantic coherence using cosine similarity benchmarks

### Requirement 4: Supabase Storage and Schema Management

**User Story:** As a database administrator, I want to store processed documents and chunks in a well-structured Supabase schema, so that the knowledge base supports efficient querying and maintains data integrity.

#### Acceptance Criteria

1. WHEN initializing storage, THE Schema_Manager SHALL create documents and chunks tables with proper relationships and indexes
2. WHEN storing document metadata, THE Document_Store SHALL record file_id, name, source_url, ingestion_timestamp, and processing_status
3. WHEN storing text chunks, THE Chunk_Store SHALL record chunk_id, document_id, content, embedding_vector, chunk_order, and semantic_metadata
4. WHEN handling storage operations, THE Transaction_Manager SHALL ensure atomic operations and maintain referential integrity
5. WHEN tracking ingestion progress, THE Status_Tracker SHALL log ingestion status per file with detailed error information

### Requirement 5: Post-Ingestion Audit and Quality Assessment

**User Story:** As a research director, I want to audit the ingested knowledge base for coverage and quality, so that I can verify the system meets research requirements and identify gaps.

#### Acceptance Criteria

1. WHEN sampling content, THE Content_Sampler SHALL select representative chunks across technical domains (ML, DRL, NLP, LLMs, finance)
2. WHEN verifying content quality, THE Quality_Auditor SHALL check technical notation preservation, content completeness, and embedding quality
3. WHEN assessing coverage, THE Coverage_Analyzer SHALL cross-reference ingested content against the system's research thesis scope
4. WHEN generating audit reports, THE Report_Generator SHALL produce Knowledge Readiness Memos with coverage scores and improvement recommendations
5. WHEN identifying gaps, THE Gap_Analyzer SHALL highlight missing domains, incomplete ingestion, and quality issues

### Requirement 6: Script Containerization and Deployment

**User Story:** As a DevOps engineer, I want all ingestion scripts to be idempotent and containerized, so that the system can be deployed reliably across different environments.

#### Acceptance Criteria

1. WHEN executing scripts, THE Script_Runner SHALL produce identical results on repeated runs without side effects
2. WHEN containerizing components, THE Container_Builder SHALL package all scripts in the ./scripts/ directory with proper dependencies
3. WHEN handling environment configuration, THE Config_Manager SHALL support environment-specific settings without hardcoded values
4. WHEN managing dependencies, THE Dependency_Manager SHALL ensure consistent library versions across all execution environments
5. WHEN validating deployment, THE Deployment_Validator SHALL verify script functionality in containerized environments

### Requirement 7: Integration with Trading Platform Architecture

**User Story:** As a platform architect, I want the knowledge base to integrate seamlessly with the existing algorithmic trading platform, so that the intelligence layer can access research capabilities effectively.

#### Acceptance Criteria

1. WHEN integrating with the platform, THE Integration_Service SHALL connect to existing Supabase and Neo4j instances without conflicts
2. WHEN supporting intelligence layer queries, THE Query_Interface SHALL provide consistent API endpoints for knowledge retrieval
3. WHEN handling concurrent access, THE Concurrency_Manager SHALL support multiple intelligence layer processes accessing the knowledge base
4. WHEN maintaining data consistency, THE Consistency_Manager SHALL ensure knowledge base updates don't interfere with trading operations
5. WHEN providing search capabilities, THE Search_Service SHALL support semantic search, keyword search, and domain-filtered queries

### Requirement 8: Intelligence Tab (F5) Multi-Source UI Integration

**User Story:** As a financial researcher, I want to manage knowledge ingestion from multiple data sources through the Intelligence tab's Documents section, so that I can seamlessly connect to various sources (Google Drive, local files, ZIP archives), browse and select documents, and monitor processing within the existing trading platform interface.

#### Acceptance Criteria

1. WHEN accessing the Intelligence tab Documents section, THE Multi_Source_Integration SHALL provide connection options for Google Drive ("Connect Google Drive" button with OAuth2 flow), local file upload ("Upload Files" button), local directory browser ("Browse Local Directory"), and ZIP file processor ("Process ZIP Archive")
2. WHEN any data source is connected, THE Unified_File_Browser SHALL display a tabbed or unified hierarchical view of accessible folders and PDF files from all connected sources with metadata (name, size, modified date, source type, processing status)
3. WHEN selecting PDFs for ingestion across sources, THE Cross_Source_Batch_Selection SHALL allow multi-select from different sources simultaneously with unified progress indicators and estimated processing time
4. WHEN ingesting documents from any source, THE Universal_Processing_Monitor SHALL show real-time progress for each document (downloading/accessing, parsing, chunking, embedding, storing) with source-specific error handling and recovery options
5. WHEN documents are processed from any source, THE Unified_Document_Library SHALL display all ingested documents with consistent metadata, tags, categories, and source attribution (Google Drive link, local path, ZIP archive name, upload timestamp)
6. WHEN querying documents, THE Source_Agnostic_RAG_Integration SHALL make documents from all sources immediately available for AI Chat queries with proper source attribution and links back to original locations
7. WHEN managing documents, THE Universal_Document_Management SHALL support editing metadata, adding tags, organizing by categories, and deleting documents regardless of original source
8. WHEN viewing document details, THE Multi_Source_Document_Preview SHALL show processing statistics (chunks created, embedding model used, quality score, source information) and allow re-processing with source-appropriate methods

### Requirement 9: Extensible Data Source Architecture

**User Story:** As a platform architect, I want an extensible data source architecture that can easily accommodate new document sources (cloud storage providers, document management systems, APIs), so that the system can grow to support additional data sources without requiring major architectural changes.

#### Acceptance Criteria

1. WHEN implementing new data sources, THE Data_Source_Interface SHALL provide a standardized connector interface that all sources must implement (authenticate, discover, download, get_metadata)
2. WHEN adding cloud storage providers, THE Cloud_Storage_Adapters SHALL support AWS S3, Azure Blob Storage, and Google Cloud Storage with consistent authentication and file access patterns
3. WHEN registering new data sources, THE Source_Registry SHALL dynamically load and manage data source connectors with proper error isolation
4. WHEN processing files from new sources, THE Universal_Pipeline SHALL work consistently regardless of source type using the standardized metadata format
5. WHEN displaying sources in the UI, THE Dynamic_Source_UI SHALL automatically generate appropriate interface elements based on source capabilities and requirements

### Requirement 10: Technology Stack and Performance Optimization

**User Story:** As a platform architect, I want to leverage the optimal technology stack for each component, so that the system achieves maximum performance while integrating seamlessly with the existing trading platform.

#### Acceptance Criteria

1. WHEN implementing core ingestion logic, THE Python_Services SHALL handle multi-source API integration, PDF parsing, and Supabase operations using asyncio for concurrent processing
2. WHEN processing high-volume embedding generation, THE Embedding_Pipeline SHALL use Python with GPU acceleration (CUDA) where available for transformer models
3. WHEN implementing performance-critical vector operations, THE Vector_Processor SHALL use NumPy/SciPy optimized operations with optional Rust bindings for mathematical computations
4. WHEN handling concurrent file processing, THE Concurrency_Manager SHALL use Python asyncio with configurable worker pools for I/O-bound operations
5. WHEN integrating with existing Rust execution-core, THE Interface_Layer SHALL provide Python-Rust FFI bindings for seamless data exchange and shared memory operations
6. WHEN implementing frontend integration, THE TypeScript_Components SHALL extend the existing Intelligence tab Documents section with multi-source connectivity, unified file browser, and processing status monitoring

### Requirement 11: Error Handling and Recovery

**User Story:** As a system operator, I want comprehensive error handling and recovery mechanisms that work consistently across all data sources, so that the ingestion process can handle failures gracefully and provide actionable diagnostics regardless of source type.

#### Acceptance Criteria

1. WHEN encountering API rate limits from any source, THE Rate_Limiter SHALL implement exponential backoff and retry mechanisms with source-specific rate limit handling
2. WHEN handling network failures, THE Network_Handler SHALL retry operations with appropriate timeouts and circuit breaker patterns for each source type
3. WHEN processing corrupted files from any source, THE File_Validator SHALL skip corrupted PDFs and log detailed error information with source attribution
4. WHEN managing partial failures across sources, THE Recovery_Manager SHALL support resuming ingestion from the last successful checkpoint per source
5. WHEN logging errors, THE Error_Logger SHALL provide structured logs with correlation IDs, source attribution, and source-specific debugging information
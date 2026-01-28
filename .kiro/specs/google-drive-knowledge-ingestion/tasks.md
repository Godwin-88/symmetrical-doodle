# Implementation Plan: Multi-Source Knowledge Base Ingestion

## Overview

This implementation plan breaks down the Multi-Source Knowledge Base Ingestion system into discrete, incremental coding tasks. Each task builds on previous work and focuses on creating a robust, three-phase pipeline: PDF discovery and inventory from multiple data sources, content ingestion with semantic processing, and quality audit with coverage analysis. The implementation emphasizes extensible data source connectors, idempotent operations, comprehensive error handling, and seamless integration with the existing algorithmic trading platform.

The system supports Google Drive, local ZIP archives, local directories, individual file uploads, and extensible cloud storage providers through a unified Intelligence tab interface.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure in ./scripts/knowledge-ingestion/
  - Set up Python virtual environment with required dependencies
  - Create configuration management system for environment-specific settings
  - Set up logging infrastructure with structured logging and correlation IDs
  - _Requirements: 6.3, 6.4, 10.5_

- [x] 2. Implement extensible data source architecture
  - [x] 2.1 Create data source interface and registry
    - Implement DataSourceInterface abstract base class
    - Create SourceRegistry for dynamic connector management
    - Add UniversalFileMetadata standardization
    - Create source type enumeration and capabilities
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ]* 2.2 Write property test for data source architecture
    - **Property 34: Extensible Data Source Architecture**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

- [x] 3. Implement Google Drive connector
  - [x] 3.1 Create Google Drive authentication service
    - Implement OAuth2 and service account authentication methods
    - Add credential validation and token refresh mechanisms
    - Create secure credential storage and retrieval
    - Integrate with DataSourceInterface
    - _Requirements: 1.1_
  
  - [ ]* 3.2 Write property test for Google Drive authentication
    - **Property 1: Multi-Source Authentication and Secure Access**
    - **Property 26: Multi-Source Authentication Integration**
    - **Validates: Requirements 1.1, 2.1, 8.1**
  
  - [x] 3.3 Implement Google Drive file discovery and metadata extraction
    - Create recursive folder scanning functionality
    - Extract comprehensive file metadata and convert to UniversalFileMetadata
    - Implement PDF filtering based on mimeType
    - Add access validation for discovered files
    - _Requirements: 1.2, 1.3, 1.5_
  
  - [ ]* 3.4 Write property tests for Google Drive discovery
    - **Property 2: Universal Metadata Completeness**
    - **Property 3: Source-Agnostic Content Filtering**
    - **Validates: Requirements 1.2, 1.3, 4.2, 4.3**

- [x] 4. Implement local data source connectors
  - [x] 4.1 Create local ZIP file connector
    - Implement LocalZipConnector with DataSourceInterface
    - Add ZIP extraction and PDF discovery functionality
    - Create metadata extraction from ZIP contents
    - Add temporary file management for extracted PDFs
    - _Requirements: 1.1, 1.2, 2.1_
  
  - [x] 4.2 Create local directory connector
    - Implement LocalDirectoryConnector with DataSourceInterface
    - Add recursive directory scanning for PDFs
    - Create metadata extraction from file system
    - Add permission validation and error handling
    - _Requirements: 1.1, 1.2, 2.1_
  
  - [x] 4.3 Create individual file upload connector
    - Implement UploadConnector for handling file streams
    - Add file validation and temporary storage
    - Create metadata extraction from uploaded files
    - Add secure file handling and cleanup
    - _Requirements: 1.1, 1.2, 2.1_
  
  - [ ]* 4.4 Write property tests for local connectors
    - **Property 1: Multi-Source Authentication and Secure Access**
    - **Property 6: Universal File Access and Processing**
    - **Validates: Requirements 1.1, 2.1, 2.2**

- [ ] 5. Implement cloud storage connectors (optional)
  - [ ] 5.1 Create AWS S3 connector
    - Implement S3Connector with DataSourceInterface
    - Add S3 authentication and bucket scanning
    - Create metadata extraction from S3 objects
    - _Requirements: 9.1, 9.2_
  
  - [ ] 5.2 Create Azure Blob Storage connector
    - Implement AzureBlobConnector with DataSourceInterface
    - Add Azure authentication and container scanning
    - Create metadata extraction from blob objects
    - _Requirements: 9.1, 9.2_
  
  - [ ]* 5.3 Write property tests for cloud storage connectors
    - **Property 1: Multi-Source Authentication and Secure Access**
    - **Property 34: Extensible Data Source Architecture**
    - **Validates: Requirements 9.1, 9.2, 9.3**

- [-] 6. Create universal PDF processing pipeline
  - [x] 6.1 Implement universal file access service
    - Create UniversalFileAccessor that works with all data source types
    - Add secure file access using appropriate methods per source (Google Drive API, local file system, ZIP extraction, cloud storage APIs)
    - Implement retry logic with exponential backoff for network failures
    - Add file validation and corruption detection across all sources
    - _Requirements: 2.1, 11.2, 11.3_
  
  - [x] 6.2 Implement universal PDF parsing with marker and pymupdf fallback
    - Integrate marker parser as primary parsing method for all sources
    - Implement pymupdf fallback for marker failures across all source types
    - Add parsing quality assessment and logging with source attribution
    - Preserve document structure and hierarchy regardless of source
    - _Requirements: 2.2, 2.5_
  
  - [ ]* 6.3 Write property test for universal file access and parsing
    - **Property 6: Universal File Access and Processing**
    - **Validates: Requirements 2.1, 2.2**
  
  - [x] 6.4 Create source-independent semantic chunking service
    - Implement intelligent text segmentation respecting document structure
    - Preserve section headers and hierarchical organization
    - Add mathematical notation and LaTeX symbol preservation
    - Create chunk metadata extraction with source attribution
    - _Requirements: 2.3, 2.4_
  
  - [ ]* 6.5 Write property tests for universal content processing
    - **Property 7: Source-Independent Semantic Chunking**
    - **Property 8: Universal Mathematical Content Preservation**
    - **Validates: Requirements 2.3, 2.4, 3.2**

- [x] 7. Checkpoint - Ensure universal PDF processing pipeline works end-to-end
  - Ensure all tests pass across all data source types, ask the user if questions arise.

- [x] 8. Implement embedding generation service
  - [x] 8.1 Create content classification system
    - Implement domain classification (ML, DRL, NLP, LLMs, finance, general)
    - Add content type detection for mathematical vs. general text
    - Create classification confidence scoring
    - Ensure consistent classification across all data sources
    - _Requirements: 3.1, 3.4_
  
  - [x] 8.2 Implement multi-model embedding router
    - Integrate OpenAI text-embedding-3-large for general content
    - Add BAAI/bge-large-en-v1.5 for financial/technical content
    - Implement sentence-transformers/all-mpnet-base-v2 for mathematical content
    - Create model selection logic based on content classification
    - _Requirements: 3.1, 3.4_
  
  - [ ]* 8.3 Write property test for model selection
    - **Property 9: Intelligent Model Selection Across Sources**
    - **Validates: Requirements 3.1, 3.4**
  
  - [x] 8.4 Create embedding quality validation
    - Implement vector dimension verification
    - Add null embedding detection
    - Create semantic coherence measurement using cosine similarity
    - Add embedding regeneration for failed quality checks
    - _Requirements: 3.5_
  
  - [ ]* 8.5 Write property test for quality validation
    - **Property 11: Cross-Source Quality Validation**
    - **Validates: Requirements 3.5, 5.2**

- [x] 9. Implement Supabase storage layer with multi-source support
  - [x] 9.1 Create database schema management with source attribution
    - Implement schema initialization for documents, chunks, and ingestion_logs tables
    - Add proper relationships, indexes, and constraints
    - Create HNSW vector indexes for efficient similarity search
    - Add schema migration and validation functionality
    - Extend schema to support multi-source metadata and attribution
    - _Requirements: 4.1_
  
  - [ ]* 9.2 Write property test for universal schema initialization
    - **Property 12: Universal Database Schema Support**
    - **Validates: Requirements 4.1**
  
  - [x] 9.3 Implement document and chunk storage services with source awareness
    - Create document metadata storage with all required fields and source attribution
    - Implement chunk storage with embedding vectors, semantic metadata, and source tracking
    - Add transaction management for atomic operations across sources
    - Create ingestion status tracking and logging with source-specific context
    - _Requirements: 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 9.4 Write property tests for multi-source storage operations
    - **Property 10: Universal Vector Storage and Indexing**
    - **Property 13: Source-Aware Transaction Atomicity**
    - **Property 14: Multi-Source Ingestion Status Tracking**
    - **Validates: Requirements 3.3, 4.4, 4.5**

- [x] 10. Create multi-source inventory and reporting system
  - [x] 10.1 Implement multi-source inventory report generation
    - Create unified Knowledge Inventory Report with PDF count across all sources
    - Add source distribution statistics and cross-source aggregation
    - Implement domain estimation based on filename and metadata analysis
    - Add inaccessible file flagging and reporting per source type
    - _Requirements: 1.4_
  
  - [x] 10.2 Create quality audit and sampling system across sources
    - Implement representative content sampling across technical domains from all sources
    - Add technical notation preservation verification
    - Create content completeness and embedding quality assessment
    - Ensure proportional sampling across all connected data sources
    - _Requirements: 5.1, 5.2_
  
  - [ ]* 10.3 Write property tests for multi-source reporting and auditing
    - **Property 4: Multi-Source Report Generation**
    - **Property 15: Cross-Source Domain Coverage Sampling**
    - **Validates: Requirements 1.4, 5.1, 5.4**

- [x] 11. Implement coverage analysis and readiness assessment
  - [x] 11.1 Create multi-source coverage analysis service
    - Implement research thesis scope cross-referencing across all sources
    - Add missing domain identification with source attribution
    - Create coverage scoring methodology that accounts for source diversity
    - _Requirements: 5.3, 5.5_
  
  - [x] 11.2 Generate Knowledge Readiness Memo with source insights
    - Create comprehensive readiness assessment reports
    - Add coverage scores and improvement recommendations
    - Implement gap analysis and remediation suggestions
    - Include source-specific insights and recommendations
    - _Requirements: 5.4_
  
  - [ ]* 11.3 Write property test for multi-source coverage analysis
    - **Property 16: Multi-Source Coverage Analysis**
    - **Validates: Requirements 5.3, 5.5**

- [x] 12. Checkpoint - Ensure complete multi-source ingestion pipeline works
  - Ensure all tests pass across all data source types, ask the user if questions arise.

- [x] 13. Implement error handling and resilience across all sources
  - [x] 13.1 Create comprehensive multi-source error handling system
    - Implement rate limiting with exponential backoff for API calls (source-specific)
    - Add network failure handling with circuit breaker patterns
    - Create corrupted file handling with detailed error logging and source attribution
    - Implement partial failure recovery with checkpoint resumption per source
    - _Requirements: 11.1, 11.2, 11.3, 11.4_
  
  - [x]* 13.2 Write property test for cross-source error handling
    - **Property 5: Cross-Source Error Handling**
    - **Validates: Requirements 1.5, 2.5, 11.1, 11.2, 11.3, 11.4, 11.5**

- [x] 14. Create containerization and deployment scripts
  - [x] 14.1 Implement script containerization with multi-source support
    - Create Dockerfile with all required dependencies for all data source types
    - Package all scripts in ./scripts/ directory structure
    - Add environment configuration management for all sources
    - Ensure consistent library versions across environments
    - _Requirements: 6.2, 6.4_
  
  - [x] 14.2 Create idempotent execution scripts for multi-source processing
    - Implement idempotent ingestion scripts that produce identical results across sources
    - Add execution state management and checkpoint handling per source
    - Create deployment validation and functionality verification for all connectors
    - _Requirements: 6.1, 6.5_
  
  - [ ]* 14.3 Write property tests for deployment and idempotency
    - **Property 17: Universal Script Idempotency**
    - **Property 18: Multi-Source Container Packaging**
    - **Property 19: Extensible Environment Configuration**
    - **Property 20: Universal Deployment Validation**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

- [x] 15. Implement platform integration layer with multi-source support
  - [x] 15.1 Create integration service for existing platform
    - Implement connection to existing Supabase and Neo4j instances
    - Add conflict detection and resolution for database operations
    - Create consistent API endpoints for knowledge retrieval across all sources
    - _Requirements: 7.1, 7.2_
  
  - [x] 15.2 Implement concurrent access management
    - Add support for multiple intelligence layer processes
    - Implement data consistency management during concurrent operations
    - Create operational isolation to prevent interference with trading operations
    - _Requirements: 7.3, 7.4_
  
  - [x] 15.3 Create search and query interface with source attribution
    - Implement semantic search using vector similarity across all sources
    - Add keyword search functionality with source filtering
    - Create domain-filtered query capabilities
    - Ensure consistent response formats with proper source attribution
    - _Requirements: 7.5_
  
  - [ ]* 15.4 Write property tests for multi-source platform integration
    - **Property 21: Multi-Source Platform Integration**
    - **Property 22: Cross-Source Concurrent Access Management**
    - **Property 23: Universal Search Functionality**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

- [ ] 16. Implement Intelligence Tab (F5) Multi-Source UI Integration
  - [x] 16.1 Create multi-source connection interface
    - Implement unified connection management for all data source types
    - Create connection status display with source-specific information
    - Add secure credential storage and management for each source type
    - Implement connection management (connect/disconnect functionality) for all sources
    - _Requirements: 8.1_
  
  - [x] 16.2 Build unified source browser component
    - Create tabbed interface for browsing files from multiple sources
    - Implement hierarchical folder navigation for applicable sources
    - Add search functionality across all connected sources
    - Create processing status indicators for files from all sources
    - _Requirements: 8.2_
  
  - [x] 16.3 Implement cross-source batch selection and ingestion interface
    - Create multi-select functionality for PDF files across different sources
    - Add unified batch ingestion controls with cross-source progress estimation
    - Implement processing options configuration per source type
    - Create ingestion job management and queuing for multi-source jobs
    - _Requirements: 8.3_
  
  - [x] 16.4 Create universal processing status monitor
    - Implement WebSocket connection for real-time progress updates across sources
    - Create detailed progress display for each processing phase per source
    - Add source-specific error handling and retry mechanisms in the UI
    - Implement processing job cancellation functionality for multi-source jobs
    - _Requirements: 8.4_
  
  - [x] 16.5 Integrate with existing document library
    - Extend DocumentAsset interface to support multi-source metadata
    - Create unified document list combining documents from all sources
    - Implement consistent metadata editing and tag management
    - Add source type indicators and original location preservation
    - _Requirements: 8.5, 8.7_
  
  - [x] 16.6 Implement RAG integration with source attribution
    - Ensure documents from all sources are immediately available for RAG queries
    - Add comprehensive source attribution in AI Chat responses
    - Implement document preview with multi-source integration
    - Create re-processing interface appropriate for each source type
    - _Requirements: 8.6, 8.8_
  
  - [ ]* 16.7 Write property tests for multi-source UI integration
    - **Property 26: Multi-Source Authentication Integration**
    - **Property 27: Unified Source Browser Completeness**
    - **Property 28: Cross-Source Batch Selection Functionality**
    - **Property 29: Universal Processing Status Monitoring**
    - **Property 30: Source-Aware Document Library Integration**
    - **Property 31: Multi-Source RAG Query Integration**
    - **Property 32: Universal Document Management Interface**
    - **Property 33: Multi-Source Document Preview and Statistics**
    - **Property 35: Frontend Multi-Source TypeScript Integration**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 10.6**

- [ ] 17. Implement backend API endpoints for multi-source UI integration
  - [x] 17.1 Create multi-source authentication endpoints
    - Implement authentication flow endpoints for all supported data source types
    - Create token storage and refresh API endpoints with source-specific handling
    - Add connection status and source information endpoints
    - Implement secure credential management for all source types
    - _Requirements: 8.1_
  
  - [x] 17.2 Create unified source browsing API
    - Implement unified file listing and navigation endpoints across all sources
    - Create cross-source search and metadata retrieval endpoints
    - Add file access validation and permission checking per source
    - Implement caching for improved performance across sources
    - _Requirements: 8.2_
  
  - [x] 17.3 Create multi-source batch ingestion management API
    - Implement job creation and configuration endpoints for cross-source processing
    - Create real-time progress tracking with WebSocket support for multi-source jobs
    - Add job management endpoints (start, pause, cancel, retry) with source awareness
    - Implement job history and status persistence with source attribution
    - _Requirements: 8.3, 8.4_
  
  - [x] 17.4 Extend document management API for multi-source support
    - Update existing document endpoints to support multi-source metadata
    - Create unified document listing with source type filtering and attribution
    - Implement document re-processing endpoints with source-appropriate methods
    - Add source-specific link preservation and validation
    - _Requirements: 8.5, 8.7, 8.8_

- [ ] 18. Implement performance optimizations for multi-source processing
  - [x] 18.1 Add asyncio concurrent processing across sources
    - Implement asyncio for I/O-bound operations (all data source APIs, Supabase)
    - Create configurable worker pools for concurrent file processing per source
    - Add GPU acceleration support for embedding generation when available
    - _Requirements: 10.1, 10.2, 10.4_
  
  - [x] 18.2 Optimize vector operations and cross-language integration
    - Implement NumPy/SciPy optimized operations for vector processing
    - Add optional Rust bindings for performance-critical mathematical computations
    - Create Python-Rust FFI bindings for seamless data exchange across sources
    - _Requirements: 10.3, 10.5_
  
  - [ ]* 18.3 Write property tests for performance optimizations
    - **Property 24: Multi-Source Technology Stack Optimization**
    - **Property 25: Universal Cross-Language Integration**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.5**

- [ ] 19. Create end-to-end integration and validation
  - [x] 19.1 Implement complete multi-source pipeline orchestration
    - Create main orchestration script that coordinates all three phases across all sources
    - Add progress tracking and status reporting throughout the multi-source pipeline
    - Implement comprehensive logging and monitoring with source attribution
    - _Requirements: All requirements integration_
  
  - [ ]* 19.2 Write integration tests for complete multi-source pipeline
    - Test end-to-end workflow from multi-source discovery to searchable knowledge base
    - Validate integration with existing algorithmic trading platform components
    - Test cross-language integration and data consistency across sources
    - Test Intelligence tab UI integration with backend services for all source types
    - _Requirements: All requirements validation_

- [x] 20. Final checkpoint - Complete multi-source system validation
  - Ensure all tests pass across all data source types, ask the user if questions arise.
  - Verify all 35 correctness properties are validated through property-based tests
  - Confirm system meets all acceptance criteria and integration requirements
  - Validate Intelligence tab integration provides seamless user experience across all sources

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Checkpoints ensure incremental validation and provide opportunities for user feedback
- The implementation follows the three-phase approach: Discovery → Ingestion → Audit
- All scripts will be containerized and placed in ./scripts/knowledge-ingestion/ directory
- The system integrates with existing Supabase and Neo4j instances without conflicts
- Intelligence tab (F5) Documents section provides the primary user interface for multi-source integration
- Frontend components extend the existing IntelligenceNew.tsx component with multi-source functionality
- Backend API endpoints integrate with the existing intelligence service architecture
- The system maintains consistency with the Bloomberg terminal aesthetic and existing UI patterns
- Extensible data source architecture allows for easy addition of new source types
- All data sources use the standardized UniversalFileMetadata format for consistency
- Cross-source batch processing enables efficient ingestion from multiple sources simultaneously
- Source attribution is maintained throughout the pipeline for proper document tracking and RAG responses
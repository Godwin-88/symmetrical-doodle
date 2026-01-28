# Document Management API for Multi-Source Support

## Overview

The Document Management API extends the existing knowledge ingestion system to provide comprehensive document management capabilities across multiple data sources. It supports unified document listing, metadata management, re-processing, and source-aware operations while maintaining backward compatibility with existing document APIs.

## Features

### Core Capabilities
- **Multi-Source Document Listing**: Unified document listing with source type filtering and attribution
- **Enhanced Metadata Management**: User-managed tags, categories, notes, and favorites
- **Source Attribution**: Complete source information preservation and validation
- **Document Re-processing**: Source-appropriate re-processing with job management
- **Preview and Statistics**: Comprehensive document preview and system statistics
- **Bulk Operations**: Efficient bulk updates and operations
- **Source Link Preservation**: Direct links to original documents where available

### Supported Data Sources
- Google Drive
- Local Directories
- Local ZIP Archives
- Individual File Uploads
- AWS S3
- Azure Blob Storage
- Google Cloud Storage

## API Endpoints

### Document Listing and Retrieval

#### POST /documents/list
List documents with multi-source filtering and attribution.

**Request Body:**
```json
{
  "source_types": ["google_drive", "local_directory"],
  "processing_status": ["completed"],
  "domain_classification": ["machine_learning"],
  "tags": ["research"],
  "categories": ["academic"],
  "is_favorite": true,
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-12-31T23:59:59Z"
  },
  "search_query": "neural networks",
  "sort_by": "created_at",
  "sort_order": "desc",
  "limit": 50,
  "offset": 0
}
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "uuid",
      "title": "Neural Networks Research",
      "file_id": "original_file_id",
      "size": 2048000,
      "mime_type": "application/pdf",
      "modified_time": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "source_info": {
        "source_type": "google_drive",
        "source_path": "/Research/ML/paper.pdf",
        "source_id": "gd_123456789",
        "access_url": "https://drive.google.com/file/d/123456789/view",
        "parent_folders": ["Research", "ML"],
        "connection_id": "conn_google_drive_001",
        "is_accessible": true,
        "access_permissions": ["read"]
      },
      "processing_status": "completed",
      "quality_score": 0.85,
      "parsing_method": "marker",
      "embedding_model": "text-embedding-3-large",
      "domain_classification": "machine_learning",
      "content_type": "research_paper",
      "language": "en",
      "tags": ["research", "ml", "neural-networks"],
      "categories": ["academic", "technical"],
      "notes": "Important research paper",
      "is_favorite": true,
      "chunk_count": 45,
      "processing_time_ms": 15000,
      "ingestion_job_id": "job_123",
      "checksum": "sha256_hash",
      "source_specific_metadata": {
        "google_drive_metadata": {
          "shared": true,
          "owner": "researcher@example.com"
        }
      }
    }
  ],
  "total_documents": 150,
  "by_source_type": {
    "google_drive": 75,
    "local_directory": 50,
    "aws_s3": 25
  },
  "by_status": {
    "completed": 140,
    "processing": 8,
    "failed": 2
  },
  "by_domain": {
    "machine_learning": 60,
    "finance": 45,
    "technology": 30,
    "other": 15
  },
  "pagination": {
    "offset": 0,
    "limit": 50,
    "has_more": true,
    "total_pages": 3
  },
  "execution_time_ms": 125
}
```

#### GET /documents/{document_id}
Get document by ID with full multi-source metadata.

**Response:**
```json
{
  "document_id": "uuid",
  "title": "Document Title",
  "source_info": {
    "source_type": "google_drive",
    "source_path": "/path/to/document.pdf",
    "access_url": "https://drive.google.com/file/d/123/view",
    "is_accessible": true
  },
  "processing_status": "completed",
  "tags": ["tag1", "tag2"],
  "categories": ["category1"],
  "is_favorite": false
}
```

### Document Metadata Management

#### PUT /documents/{document_id}
Update document metadata with multi-source awareness.

**Request Body:**
```json
{
  "title": "Updated Document Title",
  "tags": ["updated", "research"],
  "categories": ["academic", "important"],
  "notes": "Updated notes",
  "is_favorite": true,
  "domain_classification": "machine_learning"
}
```

#### POST /documents/bulk-update
Bulk update multiple documents.

**Request Body:**
```json
{
  "document_ids": ["uuid1", "uuid2", "uuid3"],
  "updates": {
    "tags": ["bulk_updated"],
    "categories": ["processed"]
  }
}
```

### Document Re-processing

#### POST /documents/reprocess
Re-process document with source-appropriate methods.

**Request Body:**
```json
{
  "document_id": "uuid",
  "processing_options": {
    "use_llm": true,
    "embedding_model": "text-embedding-3-large"
  },
  "force_reprocess": false,
  "preserve_user_metadata": true
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "reprocess_job_123",
  "message": "Document re-processing initiated successfully",
  "estimated_duration_ms": 15000
}
```

#### POST /documents/bulk-reprocess
Bulk reprocess multiple documents.

**Request Body:**
```json
{
  "document_ids": ["uuid1", "uuid2"],
  "processing_options": {
    "use_llm": true
  },
  "force_reprocess": true
}
```

### Document Preview and Information

#### POST /documents/preview
Get document preview with processing statistics and source information.

**Request Body:**
```json
{
  "document_id": "uuid",
  "include_chunks": true,
  "include_statistics": true,
  "chunk_limit": 10
}
```

**Response:**
```json
{
  "document": {
    "document_id": "uuid",
    "title": "Document Title",
    "source_info": {
      "source_type": "google_drive",
      "is_accessible": true
    }
  },
  "chunks": [
    {
      "chunk_id": "chunk_uuid",
      "content": "Chunk content preview...",
      "chunk_order": 1,
      "section_header": "Introduction",
      "token_count": 150,
      "embedding_model": "text-embedding-3-large",
      "quality_score": 0.85
    }
  ],
  "statistics": {
    "chunk_count": 45,
    "total_tokens": 12000,
    "embedding_models_used": ["text-embedding-3-large"],
    "average_chunk_quality": 0.82,
    "processing_phases": {
      "parsing": "completed",
      "chunking": "completed",
      "embedding": "completed",
      "storage": "completed"
    }
  },
  "source_validation": {
    "is_accessible": true,
    "permissions": ["read"]
  }
}
```

### Source-Specific Operations

#### GET /documents/{document_id}/validate-access
Validate document access through its source.

**Response:**
```json
{
  "document_id": "uuid",
  "is_accessible": true,
  "permissions": ["read"],
  "error": null
}
```

#### GET /documents/{document_id}/source-link
Get source-specific link for document.

**Response:**
```json
{
  "document_id": "uuid",
  "source_type": "google_drive",
  "source_path": "/Research/paper.pdf",
  "access_url": "https://drive.google.com/file/d/123/view",
  "is_accessible": true,
  "google_drive_link": "https://drive.google.com/file/d/123/view",
  "can_open_in_browser": true
}
```

#### GET /documents/by-source/{source_type}
List documents from a specific source type.

**Query Parameters:**
- `connection_id`: Filter by connection ID
- `processing_status`: Filter by processing status
- `limit`: Maximum results (default: 50)
- `offset`: Pagination offset (default: 0)

### Document Deletion

#### POST /documents/delete
Delete documents with source-aware cleanup.

**Request Body:**
```json
{
  "document_ids": ["uuid1", "uuid2"],
  "delete_from_source": false,
  "confirm_deletion": true
}
```

**Response:**
```json
{
  "deleted_documents": ["uuid1", "uuid2"],
  "failed_deletions": [],
  "source_deletions": [
    {
      "document_id": "uuid1",
      "source_type": "google_drive",
      "success": false,
      "message": "Source deletion not implemented for google_drive"
    }
  ],
  "total_deleted": 2
}
```

### Statistics and Analytics

#### GET /documents/statistics
Get comprehensive document statistics across all sources.

**Response:**
```json
{
  "total_documents": 1500,
  "by_source_type": {
    "google_drive": 750,
    "local_directory": 400,
    "aws_s3": 200,
    "individual_upload": 150
  },
  "by_processing_status": {
    "completed": 1350,
    "processing": 100,
    "failed": 50
  },
  "by_domain": {
    "machine_learning": 600,
    "finance": 450,
    "technology": 300,
    "other": 150
  },
  "by_quality_score": {
    "high": 900,
    "medium": 450,
    "low": 100,
    "unknown": 50
  },
  "processing_statistics": {
    "total_chunks": 67500,
    "average_processing_time": 12000,
    "success_rate": 96.7
  },
  "storage_statistics": {
    "total_size_bytes": 15360000000,
    "average_document_size": 10240000
  },
  "recent_activity": [
    {
      "document_id": "uuid",
      "title": "Recent Document",
      "action": "updated",
      "timestamp": "2024-01-15T10:30:00Z",
      "source_type": "google_drive"
    }
  ]
}
```

## Data Models

### DocumentMetadata
Complete document metadata with multi-source support.

```typescript
interface DocumentMetadata {
  document_id: string;
  title: string;
  file_id: string;
  size: number;
  mime_type: string;
  modified_time: string;
  created_at: string;
  updated_at: string;
  
  // Source information
  source_info: DocumentSourceInfo;
  
  // Processing information
  processing_status: string;
  quality_score?: number;
  parsing_method?: string;
  embedding_model?: string;
  
  // Content classification
  domain_classification?: string;
  content_type?: string;
  language: string;
  
  // User-managed metadata
  tags: string[];
  categories: string[];
  notes?: string;
  is_favorite: boolean;
  
  // Processing statistics
  chunk_count: number;
  processing_time_ms?: number;
  ingestion_job_id?: string;
  
  // Additional metadata
  checksum?: string;
  source_specific_metadata: Record<string, any>;
}
```

### DocumentSourceInfo
Source information for a document.

```typescript
interface DocumentSourceInfo {
  source_type: string;
  source_path: string;
  source_id: string;
  access_url?: string;
  parent_folders: string[];
  connection_id?: string;
  last_accessed?: string;
  is_accessible: boolean;
  access_permissions: string[];
}
```

## Database Schema Extensions

The document management API requires additional fields in the documents table:

```sql
-- Source attribution fields
ALTER TABLE documents 
ADD COLUMN source_type VARCHAR(50) DEFAULT 'individual_upload',
ADD COLUMN source_path TEXT,
ADD COLUMN source_id VARCHAR(255),
ADD COLUMN connection_id VARCHAR(255),
ADD COLUMN parent_folders JSONB DEFAULT '[]'::jsonb,
ADD COLUMN access_url TEXT,
ADD COLUMN is_accessible BOOLEAN DEFAULT true,
ADD COLUMN access_permissions JSONB DEFAULT '[]'::jsonb,
ADD COLUMN last_accessed TIMESTAMP WITH TIME ZONE,
ADD COLUMN checksum VARCHAR(64);

-- User-managed metadata fields
ALTER TABLE documents 
ADD COLUMN tags JSONB DEFAULT '[]'::jsonb,
ADD COLUMN categories JSONB DEFAULT '[]'::jsonb,
ADD COLUMN notes TEXT,
ADD COLUMN is_favorite BOOLEAN DEFAULT false,
ADD COLUMN content_type VARCHAR(100),
ADD COLUMN language VARCHAR(10) DEFAULT 'en';

-- Enhanced processing fields
ALTER TABLE documents 
ADD COLUMN embedding_model VARCHAR(100),
ADD COLUMN chunk_count INTEGER DEFAULT 0,
ADD COLUMN processing_time_ms INTEGER,
ADD COLUMN ingestion_job_id VARCHAR(255),
ADD COLUMN modified_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN mime_type VARCHAR(100) DEFAULT 'application/pdf';

-- Source-specific metadata
ALTER TABLE documents 
ADD COLUMN source_specific_metadata JSONB DEFAULT '{}'::jsonb;
```

## Integration with Existing Systems

### Intelligence Tab Integration
The document management API integrates seamlessly with the existing Intelligence tab (F5) Documents section:

- **Unified Document Library**: All documents appear in a single, filterable list regardless of source
- **Source Attribution**: Clear indicators show document source with appropriate icons and links
- **Metadata Management**: In-place editing of tags, categories, and notes
- **Re-processing**: One-click re-processing with progress monitoring
- **Preview**: Rich document preview with chunk information and statistics

### RAG Integration
Documents from all sources are immediately available for RAG queries with proper source attribution:

- **Source Links**: AI responses include links back to original documents
- **Attribution**: Clear indication of which sources contributed to responses
- **Access Validation**: Real-time validation of document accessibility

### Batch Processing Integration
The API integrates with the existing batch processing system:

- **Re-processing Jobs**: Document re-processing creates standard batch jobs
- **Progress Monitoring**: Real-time progress updates via WebSocket
- **Job Management**: Full job control (start, pause, cancel, retry)

## Error Handling

The API provides comprehensive error handling with specific error codes and messages:

### Common Error Responses

```json
{
  "detail": "Document not found",
  "status_code": 404
}
```

```json
{
  "detail": "Document source is not accessible for re-processing",
  "status_code": 400
}
```

```json
{
  "detail": "Authentication service not initialized",
  "status_code": 503
}
```

### Validation Errors

```json
{
  "detail": "Deletion must be confirmed",
  "status_code": 400
}
```

## Performance Considerations

### Indexing Strategy
The API uses comprehensive indexing for optimal query performance:

- **Source Type Indexes**: Fast filtering by source type
- **Composite Indexes**: Optimized for common query patterns
- **JSONB Indexes**: Efficient querying of metadata fields
- **Full-Text Search**: Support for content search across documents

### Caching
- **Metadata Caching**: Frequently accessed document metadata is cached
- **Statistics Caching**: System statistics are cached with periodic refresh
- **Source Validation Caching**: Access validation results are cached per connection

### Pagination
All list endpoints support efficient pagination:
- **Offset-based Pagination**: Standard offset/limit pagination
- **Cursor-based Pagination**: Available for large datasets
- **Total Count**: Accurate total counts for UI pagination

## Security Considerations

### Access Control
- **Source-based Access**: Documents inherit access controls from their sources
- **Connection Validation**: Real-time validation of source accessibility
- **Permission Checking**: Granular permission checking per document

### Data Privacy
- **Source Attribution**: Complete audit trail of document sources
- **Metadata Isolation**: User metadata is isolated per user/tenant
- **Secure Deletion**: Proper cleanup of all related data on deletion

## Testing

The API includes comprehensive test coverage:

### Test Categories
- **Unit Tests**: Individual function and method testing
- **Integration Tests**: End-to-end API testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Access control and validation testing

### Test Data
- **Multi-Source Test Documents**: Documents from all supported sources
- **Metadata Variations**: Various combinations of tags, categories, and attributes
- **Processing States**: Documents in different processing states

## Deployment

### Prerequisites
- **Schema Migration**: Run multi-source schema migration
- **Service Dependencies**: Ensure all required services are initialized
- **Connection Setup**: Configure connections to data sources

### Configuration
```python
# Enable document management API
ENABLE_DOCUMENT_MANAGEMENT = True

# Configure source types
SUPPORTED_SOURCE_TYPES = [
    'google_drive',
    'local_directory', 
    'local_zip',
    'individual_upload',
    'aws_s3',
    'azure_blob',
    'google_cloud_storage'
]

# Performance settings
DOCUMENT_LIST_DEFAULT_LIMIT = 50
DOCUMENT_LIST_MAX_LIMIT = 1000
METADATA_CACHE_TTL = 300  # 5 minutes
```

### Monitoring
- **API Metrics**: Request/response times, error rates
- **Document Statistics**: Processing success rates, source distribution
- **Performance Metrics**: Query performance, cache hit rates
- **Error Tracking**: Detailed error logging and alerting

## Future Enhancements

### Planned Features
- **Advanced Search**: Full-text search across document content
- **Document Versioning**: Track document changes over time
- **Collaborative Features**: Document sharing and collaboration
- **AI-Powered Categorization**: Automatic document categorization
- **Workflow Integration**: Document approval and review workflows

### API Versioning
The API follows semantic versioning with backward compatibility guarantees:
- **v1.0**: Initial multi-source support
- **v1.1**: Enhanced search capabilities
- **v2.0**: Advanced workflow features

## Support and Documentation

### API Documentation
- **OpenAPI Specification**: Complete API specification available
- **Interactive Documentation**: Swagger UI for API exploration
- **Code Examples**: Sample code in multiple languages

### Support Channels
- **Documentation**: Comprehensive guides and tutorials
- **Issue Tracking**: GitHub issues for bug reports and feature requests
- **Community**: Developer community for questions and discussions
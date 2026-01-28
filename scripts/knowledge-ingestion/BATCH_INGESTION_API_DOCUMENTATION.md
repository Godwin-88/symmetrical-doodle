# Batch Ingestion Management API Documentation

## Overview

The Batch Ingestion Management API provides comprehensive multi-source batch processing capabilities for the knowledge base ingestion system. It enables users to create, manage, and monitor batch ingestion jobs across multiple data sources with real-time progress tracking via WebSocket connections.

## Features

- **Cross-Source Batch Job Creation**: Create jobs that process files from multiple data sources simultaneously
- **Real-Time Progress Tracking**: WebSocket-based live updates for job and file-level progress
- **Job Control Operations**: Start, pause, resume, cancel, and retry batch jobs
- **Job History and Persistence**: Complete job history with source attribution and detailed statistics
- **Error Handling and Recovery**: Comprehensive error handling with retry mechanisms
- **Performance Metrics**: Detailed job statistics and performance analytics
- **Queue Management**: Priority-based job queuing with concurrent execution limits

## Architecture

### Core Components

1. **BatchIngestionManager**: Central service managing job lifecycle and execution
2. **Multi-Source API Endpoints**: FastAPI endpoints for job management
3. **WebSocket Manager**: Real-time progress broadcasting system
4. **Job Persistence**: Database storage for job history and status
5. **Processing Pipeline**: Unified processing pipeline supporting all data source types

### Data Flow

```
Client Request → API Endpoint → BatchIngestionManager → Job Queue → Processing Pipeline → Storage → WebSocket Updates
```

## API Endpoints

### Base URL
```
http://localhost:8001
```

### Health Check
```http
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "multi-source-auth-api",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Batch Job Management

#### Create Batch Job
```http
POST /batch/jobs
```

Creates a new batch ingestion job with cross-source file selection.

**Request Body:**
```json
{
  "user_id": "user123",
  "name": "Research Papers Batch",
  "description": "Batch ingestion of ML research papers from multiple sources",
  "priority": "normal",
  "file_selections": [
    {
      "connection_id": "gdrive_conn_1",
      "source_type": "google_drive",
      "file_ids": ["1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms", "1mGVIS7_2Dpkqf-9njVHoecgWQjiGHy7dnpgZg0jgHDU"]
    },
    {
      "connection_id": "local_zip_conn_1",
      "source_type": "local_zip",
      "file_ids": ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
    }
  ],
  "processing_options": {
    "use_llm_parsing": false,
    "embedding_model_preference": "text-embedding-3-large",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "quality_threshold": 0.7,
    "retry_failed_files": true,
    "max_retries": 3,
    "skip_existing": true,
    "parallel_processing": true,
    "max_workers": 4,
    "timeout_seconds": 300
  }
}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "user_id": "user123",
  "name": "Research Papers Batch",
  "description": "Batch ingestion of ML research papers from multiple sources",
  "status": "pending",
  "priority": "normal",
  "created_at": "2024-01-15T10:30:00Z",
  "total_files": 5,
  "completed_files": 0,
  "failed_files": 0,
  "skipped_files": 0,
  "progress_percentage": 0.0,
  "estimated_duration_ms": 150000,
  "source_progress": [
    {
      "source_type": "google_drive",
      "connection_id": "gdrive_conn_1",
      "total_files": 2,
      "completed_files": 0,
      "failed_files": 0,
      "skipped_files": 0,
      "progress_percentage": 0.0,
      "files": [...]
    },
    {
      "source_type": "local_zip",
      "connection_id": "local_zip_conn_1",
      "total_files": 3,
      "completed_files": 0,
      "failed_files": 0,
      "skipped_files": 0,
      "progress_percentage": 0.0,
      "files": [...]
    }
  ]
}
```

#### Get Job Details
```http
GET /batch/jobs/{job_id}?user_id={user_id}
```

Retrieves detailed information about a specific batch job.

**Response:**
```json
{
  "job_id": "job_abc123",
  "user_id": "user123",
  "name": "Research Papers Batch",
  "status": "running",
  "progress_percentage": 60.0,
  "total_files": 5,
  "completed_files": 3,
  "failed_files": 0,
  "skipped_files": 0,
  "started_at": "2024-01-15T10:31:00Z",
  "estimated_duration_ms": 150000,
  "source_progress": [...]
}
```

#### Control Job Operations
```http
POST /batch/jobs/{job_id}/control
```

Controls job execution with various operations.

**Request Body:**
```json
{
  "user_id": "user123",
  "action": "start",  // "start", "pause", "resume", "cancel", "retry"
  "retry_failed_only": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Job start successful"
}
```

#### List Jobs
```http
POST /batch/jobs/list
```

Lists batch jobs for a user with optional filtering.

**Request Body:**
```json
{
  "user_id": "user123",
  "status_filter": ["running", "completed"],
  "limit": 10,
  "offset": 0
}
```

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_abc123",
      "name": "Research Papers Batch",
      "status": "running",
      "progress_percentage": 60.0,
      "created_at": "2024-01-15T10:30:00Z",
      "total_files": 5,
      "completed_files": 3
    }
  ],
  "total_jobs": 1,
  "has_more": false
}
```

#### Get Statistics
```http
GET /batch/statistics?user_id={user_id}
```

Retrieves job execution statistics.

**Response:**
```json
{
  "total_jobs": 15,
  "active_jobs": 2,
  "completed_jobs": 10,
  "failed_jobs": 2,
  "cancelled_jobs": 1,
  "total_files_processed": 150,
  "average_processing_time_ms": 45000,
  "success_rate": 92.5,
  "by_source_type": {
    "google_drive": 80,
    "local_zip": 45,
    "local_directory": 25
  },
  "by_priority": {
    "normal": 12,
    "high": 2,
    "urgent": 1
  },
  "by_status": {
    "completed": 10,
    "running": 2,
    "failed": 2,
    "cancelled": 1
  }
}
```

### WebSocket Real-Time Updates

#### Connect to WebSocket
```
ws://localhost:8001/batch/ws/{user_id}?job_id={job_id}
```

Establishes WebSocket connection for real-time progress updates.

**Connection Parameters:**
- `user_id`: User identifier (required)
- `job_id`: Specific job to monitor (optional, monitors all user jobs if omitted)

#### Message Types

**Connection Established:**
```json
{
  "type": "connection_established",
  "user_id": "user123",
  "job_id": null,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Job Update:**
```json
{
  "type": "job_update",
  "job": {
    "job_id": "job_abc123",
    "status": "running",
    "progress_percentage": 60.0,
    "completed_files": 3,
    "total_files": 5
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

**File Progress:**
```json
{
  "type": "file_progress",
  "job_id": "job_abc123",
  "source_type": "google_drive",
  "connection_id": "gdrive_conn_1",
  "file": {
    "file_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    "status": "embedding",
    "progress": 80,
    "current_step": "Generating embeddings",
    "file_metadata": {
      "name": "research_paper.pdf",
      "size": 2048576,
      "source_type": "google_drive"
    }
  },
  "timestamp": "2024-01-15T10:35:30Z"
}
```

**Ping/Pong:**
```json
// Client sends:
{
  "type": "ping",
  "timestamp": "2024-01-15T10:35:00Z"
}

// Server responds:
{
  "type": "pong",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

## Data Models

### Job Status
- `pending`: Job created but not started
- `queued`: Job queued for execution
- `running`: Job currently executing
- `paused`: Job execution paused
- `completed`: Job completed successfully
- `failed`: Job failed with errors
- `cancelled`: Job cancelled by user
- `retrying`: Job being retried after failure

### Job Priority
- `low`: Low priority job
- `normal`: Normal priority job (default)
- `high`: High priority job
- `urgent`: Urgent priority job

### File Processing Status
- `pending`: File queued for processing
- `accessing`: Accessing file from source
- `downloading`: Downloading file content
- `parsing`: Parsing PDF content
- `chunking`: Creating semantic chunks
- `embedding`: Generating embeddings
- `storing`: Storing in database
- `completed`: File processing completed
- `failed`: File processing failed
- `skipped`: File skipped (already processed)

### Processing Options
```json
{
  "use_llm_parsing": false,
  "embedding_model_preference": "text-embedding-3-large",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "quality_threshold": 0.7,
  "retry_failed_files": true,
  "max_retries": 3,
  "skip_existing": true,
  "preserve_structure": true,
  "extract_math": true,
  "parallel_processing": true,
  "max_workers": 4,
  "timeout_seconds": 300,
  "source_specific_options": {}
}
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service unavailable

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

1. **Job Not Found**: Job ID doesn't exist or user doesn't have access
2. **Invalid Action**: Attempting invalid job control operation
3. **Service Unavailable**: Required services not initialized
4. **Invalid File Selection**: Selected files not accessible or don't exist
5. **Processing Timeout**: File processing exceeded timeout limits

## Configuration

### Environment Variables
```bash
# Server configuration
MAX_CONCURRENT_JOBS=4
MAX_CONCURRENT_PROCESSING=3

# Processing configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_MARKER_LLM=false

# Database configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# API keys
OPENAI_API_KEY=your_openai_key
```

### Configuration File (config.yaml)
```yaml
processing:
  use_marker_llm: false
  chunk_size: 1000
  chunk_overlap: 200
  max_file_size_mb: 100

embeddings:
  openai_model: "text-embedding-3-large"
  batch_size: 32
  use_gpu: true

max_concurrent_jobs: 4
max_concurrent_processing: 3
```

## Usage Examples

### Python Client Example
```python
import asyncio
import aiohttp
import json

async def create_and_monitor_job():
    async with aiohttp.ClientSession() as session:
        # Create job
        job_request = {
            "user_id": "user123",
            "name": "Test Batch Job",
            "file_selections": [
                {
                    "connection_id": "conn1",
                    "source_type": "local_directory",
                    "file_ids": ["file1.pdf", "file2.pdf"]
                }
            ]
        }
        
        async with session.post("http://localhost:8001/batch/jobs", json=job_request) as response:
            job = await response.json()
            job_id = job["job_id"]
            print(f"Created job: {job_id}")
        
        # Start job
        control_request = {"user_id": "user123", "action": "start"}
        async with session.post(f"http://localhost:8001/batch/jobs/{job_id}/control", json=control_request) as response:
            result = await response.json()
            print(f"Job started: {result['success']}")
        
        # Monitor progress
        while True:
            async with session.get(f"http://localhost:8001/batch/jobs/{job_id}", params={"user_id": "user123"}) as response:
                job_status = await response.json()
                print(f"Progress: {job_status['progress_percentage']:.1f}%")
                
                if job_status['status'] in ['completed', 'failed', 'cancelled']:
                    break
                    
            await asyncio.sleep(5)

# Run the example
asyncio.run(create_and_monitor_job())
```

### JavaScript/TypeScript Client Example
```typescript
interface BatchJob {
  job_id: string;
  status: string;
  progress_percentage: number;
  total_files: number;
  completed_files: number;
}

class BatchIngestionClient {
  private baseUrl: string;
  private websocket: WebSocket | null = null;

  constructor(baseUrl: string = 'http://localhost:8001') {
    this.baseUrl = baseUrl;
  }

  async createJob(userId: string, name: string, fileSelections: any[]): Promise<BatchJob> {
    const response = await fetch(`${this.baseUrl}/batch/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        name: name,
        file_selections: fileSelections
      })
    });
    
    return await response.json();
  }

  async startJob(jobId: string, userId: string): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/batch/jobs/${jobId}/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        action: 'start'
      })
    });
    
    const result = await response.json();
    return result.success;
  }

  connectWebSocket(userId: string, onMessage: (data: any) => void): void {
    this.websocket = new WebSocket(`ws://localhost:8001/batch/ws/${userId}`);
    
    this.websocket.onopen = () => {
      console.log('WebSocket connected');
    };
    
    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };
    
    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }
}

// Usage
const client = new BatchIngestionClient();

// Create and start job
const job = await client.createJob('user123', 'Test Job', fileSelections);
await client.startJob(job.job_id, 'user123');

// Monitor progress
client.connectWebSocket('user123', (data) => {
  if (data.type === 'job_update') {
    console.log(`Job progress: ${data.job.progress_percentage}%`);
  }
});
```

## Performance Considerations

### Concurrency Limits
- Maximum concurrent jobs: 4 (configurable)
- Maximum workers per job: 4 (configurable)
- Maximum concurrent processing: 3 (configurable)

### Memory Usage
- Each job maintains file metadata in memory
- WebSocket connections consume minimal memory
- Embedding vectors are streamed to database

### Scalability
- Horizontal scaling supported through load balancing
- Database connection pooling for high throughput
- Asynchronous processing for optimal resource utilization

## Monitoring and Logging

### Log Levels
- `DEBUG`: Detailed processing information
- `INFO`: General operational messages
- `WARNING`: Non-critical issues
- `ERROR`: Error conditions requiring attention

### Metrics
- Job completion rates
- Processing times per file
- Error rates by source type
- WebSocket connection counts
- Queue depths and processing backlogs

### Health Checks
- API endpoint health: `/health`
- Database connectivity
- Service dependencies status
- Queue processing status

## Security Considerations

### Authentication
- User-based job isolation
- Connection-based access control
- Secure credential storage

### Data Protection
- Encrypted credential storage
- Secure file access patterns
- Input validation and sanitization

### Rate Limiting
- Per-user job creation limits
- WebSocket connection limits
- API request rate limiting

## Troubleshooting

### Common Issues

1. **Jobs Stuck in Queue**
   - Check concurrent job limits
   - Verify service dependencies
   - Review error logs

2. **WebSocket Connection Failures**
   - Verify network connectivity
   - Check firewall settings
   - Review WebSocket endpoint logs

3. **File Processing Failures**
   - Check file accessibility
   - Verify source connections
   - Review processing timeouts

4. **High Memory Usage**
   - Monitor job queue sizes
   - Check for memory leaks
   - Review concurrent processing limits

### Debug Commands
```bash
# Check service status
curl http://localhost:8001/health

# Get job statistics
curl "http://localhost:8001/batch/statistics?user_id=user123"

# List active jobs
curl -X POST http://localhost:8001/batch/jobs/list \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "status_filter": ["running"]}'
```

## API Versioning

Current API version: `v1`

Future versions will maintain backward compatibility with clear migration paths for breaking changes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs
3. Verify configuration settings
4. Test with minimal examples

## Changelog

### Version 1.0.0
- Initial release
- Multi-source batch job creation
- Real-time WebSocket progress tracking
- Job control operations (start, pause, resume, cancel, retry)
- Comprehensive job statistics and history
- Error handling and recovery mechanisms
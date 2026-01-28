# Multi-Source Pipeline Orchestration Documentation

## Overview

The Multi-Source Pipeline Orchestration system provides a comprehensive solution for coordinating knowledge base ingestion across all supported data sources. It orchestrates the complete pipeline from discovery through ingestion to quality audit, with comprehensive progress tracking, monitoring, and error handling.

## Architecture

### Core Components

1. **MultiSourcePipelineOrchestrator**: Main orchestration engine
2. **OrchestrationMonitor**: Monitoring and metrics collection
3. **OrchestrationErrorHandler**: Error handling and recovery
4. **OrchestrationConfigManager**: Configuration management

### Pipeline Phases

1. **Initialization**: Service setup and dependency injection
2. **Discovery**: File discovery across all connected sources
3. **Ingestion**: Multi-source batch processing with async optimizations
4. **Audit**: Quality assessment and coverage analysis
5. **Completion**: Final reporting and cleanup

## Features

### Multi-Source Support
- Google Drive (OAuth2 and service account)
- Local directories and ZIP archives
- Individual file uploads
- Cloud storage (AWS S3, Azure Blob, Google Cloud Storage)
- Extensible connector architecture

### Performance Optimization
- Async concurrent processing across sources
- GPU acceleration for embedding generation
- Intelligent resource management and backpressure handling
- Adaptive batching based on system performance
- Cross-source optimization strategies

### Monitoring and Logging
- Real-time progress tracking with source attribution
- Comprehensive performance metrics collection
- WebSocket-based live updates
- Structured logging with correlation IDs
- Alert system for performance and error conditions

### Error Handling and Recovery
- Intelligent error classification and recovery strategies
- Circuit breaker patterns for failing sources
- Exponential backoff and retry mechanisms
- Graceful degradation and fallback options
- Comprehensive error tracking and reporting

### Quality Assurance
- Automated quality audit across all sources
- Coverage analysis with domain-specific insights
- Knowledge readiness assessment
- Source-aware quality metrics

## Configuration

### Basic Configuration

```python
from main_orchestrator import OrchestrationConfig

config = OrchestrationConfig(
    max_concurrent_sources=4,
    max_concurrent_files_per_source=16,
    enable_cross_source_optimization=True,
    progress_update_interval_seconds=5,
    enable_websocket_updates=True,
    enable_quality_audit=True,
    enable_coverage_analysis=True
)
```

### Advanced Configuration

```python
from config.orchestration_config import (
    OrchestrationConfiguration,
    PerformanceConfiguration,
    MonitoringConfiguration,
    ErrorHandlingConfiguration,
    QualityConfiguration
)

# Performance optimization
performance_config = PerformanceConfiguration(
    max_concurrent_sources=8,
    max_concurrent_files_per_source=32,
    enable_resource_monitoring=True,
    enable_adaptive_batching=True,
    enable_gpu_acceleration=True,
    memory_threshold_mb=8192,
    cpu_threshold_percent=80.0
)

# Monitoring configuration
monitoring_config = MonitoringConfiguration(
    progress_update_interval_seconds=2,
    reporting_level=ProgressReportingLevel.DETAILED,
    enable_websocket_updates=True,
    enable_performance_logging=True,
    metrics_collection_interval_seconds=5
)

# Error handling configuration
error_config = ErrorHandlingConfiguration(
    max_retries_per_source=3,
    retry_delay_seconds=5.0,
    exponential_backoff=True,
    continue_on_source_failure=True,
    enable_graceful_degradation=True
)

# Complete configuration
orchestration_config = OrchestrationConfiguration(
    performance=performance_config,
    monitoring=monitoring_config,
    error_handling=error_config
)
```

## Usage Examples

### Basic Usage

```python
import asyncio
from main_orchestrator import get_orchestrator, OrchestrationConfig
from services.enhanced_batch_manager import EnhancedProcessingOptions

async def basic_orchestration():
    # Create configuration
    config = OrchestrationConfig(
        max_concurrent_sources=2,
        enable_quality_audit=True
    )
    
    # Get orchestrator
    orchestrator = await get_orchestrator(config)
    
    # Define source selections
    source_selections = [
        {
            'source_type': 'google_drive',
            'connection_id': 'my_gdrive_connection',
            'source_name': 'Research Papers',
            'recursive': True
        },
        {
            'source_type': 'local_directory',
            'connection_id': 'local_docs',
            'source_name': 'Local Documents',
            'recursive': True
        }
    ]
    
    # Create processing options
    processing_options = EnhancedProcessingOptions(
        enable_async_processing=True,
        enable_gpu_acceleration=True,
        max_concurrent_files=16
    )
    
    # Run orchestration
    progress = await orchestrator.orchestrate_complete_pipeline(
        user_id="user123",
        source_selections=source_selections,
        processing_options=processing_options
    )
    
    # Check results
    if progress.overall_status == OrchestrationStatus.COMPLETED:
        print(f"Successfully processed {progress.processed_files} files")
        print(f"Success rate: {(progress.processed_files / progress.total_files) * 100:.1f}%")
    else:
        print(f"Orchestration failed: {progress.last_error}")

# Run the orchestration
asyncio.run(basic_orchestration())
```

### Advanced Usage with Monitoring

```python
import asyncio
from main_orchestrator import get_orchestrator
from services.orchestration_monitoring import get_orchestration_monitor
from config.orchestration_config import load_orchestration_config

async def advanced_orchestration_with_monitoring():
    # Load configuration from file
    config = load_orchestration_config(
        config_file="orchestration_production.yaml",
        environment="production"
    )
    
    # Start monitoring
    monitor = get_orchestration_monitor(config.monitoring)
    await monitor.start_monitoring()
    
    # Add custom alert handler
    def custom_alert_handler(alert):
        print(f"ALERT: {alert.title} - {alert.message}")
        # Send to external monitoring system
    
    monitor.add_alert_handler(custom_alert_handler)
    
    # Get orchestrator
    orchestrator = await get_orchestrator(config)
    
    # Add WebSocket connection for real-time updates
    class MockWebSocket:
        async def send_json(self, data):
            print(f"WebSocket update: {data['type']}")
    
    websocket = MockWebSocket()
    await orchestrator.add_websocket_connection("user123", websocket)
    
    # Define multiple sources
    source_selections = [
        {
            'source_type': 'google_drive',
            'connection_id': 'research_drive',
            'source_name': 'Research Papers'
        },
        {
            'source_type': 'aws_s3',
            'connection_id': 's3_bucket',
            'source_name': 'S3 Documents'
        },
        {
            'source_type': 'local_zip',
            'connection_id': 'archive_files',
            'source_name': 'Archive Collection'
        }
    ]
    
    # High-performance processing options
    processing_options = EnhancedProcessingOptions(
        enable_async_processing=True,
        max_concurrent_files=32,
        enable_gpu_acceleration=True,
        enable_adaptive_batching=True,
        enable_resource_monitoring=True,
        gpu_batch_size=64
    )
    
    try:
        # Run orchestration
        progress = await orchestrator.orchestrate_complete_pipeline(
            user_id="user123",
            source_selections=source_selections,
            processing_options=processing_options
        )
        
        # Get final metrics
        metrics = monitor.get_current_metrics()
        print(f"Final metrics: {metrics}")
        
        # Get error statistics
        if hasattr(orchestrator, '_error_handler'):
            error_stats = orchestrator._error_handler.get_error_statistics()
            print(f"Error statistics: {error_stats}")
        
        return progress
        
    finally:
        # Cleanup
        await monitor.stop_monitoring()
        await orchestrator.shutdown()

# Run advanced orchestration
asyncio.run(advanced_orchestration_with_monitoring())
```

### Command Line Usage

```bash
# Basic orchestration
python main_orchestrator.py \
    --user-id user123 \
    --sources "google_drive:conn1,local_directory:conn2" \
    --enable-audit

# High-performance orchestration
python main_orchestrator.py \
    --user-id user123 \
    --sources "google_drive:research,aws_s3:documents,local_zip:archives" \
    --max-concurrent-sources 8 \
    --max-concurrent-files 32 \
    --enable-audit \
    --config production_config.json

# Development mode with verbose logging
python main_orchestrator.py \
    --user-id dev_user \
    --sources "local_directory:test_docs" \
    --max-concurrent-sources 2 \
    --log-level DEBUG \
    --disable-websocket
```

## API Integration

### REST API Endpoints

The orchestration system integrates with the existing API infrastructure:

```python
from fastapi import FastAPI, BackgroundTasks
from main_orchestrator import get_orchestrator

app = FastAPI()

@app.post("/api/orchestration/start")
async def start_orchestration(
    request: OrchestrationRequest,
    background_tasks: BackgroundTasks
):
    """Start multi-source pipeline orchestration"""
    orchestrator = await get_orchestrator()
    
    # Start orchestration in background
    background_tasks.add_task(
        orchestrator.orchestrate_complete_pipeline,
        request.user_id,
        request.source_selections,
        request.processing_options
    )
    
    return {"status": "started", "orchestration_id": "..."}

@app.get("/api/orchestration/{orchestration_id}/progress")
async def get_orchestration_progress(orchestration_id: str):
    """Get orchestration progress"""
    orchestrator = await get_orchestrator()
    progress = await orchestrator.get_orchestration_progress()
    
    return {
        "orchestration_id": progress.orchestration_id,
        "status": progress.overall_status.value,
        "current_phase": progress.current_phase.value,
        "total_files": progress.total_files,
        "processed_files": progress.processed_files,
        "throughput": progress.throughput_files_per_second,
        "source_states": {
            key: {
                "source_name": state.source_name,
                "processed_files": state.processed_files,
                "total_files": state.total_files,
                "status": state.phase_status.value
            }
            for key, state in progress.source_states.items()
        }
    }
```

### WebSocket Integration

```python
from fastapi import WebSocket
from main_orchestrator import get_orchestrator

@app.websocket("/ws/orchestration/{user_id}")
async def orchestration_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time orchestration updates"""
    await websocket.accept()
    
    orchestrator = await get_orchestrator()
    await orchestrator.add_websocket_connection(user_id, websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await orchestrator.remove_websocket_connection(user_id, websocket)
```

## Monitoring and Metrics

### Performance Metrics

The orchestration system collects comprehensive metrics:

- **System Metrics**: CPU, memory, disk I/O, network usage
- **Processing Metrics**: Throughput, queue sizes, active workers
- **Source Metrics**: Per-source processing statistics
- **Error Metrics**: Error rates, recovery success rates
- **Quality Metrics**: Content quality scores, embedding quality

### Alerts and Notifications

Configurable alerts for:
- High resource usage (CPU, memory)
- Processing stalls or failures
- High error rates per source
- Performance degradation
- Circuit breaker activation

### Logging

Structured logging with:
- Correlation IDs for request tracking
- Source attribution for all operations
- Performance timing information
- Error context and stack traces
- Audit trail for all operations

## Error Handling

### Error Classification

Errors are automatically classified into categories:
- **Authentication**: OAuth failures, expired tokens
- **Network**: Connection timeouts, DNS failures
- **Rate Limiting**: API quota exceeded
- **Permission**: Access denied, insufficient permissions
- **Data Corruption**: Malformed files, parsing errors
- **Resource Exhaustion**: Out of memory, disk space

### Recovery Strategies

- **Retry**: Simple retry for transient errors
- **Retry with Backoff**: Exponential backoff for rate limits
- **Skip and Continue**: Skip problematic files
- **Fallback Method**: Use alternative processing methods
- **Graceful Degradation**: Reduce performance to continue
- **Circuit Breaker**: Temporarily disable failing sources

## Performance Optimization

### Async Processing

- Concurrent file processing across sources
- Non-blocking I/O operations
- Intelligent worker pool management
- Resource-aware task scheduling

### GPU Acceleration

- GPU-accelerated embedding generation
- Batch processing optimization
- Memory management for large models
- Fallback to CPU when GPU unavailable

### Resource Management

- Dynamic concurrency adjustment
- Memory usage monitoring
- Backpressure handling
- Adaptive batching based on system load

## Quality Assurance

### Automated Quality Audit

- Content preservation verification
- Embedding quality assessment
- Mathematical notation preservation
- Cross-source consistency checks

### Coverage Analysis

- Domain coverage assessment
- Gap identification and reporting
- Research scope alignment verification
- Recommendation generation

### Knowledge Readiness Assessment

- Overall system readiness scoring
- Integration compatibility verification
- Performance benchmark comparison
- Improvement recommendations

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce concurrent file processing
   - Enable adaptive batching
   - Check for memory leaks in processing

2. **Slow Processing**
   - Enable GPU acceleration
   - Increase concurrent workers
   - Check network connectivity

3. **Authentication Failures**
   - Verify credentials and permissions
   - Check token expiration
   - Validate OAuth2 configuration

4. **Source Connection Issues**
   - Check network connectivity
   - Verify API endpoints
   - Review rate limiting settings

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

config = OrchestrationConfig(
    enable_detailed_logging=True,
    progress_update_interval_seconds=1
)
```

### Health Checks

Monitor system health:

```python
# Get current metrics
metrics = monitor.get_current_metrics()
print(f"System health: {metrics}")

# Check for active alerts
alerts = monitor.get_alerts()
for alert in alerts:
    print(f"Alert: {alert.title} - {alert.message}")
```

## Best Practices

### Configuration

1. **Start Conservative**: Begin with lower concurrency settings
2. **Monitor Resources**: Enable resource monitoring in production
3. **Use Environment-Specific Configs**: Separate dev/staging/production configs
4. **Enable Error Recovery**: Use graceful degradation and retry mechanisms

### Performance

1. **Enable GPU Acceleration**: For embedding-heavy workloads
2. **Use Adaptive Batching**: Let the system optimize batch sizes
3. **Monitor Throughput**: Track files per second metrics
4. **Balance Concurrency**: Don't exceed system capabilities

### Monitoring

1. **Enable WebSocket Updates**: For real-time progress tracking
2. **Set Up Alerts**: Configure alerts for critical conditions
3. **Use Correlation IDs**: Track requests across the system
4. **Monitor Error Rates**: Watch for patterns in failures

### Error Handling

1. **Enable Circuit Breakers**: Protect against cascading failures
2. **Use Exponential Backoff**: For rate-limited APIs
3. **Continue on Failures**: Don't let single source failures stop everything
4. **Log Error Context**: Include sufficient context for debugging

## Integration Examples

### With Existing Intelligence Layer

```python
# Integration with existing intelligence service
from intelligence_layer.rag_service import RAGService

async def integrate_with_intelligence():
    # Run orchestration
    orchestrator = await get_orchestrator()
    progress = await orchestrator.orchestrate_complete_pipeline(...)
    
    # Update RAG service with new documents
    if progress.overall_status == OrchestrationStatus.COMPLETED:
        rag_service = RAGService()
        await rag_service.refresh_knowledge_base()
        print("Knowledge base updated successfully")
```

### With Frontend UI

```typescript
// Frontend integration example
class OrchestrationService {
    private websocket: WebSocket;
    
    async startOrchestration(sources: SourceSelection[]): Promise<string> {
        const response = await fetch('/api/orchestration/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                source_selections: sources,
                processing_options: {
                    enable_async_processing: true,
                    enable_gpu_acceleration: true
                }
            })
        });
        
        const result = await response.json();
        this.connectWebSocket(result.orchestration_id);
        return result.orchestration_id;
    }
    
    private connectWebSocket(orchestrationId: string) {
        this.websocket = new WebSocket(`/ws/orchestration/${this.userId}`);
        
        this.websocket.onmessage = (event) => {
            const update = JSON.parse(event.data);
            this.handleProgressUpdate(update);
        };
    }
    
    private handleProgressUpdate(update: any) {
        // Update UI with progress information
        console.log('Progress update:', update);
    }
}
```

This documentation provides comprehensive guidance for using the Multi-Source Pipeline Orchestration system effectively across different scenarios and integration patterns.
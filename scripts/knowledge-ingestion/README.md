# Google Drive Knowledge Base Ingestion System

A comprehensive system for ingesting technical PDF documents from Google Drive into a searchable knowledge base for algorithmic trading platforms.

## Overview

This system implements a three-phase pipeline:

1. **Discovery & Inventory**: Scan Google Drive folders and catalog PDF files
2. **Ingestion Pipeline**: Extract, process, and embed PDF content into Supabase
3. **Audit & Profile**: Quality assessment and coverage analysis

## Features

- **Multi-model Embedding**: Intelligent model selection based on content type
- **High-fidelity PDF Parsing**: Marker with PyMuPDF fallback
- **Semantic Chunking**: Preserves document structure and mathematical notation
- **Structured Logging**: Correlation IDs and comprehensive error tracking
- **Environment Configuration**: Flexible configuration management
- **Idempotent Operations**: Safe to run multiple times
- **Performance Optimization**: Async processing and GPU acceleration

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python setup_environment.py

# Activate environment
source activate.sh  # Unix/Linux/macOS
# or
activate.bat       # Windows
```

### 2. Configuration

Copy and configure environment variables:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your credentials:

```env
# Google Drive API
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google-drive-credentials.json
GOOGLE_DRIVE_FOLDER_IDS=your_folder_id_1,your_folder_id_2

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key
```

### 3. Google Drive Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create credentials (Service Account recommended)
5. Download JSON key file to `./credentials/google-drive-credentials.json`

### 4. Run the System

```bash
# Validate configuration
python main.py --validate-only

# Run ingestion (development)
python main.py --environment development

# Run ingestion (production)
python main.py --environment production
```

## Project Structure

```
scripts/knowledge-ingestion/
├── core/                   # Core infrastructure
│   ├── config.py          # Configuration management
│   └── logging.py         # Structured logging
├── services/              # Service layer
├── models/                # Data models
├── utils/                 # Utility functions
├── tests/                 # Test suite
├── config/                # Configuration files
│   ├── .env.example       # Environment template
│   ├── config.yaml        # Base configuration
│   ├── config.development.yaml
│   └── config.production.yaml
├── credentials/           # API credentials (gitignored)
├── logs/                  # Log files
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── setup_environment.py  # Environment setup script
└── main.py               # Main entry point
```

## Configuration

The system supports multiple configuration methods:

1. **Environment Variables**: `.env` files
2. **YAML Configuration**: `config.yaml` files
3. **Environment-specific Overrides**: `config.{environment}.yaml`

### Configuration Hierarchy

1. Base YAML configuration (`config.yaml`)
2. Environment-specific YAML (`config.development.yaml`)
3. Environment variables (`.env`)
4. Environment-specific variables (`.env.development`)

## Logging

The system uses structured logging with correlation IDs:

- **JSON Format**: Production environments
- **Console Format**: Development environments
- **Correlation IDs**: Track operations across components
- **Performance Metrics**: Operation timing and success rates
- **Error Context**: Detailed error information with stack traces

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General operational information
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical error conditions

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=knowledge_ingestion

# Run property-based tests
pytest tests/ -k "property"
```

### Code Quality

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .

# Security scanning
bandit -r .
```

## Environment Variables Reference

### Required Variables

- `GOOGLE_DRIVE_CREDENTIALS_PATH`: Path to Google Drive API credentials
- `GOOGLE_DRIVE_FOLDER_IDS`: Comma-separated list of folder IDs to scan
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase API key

### Optional Variables

- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `ENVIRONMENT`: Environment name (development/staging/production)
- `DEBUG`: Enable debug mode (true/false)
- `MAX_CONCURRENT_DOWNLOADS`: Maximum concurrent downloads (default: 5)
- `MAX_CONCURRENT_PROCESSING`: Maximum concurrent processing (default: 3)

## Troubleshooting

### Common Issues

1. **Google Drive Authentication Failed**
   - Verify credentials file exists and is valid
   - Check API is enabled in Google Cloud Console
   - Ensure service account has access to target folders

2. **Supabase Connection Failed**
   - Verify URL and API key are correct
   - Check network connectivity
   - Ensure pgvector extension is enabled

3. **PDF Processing Errors**
   - Check file permissions and accessibility
   - Verify PDF files are not corrupted
   - Monitor memory usage for large files

4. **Embedding Generation Failed**
   - Verify API keys are valid
   - Check rate limits and quotas
   - Monitor GPU memory if using local models

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
python main.py --environment development
```

## Performance Tuning

### Concurrency Settings

Adjust based on your system resources:

```yaml
# config.yaml
max_concurrent_downloads: 10    # I/O bound
max_concurrent_processing: 4    # CPU bound
```

### Memory Management

For large PDF collections:

```yaml
processing:
  max_file_size_mb: 50         # Reduce for memory constraints
  chunk_size: 500              # Smaller chunks use less memory
```

### GPU Acceleration

Enable GPU for embedding generation:

```yaml
embeddings:
  use_gpu: true
  batch_size: 64               # Increase for better GPU utilization
```

## License

This project is part of the Algorithmic Trading Platform and follows the same licensing terms.
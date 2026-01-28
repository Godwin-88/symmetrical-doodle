# Realistic Testing Setup Guide

## Overview

This guide helps you set up realistic end-to-end testing using your actual Google Drive folder with PDF files. This provides much more meaningful validation than mock data.

## Prerequisites

### 1. Google Drive Folder with PDFs
- Create or use an existing Google Drive folder containing PDF files
- Ensure the folder contains technical/academic papers for realistic domain classification
- Mix of different file sizes and types for comprehensive testing

### 2. Google Drive API Credentials

#### Option A: Service Account (Recommended for automated testing)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create a Service Account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Download the JSON credentials file
5. Share your Google Drive folder with the service account email
6. Save credentials as `scripts/knowledge-ingestion/credentials/google-drive-credentials.json`

#### Option B: OAuth2 (For interactive testing)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Drive API
3. Create OAuth2 credentials (Desktop application)
4. Download the JSON credentials file
5. Save as `scripts/knowledge-ingestion/credentials/google-drive-credentials.json`

## Configuration Setup

### 1. Get Your Google Drive Folder ID

From your Google Drive folder URL:
```
https://drive.google.com/drive/folders/1ABC123XYZ456DEF789
                                      ^^^^^^^^^^^^^^^^^^^
                                      This is your folder ID
```

### 2. Configure Environment Variables

Edit `scripts/knowledge-ingestion/config/.env`:

```bash
# Environment
ENVIRONMENT=development
DEBUG=true

# Google Drive API
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google-drive-credentials.json
# Replace with your actual folder ID(s) - comma-separated for multiple folders
GOOGLE_DRIVE_FOLDER_IDS=1ABC123XYZ456DEF789

# Optional: Supabase for full pipeline testing
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Optional: OpenAI for embedding testing
OPENAI_API_KEY=your_openai_api_key

# Processing settings
MAX_CONCURRENT_DOWNLOADS=5
MAX_CONCURRENT_PROCESSING=3
MAX_RETRIES=3
RETRY_DELAY=1.0
```

### 3. Create Credentials Directory

```bash
mkdir -p scripts/knowledge-ingestion/credentials
# Copy your Google Drive credentials JSON file here
```

## Running Realistic Tests

### 1. Basic Discovery Test
```bash
cd scripts/knowledge-ingestion
python test_realistic_end_to_end.py
```

### 2. Full Pipeline Test (with Supabase)
```bash
# After configuring Supabase credentials
python test_realistic_end_to_end.py
```

## Expected Test Outcomes

### Successful Test Results
- ✅ **Authentication**: Successful Google Drive API connection
- ✅ **Discovery**: Real PDF files discovered and cataloged
- ✅ **Domain Classification**: Automatic classification of your PDFs
- ✅ **Inventory Report**: Comprehensive analysis of your document collection
- ✅ **Quality Assessment**: Real content quality metrics

### Generated Reports
The test will create several output files:
- `test_outputs/realistic_inventory_report.json` - Detailed inventory data
- `test_outputs/realistic_inventory_summary.txt` - Human-readable summary
- `test_outputs/realistic_quality_audit.json` - Quality assessment results

### Sample Output
```
REALISTIC END-TO-END TEST RESULTS
================================================================================
✅ Google Drive Authentication: SUCCESS
✅ PDF Discovery: 25 PDFs found
✅ Inventory Report: Generated with 4 domains
✅ Quality Audit: Completed with realistic data
✅ Reports saved to: /path/to/test_outputs

📊 Key Metrics:
   - Total PDFs: 25
   - Total Size: 156.7 MB
   - Accessibility Rate: 96.0%
   - Estimated Processing Time: 1.3 hours

Domain Distribution:
   - Machine Learning: 8 files (32.0%)
   - Finance & Trading: 6 files (24.0%)
   - Deep Reinforcement Learning: 5 files (20.0%)
   - Natural Language Processing: 4 files (16.0%)
   - General Technical: 2 files (8.0%)
```

## Benefits of Realistic Testing

### 1. **Validation of Real-World Performance**
- Test with actual file sizes and structures
- Validate domain classification accuracy
- Assess processing time estimates

### 2. **Error Handling Verification**
- Test with real permission scenarios
- Handle actual file corruption or access issues
- Validate retry mechanisms with real network conditions

### 3. **Quality Assessment**
- Analyze real mathematical notation and technical content
- Test embedding generation with actual academic papers
- Validate content preservation through real PDF parsing

### 4. **Scalability Testing**
- Test with your actual document collection size
- Validate memory usage with real file sizes
- Assess processing time with realistic workloads

## Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
Error: Failed to authenticate with Google Drive
```
**Solution**: 
- Verify credentials file path and format
- Ensure Google Drive API is enabled
- Check service account permissions

#### 2. Folder Access Issues
```
Error: Access denied to folder
```
**Solution**:
- Share folder with service account email
- Verify folder ID is correct
- Check folder permissions

#### 3. No PDFs Found
```
Discovery completed: 0 PDFs found
```
**Solution**:
- Verify folder contains PDF files
- Check folder ID is correct
- Ensure PDFs are not in subfolders (or enable recursive search)

### Debug Mode
Enable detailed logging by setting:
```bash
DEBUG=true
```

This will provide detailed information about:
- Authentication process
- File discovery steps
- Domain classification decisions
- Quality assessment details

## Next Steps

After successful realistic testing:

1. **Review Generated Reports**: Analyze your document collection insights
2. **Configure Full Pipeline**: Add Supabase and OpenAI credentials
3. **Production Setup**: Configure for your production environment
4. **Monitoring**: Set up logging and monitoring for production use

## Security Considerations

- **Credentials**: Never commit credentials to version control
- **Access**: Use least-privilege service accounts
- **Monitoring**: Monitor API usage and costs
- **Data**: Ensure compliance with data handling requirements
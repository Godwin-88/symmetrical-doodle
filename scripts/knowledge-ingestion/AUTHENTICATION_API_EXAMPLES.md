
# Multi-Source Authentication API Usage Examples

## 1. Local Directory Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/local-directory" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "directory_path": "/path/to/pdf/directory"
  }'
```

## 2. Local ZIP File Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/local-zip" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "zip_path": "/path/to/documents.zip"
  }'
```

## 3. Google Drive Service Account Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/google-drive/service-account" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "service_account_info": {
      "type": "service_account",
      "project_id": "your-project-id",
      "private_key_id": "key-id",
      "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
      "client_email": "service-account@your-project.iam.gserviceaccount.com",
      "client_id": "123456789",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token"
    }
  }'
```

## 4. AWS S3 Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/aws-s3" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "region": "us-east-1"
  }'
```

## 5. List Available Source Types

```bash
curl -X GET "http://localhost:8000/multi-source/sources"
```

## 6. Get Connection Status

```bash
curl -X GET "http://localhost:8000/multi-source/connections/{connection_id}"
```

## 7. List All Connections

```bash
curl -X GET "http://localhost:8000/multi-source/connections?user_id=user123"
```

## 8. Disconnect from Source

```bash
curl -X DELETE "http://localhost:8000/multi-source/connections/{connection_id}"
```

## 9. Get Authentication Statistics

```bash
curl -X GET "http://localhost:8000/multi-source/statistics"
```

## Response Format

All authentication endpoints return a response in this format:

```json
{
  "success": true,
  "status": "authenticated",
  "connection_id": "local_dir_abc123",
  "user_info": {
    "path": "/path/to/directory",
    "type": "local_directory"
  },
  "permissions": ["read"],
  "expires_at": null,
  "error": null,
  "metadata": {
    "pdf_count": 42,
    "path": "/path/to/directory"
  }
}
```

## Error Handling

Failed authentication returns:

```json
{
  "success": false,
  "status": "invalid",
  "connection_id": null,
  "user_info": null,
  "permissions": [],
  "expires_at": null,
  "error": "Directory does not exist: /invalid/path",
  "metadata": {}
}
```

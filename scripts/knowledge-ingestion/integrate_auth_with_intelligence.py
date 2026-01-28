"""
Integration Script for Multi-Source Authentication with Intelligence Layer

This script demonstrates how to integrate the multi-source authentication
endpoints with the existing intelligence layer FastAPI application.
"""

import asyncio
import sys
from pathlib import Path

# Add the intelligence layer to the path
intelligence_path = Path(__file__).parent.parent.parent / "intelligence-layer" / "src"
sys.path.insert(0, str(intelligence_path))

try:
    from intelligence_layer.main import app as intelligence_app
    from services.intelligence_integration import integrate_with_intelligence_layer
    
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Intelligence layer not available: {e}")
    INTELLIGENCE_AVAILABLE = False


async def integrate_authentication():
    """Integrate multi-source authentication with intelligence layer"""
    
    if not INTELLIGENCE_AVAILABLE:
        print("Intelligence layer not available. Creating standalone authentication API...")
        
        # Create standalone authentication API
        from services.multi_source_api_endpoints import start_multi_source_auth_api
        
        print("Starting standalone multi-source authentication API on port 8001...")
        await start_multi_source_auth_api(host="0.0.0.0", port=8001)
        
    else:
        print("Integrating multi-source authentication with intelligence layer...")
        
        # Integrate with existing intelligence layer
        success = await integrate_with_intelligence_layer(intelligence_app)
        
        if success:
            print("✓ Multi-source authentication successfully integrated with intelligence layer")
            print("Authentication endpoints available at:")
            print("  - /multi-source/health")
            print("  - /multi-source/auth/google-drive/oauth2")
            print("  - /multi-source/auth/google-drive/service-account")
            print("  - /multi-source/auth/aws-s3")
            print("  - /multi-source/auth/azure-blob")
            print("  - /multi-source/auth/google-cloud-storage")
            print("  - /multi-source/auth/local-directory")
            print("  - /multi-source/auth/local-zip")
            print("  - /multi-source/auth/upload-setup")
            print("  - /multi-source/connections")
            print("  - /multi-source/sources")
            print("  - /multi-source/statistics")
            
            # Start the intelligence layer with integrated authentication
            import uvicorn
            
            print("\nStarting intelligence layer with integrated authentication on port 8000...")
            config = uvicorn.Config(
                app=intelligence_app,
                host="0.0.0.0",
                port=8000,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        else:
            print("✗ Failed to integrate multi-source authentication with intelligence layer")
            return False
    
    return True


def test_integration():
    """Test the integration without starting the server"""
    
    if not INTELLIGENCE_AVAILABLE:
        print("Intelligence layer not available for testing")
        return False
    
    print("Testing integration with intelligence layer...")
    
    try:
        # Test importing the integration module
        from services.intelligence_integration import get_integration
        
        integration = get_integration()
        print("✓ Integration service created successfully")
        
        # Test that we can add routes (without actually starting the server)
        from fastapi import FastAPI
        test_app = FastAPI()
        
        integration.add_routes_to_app(test_app)
        print("✓ Routes added to test app successfully")
        
        # Check that routes were added
        route_paths = [route.path for route in test_app.routes]
        expected_routes = [
            "/multi-source/health",
            "/multi-source/auth/google-drive/oauth2",
            "/multi-source/auth/local-directory",
            "/multi-source/connections",
            "/multi-source/sources"
        ]
        
        for expected_route in expected_routes:
            if expected_route in route_paths:
                print(f"✓ Route {expected_route} found")
            else:
                print(f"✗ Route {expected_route} not found")
                return False
        
        print("✓ All expected routes found")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def create_example_usage():
    """Create example usage documentation"""
    
    example_usage = """
# Multi-Source Authentication API Usage Examples

## 1. Local Directory Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/local-directory" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user123",
    "directory_path": "/path/to/pdf/directory"
  }'
```

## 2. Local ZIP File Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/local-zip" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user123",
    "zip_path": "/path/to/documents.zip"
  }'
```

## 3. Google Drive Service Account Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/google-drive/service-account" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user123",
    "service_account_info": {
      "type": "service_account",
      "project_id": "your-project-id",
      "private_key_id": "key-id",
      "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n",
      "client_email": "service-account@your-project.iam.gserviceaccount.com",
      "client_id": "123456789",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token"
    }
  }'
```

## 4. AWS S3 Authentication

```bash
curl -X POST "http://localhost:8000/multi-source/auth/aws-s3" \\
  -H "Content-Type: application/json" \\
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
"""
    
    # Write example usage to file
    example_file = Path(__file__).parent / "AUTHENTICATION_API_EXAMPLES.md"
    with open(example_file, 'w') as f:
        f.write(example_usage)
    
    print(f"✓ Example usage documentation created: {example_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Source Authentication Integration")
    parser.add_argument("--test", action="store_true", help="Test integration without starting server")
    parser.add_argument("--examples", action="store_true", help="Create example usage documentation")
    parser.add_argument("--standalone", action="store_true", help="Run standalone authentication API")
    
    args = parser.parse_args()
    
    if args.examples:
        create_example_usage()
        return
    
    if args.test:
        success = test_integration()
        if success:
            print("\n✓ Integration test completed successfully")
        else:
            print("\n✗ Integration test failed")
            sys.exit(1)
        return
    
    if args.standalone:
        print("Starting standalone multi-source authentication API...")
        asyncio.run(integrate_authentication())
        return
    
    # Default: try to integrate with intelligence layer
    print("Starting multi-source authentication integration...")
    asyncio.run(integrate_authentication())


if __name__ == "__main__":
    main()
"""
Multi-Source Authentication Service

This module provides authentication and credential management for all supported data sources:
- Google Drive OAuth2 and service account authentication
- Cloud storage authentication (AWS S3, Azure Blob, Google Cloud Storage)
- Local source validation and setup
- Token storage, refresh, and management
- Connection status monitoring and reporting
- Secure credential management with encryption
"""

import asyncio
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import base64

# Encryption and security
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Google Drive authentication
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# AWS S3
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Azure Blob Storage
try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Google Cloud Storage
try:
    from google.cloud import storage as gcs
    from google.cloud.exceptions import GoogleCloudError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from core.config import get_settings
from core.logging import get_logger


class DataSourceType(Enum):
    """Supported data source types"""
    GOOGLE_DRIVE = "google_drive"
    LOCAL_ZIP = "local_zip"
    LOCAL_DIRECTORY = "local_directory"
    INDIVIDUAL_UPLOAD = "individual_upload"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GOOGLE_CLOUD_STORAGE = "google_cloud_storage"


class AuthenticationStatus(Enum):
    """Authentication status states"""
    NOT_AUTHENTICATED = "not_authenticated"
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    status: AuthenticationStatus
    connection_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    permissions: List[str] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConnectionInfo:
    """Information about a data source connection"""
    connection_id: str
    source_type: DataSourceType
    user_id: str
    status: AuthenticationStatus
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None
    user_info: Dict[str, Any] = None
    permissions: List[str] = None
    quota_info: Dict[str, Any] = None
    error_info: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.user_info is None:
            self.user_info = {}
        if self.permissions is None:
            self.permissions = []
        if self.quota_info is None:
            self.quota_info = {}
        if self.error_info is None:
            self.error_info = {}
        if self.metadata is None:
            self.metadata = {}


class CredentialManager:
    """Secure credential storage and management"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Credential storage directory
        credentials_dir = getattr(self.settings, 'multi_source_auth', None)
        if credentials_dir and hasattr(credentials_dir, 'credentials_dir'):
            self.credentials_dir = Path(credentials_dir.credentials_dir)
        else:
            # Fallback to environment variable or default
            self.credentials_dir = Path(os.getenv('CREDENTIALS_DIR', './credentials'))
        
        self.credentials_dir.mkdir(exist_ok=True, mode=0o700)
        
        # Connection registry file
        self.connections_file = self.credentials_dir / "connections.json"
        
        # Initialize encryption
        if encryption_key:
            self._fernet = Fernet(encryption_key)
        else:
            self._fernet = self._generate_encryption_key()
    
    def _generate_encryption_key(self) -> Fernet:
        """Generate or load encryption key"""
        key_file = self.credentials_dir / ".encryption_key"
        
        if key_file.exists():
            # Load existing key
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
        
        return Fernet(key)
    
    def encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt credentials for secure storage"""
        try:
            credentials_json = json.dumps(credentials)
            encrypted = self._fernet.encrypt(credentials_json.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Failed to encrypt credentials: {e}")
            raise
    
    def decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        """Decrypt stored credentials"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_credentials.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Failed to decrypt credentials: {e}")
            raise
    
    def store_connection(self, connection: ConnectionInfo) -> bool:
        """Store connection information"""
        try:
            connections = self.load_connections()
            connections[connection.connection_id] = asdict(connection)
            
            # Convert datetime objects to ISO strings
            conn_dict = connections[connection.connection_id]
            for key, value in conn_dict.items():
                if isinstance(value, datetime):
                    conn_dict[key] = value.isoformat()
                elif key == 'source_type' and isinstance(value, DataSourceType):
                    conn_dict[key] = value.value
                elif key == 'status' and isinstance(value, AuthenticationStatus):
                    conn_dict[key] = value.value
            
            with open(self.connections_file, 'w') as f:
                json.dump(connections, f, indent=2)
            
            os.chmod(self.connections_file, 0o600)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store connection: {e}")
            return False
    
    def load_connections(self) -> Dict[str, Dict[str, Any]]:
        """Load all stored connections"""
        try:
            if not self.connections_file.exists():
                return {}
            
            with open(self.connections_file, 'r') as f:
                connections = json.load(f)
            
            # Convert ISO strings back to datetime objects
            for conn_id, conn_data in connections.items():
                for key, value in conn_data.items():
                    if key in ['created_at', 'last_accessed', 'expires_at'] and value:
                        conn_data[key] = datetime.fromisoformat(value)
                    elif key == 'source_type':
                        conn_data[key] = DataSourceType(value)
                    elif key == 'status':
                        conn_data[key] = AuthenticationStatus(value)
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Failed to load connections: {e}")
            return {}
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get specific connection information"""
        connections = self.load_connections()
        conn_data = connections.get(connection_id)
        
        if conn_data:
            return ConnectionInfo(**conn_data)
        return None
    
    def delete_connection(self, connection_id: str) -> bool:
        """Delete connection information"""
        try:
            connections = self.load_connections()
            if connection_id in connections:
                del connections[connection_id]
                
                with open(self.connections_file, 'w') as f:
                    json.dump(connections, f, indent=2)
                
                # Also delete credential files
                cred_file = self.credentials_dir / f"{connection_id}.cred"
                if cred_file.exists():
                    cred_file.unlink()
                
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete connection: {e}")
            return False
    
    def store_credentials(self, connection_id: str, credentials: Dict[str, Any]) -> bool:
        """Store encrypted credentials for a connection"""
        try:
            encrypted = self.encrypt_credentials(credentials)
            cred_file = self.credentials_dir / f"{connection_id}.cred"
            
            with open(cred_file, 'w') as f:
                f.write(encrypted)
            
            os.chmod(cred_file, 0o600)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store credentials: {e}")
            return False
    
    def load_credentials(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt credentials for a connection"""
        try:
            cred_file = self.credentials_dir / f"{connection_id}.cred"
            
            if not cred_file.exists():
                return None
            
            with open(cred_file, 'r') as f:
                encrypted = f.read()
            
            return self.decrypt_credentials(encrypted)
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            return None


class GoogleDriveAuthenticator:
    """Google Drive authentication handler"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Google Drive API scopes
        self.scopes = [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
    
    async def authenticate_oauth2(self, user_id: str, client_config: Dict[str, Any]) -> AuthenticationResult:
        """Authenticate using OAuth2 flow"""
        try:
            # Create OAuth2 flow
            flow = Flow.from_client_config(
                client_config,
                scopes=self.scopes,
                redirect_uri='http://localhost:8080/auth/callback'
            )
            
            # Generate authorization URL
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # For now, return the auth URL for manual completion
            # In a full implementation, this would handle the callback
            connection_id = f"gd_oauth_{secrets.token_urlsafe(16)}"
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                metadata={
                    'auth_url': auth_url,
                    'state': state,
                    'flow_config': client_config
                }
            )
            
        except Exception as e:
            self.logger.error(f"OAuth2 authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def authenticate_service_account(self, user_id: str, service_account_info: Dict[str, Any]) -> AuthenticationResult:
        """Authenticate using service account"""
        try:
            from google.oauth2 import service_account
            
            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=self.scopes
            )
            
            # Test the credentials
            service = build('drive', 'v3', credentials=credentials)
            about = service.about().get(fields='user').execute()
            
            # Create connection
            connection_id = f"gd_sa_{secrets.token_urlsafe(16)}"
            
            # Store credentials
            self.credential_manager.store_credentials(connection_id, {
                'type': 'service_account',
                'service_account_info': service_account_info
            })
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.GOOGLE_DRIVE,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'email': about.get('user', {}).get('emailAddress', 'service-account'),
                    'name': about.get('user', {}).get('displayName', 'Service Account'),
                    'type': 'service_account'
                },
                permissions=['drive.readonly', 'drive.metadata.readonly']
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions
            )
            
        except Exception as e:
            self.logger.error(f"Service account authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def refresh_token(self, connection_id: str) -> AuthenticationResult:
        """Refresh OAuth2 token"""
        try:
            credentials_data = self.credential_manager.load_credentials(connection_id)
            if not credentials_data:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error="Credentials not found"
                )
            
            if credentials_data.get('type') == 'oauth2':
                # Refresh OAuth2 token
                credentials = Credentials.from_authorized_user_info(
                    credentials_data['oauth2_info']
                )
                
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                    
                    # Update stored credentials
                    credentials_data['oauth2_info'] = {
                        'token': credentials.token,
                        'refresh_token': credentials.refresh_token,
                        'token_uri': credentials.token_uri,
                        'client_id': credentials.client_id,
                        'client_secret': credentials.client_secret,
                        'scopes': credentials.scopes
                    }
                    
                    self.credential_manager.store_credentials(connection_id, credentials_data)
                    
                    return AuthenticationResult(
                        success=True,
                        status=AuthenticationStatus.AUTHENTICATED,
                        connection_id=connection_id,
                        expires_at=credentials.expiry
                    )
            
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error="Token refresh not supported for this credential type"
            )
            
        except RefreshError as e:
            self.logger.error(f"Token refresh failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.EXPIRED,
                error=str(e)
            )
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def validate_connection(self, connection_id: str) -> AuthenticationResult:
        """Validate existing connection"""
        try:
            credentials_data = self.credential_manager.load_credentials(connection_id)
            if not credentials_data:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error="Credentials not found"
                )
            
            # Build service and test connection
            if credentials_data.get('type') == 'service_account':
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_data['service_account_info'],
                    scopes=self.scopes
                )
            elif credentials_data.get('type') == 'oauth2':
                credentials = Credentials.from_authorized_user_info(
                    credentials_data['oauth2_info']
                )
            else:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error="Unknown credential type"
                )
            
            # Test the connection
            service = build('drive', 'v3', credentials=credentials)
            about = service.about().get(fields='user,storageQuota').execute()
            
            # Update connection info
            connection = self.credential_manager.get_connection(connection_id)
            if connection:
                connection.last_accessed = datetime.now(timezone.utc)
                connection.status = AuthenticationStatus.AUTHENTICATED
                connection.quota_info = about.get('storageQuota', {})
                self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=about.get('user', {}),
                metadata={'quota': about.get('storageQuota', {})}
            )
            
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )


class CloudStorageAuthenticator:
    """Cloud storage authentication handler"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.logger = get_logger(__name__)
        self.settings = get_settings()
    
    async def authenticate_aws_s3(self, user_id: str, aws_config: Dict[str, Any]) -> AuthenticationResult:
        """Authenticate with AWS S3"""
        if not AWS_AVAILABLE:
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error="AWS SDK not available"
            )
        
        try:
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['access_key_id'],
                aws_secret_access_key=aws_config['secret_access_key'],
                region_name=aws_config.get('region', 'us-east-1')
            )
            
            # Test connection by listing buckets
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            
            # Create connection
            connection_id = f"s3_{secrets.token_urlsafe(16)}"
            
            # Store credentials
            self.credential_manager.store_credentials(connection_id, {
                'type': 'aws_s3',
                'aws_config': aws_config
            })
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.AWS_S3,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'aws_account': aws_config.get('account_id', 'unknown'),
                    'region': aws_config.get('region', 'us-east-1')
                },
                permissions=['s3:ListBucket', 's3:GetObject'],
                metadata={'buckets': buckets}
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata={'buckets': buckets}
            )
            
        except (ClientError, NoCredentialsError) as e:
            self.logger.error(f"AWS S3 authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def authenticate_azure_blob(self, user_id: str, azure_config: Dict[str, Any]) -> AuthenticationResult:
        """Authenticate with Azure Blob Storage"""
        if not AZURE_AVAILABLE:
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error="Azure SDK not available"
            )
        
        try:
            # Create blob service client
            blob_service_client = BlobServiceClient(
                account_url=azure_config['account_url'],
                credential=azure_config['credential']
            )
            
            # Test connection by listing containers
            containers = []
            for container in blob_service_client.list_containers():
                containers.append(container['name'])
            
            # Create connection
            connection_id = f"azure_{secrets.token_urlsafe(16)}"
            
            # Store credentials
            self.credential_manager.store_credentials(connection_id, {
                'type': 'azure_blob',
                'azure_config': azure_config
            })
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.AZURE_BLOB,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'account_name': azure_config.get('account_name', 'unknown'),
                    'account_url': azure_config['account_url']
                },
                permissions=['blob:read', 'container:list'],
                metadata={'containers': containers}
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata={'containers': containers}
            )
            
        except AzureError as e:
            self.logger.error(f"Azure Blob authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def authenticate_gcs(self, user_id: str, gcs_config: Dict[str, Any]) -> AuthenticationResult:
        """Authenticate with Google Cloud Storage"""
        if not GCS_AVAILABLE:
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error="Google Cloud SDK not available"
            )
        
        try:
            # Create GCS client
            if 'service_account_info' in gcs_config:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_info(
                    gcs_config['service_account_info']
                )
                client = gcs.Client(credentials=credentials)
            else:
                client = gcs.Client()
            
            # Test connection by listing buckets
            buckets = [bucket.name for bucket in client.list_buckets()]
            
            # Create connection
            connection_id = f"gcs_{secrets.token_urlsafe(16)}"
            
            # Store credentials
            self.credential_manager.store_credentials(connection_id, {
                'type': 'google_cloud_storage',
                'gcs_config': gcs_config
            })
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.GOOGLE_CLOUD_STORAGE,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'project_id': gcs_config.get('project_id', client.project),
                    'type': 'service_account' if 'service_account_info' in gcs_config else 'default'
                },
                permissions=['storage.buckets.list', 'storage.objects.get'],
                metadata={'buckets': buckets}
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata={'buckets': buckets}
            )
            
        except GoogleCloudError as e:
            self.logger.error(f"Google Cloud Storage authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )


class LocalSourceValidator:
    """Local source validation handler"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.logger = get_logger(__name__)
        self.settings = get_settings()
    
    async def validate_local_directory(self, user_id: str, directory_path: str) -> AuthenticationResult:
        """Validate local directory access"""
        try:
            path = Path(directory_path)
            
            if not path.exists():
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"Directory does not exist: {directory_path}"
                )
            
            if not path.is_dir():
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"Path is not a directory: {directory_path}"
                )
            
            if not os.access(path, os.R_OK):
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"No read permission for directory: {directory_path}"
                )
            
            # Count PDF files
            pdf_count = len(list(path.rglob("*.pdf")))
            
            # Create connection
            connection_id = f"local_dir_{secrets.token_urlsafe(16)}"
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.LOCAL_DIRECTORY,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'path': str(path.absolute()),
                    'type': 'local_directory'
                },
                permissions=['read'],
                metadata={
                    'pdf_count': pdf_count,
                    'path': str(path.absolute())
                }
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata=connection.metadata
            )
            
        except Exception as e:
            self.logger.error(f"Local directory validation failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def validate_zip_file(self, user_id: str, zip_path: str) -> AuthenticationResult:
        """Validate ZIP file access"""
        try:
            import zipfile
            
            path = Path(zip_path)
            
            if not path.exists():
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"ZIP file does not exist: {zip_path}"
                )
            
            if not path.is_file():
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"Path is not a file: {zip_path}"
                )
            
            if not os.access(path, os.R_OK):
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"No read permission for file: {zip_path}"
                )
            
            # Test ZIP file and count PDFs
            pdf_count = 0
            try:
                with zipfile.ZipFile(path, 'r') as zip_file:
                    for file_info in zip_file.filelist:
                        if file_info.filename.lower().endswith('.pdf'):
                            pdf_count += 1
            except zipfile.BadZipFile:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error=f"Invalid ZIP file: {zip_path}"
                )
            
            # Create connection
            connection_id = f"local_zip_{secrets.token_urlsafe(16)}"
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.LOCAL_ZIP,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'path': str(path.absolute()),
                    'type': 'local_zip',
                    'size': path.stat().st_size
                },
                permissions=['read'],
                metadata={
                    'pdf_count': pdf_count,
                    'path': str(path.absolute()),
                    'file_size': path.stat().st_size
                }
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata=connection.metadata
            )
            
        except Exception as e:
            self.logger.error(f"ZIP file validation failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def setup_upload_handler(self, user_id: str) -> AuthenticationResult:
        """Setup individual file upload handler"""
        try:
            # Create connection for upload handling
            connection_id = f"upload_{secrets.token_urlsafe(16)}"
            
            # Create upload directory
            upload_dir_setting = getattr(self.settings, 'multi_source_auth', None)
            if upload_dir_setting and hasattr(upload_dir_setting, 'upload_dir'):
                upload_base = Path(upload_dir_setting.upload_dir)
                max_upload_size = getattr(upload_dir_setting, 'max_upload_size', 100 * 1024 * 1024)
            else:
                upload_base = Path(os.getenv('UPLOAD_DIR', './uploads'))
                max_upload_size = int(os.getenv('MAX_UPLOAD_SIZE', str(100 * 1024 * 1024)))
            
            upload_dir = upload_base / user_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Store connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                source_type=DataSourceType.INDIVIDUAL_UPLOAD,
                user_id=user_id,
                status=AuthenticationStatus.AUTHENTICATED,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                user_info={
                    'upload_dir': str(upload_dir),
                    'type': 'individual_upload'
                },
                permissions=['upload', 'read'],
                metadata={
                    'upload_dir': str(upload_dir),
                    'max_file_size': max_upload_size
                }
            )
            
            self.credential_manager.store_connection(connection)
            
            return AuthenticationResult(
                success=True,
                status=AuthenticationStatus.AUTHENTICATED,
                connection_id=connection_id,
                user_info=connection.user_info,
                permissions=connection.permissions,
                metadata=connection.metadata
            )
            
        except Exception as e:
            self.logger.error(f"Upload handler setup failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )


class MultiSourceAuthenticationService:
    """Main multi-source authentication service"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Initialize credential manager
        self.credential_manager = CredentialManager()
        
        # Initialize authenticators
        self.google_drive_auth = GoogleDriveAuthenticator(self.credential_manager)
        self.cloud_storage_auth = CloudStorageAuthenticator(self.credential_manager)
        self.local_source_validator = LocalSourceValidator(self.credential_manager)
    
    async def authenticate_source(
        self,
        source_type: DataSourceType,
        user_id: str,
        auth_config: Dict[str, Any]
    ) -> AuthenticationResult:
        """Authenticate with a data source"""
        try:
            if source_type == DataSourceType.GOOGLE_DRIVE:
                if 'service_account_info' in auth_config:
                    return await self.google_drive_auth.authenticate_service_account(
                        user_id, auth_config['service_account_info']
                    )
                elif 'client_config' in auth_config:
                    return await self.google_drive_auth.authenticate_oauth2(
                        user_id, auth_config['client_config']
                    )
                else:
                    return AuthenticationResult(
                        success=False,
                        status=AuthenticationStatus.INVALID,
                        error="Missing authentication configuration"
                    )
            
            elif source_type == DataSourceType.AWS_S3:
                return await self.cloud_storage_auth.authenticate_aws_s3(
                    user_id, auth_config
                )
            
            elif source_type == DataSourceType.AZURE_BLOB:
                return await self.cloud_storage_auth.authenticate_azure_blob(
                    user_id, auth_config
                )
            
            elif source_type == DataSourceType.GOOGLE_CLOUD_STORAGE:
                return await self.cloud_storage_auth.authenticate_gcs(
                    user_id, auth_config
                )
            
            elif source_type == DataSourceType.LOCAL_DIRECTORY:
                return await self.local_source_validator.validate_local_directory(
                    user_id, auth_config['directory_path']
                )
            
            elif source_type == DataSourceType.LOCAL_ZIP:
                return await self.local_source_validator.validate_zip_file(
                    user_id, auth_config['zip_path']
                )
            
            elif source_type == DataSourceType.INDIVIDUAL_UPLOAD:
                return await self.local_source_validator.setup_upload_handler(user_id)
            
            else:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.ERROR,
                    error=f"Unsupported source type: {source_type}"
                )
                
        except Exception as e:
            self.logger.error(f"Authentication failed for {source_type}: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def refresh_connection(self, connection_id: str) -> AuthenticationResult:
        """Refresh connection credentials"""
        try:
            connection = self.credential_manager.get_connection(connection_id)
            if not connection:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error="Connection not found"
                )
            
            if connection.source_type == DataSourceType.GOOGLE_DRIVE:
                return await self.google_drive_auth.refresh_token(connection_id)
            else:
                # For other sources, validate the existing connection
                return await self.validate_connection(connection_id)
                
        except Exception as e:
            self.logger.error(f"Connection refresh failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def validate_connection(self, connection_id: str) -> AuthenticationResult:
        """Validate existing connection"""
        try:
            connection = self.credential_manager.get_connection(connection_id)
            if not connection:
                return AuthenticationResult(
                    success=False,
                    status=AuthenticationStatus.INVALID,
                    error="Connection not found"
                )
            
            if connection.source_type == DataSourceType.GOOGLE_DRIVE:
                return await self.google_drive_auth.validate_connection(connection_id)
            else:
                # For other sources, assume valid if stored
                connection.last_accessed = datetime.now(timezone.utc)
                self.credential_manager.store_connection(connection)
                
                return AuthenticationResult(
                    success=True,
                    status=AuthenticationStatus.AUTHENTICATED,
                    connection_id=connection_id,
                    user_info=connection.user_info,
                    permissions=connection.permissions
                )
                
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return AuthenticationResult(
                success=False,
                status=AuthenticationStatus.ERROR,
                error=str(e)
            )
    
    async def disconnect_source(self, connection_id: str) -> bool:
        """Disconnect from a data source"""
        try:
            return self.credential_manager.delete_connection(connection_id)
        except Exception as e:
            self.logger.error(f"Disconnect failed: {e}")
            return False
    
    async def get_connection_status(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection status information"""
        return self.credential_manager.get_connection(connection_id)
    
    async def list_connections(self, user_id: Optional[str] = None) -> List[ConnectionInfo]:
        """List all connections, optionally filtered by user"""
        try:
            connections_data = self.credential_manager.load_connections()
            connections = []
            
            for conn_id, conn_data in connections_data.items():
                if user_id is None or conn_data.get('user_id') == user_id:
                    connections.append(ConnectionInfo(**conn_data))
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Failed to list connections: {e}")
            return []
    
    async def get_source_capabilities(self, source_type: DataSourceType) -> Dict[str, Any]:
        """Get capabilities for a data source type"""
        capabilities = {
            DataSourceType.GOOGLE_DRIVE: {
                'can_browse': True,
                'can_search': True,
                'can_upload': False,
                'can_download': True,
                'supports_auth': True,
                'supports_batch': True,
                'auth_methods': ['oauth2', 'service_account']
            },
            DataSourceType.LOCAL_ZIP: {
                'can_browse': True,
                'can_search': False,
                'can_upload': False,
                'can_download': True,
                'supports_auth': False,
                'supports_batch': True,
                'auth_methods': ['file_path']
            },
            DataSourceType.LOCAL_DIRECTORY: {
                'can_browse': True,
                'can_search': False,
                'can_upload': False,
                'can_download': True,
                'supports_auth': False,
                'supports_batch': True,
                'auth_methods': ['directory_path']
            },
            DataSourceType.INDIVIDUAL_UPLOAD: {
                'can_browse': False,
                'can_search': False,
                'can_upload': True,
                'can_download': False,
                'supports_auth': False,
                'supports_batch': True,
                'auth_methods': ['none']
            },
            DataSourceType.AWS_S3: {
                'can_browse': True,
                'can_search': True,
                'can_upload': False,
                'can_download': True,
                'supports_auth': True,
                'supports_batch': True,
                'auth_methods': ['access_key'],
                'available': AWS_AVAILABLE
            },
            DataSourceType.AZURE_BLOB: {
                'can_browse': True,
                'can_search': True,
                'can_upload': False,
                'can_download': True,
                'supports_auth': True,
                'supports_batch': True,
                'auth_methods': ['connection_string', 'sas_token'],
                'available': AZURE_AVAILABLE
            },
            DataSourceType.GOOGLE_CLOUD_STORAGE: {
                'can_browse': True,
                'can_search': True,
                'can_upload': False,
                'can_download': True,
                'supports_auth': True,
                'supports_batch': True,
                'auth_methods': ['service_account', 'default'],
                'available': GCS_AVAILABLE
            }
        }
        
        return capabilities.get(source_type, {})


# Global service instance
_auth_service: Optional[MultiSourceAuthenticationService] = None


def get_auth_service() -> MultiSourceAuthenticationService:
    """Get or create global authentication service instance"""
    global _auth_service
    
    if _auth_service is None:
        _auth_service = MultiSourceAuthenticationService()
    
    return _auth_service
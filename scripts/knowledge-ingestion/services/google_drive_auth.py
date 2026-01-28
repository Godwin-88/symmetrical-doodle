"""
Google Drive Authentication Service

Implements OAuth2 and service account authentication methods with credential validation,
token refresh mechanisms, and secure credential storage and retrieval.

Requirements: 1.1
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..core.config import get_settings
from ..core.logging import get_logger


class AuthMethod(Enum):
    """Authentication method types"""
    SERVICE_ACCOUNT = "service_account"
    OAUTH2 = "oauth2"


@dataclass
class AuthResult:
    """Authentication result with status and credentials"""
    success: bool
    credentials: Optional[Credentials] = None
    auth_method: Optional[AuthMethod] = None
    error_message: Optional[str] = None
    expires_at: Optional[datetime] = None


class GoogleDriveAuthService:
    """
    Google Drive authentication service supporting both OAuth2 and service account methods.
    Provides credential validation, token refresh, and secure storage.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._credentials: Optional[Credentials] = None
        self._auth_method: Optional[AuthMethod] = None
        self._token_file = Path(self.settings.google_drive.credentials_path).parent / "token.pickle"
        
    async def authenticate(self) -> AuthResult:
        """
        Authenticate with Google Drive API using configured method.
        
        Returns:
            AuthResult with success status and credentials
        """
        self.logger.info("Starting Google Drive authentication")
        
        # Try service account first if configured
        if self.settings.google_drive.service_account_file:
            result = await self._authenticate_service_account()
            if result.success:
                self._credentials = result.credentials
                self._auth_method = AuthMethod.SERVICE_ACCOUNT
                self.logger.info("Successfully authenticated using service account")
                return result
            else:
                self.logger.warning(f"Service account authentication failed: {result.error_message}")
        
        # Try OAuth2 if service account failed or not configured
        if self.settings.google_drive.oauth_client_id and self.settings.google_drive.oauth_client_secret:
            result = await self._authenticate_oauth2()
            if result.success:
                self._credentials = result.credentials
                self._auth_method = AuthMethod.OAUTH2
                self.logger.info("Successfully authenticated using OAuth2")
                return result
            else:
                self.logger.error(f"OAuth2 authentication failed: {result.error_message}")
        
        # Try credentials file path (could be service account or OAuth2 client secrets)
        if self.settings.google_drive.credentials_path:
            result = await self._authenticate_from_file()
            if result.success:
                self._credentials = result.credentials
                self._auth_method = result.auth_method
                self.logger.info(f"Successfully authenticated using credentials file with {result.auth_method.value}")
                return result
            else:
                self.logger.error(f"File-based authentication failed: {result.error_message}")
        
        error_msg = "No valid authentication method configured or all methods failed"
        self.logger.error(error_msg)
        return AuthResult(success=False, error_message=error_msg)
    
    async def _authenticate_service_account(self) -> AuthResult:
        """Authenticate using service account credentials"""
        try:
            service_account_file = self.settings.google_drive.service_account_file
            if not os.path.exists(service_account_file):
                return AuthResult(
                    success=False,
                    error_message=f"Service account file not found: {service_account_file}"
                )
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=self.settings.google_drive.scopes
            )
            
            # Validate credentials by making a test API call
            validation_result = await self._validate_credentials(credentials)
            if not validation_result:
                return AuthResult(
                    success=False,
                    error_message="Service account credentials validation failed"
                )
            
            return AuthResult(
                success=True,
                credentials=credentials,
                auth_method=AuthMethod.SERVICE_ACCOUNT,
                expires_at=None  # Service account credentials don't expire
            )
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"Service account authentication error: {str(e)}"
            )
    
    async def _authenticate_oauth2(self) -> AuthResult:
        """Authenticate using OAuth2 flow"""
        try:
            # Check for existing token
            if self._token_file.exists():
                with open(self._token_file, 'rb') as token:
                    credentials = pickle.load(token)
                
                # Check if credentials are valid and not expired
                if credentials and credentials.valid:
                    return AuthResult(
                        success=True,
                        credentials=credentials,
                        auth_method=AuthMethod.OAUTH2,
                        expires_at=credentials.expiry
                    )
                
                # Try to refresh expired credentials
                if credentials and credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                        await self._save_token(credentials)
                        return AuthResult(
                            success=True,
                            credentials=credentials,
                            auth_method=AuthMethod.OAUTH2,
                            expires_at=credentials.expiry
                        )
                    except Exception as e:
                        self.logger.warning(f"Token refresh failed: {str(e)}")
            
            # Create OAuth2 client configuration
            client_config = {
                "installed": {
                    "client_id": self.settings.google_drive.oauth_client_id,
                    "client_secret": self.settings.google_drive.oauth_client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"]
                }
            }
            
            # Run OAuth2 flow
            flow = InstalledAppFlow.from_client_config(
                client_config,
                scopes=self.settings.google_drive.scopes
            )
            
            # Use local server for OAuth2 flow
            credentials = flow.run_local_server(port=0)
            
            # Save token for future use
            await self._save_token(credentials)
            
            return AuthResult(
                success=True,
                credentials=credentials,
                auth_method=AuthMethod.OAUTH2,
                expires_at=credentials.expiry
            )
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"OAuth2 authentication error: {str(e)}"
            )
    
    async def _authenticate_from_file(self) -> AuthResult:
        """Authenticate from credentials file (auto-detect type)"""
        try:
            credentials_path = self.settings.google_drive.credentials_path
            if not os.path.exists(credentials_path):
                return AuthResult(
                    success=False,
                    error_message=f"Credentials file not found: {credentials_path}"
                )
            
            # Read and parse credentials file
            with open(credentials_path, 'r') as f:
                cred_data = json.load(f)
            
            # Detect credential type
            if "type" in cred_data and cred_data["type"] == "service_account":
                # Service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=self.settings.google_drive.scopes
                )
                
                validation_result = await self._validate_credentials(credentials)
                if not validation_result:
                    return AuthResult(
                        success=False,
                        error_message="Service account credentials validation failed"
                    )
                
                return AuthResult(
                    success=True,
                    credentials=credentials,
                    auth_method=AuthMethod.SERVICE_ACCOUNT,
                    expires_at=None
                )
            
            elif "installed" in cred_data or "web" in cred_data:
                # OAuth2 client secrets
                # Check for existing token first
                if self._token_file.exists():
                    with open(self._token_file, 'rb') as token:
                        credentials = pickle.load(token)
                    
                    if credentials and credentials.valid:
                        return AuthResult(
                            success=True,
                            credentials=credentials,
                            auth_method=AuthMethod.OAUTH2,
                            expires_at=credentials.expiry
                        )
                    
                    # Try to refresh
                    if credentials and credentials.expired and credentials.refresh_token:
                        try:
                            credentials.refresh(Request())
                            await self._save_token(credentials)
                            return AuthResult(
                                success=True,
                                credentials=credentials,
                                auth_method=AuthMethod.OAUTH2,
                                expires_at=credentials.expiry
                            )
                        except Exception as e:
                            self.logger.warning(f"Token refresh failed: {str(e)}")
                
                # Run OAuth2 flow
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path,
                    scopes=self.settings.google_drive.scopes
                )
                credentials = flow.run_local_server(port=0)
                await self._save_token(credentials)
                
                return AuthResult(
                    success=True,
                    credentials=credentials,
                    auth_method=AuthMethod.OAUTH2,
                    expires_at=credentials.expiry
                )
            
            else:
                return AuthResult(
                    success=False,
                    error_message="Unknown credentials file format"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                error_message=f"File-based authentication error: {str(e)}"
            )
    
    async def _validate_credentials(self, credentials: Credentials) -> bool:
        """
        Validate credentials by making a test API call.
        
        Args:
            credentials: Google credentials to validate
            
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            # Build Drive service
            service = build('drive', 'v3', credentials=credentials)
            
            # Make a simple API call to validate credentials
            result = service.about().get(fields="user").execute()
            
            if result and "user" in result:
                self.logger.info(f"Credentials validated for user: {result['user'].get('emailAddress', 'Unknown')}")
                return True
            
            return False
            
        except HttpError as e:
            self.logger.error(f"Credentials validation failed with HTTP error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Credentials validation failed: {str(e)}")
            return False
    
    async def _save_token(self, credentials: Credentials):
        """Save OAuth2 token to file for future use"""
        try:
            # Ensure directory exists
            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._token_file, 'wb') as token:
                pickle.dump(credentials, token)
            
            # Set restrictive permissions
            os.chmod(self._token_file, 0o600)
            
            self.logger.info(f"Token saved to {self._token_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save token: {str(e)}")
    
    async def refresh_credentials(self) -> AuthResult:
        """
        Refresh current credentials if they are expired or about to expire.
        
        Returns:
            AuthResult with refreshed credentials or error
        """
        if not self._credentials:
            return AuthResult(
                success=False,
                error_message="No credentials to refresh"
            )
        
        if self._auth_method == AuthMethod.SERVICE_ACCOUNT:
            # Service account credentials don't need refresh
            return AuthResult(
                success=True,
                credentials=self._credentials,
                auth_method=self._auth_method
            )
        
        if self._auth_method == AuthMethod.OAUTH2:
            try:
                # Check if refresh is needed
                if self._credentials.valid and not self._credentials_need_refresh():
                    return AuthResult(
                        success=True,
                        credentials=self._credentials,
                        auth_method=self._auth_method,
                        expires_at=self._credentials.expiry
                    )
                
                # Refresh credentials
                if self._credentials.refresh_token:
                    self._credentials.refresh(Request())
                    await self._save_token(self._credentials)
                    
                    return AuthResult(
                        success=True,
                        credentials=self._credentials,
                        auth_method=self._auth_method,
                        expires_at=self._credentials.expiry
                    )
                else:
                    return AuthResult(
                        success=False,
                        error_message="No refresh token available, re-authentication required"
                    )
                    
            except Exception as e:
                return AuthResult(
                    success=False,
                    error_message=f"Credential refresh failed: {str(e)}"
                )
        
        return AuthResult(
            success=False,
            error_message="Unknown authentication method"
        )
    
    def _credentials_need_refresh(self) -> bool:
        """Check if credentials need refresh (expire within 5 minutes)"""
        if not self._credentials or not self._credentials.expiry:
            return False
        
        # Refresh if expiring within 5 minutes
        return self._credentials.expiry <= datetime.utcnow() + timedelta(minutes=5)
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get current credentials"""
        return self._credentials
    
    def get_auth_method(self) -> Optional[AuthMethod]:
        """Get current authentication method"""
        return self._auth_method
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid credentials"""
        return (
            self._credentials is not None and 
            self._credentials.valid and
            not self._credentials_need_refresh()
        )
    
    async def get_drive_service(self):
        """
        Get authenticated Google Drive service instance.
        
        Returns:
            Google Drive service instance or None if not authenticated
        """
        if not self.is_authenticated():
            auth_result = await self.refresh_credentials()
            if not auth_result.success:
                self.logger.error("Failed to get valid credentials for Drive service")
                return None
        
        try:
            return build('drive', 'v3', credentials=self._credentials)
        except Exception as e:
            self.logger.error(f"Failed to build Drive service: {str(e)}")
            return None
    
    def get_auth_info(self) -> Dict[str, Any]:
        """
        Get current authentication information.
        
        Returns:
            Dictionary with authentication status and details
        """
        if not self._credentials:
            return {
                "authenticated": False,
                "auth_method": None,
                "expires_at": None,
                "needs_refresh": False
            }
        
        return {
            "authenticated": self.is_authenticated(),
            "auth_method": self._auth_method.value if self._auth_method else None,
            "expires_at": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
            "needs_refresh": self._credentials_need_refresh(),
            "valid": self._credentials.valid if self._credentials else False
        }
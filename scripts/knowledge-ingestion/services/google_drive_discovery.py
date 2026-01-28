"""
Google Drive File Discovery and Metadata Extraction Service

Implements recursive folder scanning, comprehensive metadata extraction,
PDF filtering, and access validation for discovered files.

Requirements: 1.2, 1.3, 1.5
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from googleapiclient.errors import HttpError
from googleapiclient.discovery import Resource

from .google_drive_auth import GoogleDriveAuthService
from ..core.config import get_settings
from ..core.logging import get_logger


class AccessStatus(Enum):
    """File access status"""
    ACCESSIBLE = "accessible"
    RESTRICTED = "restricted"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class PDFMetadata:
    """PDF file metadata extracted from Google Drive"""
    file_id: str
    name: str
    mime_type: str
    modified_time: datetime
    size: int
    web_view_link: str
    access_status: AccessStatus
    domain_classification: Optional[str] = None
    parent_folders: List[str] = field(default_factory=list)
    owners: List[str] = field(default_factory=list)
    created_time: Optional[datetime] = None
    md5_checksum: Optional[str] = None
    version: Optional[str] = None
    shared: bool = False
    download_url: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Result of file discovery operation"""
    success: bool
    pdfs_found: List[PDFMetadata] = field(default_factory=list)
    total_files_scanned: int = 0
    inaccessible_files: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    folders_scanned: List[str] = field(default_factory=list)


class GoogleDriveDiscoveryService:
    """
    Google Drive file discovery service with recursive folder scanning,
    metadata extraction, PDF filtering, and access validation.
    """
    
    def __init__(self, auth_service: GoogleDriveAuthService):
        self.auth_service = auth_service
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._drive_service: Optional[Resource] = None
        
    async def discover_pdfs(self, folder_ids: Optional[List[str]] = None) -> DiscoveryResult:
        """
        Discover all PDF files in specified folders with comprehensive metadata extraction.
        
        Args:
            folder_ids: List of Google Drive folder IDs to scan. If None, uses config.
            
        Returns:
            DiscoveryResult with found PDFs and operation details
        """
        self.logger.info("Starting PDF discovery process")
        
        # Get Drive service
        self._drive_service = await self.auth_service.get_drive_service()
        if not self._drive_service:
            return DiscoveryResult(
                success=False,
                errors=["Failed to get authenticated Drive service"]
            )
        
        # Use provided folder IDs or get from config
        target_folders = folder_ids or self.settings.google_drive.folder_ids
        if not target_folders:
            return DiscoveryResult(
                success=False,
                errors=["No folder IDs specified for discovery"]
            )
        
        result = DiscoveryResult(success=True)
        
        # Discover files in each folder
        for folder_id in target_folders:
            self.logger.info(f"Scanning folder: {folder_id}")
            
            try:
                folder_result = await self._scan_folder_recursive(folder_id)
                
                # Merge results
                result.pdfs_found.extend(folder_result.pdfs_found)
                result.total_files_scanned += folder_result.total_files_scanned
                result.inaccessible_files.extend(folder_result.inaccessible_files)
                result.errors.extend(folder_result.errors)
                result.folders_scanned.extend(folder_result.folders_scanned)
                
            except Exception as e:
                error_msg = f"Error scanning folder {folder_id}: {str(e)}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Remove duplicates based on file_id
        seen_ids = set()
        unique_pdfs = []
        for pdf in result.pdfs_found:
            if pdf.file_id not in seen_ids:
                unique_pdfs.append(pdf)
                seen_ids.add(pdf.file_id)
        
        result.pdfs_found = unique_pdfs
        
        self.logger.info(
            f"Discovery completed: {len(result.pdfs_found)} PDFs found, "
            f"{result.total_files_scanned} total files scanned, "
            f"{len(result.inaccessible_files)} inaccessible files"
        )
        
        return result
    
    async def _scan_folder_recursive(self, folder_id: str, visited: Optional[Set[str]] = None) -> DiscoveryResult:
        """
        Recursively scan a folder and its subfolders for PDF files.
        
        Args:
            folder_id: Google Drive folder ID to scan
            visited: Set of already visited folder IDs to prevent infinite loops
            
        Returns:
            DiscoveryResult for this folder and its subfolders
        """
        if visited is None:
            visited = set()
        
        if folder_id in visited:
            self.logger.warning(f"Circular reference detected, skipping folder: {folder_id}")
            return DiscoveryResult(success=True)
        
        visited.add(folder_id)
        result = DiscoveryResult(success=True)
        result.folders_scanned.append(folder_id)
        
        try:
            # Get all files in the folder
            files = await self._list_files_in_folder(folder_id)
            result.total_files_scanned += len(files)
            
            # Process each file
            for file_info in files:
                try:
                    # Check if it's a folder
                    if file_info.get('mimeType') == 'application/vnd.google-apps.folder':
                        # Recursively scan subfolder
                        subfolder_result = await self._scan_folder_recursive(
                            file_info['id'], visited.copy()
                        )
                        
                        # Merge subfolder results
                        result.pdfs_found.extend(subfolder_result.pdfs_found)
                        result.total_files_scanned += subfolder_result.total_files_scanned
                        result.inaccessible_files.extend(subfolder_result.inaccessible_files)
                        result.errors.extend(subfolder_result.errors)
                        result.folders_scanned.extend(subfolder_result.folders_scanned)
                        
                    # Check if it's a PDF
                    elif file_info.get('mimeType') == 'application/pdf':
                        pdf_metadata = await self._extract_pdf_metadata(file_info)
                        if pdf_metadata:
                            result.pdfs_found.append(pdf_metadata)
                        else:
                            result.inaccessible_files.append({
                                'file_id': file_info.get('id'),
                                'name': file_info.get('name'),
                                'reason': 'Failed to extract metadata'
                            })
                
                except Exception as e:
                    error_msg = f"Error processing file {file_info.get('id', 'unknown')}: {str(e)}"
                    self.logger.warning(error_msg)
                    result.errors.append(error_msg)
                    
                    result.inaccessible_files.append({
                        'file_id': file_info.get('id'),
                        'name': file_info.get('name', 'unknown'),
                        'reason': str(e)
                    })
        
        except Exception as e:
            error_msg = f"Error scanning folder {folder_id}: {str(e)}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
            result.success = False
        
        return result
    
    async def _list_files_in_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """
        List all files in a specific folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            List of file information dictionaries
        """
        files = []
        page_token = None
        
        while True:
            try:
                # Query for files in the folder
                query = f"'{folder_id}' in parents and trashed=false"
                
                request = self._drive_service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, "
                           "webViewLink, parents, owners, createdTime, md5Checksum, version, shared)",
                    pageSize=1000,
                    pageToken=page_token
                )
                
                response = request.execute()
                files.extend(response.get('files', []))
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            except HttpError as e:
                if e.resp.status == 404:
                    self.logger.warning(f"Folder not found: {folder_id}")
                    break
                elif e.resp.status == 403:
                    self.logger.warning(f"Access denied to folder: {folder_id}")
                    break
                else:
                    raise e
        
        return files
    
    async def _extract_pdf_metadata(self, file_info: Dict[str, Any]) -> Optional[PDFMetadata]:
        """
        Extract comprehensive metadata from a PDF file.
        
        Args:
            file_info: File information from Google Drive API
            
        Returns:
            PDFMetadata object or None if extraction fails
        """
        try:
            # Validate access to the file
            access_status = await self._validate_file_access(file_info['id'])
            
            # Parse datetime fields
            modified_time = None
            if file_info.get('modifiedTime'):
                modified_time = datetime.fromisoformat(
                    file_info['modifiedTime'].replace('Z', '+00:00')
                )
            
            created_time = None
            if file_info.get('createdTime'):
                created_time = datetime.fromisoformat(
                    file_info['createdTime'].replace('Z', '+00:00')
                )
            
            # Extract owner information
            owners = []
            if file_info.get('owners'):
                owners = [owner.get('emailAddress', owner.get('displayName', 'Unknown')) 
                         for owner in file_info['owners']]
            
            # Create download URL
            download_url = f"https://drive.google.com/uc?id={file_info['id']}&export=download"
            
            # Estimate domain classification based on filename
            domain_classification = self._estimate_domain_from_filename(file_info.get('name', ''))
            
            return PDFMetadata(
                file_id=file_info['id'],
                name=file_info.get('name', 'Unknown'),
                mime_type=file_info.get('mimeType', 'application/pdf'),
                modified_time=modified_time or datetime.now(),
                size=int(file_info.get('size', 0)),
                web_view_link=file_info.get('webViewLink', ''),
                access_status=access_status,
                domain_classification=domain_classification,
                parent_folders=file_info.get('parents', []),
                owners=owners,
                created_time=created_time,
                md5_checksum=file_info.get('md5Checksum'),
                version=file_info.get('version'),
                shared=file_info.get('shared', False),
                download_url=download_url
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for file {file_info.get('id')}: {str(e)}")
            return None
    
    async def _validate_file_access(self, file_id: str) -> AccessStatus:
        """
        Validate access to a specific file.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            AccessStatus indicating file accessibility
        """
        try:
            # Try to get file metadata to validate access
            request = self._drive_service.files().get(
                fileId=file_id,
                fields="id, name, mimeType"
            )
            
            response = request.execute()
            
            if response and response.get('id') == file_id:
                return AccessStatus.ACCESSIBLE
            else:
                return AccessStatus.ERROR
                
        except HttpError as e:
            if e.resp.status == 404:
                return AccessStatus.NOT_FOUND
            elif e.resp.status == 403:
                return AccessStatus.RESTRICTED
            else:
                self.logger.warning(f"HTTP error validating access to {file_id}: {e}")
                return AccessStatus.ERROR
        except Exception as e:
            self.logger.warning(f"Error validating access to {file_id}: {str(e)}")
            return AccessStatus.ERROR
    
    def _estimate_domain_from_filename(self, filename: str) -> Optional[str]:
        """
        Estimate domain classification based on filename analysis.
        
        Args:
            filename: PDF filename
            
        Returns:
            Estimated domain or None
        """
        if not filename:
            return None
        
        filename_lower = filename.lower()
        
        # Define domain keywords
        domain_keywords = {
            'ml': ['machine learning', 'ml', 'neural', 'deep learning', 'ai', 'artificial intelligence'],
            'drl': ['reinforcement learning', 'rl', 'drl', 'deep reinforcement', 'q-learning', 'policy'],
            'nlp': ['nlp', 'natural language', 'text processing', 'language model', 'bert', 'transformer'],
            'llm': ['llm', 'large language', 'gpt', 'language model', 'chatgpt', 'generative'],
            'finance': ['finance', 'trading', 'market', 'portfolio', 'investment', 'risk', 'quantitative', 'algorithmic trading']
        }
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in filename_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    async def get_folder_info(self, folder_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Folder information dictionary or None if not accessible
        """
        try:
            if not self._drive_service:
                self._drive_service = await self.auth_service.get_drive_service()
                if not self._drive_service:
                    return None
            
            request = self._drive_service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType, modifiedTime, parents, owners, shared"
            )
            
            response = request.execute()
            return response
            
        except HttpError as e:
            self.logger.error(f"Failed to get folder info for {folder_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting folder info for {folder_id}: {str(e)}")
            return None
    
    async def validate_folder_access(self, folder_ids: List[str]) -> Dict[str, bool]:
        """
        Validate access to multiple folders.
        
        Args:
            folder_ids: List of Google Drive folder IDs
            
        Returns:
            Dictionary mapping folder IDs to access status (True/False)
        """
        access_results = {}
        
        for folder_id in folder_ids:
            folder_info = await self.get_folder_info(folder_id)
            access_results[folder_id] = folder_info is not None
        
        return access_results
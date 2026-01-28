"""
Local ZIP File Discovery Service

Replaces Google Drive discovery with local ZIP file processing.
Extracts PDFs from ZIP files and creates metadata for ingestion.

Requirements: 1.2, 1.3, 1.5 (adapted for local files)
"""

import asyncio
import zipfile
import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import hashlib
import mimetypes

# Simplified imports for local processing
try:
    from ..core.config import get_settings
    from ..core.logging import get_logger, log_context
except ImportError:
    # Fallback for testing
    import logging
    
    def get_settings():
        return None
    
    def get_logger(name, component=None):
        return logging.getLogger(name)
    
    class log_context:
        def __init__(self, component, operation, correlation_id=None):
            self.correlation_id = correlation_id or "local"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


class AccessStatus(Enum):
    """File access status"""
    ACCESSIBLE = "accessible"
    RESTRICTED = "restricted"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class PDFMetadata:
    """PDF file metadata extracted from local ZIP file"""
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
    local_path: Optional[str] = None  # Path within ZIP or extracted location


@dataclass
class DiscoveryResult:
    """Result of local file discovery operation"""
    success: bool
    pdfs_found: List[PDFMetadata] = field(default_factory=list)
    total_files_scanned: int = 0
    inaccessible_files: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    folders_scanned: List[str] = field(default_factory=list)
    zip_files_processed: List[str] = field(default_factory=list)


class LocalZipDiscoveryService:
    """
    Local ZIP file discovery service for processing PDFs from ZIP archives.
    Replaces Google Drive discovery for local testing and development.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="local_zip_discovery")
        
    async def discover_pdfs_from_zip(
        self, 
        zip_path: str,
        extract_to: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> DiscoveryResult:
        """
        Discover all PDF files in a ZIP archive.
        
        Args:
            zip_path: Path to the ZIP file
            extract_to: Optional directory to extract files to
            correlation_id: Correlation ID for logging
            
        Returns:
            DiscoveryResult with found PDFs and operation details
        """
        with log_context("local_zip_discovery", "discover_pdfs", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting local ZIP PDF discovery: {zip_path}")
            
            zip_file_path = Path(zip_path)
            if not zip_file_path.exists():
                return DiscoveryResult(
                    success=False,
                    errors=[f"ZIP file not found: {zip_path}"]
                )
            
            result = DiscoveryResult(success=True)
            result.zip_files_processed.append(str(zip_file_path))
            
            # Set up extraction directory
            if extract_to is None:
                extract_to = zip_file_path.parent / f"{zip_file_path.stem}_extracted"
            
            extract_path = Path(extract_to)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # Process ZIP file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # Get list of all files in ZIP
                    file_list = zip_ref.namelist()
                    result.total_files_scanned = len(file_list)
                    
                    self.logger.info(f"Found {len(file_list)} files in ZIP archive")
                    
                    # Process each file
                    for file_path in file_list:
                        try:
                            # Skip directories
                            if file_path.endswith('/'):
                                continue
                            
                            # Check if it's a PDF
                            if self._is_pdf_file(file_path):
                                pdf_metadata = await self._extract_pdf_metadata_from_zip(
                                    zip_ref, file_path, extract_path, zip_file_path
                                )
                                
                                if pdf_metadata:
                                    result.pdfs_found.append(pdf_metadata)
                                    self.logger.info(f"Found PDF: {pdf_metadata.name}")
                                else:
                                    result.inaccessible_files.append({
                                        'file_id': self._generate_file_id(file_path),
                                        'name': Path(file_path).name,
                                        'reason': 'Failed to extract metadata'
                                    })
                        
                        except Exception as e:
                            error_msg = f"Error processing file {file_path}: {str(e)}"
                            self.logger.warning(error_msg)
                            result.errors.append(error_msg)
                            
                            result.inaccessible_files.append({
                                'file_id': self._generate_file_id(file_path),
                                'name': Path(file_path).name,
                                'reason': str(e)
                            })
                
                # Track folders scanned
                folders_in_zip = set()
                for pdf in result.pdfs_found:
                    folders_in_zip.update(pdf.parent_folders)
                result.folders_scanned = list(folders_in_zip)
                
                self.logger.info(
                    f"Discovery completed: {len(result.pdfs_found)} PDFs found, "
                    f"{result.total_files_scanned} total files scanned, "
                    f"{len(result.inaccessible_files)} inaccessible files"
                )
                
                return result
                
            except Exception as e:
                error_msg = f"Error processing ZIP file {zip_path}: {str(e)}"
                self.logger.error(error_msg)
                return DiscoveryResult(
                    success=False,
                    errors=[error_msg]
                )
    
    async def discover_pdfs_from_directory(
        self, 
        directory_path: str,
        recursive: bool = True,
        correlation_id: Optional[str] = None
    ) -> DiscoveryResult:
        """
        Discover all PDF files in a local directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to scan subdirectories
            correlation_id: Correlation ID for logging
            
        Returns:
            DiscoveryResult with found PDFs
        """
        with log_context("local_zip_discovery", "discover_directory", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting directory PDF discovery: {directory_path}")
            
            dir_path = Path(directory_path)
            if not dir_path.exists():
                return DiscoveryResult(
                    success=False,
                    errors=[f"Directory not found: {directory_path}"]
                )
            
            result = DiscoveryResult(success=True)
            result.folders_scanned.append(str(dir_path))
            
            try:
                # Get all files
                if recursive:
                    files = list(dir_path.rglob("*"))
                else:
                    files = list(dir_path.glob("*"))
                
                pdf_files = [f for f in files if f.is_file() and self._is_pdf_file(str(f))]
                result.total_files_scanned = len(files)
                
                self.logger.info(f"Found {len(pdf_files)} PDF files in directory")
                
                # Process each PDF
                for pdf_file in pdf_files:
                    try:
                        pdf_metadata = await self._extract_pdf_metadata_from_file(pdf_file, dir_path)
                        if pdf_metadata:
                            result.pdfs_found.append(pdf_metadata)
                        else:
                            result.inaccessible_files.append({
                                'file_id': self._generate_file_id(str(pdf_file)),
                                'name': pdf_file.name,
                                'reason': 'Failed to extract metadata'
                            })
                    
                    except Exception as e:
                        error_msg = f"Error processing file {pdf_file}: {str(e)}"
                        self.logger.warning(error_msg)
                        result.errors.append(error_msg)
                        
                        result.inaccessible_files.append({
                            'file_id': self._generate_file_id(str(pdf_file)),
                            'name': pdf_file.name,
                            'reason': str(e)
                        })
                
                return result
                
            except Exception as e:
                error_msg = f"Error scanning directory {directory_path}: {str(e)}"
                self.logger.error(error_msg)
                return DiscoveryResult(
                    success=False,
                    errors=[error_msg]
                )
    
    def _is_pdf_file(self, file_path: str) -> bool:
        """Check if a file is a PDF based on extension and mime type"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() != '.pdf':
            return False
        
        # Check mime type if possible
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type == 'application/pdf'
    
    async def _extract_pdf_metadata_from_zip(
        self, 
        zip_ref: zipfile.ZipFile, 
        file_path: str, 
        extract_path: Path,
        zip_file_path: Path
    ) -> Optional[PDFMetadata]:
        """Extract metadata from a PDF file within a ZIP archive"""
        try:
            # Get file info from ZIP
            zip_info = zip_ref.getinfo(file_path)
            
            # Extract file temporarily for size calculation
            file_data = zip_ref.read(file_path)
            file_size = len(file_data)
            
            # Generate file ID
            file_id = self._generate_file_id(file_path)
            
            # Get timestamps
            zip_date = datetime(*zip_info.date_time, tzinfo=timezone.utc)
            
            # Extract to temporary location for processing
            extracted_path = extract_path / Path(file_path).name
            with open(extracted_path, 'wb') as f:
                f.write(file_data)
            
            # Calculate MD5 checksum
            md5_hash = hashlib.md5(file_data).hexdigest()
            
            # Estimate domain classification
            domain_classification = self._estimate_domain_from_filename(Path(file_path).name)
            
            # Get parent folders from ZIP path
            parent_folders = []
            if '/' in file_path:
                parent_folders = Path(file_path).parent.parts
            
            return PDFMetadata(
                file_id=file_id,
                name=Path(file_path).name,
                mime_type='application/pdf',
                modified_time=zip_date,
                size=file_size,
                web_view_link=f"file://{extracted_path}",
                access_status=AccessStatus.ACCESSIBLE,
                domain_classification=domain_classification,
                parent_folders=list(parent_folders),
                owners=['local_user'],
                created_time=zip_date,
                md5_checksum=md5_hash,
                version='1.0',
                shared=False,
                download_url=f"file://{extracted_path}",
                local_path=str(extracted_path)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for {file_path}: {str(e)}")
            return None
    
    async def _extract_pdf_metadata_from_file(
        self, 
        pdf_file: Path, 
        base_path: Path
    ) -> Optional[PDFMetadata]:
        """Extract metadata from a local PDF file"""
        try:
            # Get file stats
            stat = pdf_file.stat()
            
            # Generate file ID
            file_id = self._generate_file_id(str(pdf_file))
            
            # Get timestamps
            modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            created_time = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            
            # Calculate MD5 checksum
            with open(pdf_file, 'rb') as f:
                file_data = f.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
            
            # Estimate domain classification
            domain_classification = self._estimate_domain_from_filename(pdf_file.name)
            
            # Get relative path for parent folders
            try:
                relative_path = pdf_file.relative_to(base_path)
                parent_folders = list(relative_path.parent.parts) if relative_path.parent.parts != ('.',) else []
            except ValueError:
                parent_folders = []
            
            return PDFMetadata(
                file_id=file_id,
                name=pdf_file.name,
                mime_type='application/pdf',
                modified_time=modified_time,
                size=stat.st_size,
                web_view_link=f"file://{pdf_file}",
                access_status=AccessStatus.ACCESSIBLE,
                domain_classification=domain_classification,
                parent_folders=parent_folders,
                owners=['local_user'],
                created_time=created_time,
                md5_checksum=md5_hash,
                version='1.0',
                shared=False,
                download_url=f"file://{pdf_file}",
                local_path=str(pdf_file)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for {pdf_file}: {str(e)}")
            return None
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate a unique file ID based on file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:16]
    
    def _estimate_domain_from_filename(self, filename: str) -> Optional[str]:
        """Estimate domain classification based on filename analysis"""
        if not filename:
            return None
        
        filename_lower = filename.lower()
        
        # Define domain keywords
        domain_keywords = {
            'Machine Learning': ['machine learning', 'ml', 'neural', 'deep learning', 'ai', 'artificial intelligence'],
            'Deep Reinforcement Learning': ['reinforcement learning', 'rl', 'drl', 'deep reinforcement', 'q-learning', 'policy'],
            'Natural Language Processing': ['nlp', 'natural language', 'text processing', 'language model', 'bert', 'transformer'],
            'Large Language Models': ['llm', 'large language', 'gpt', 'language model', 'chatgpt', 'generative'],
            'Finance & Trading': ['finance', 'trading', 'market', 'portfolio', 'investment', 'risk', 'quantitative', 'algorithmic trading', 'taleb', 'black swan', 'antifragile', 'fooled by randomness']
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
        
        return 'General Technical'


# Utility functions for backward compatibility
async def discover_pdfs_from_zip(zip_path: str, extract_to: Optional[str] = None) -> DiscoveryResult:
    """Convenience function for ZIP discovery"""
    service = LocalZipDiscoveryService()
    return await service.discover_pdfs_from_zip(zip_path, extract_to)


async def discover_pdfs_from_directory(directory_path: str, recursive: bool = True) -> DiscoveryResult:
    """Convenience function for directory discovery"""
    service = LocalZipDiscoveryService()
    return await service.discover_pdfs_from_directory(directory_path, recursive)
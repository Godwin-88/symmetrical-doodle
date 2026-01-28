"""
Secure PDF Download Service

Implements authenticated download functionality using google-api-python-client,
retry logic with exponential backoff for network failures, and file validation
and corruption detection.

Requirements: 2.1, 10.2, 10.3
"""

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import random

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io

from .google_drive_auth import GoogleDriveAuthService
from .google_drive_discovery import PDFMetadata, AccessStatus
from ..core.config import get_settings
from ..core.logging import get_logger


class DownloadStatus(Enum):
    """Download operation status"""
    SUCCESS = "success"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    ACCESS_DENIED = "access_denied"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class DownloadResult:
    """Result of PDF download operation"""
    success: bool
    status: DownloadStatus
    file_content: Optional[bytes] = None
    file_size: int = 0
    download_time_ms: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    md5_checksum: Optional[str] = None
    validation_passed: bool = False


class PDFDownloadService:
    """
    Secure PDF download service with authentication, retry logic,
    and comprehensive file validation.
    """
    
    def __init__(self, auth_service: GoogleDriveAuthService):
        self.auth_service = auth_service
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._drive_service = None
        
    async def download_pdf(self, pdf_metadata: PDFMetadata) -> DownloadResult:
        """
        Download a PDF file with authentication, retry logic, and validation.
        
        Args:
            pdf_metadata: PDF metadata containing file information
            
        Returns:
            DownloadResult with download status and content
        """
        start_time = time.time()
        
        self.logger.info(f"Starting download for PDF: {pdf_metadata.name} (ID: {pdf_metadata.file_id})")
        
        # Check if file is accessible
        if pdf_metadata.access_status != AccessStatus.ACCESSIBLE:
            return DownloadResult(
                success=False,
                status=DownloadStatus.ACCESS_DENIED,
                error_message=f"File not accessible: {pdf_metadata.access_status.value}"
            )
        
        # Check file size limits
        max_size_bytes = self.settings.processing.max_file_size_mb * 1024 * 1024
        if pdf_metadata.size > max_size_bytes:
            return DownloadResult(
                success=False,
                status=DownloadStatus.FAILED,
                error_message=f"File too large: {pdf_metadata.size} bytes (max: {max_size_bytes})"
            )
        
        # Get authenticated Drive service
        self._drive_service = await self.auth_service.get_drive_service()
        if not self._drive_service:
            return DownloadResult(
                success=False,
                status=DownloadStatus.FAILED,
                error_message="Failed to get authenticated Drive service"
            )
        
        # Attempt download with retry logic
        result = await self._download_with_retry(pdf_metadata)
        
        # Calculate download time
        download_time_ms = int((time.time() - start_time) * 1000)
        result.download_time_ms = download_time_ms
        
        if result.success:
            self.logger.info(
                f"Successfully downloaded PDF: {pdf_metadata.name} "
                f"({result.file_size} bytes in {download_time_ms}ms)"
            )
        else:
            self.logger.error(
                f"Failed to download PDF: {pdf_metadata.name} - {result.error_message}"
            )
        
        return result
    
    async def _download_with_retry(self, pdf_metadata: PDFMetadata) -> DownloadResult:
        """
        Download PDF with exponential backoff retry logic.
        
        Args:
            pdf_metadata: PDF metadata
            
        Returns:
            DownloadResult with final download status
        """
        max_retries = self.settings.max_retries
        base_delay = self.settings.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                result = await self._perform_download(pdf_metadata)
                result.retry_count = attempt
                
                if result.success:
                    return result
                
                # Don't retry for certain error types
                if result.status in [DownloadStatus.ACCESS_DENIED, DownloadStatus.NOT_FOUND, DownloadStatus.CORRUPTED]:
                    return result
                
                # If this was the last attempt, return the result
                if attempt == max_retries:
                    return result
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed for {pdf_metadata.name}: {result.error_message}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                error_msg = f"Unexpected error during download attempt {attempt + 1}: {str(e)}"
                self.logger.error(error_msg)
                
                if attempt == max_retries:
                    return DownloadResult(
                        success=False,
                        status=DownloadStatus.FAILED,
                        error_message=error_msg,
                        retry_count=attempt
                    )
                
                # Wait before retry
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
        
        return DownloadResult(
            success=False,
            status=DownloadStatus.FAILED,
            error_message="Max retries exceeded",
            retry_count=max_retries
        )
    
    async def _perform_download(self, pdf_metadata: PDFMetadata) -> DownloadResult:
        """
        Perform the actual PDF download operation.
        
        Args:
            pdf_metadata: PDF metadata
            
        Returns:
            DownloadResult for this download attempt
        """
        try:
            # Get file from Drive API
            request = self._drive_service.files().get_media(fileId=pdf_metadata.file_id)
            
            # Create in-memory buffer for download
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    self.logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            # Get downloaded content
            file_content = file_buffer.getvalue()
            file_buffer.close()
            
            if not file_content:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.FAILED,
                    error_message="Downloaded file is empty"
                )
            
            # Validate the downloaded file
            validation_result = await self._validate_pdf_content(file_content, pdf_metadata)
            
            return DownloadResult(
                success=validation_result.success,
                status=validation_result.status,
                file_content=file_content if validation_result.success else None,
                file_size=len(file_content),
                error_message=validation_result.error_message,
                md5_checksum=validation_result.md5_checksum,
                validation_passed=validation_result.validation_passed
            )
            
        except HttpError as e:
            if e.resp.status == 404:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.NOT_FOUND,
                    error_message=f"File not found: {pdf_metadata.file_id}"
                )
            elif e.resp.status == 403:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.ACCESS_DENIED,
                    error_message=f"Access denied to file: {pdf_metadata.file_id}"
                )
            elif e.resp.status == 429:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.RATE_LIMITED,
                    error_message="Rate limit exceeded"
                )
            else:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.FAILED,
                    error_message=f"HTTP error {e.resp.status}: {str(e)}"
                )
        
        except Exception as e:
            return DownloadResult(
                success=False,
                status=DownloadStatus.FAILED,
                error_message=f"Download error: {str(e)}"
            )
    
    async def _validate_pdf_content(self, content: bytes, pdf_metadata: PDFMetadata) -> DownloadResult:
        """
        Validate downloaded PDF content for corruption and integrity.
        
        Args:
            content: Downloaded file content
            pdf_metadata: Original PDF metadata
            
        Returns:
            DownloadResult with validation status
        """
        try:
            # Check if content starts with PDF header
            if not content.startswith(b'%PDF-'):
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.CORRUPTED,
                    error_message="File does not have valid PDF header",
                    validation_passed=False
                )
            
            # Check if content ends with PDF footer (%%EOF or similar)
            content_end = content[-100:].decode('latin-1', errors='ignore')
            if '%%EOF' not in content_end and 'endobj' not in content_end:
                self.logger.warning(f"PDF may be incomplete - no EOF marker found: {pdf_metadata.name}")
            
            # Calculate MD5 checksum
            md5_hash = hashlib.md5(content).hexdigest()
            
            # Compare with expected checksum if available
            if pdf_metadata.md5_checksum and pdf_metadata.md5_checksum != md5_hash:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.CORRUPTED,
                    error_message=f"MD5 checksum mismatch. Expected: {pdf_metadata.md5_checksum}, Got: {md5_hash}",
                    md5_checksum=md5_hash,
                    validation_passed=False
                )
            
            # Check file size matches expected size
            if pdf_metadata.size > 0 and len(content) != pdf_metadata.size:
                size_diff = abs(len(content) - pdf_metadata.size)
                # Allow small differences (metadata might be slightly off)
                if size_diff > 1024:  # More than 1KB difference
                    return DownloadResult(
                        success=False,
                        status=DownloadStatus.CORRUPTED,
                        error_message=f"File size mismatch. Expected: {pdf_metadata.size}, Got: {len(content)}",
                        md5_checksum=md5_hash,
                        validation_passed=False
                    )
            
            # Basic PDF structure validation
            validation_errors = await self._validate_pdf_structure(content)
            if validation_errors:
                return DownloadResult(
                    success=False,
                    status=DownloadStatus.CORRUPTED,
                    error_message=f"PDF structure validation failed: {'; '.join(validation_errors)}",
                    md5_checksum=md5_hash,
                    validation_passed=False
                )
            
            return DownloadResult(
                success=True,
                status=DownloadStatus.SUCCESS,
                md5_checksum=md5_hash,
                validation_passed=True
            )
            
        except Exception as e:
            return DownloadResult(
                success=False,
                status=DownloadStatus.FAILED,
                error_message=f"Validation error: {str(e)}",
                validation_passed=False
            )
    
    async def _validate_pdf_structure(self, content: bytes) -> list[str]:
        """
        Perform basic PDF structure validation.
        
        Args:
            content: PDF file content
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Convert to string for pattern matching (using latin-1 to preserve bytes)
            content_str = content.decode('latin-1', errors='ignore')
            
            # Check for required PDF elements
            if 'xref' not in content_str:
                errors.append("Missing xref table")
            
            if 'trailer' not in content_str:
                errors.append("Missing trailer")
            
            # Check for common PDF objects
            if '/Type' not in content_str:
                errors.append("No PDF objects found")
            
            # Check for catalog object
            if '/Catalog' not in content_str:
                errors.append("Missing document catalog")
            
            # Check for pages
            if '/Pages' not in content_str and '/Page' not in content_str:
                errors.append("No pages found in document")
            
            # Check for excessive null bytes (sign of corruption)
            null_count = content_str.count('\x00')
            if null_count > len(content_str) * 0.1:  # More than 10% null bytes
                errors.append("Excessive null bytes detected (possible corruption)")
            
        except Exception as e:
            errors.append(f"Structure validation error: {str(e)}")
        
        return errors
    
    async def download_multiple_pdfs(self, pdf_list: list[PDFMetadata], 
                                   max_concurrent: Optional[int] = None) -> Dict[str, DownloadResult]:
        """
        Download multiple PDFs concurrently with rate limiting.
        
        Args:
            pdf_list: List of PDF metadata to download
            max_concurrent: Maximum concurrent downloads (uses config if None)
            
        Returns:
            Dictionary mapping file IDs to download results
        """
        if not pdf_list:
            return {}
        
        max_concurrent = max_concurrent or self.settings.max_concurrent_downloads
        
        self.logger.info(f"Starting batch download of {len(pdf_list)} PDFs with max {max_concurrent} concurrent")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(pdf_metadata: PDFMetadata) -> Tuple[str, DownloadResult]:
            async with semaphore:
                result = await self.download_pdf(pdf_metadata)
                return pdf_metadata.file_id, result
        
        # Create download tasks
        tasks = [download_with_semaphore(pdf) for pdf in pdf_list]
        
        # Execute downloads
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        download_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Download task failed with exception: {str(result)}")
                continue
            
            file_id, download_result = result
            download_results[file_id] = download_result
        
        # Log summary
        successful = sum(1 for r in download_results.values() if r.success)
        failed = len(download_results) - successful
        
        self.logger.info(f"Batch download completed: {successful} successful, {failed} failed")
        
        return download_results
    
    async def save_pdf_to_file(self, content: bytes, file_path: Path) -> bool:
        """
        Save PDF content to a file.
        
        Args:
            content: PDF file content
            file_path: Path where to save the file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            self.logger.info(f"PDF saved to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save PDF to {file_path}: {str(e)}")
            return False
    
    async def create_temporary_file(self, content: bytes, suffix: str = ".pdf") -> Optional[Path]:
        """
        Create a temporary file with PDF content.
        
        Args:
            content: PDF file content
            suffix: File suffix (default: .pdf)
            
        Returns:
            Path to temporary file or None if failed
        """
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = Path(temp_file.name)
            
            # Write content
            temp_file.write(content)
            temp_file.close()
            
            self.logger.debug(f"Created temporary PDF file: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary file: {str(e)}")
            return None
    
    def get_download_stats(self, results: Dict[str, DownloadResult]) -> Dict[str, Any]:
        """
        Generate download statistics from results.
        
        Args:
            results: Dictionary of download results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        total_files = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total_files - successful
        
        total_size = sum(r.file_size for r in results.values() if r.success)
        total_time = sum(r.download_time_ms for r in results.values())
        avg_time = total_time / total_files if total_files > 0 else 0
        
        # Count by status
        status_counts = {}
        for result in results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count retries
        total_retries = sum(r.retry_count for r in results.values())
        
        return {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_files if total_files > 0 else 0,
            "total_size_bytes": total_size,
            "total_time_ms": total_time,
            "average_time_ms": avg_time,
            "status_breakdown": status_counts,
            "total_retries": total_retries,
            "average_retries": total_retries / total_files if total_files > 0 else 0
        }
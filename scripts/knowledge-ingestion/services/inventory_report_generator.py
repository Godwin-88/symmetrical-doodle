"""
Knowledge Inventory Report Generation Service

Implements comprehensive inventory report generation with PDF count,
domain distribution, and inaccessible file flagging and reporting.

Requirements: 1.4
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re

# Simplified imports for testing
try:
    from .google_drive_discovery import PDFMetadata, DiscoveryResult, AccessStatus
    from ..core.config import get_settings
    from ..core.logging import get_logger, log_context
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    from enum import Enum
    from datetime import datetime
    from typing import List, Dict, Any, Optional
    import logging
    
    class AccessStatus(Enum):
        ACCESSIBLE = "accessible"
        RESTRICTED = "restricted"
        NOT_FOUND = "not_found"
        ERROR = "error"
    
    @dataclass
    class PDFMetadata:
        file_id: str
        name: str
        mime_type: str
        modified_time: datetime
        size: int
        web_view_link: str
        access_status: AccessStatus
        domain_classification: Optional[str] = None
    
    @dataclass
    class DiscoveryResult:
        success: bool
        pdfs_found: List[PDFMetadata]
        total_files_scanned: int
        inaccessible_files: List[Dict[str, Any]]
        folders_scanned: List[str]
    
    def get_settings():
        return None
    
    def get_logger(name, component=None):
        return logging.getLogger(name)
    
    class log_context:
        def __init__(self, component, operation, correlation_id=None):
            self.correlation_id = correlation_id or "test"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


class DomainCategory(Enum):
    """Technical domain categories"""
    ML = "Machine Learning"
    DRL = "Deep Reinforcement Learning"
    NLP = "Natural Language Processing"
    LLM = "Large Language Models"
    FINANCE = "Finance & Trading"
    GENERAL = "General Technical"
    UNKNOWN = "Unknown"


@dataclass
class DomainStats:
    """Statistics for a specific domain"""
    category: DomainCategory
    count: int
    percentage: float
    file_ids: List[str] = field(default_factory=list)
    sample_files: List[str] = field(default_factory=list)  # Sample filenames
    total_size_mb: float = 0.0
    avg_size_mb: float = 0.0


@dataclass
class AccessibilityStats:
    """File accessibility statistics"""
    accessible_count: int = 0
    restricted_count: int = 0
    not_found_count: int = 0
    error_count: int = 0
    total_count: int = 0
    
    @property
    def accessibility_rate(self) -> float:
        """Calculate accessibility rate as percentage"""
        if self.total_count == 0:
            return 0.0
        return (self.accessible_count / self.total_count) * 100


@dataclass
class InaccessibleFile:
    """Information about an inaccessible file"""
    file_id: str
    name: str
    reason: str
    access_status: AccessStatus
    size: Optional[int] = None
    modified_time: Optional[datetime] = None
    estimated_domain: Optional[str] = None


@dataclass
class FolderSummary:
    """Summary statistics for a folder"""
    folder_id: str
    folder_name: Optional[str]
    pdf_count: int
    total_size_mb: float
    domain_distribution: Dict[str, int]
    accessibility_rate: float
    last_modified: Optional[datetime] = None


@dataclass
class KnowledgeInventoryReport:
    """Comprehensive knowledge inventory report"""
    # Report metadata
    report_id: str
    generated_at: datetime
    correlation_id: Optional[str] = None
    
    # Discovery summary
    total_pdfs_found: int = 0
    total_files_scanned: int = 0
    folders_scanned: List[str] = field(default_factory=list)
    
    # Domain distribution
    domain_stats: List[DomainStats] = field(default_factory=list)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Accessibility analysis
    accessibility_stats: AccessibilityStats = field(default_factory=AccessibilityStats)
    inaccessible_files: List[InaccessibleFile] = field(default_factory=list)
    
    # Folder analysis
    folder_summaries: List[FolderSummary] = field(default_factory=list)
    
    # Size analysis
    total_size_mb: float = 0.0
    avg_file_size_mb: float = 0.0
    largest_files: List[Tuple[str, str, float]] = field(default_factory=list)  # (name, id, size_mb)
    
    # Quality indicators
    files_with_metadata: int = 0
    files_with_domain_classification: int = 0
    metadata_completeness_rate: float = 0.0
    
    # Processing recommendations
    estimated_processing_time_hours: float = 0.0
    recommended_batch_size: int = 0
    potential_issues: List[str] = field(default_factory=list)
    
    # Raw data (optional, for detailed analysis)
    include_raw_data: bool = False
    raw_pdf_metadata: List[PDFMetadata] = field(default_factory=list)


class DomainEstimator:
    """Advanced domain estimation based on filename and metadata analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__, component="domain_estimator")
        
        # Enhanced domain keyword patterns
        self.domain_patterns = {
            DomainCategory.ML: {
                'keywords': [
                    'machine learning', 'ml', 'neural network', 'deep learning', 'ai', 
                    'artificial intelligence', 'supervised learning', 'unsupervised learning',
                    'classification', 'regression', 'clustering', 'feature engineering',
                    'model training', 'cross validation', 'overfitting', 'gradient descent',
                    'random forest', 'svm', 'support vector', 'decision tree', 'ensemble'
                ],
                'patterns': [
                    r'\bml\b', r'\bai\b', r'neural', r'deep.?learning', r'machine.?learning',
                    r'supervised', r'unsupervised', r'classification', r'regression'
                ]
            },
            DomainCategory.DRL: {
                'keywords': [
                    'reinforcement learning', 'rl', 'drl', 'deep reinforcement', 
                    'q-learning', 'policy gradient', 'actor critic', 'markov decision',
                    'mdp', 'bellman', 'temporal difference', 'monte carlo', 'sarsa',
                    'dqn', 'a3c', 'ppo', 'trpo', 'ddpg', 'td3', 'sac', 'rainbow'
                ],
                'patterns': [
                    r'\brl\b', r'\bdrl\b', r'reinforcement', r'q.?learning', r'policy.?gradient',
                    r'actor.?critic', r'markov', r'\bmdp\b', r'bellman', r'temporal.?difference'
                ]
            },
            DomainCategory.NLP: {
                'keywords': [
                    'nlp', 'natural language processing', 'text processing', 'language model',
                    'tokenization', 'parsing', 'sentiment analysis', 'named entity',
                    'pos tagging', 'word embedding', 'word2vec', 'glove', 'fasttext',
                    'text classification', 'text generation', 'machine translation',
                    'information extraction', 'question answering', 'text summarization'
                ],
                'patterns': [
                    r'\bnlp\b', r'natural.?language', r'text.?processing', r'language.?model',
                    r'tokenization', r'sentiment', r'named.?entity', r'word.?embedding'
                ]
            },
            DomainCategory.LLM: {
                'keywords': [
                    'llm', 'large language model', 'transformer', 'attention', 'bert', 'gpt',
                    'chatgpt', 'generative', 'pre-trained', 'fine-tuning', 'prompt engineering',
                    'in-context learning', 'few-shot', 'zero-shot', 'instruction tuning',
                    'rlhf', 'constitutional ai', 'chain of thought', 'reasoning'
                ],
                'patterns': [
                    r'\bllm\b', r'large.?language', r'\bgpt\b', r'\bbert\b', r'transformer',
                    r'attention', r'generative', r'pre.?trained', r'fine.?tuning'
                ]
            },
            DomainCategory.FINANCE: {
                'keywords': [
                    'finance', 'trading', 'market', 'portfolio', 'investment', 'risk management',
                    'quantitative finance', 'algorithmic trading', 'high frequency trading',
                    'derivatives', 'options', 'futures', 'bonds', 'equity', 'forex',
                    'volatility', 'arbitrage', 'hedge fund', 'asset pricing', 'capm',
                    'black scholes', 'monte carlo simulation', 'var', 'value at risk',
                    'backtesting', 'alpha', 'beta', 'sharpe ratio', 'drawdown'
                ],
                'patterns': [
                    r'finance', r'trading', r'market', r'portfolio', r'investment',
                    r'quantitative', r'algorithmic.?trading', r'derivatives', r'options',
                    r'volatility', r'arbitrage', r'black.?scholes', r'sharpe.?ratio'
                ]
            }
        }
    
    def estimate_domain(self, pdf_metadata: PDFMetadata) -> DomainCategory:
        """
        Estimate domain classification based on filename and metadata analysis.
        
        Args:
            pdf_metadata: PDF metadata to analyze
            
        Returns:
            Estimated domain category
        """
        filename = pdf_metadata.name.lower() if pdf_metadata.name else ""
        
        # Score each domain
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in filename:
                    score += 1
            
            # Pattern matching (regex)
            for pattern in patterns['patterns']:
                if re.search(pattern, filename, re.IGNORECASE):
                    score += 2  # Regex patterns get higher weight
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score, or GENERAL if no clear match
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            
            # Only return specific domain if score is significant
            if domain_scores[best_domain] >= 2:
                return best_domain
        
        # Check if it looks technical but doesn't match specific domains
        technical_indicators = [
            'algorithm', 'method', 'analysis', 'research', 'study', 'paper',
            'technical', 'mathematical', 'statistical', 'computational'
        ]
        
        if any(indicator in filename for indicator in technical_indicators):
            return DomainCategory.GENERAL
        
        return DomainCategory.UNKNOWN
    
    def get_domain_confidence(self, pdf_metadata: PDFMetadata, estimated_domain: DomainCategory) -> float:
        """
        Calculate confidence score for domain estimation.
        
        Args:
            pdf_metadata: PDF metadata
            estimated_domain: Estimated domain
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if estimated_domain == DomainCategory.UNKNOWN:
            return 0.0
        
        filename = pdf_metadata.name.lower() if pdf_metadata.name else ""
        
        if estimated_domain in self.domain_patterns:
            patterns = self.domain_patterns[estimated_domain]
            
            # Count matches
            keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in filename)
            pattern_matches = sum(1 for pattern in patterns['patterns'] 
                                if re.search(pattern, filename, re.IGNORECASE))
            
            total_matches = keyword_matches + (pattern_matches * 2)
            max_possible = len(patterns['keywords']) + (len(patterns['patterns']) * 2)
            
            return min(total_matches / max_possible, 1.0)
        
        return 0.5  # Default confidence for GENERAL


class InventoryReportGenerator:
    """
    Knowledge Inventory Report Generator
    
    Creates comprehensive inventory reports with PDF count, domain distribution,
    and inaccessible file flagging and reporting.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="inventory_report_generator")
        self.domain_estimator = DomainEstimator()
    
    async def generate_inventory_report(
        self, 
        discovery_result: DiscoveryResult,
        include_raw_data: bool = False,
        correlation_id: Optional[str] = None
    ) -> KnowledgeInventoryReport:
        """
        Generate comprehensive knowledge inventory report.
        
        Args:
            discovery_result: Result from Google Drive discovery
            include_raw_data: Whether to include raw PDF metadata in report
            correlation_id: Correlation ID for logging
            
        Returns:
            Comprehensive inventory report
        """
        with log_context("inventory_report_generator", "generate_report", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting inventory report generation: {len(discovery_result.pdfs_found)} PDFs found")
            
            # Create report structure
            report = KnowledgeInventoryReport(
                report_id=f"inventory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(timezone.utc),
                correlation_id=ctx.correlation_id,
                include_raw_data=include_raw_data
            )
            
            # Basic statistics
            report.total_pdfs_found = len(discovery_result.pdfs_found)
            report.total_files_scanned = discovery_result.total_files_scanned
            report.folders_scanned = discovery_result.folders_scanned.copy()
            
            if include_raw_data:
                report.raw_pdf_metadata = discovery_result.pdfs_found.copy()
            
            # Analyze domains
            await self._analyze_domains(discovery_result.pdfs_found, report)
            
            # Analyze accessibility
            await self._analyze_accessibility(discovery_result, report)
            
            # Analyze file sizes
            await self._analyze_file_sizes(discovery_result.pdfs_found, report)
            
            # Analyze metadata completeness
            await self._analyze_metadata_completeness(discovery_result.pdfs_found, report)
            
            # Generate processing recommendations
            await self._generate_processing_recommendations(report)
            
            # Analyze folders (if folder information is available)
            await self._analyze_folders(discovery_result, report)
            
            self.logger.info(f"Inventory report generation completed: {report.total_pdfs_found} PDFs, {len(report.domain_stats)} domains, {report.accessibility_stats.accessibility_rate:.1f}% accessible")
            
            return report
    
    async def _analyze_domains(self, pdfs: List[PDFMetadata], report: KnowledgeInventoryReport):
        """Analyze domain distribution of PDFs"""
        domain_counts = {}
        domain_file_mapping = {}
        domain_sizes = {}
        
        for pdf in pdfs:
            # Use existing classification or estimate
            if pdf.domain_classification:
                try:
                    domain = DomainCategory(pdf.domain_classification)
                except ValueError:
                    domain = DomainCategory.UNKNOWN
            else:
                domain = self.domain_estimator.estimate_domain(pdf)
            
            # Update counts
            domain_key = domain.value
            domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
            
            # Track file IDs
            if domain_key not in domain_file_mapping:
                domain_file_mapping[domain_key] = []
            domain_file_mapping[domain_key].append(pdf.file_id)
            
            # Track sizes
            file_size_mb = pdf.size / (1024 * 1024) if pdf.size else 0
            domain_sizes[domain_key] = domain_sizes.get(domain_key, 0) + file_size_mb
        
        # Create domain statistics
        total_files = len(pdfs)
        for domain_name, count in domain_counts.items():
            try:
                domain_category = DomainCategory(domain_name)
            except ValueError:
                domain_category = DomainCategory.UNKNOWN
            
            percentage = (count / total_files * 100) if total_files > 0 else 0
            
            # Get sample filenames
            sample_files = []
            file_ids = domain_file_mapping[domain_name][:5]  # First 5 as samples
            for pdf in pdfs:
                if pdf.file_id in file_ids and len(sample_files) < 5:
                    sample_files.append(pdf.name or "Unknown")
            
            domain_stat = DomainStats(
                category=domain_category,
                count=count,
                percentage=percentage,
                file_ids=domain_file_mapping[domain_name],
                sample_files=sample_files,
                total_size_mb=domain_sizes.get(domain_name, 0),
                avg_size_mb=domain_sizes.get(domain_name, 0) / count if count > 0 else 0
            )
            
            report.domain_stats.append(domain_stat)
            report.domain_distribution[domain_name] = count
        
        # Sort by count (descending)
        report.domain_stats.sort(key=lambda x: x.count, reverse=True)
    
    async def _analyze_accessibility(self, discovery_result: DiscoveryResult, report: KnowledgeInventoryReport):
        """Analyze file accessibility statistics"""
        accessibility_stats = AccessibilityStats()
        
        # Count accessible files
        for pdf in discovery_result.pdfs_found:
            accessibility_stats.total_count += 1
            
            if pdf.access_status == AccessStatus.ACCESSIBLE:
                accessibility_stats.accessible_count += 1
            elif pdf.access_status == AccessStatus.RESTRICTED:
                accessibility_stats.restricted_count += 1
            elif pdf.access_status == AccessStatus.NOT_FOUND:
                accessibility_stats.not_found_count += 1
            else:
                accessibility_stats.error_count += 1
        
        # Add inaccessible files from discovery result
        for inaccessible in discovery_result.inaccessible_files:
            accessibility_stats.total_count += 1
            
            # Determine access status from reason
            reason = inaccessible.get('reason', '').lower()
            if 'not found' in reason or '404' in reason:
                access_status = AccessStatus.NOT_FOUND
                accessibility_stats.not_found_count += 1
            elif 'access denied' in reason or '403' in reason or 'restricted' in reason:
                access_status = AccessStatus.RESTRICTED
                accessibility_stats.restricted_count += 1
            else:
                access_status = AccessStatus.ERROR
                accessibility_stats.error_count += 1
            
            # Create inaccessible file record
            inaccessible_file = InaccessibleFile(
                file_id=inaccessible.get('file_id', ''),
                name=inaccessible.get('name', 'Unknown'),
                reason=inaccessible.get('reason', 'Unknown error'),
                access_status=access_status,
                estimated_domain=self.domain_estimator.estimate_domain(
                    PDFMetadata(
                        file_id=inaccessible.get('file_id', ''),
                        name=inaccessible.get('name', ''),
                        mime_type='application/pdf',
                        modified_time=datetime.now(),
                        size=0,
                        web_view_link='',
                        access_status=access_status
                    )
                ).value
            )
            
            report.inaccessible_files.append(inaccessible_file)
        
        report.accessibility_stats = accessibility_stats
    
    async def _analyze_file_sizes(self, pdfs: List[PDFMetadata], report: KnowledgeInventoryReport):
        """Analyze file size statistics"""
        if not pdfs:
            return
        
        total_size_bytes = sum(pdf.size for pdf in pdfs if pdf.size)
        report.total_size_mb = total_size_bytes / (1024 * 1024)
        report.avg_file_size_mb = report.total_size_mb / len(pdfs) if pdfs else 0
        
        # Find largest files
        sized_files = [(pdf.name or "Unknown", pdf.file_id, pdf.size / (1024 * 1024)) 
                      for pdf in pdfs if pdf.size]
        sized_files.sort(key=lambda x: x[2], reverse=True)
        report.largest_files = sized_files[:10]  # Top 10 largest files
    
    async def _analyze_metadata_completeness(self, pdfs: List[PDFMetadata], report: KnowledgeInventoryReport):
        """Analyze metadata completeness"""
        if not pdfs:
            return
        
        files_with_complete_metadata = 0
        files_with_domain_classification = 0
        
        for pdf in pdfs:
            # Check for complete metadata (basic required fields)
            has_complete_metadata = all([
                pdf.file_id,
                pdf.name,
                pdf.mime_type,
                pdf.modified_time,
                pdf.web_view_link
            ])
            
            if has_complete_metadata:
                files_with_complete_metadata += 1
            
            if pdf.domain_classification:
                files_with_domain_classification += 1
        
        report.files_with_metadata = files_with_complete_metadata
        report.files_with_domain_classification = files_with_domain_classification
        report.metadata_completeness_rate = (files_with_complete_metadata / len(pdfs)) * 100
    
    async def _generate_processing_recommendations(self, report: KnowledgeInventoryReport):
        """Generate processing recommendations based on analysis"""
        # Estimate processing time (rough calculation)
        # Assume 30 seconds per MB for processing (parsing + embedding)
        estimated_time_minutes = report.total_size_mb * 0.5  # 30 seconds per MB
        report.estimated_processing_time_hours = estimated_time_minutes / 60
        
        # Recommend batch size based on total files and size
        if report.total_pdfs_found <= 100:
            report.recommended_batch_size = 10
        elif report.total_pdfs_found <= 500:
            report.recommended_batch_size = 25
        else:
            report.recommended_batch_size = 50
        
        # Identify potential issues
        potential_issues = []
        
        if report.accessibility_stats.accessibility_rate < 90:
            potential_issues.append(
                f"Low accessibility rate ({report.accessibility_stats.accessibility_rate:.1f}%) - "
                f"{report.accessibility_stats.restricted_count + report.accessibility_stats.not_found_count + report.accessibility_stats.error_count} "
                "files may not be processable"
            )
        
        if report.avg_file_size_mb > 50:
            potential_issues.append(
                f"Large average file size ({report.avg_file_size_mb:.1f} MB) may require "
                "increased processing time and memory"
            )
        
        if report.metadata_completeness_rate < 95:
            potential_issues.append(
                f"Incomplete metadata for {100 - report.metadata_completeness_rate:.1f}% of files "
                "may affect processing quality"
            )
        
        # Check domain distribution balance
        if report.domain_stats:
            max_domain_percentage = max(stat.percentage for stat in report.domain_stats)
            if max_domain_percentage > 70:
                potential_issues.append(
                    f"Heavily skewed domain distribution ({max_domain_percentage:.1f}% in one domain) "
                    "may affect embedding model selection"
                )
        
        report.potential_issues = potential_issues
    
    async def _analyze_folders(self, discovery_result: DiscoveryResult, report: KnowledgeInventoryReport):
        """Analyze folder-level statistics"""
        # Group PDFs by parent folders
        folder_stats = {}
        
        for pdf in discovery_result.pdfs_found:
            for parent_folder in pdf.parent_folders:
                if parent_folder not in folder_stats:
                    folder_stats[parent_folder] = {
                        'pdfs': [],
                        'total_size': 0,
                        'domains': {},
                        'accessible_count': 0
                    }
                
                folder_stats[parent_folder]['pdfs'].append(pdf)
                folder_stats[parent_folder]['total_size'] += pdf.size or 0
                
                # Track domain
                domain = pdf.domain_classification or 'Unknown'
                folder_stats[parent_folder]['domains'][domain] = \
                    folder_stats[parent_folder]['domains'].get(domain, 0) + 1
                
                # Track accessibility
                if pdf.access_status == AccessStatus.ACCESSIBLE:
                    folder_stats[parent_folder]['accessible_count'] += 1
        
        # Create folder summaries
        for folder_id, stats in folder_stats.items():
            pdf_count = len(stats['pdfs'])
            accessibility_rate = (stats['accessible_count'] / pdf_count * 100) if pdf_count > 0 else 0
            
            # Find most recent modification
            last_modified = None
            if stats['pdfs']:
                last_modified = max(pdf.modified_time for pdf in stats['pdfs'] if pdf.modified_time)
            
            folder_summary = FolderSummary(
                folder_id=folder_id,
                folder_name=None,  # Would need additional API call to get folder name
                pdf_count=pdf_count,
                total_size_mb=stats['total_size'] / (1024 * 1024),
                domain_distribution=stats['domains'],
                accessibility_rate=accessibility_rate,
                last_modified=last_modified
            )
            
            report.folder_summaries.append(folder_summary)
        
        # Sort by PDF count (descending)
        report.folder_summaries.sort(key=lambda x: x.pdf_count, reverse=True)
    
    async def save_report_to_file(self, report: KnowledgeInventoryReport, output_path: Path) -> bool:
        """
        Save inventory report to JSON file.
        
        Args:
            report: Inventory report to save
            output_path: Path to save the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert report to dictionary for JSON serialization
            report_dict = self._report_to_dict(report)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Inventory report saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error("Failed to save inventory report", 
                            output_path=str(output_path), error=str(e))
            return False
    
    def _report_to_dict(self, report: KnowledgeInventoryReport) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization"""
        return {
            'report_metadata': {
                'report_id': report.report_id,
                'generated_at': report.generated_at.isoformat(),
                'correlation_id': report.correlation_id
            },
            'discovery_summary': {
                'total_pdfs_found': report.total_pdfs_found,
                'total_files_scanned': report.total_files_scanned,
                'folders_scanned': report.folders_scanned
            },
            'domain_analysis': {
                'domain_distribution': report.domain_distribution,
                'domain_stats': [
                    {
                        'category': stat.category.value,
                        'count': stat.count,
                        'percentage': round(stat.percentage, 2),
                        'total_size_mb': round(stat.total_size_mb, 2),
                        'avg_size_mb': round(stat.avg_size_mb, 2),
                        'sample_files': stat.sample_files[:3]  # Limit samples in output
                    }
                    for stat in report.domain_stats
                ]
            },
            'accessibility_analysis': {
                'accessibility_rate': round(report.accessibility_stats.accessibility_rate, 2),
                'accessible_count': report.accessibility_stats.accessible_count,
                'restricted_count': report.accessibility_stats.restricted_count,
                'not_found_count': report.accessibility_stats.not_found_count,
                'error_count': report.accessibility_stats.error_count,
                'total_count': report.accessibility_stats.total_count,
                'inaccessible_files': [
                    {
                        'file_id': f.file_id,
                        'name': f.name,
                        'reason': f.reason,
                        'access_status': f.access_status.value,
                        'estimated_domain': f.estimated_domain
                    }
                    for f in report.inaccessible_files
                ]
            },
            'size_analysis': {
                'total_size_mb': round(report.total_size_mb, 2),
                'avg_file_size_mb': round(report.avg_file_size_mb, 2),
                'largest_files': [
                    {
                        'name': name,
                        'file_id': file_id,
                        'size_mb': round(size_mb, 2)
                    }
                    for name, file_id, size_mb in report.largest_files
                ]
            },
            'metadata_analysis': {
                'files_with_metadata': report.files_with_metadata,
                'files_with_domain_classification': report.files_with_domain_classification,
                'metadata_completeness_rate': round(report.metadata_completeness_rate, 2)
            },
            'processing_recommendations': {
                'estimated_processing_time_hours': round(report.estimated_processing_time_hours, 2),
                'recommended_batch_size': report.recommended_batch_size,
                'potential_issues': report.potential_issues
            },
            'folder_analysis': [
                {
                    'folder_id': summary.folder_id,
                    'folder_name': summary.folder_name,
                    'pdf_count': summary.pdf_count,
                    'total_size_mb': round(summary.total_size_mb, 2),
                    'domain_distribution': summary.domain_distribution,
                    'accessibility_rate': round(summary.accessibility_rate, 2),
                    'last_modified': summary.last_modified.isoformat() if summary.last_modified else None
                }
                for summary in report.folder_summaries
            ]
        }
    
    async def generate_summary_report(self, report: KnowledgeInventoryReport) -> str:
        """
        Generate a human-readable summary of the inventory report.
        
        Args:
            report: Inventory report
            
        Returns:
            Formatted summary string
        """
        summary_lines = [
            "=" * 60,
            "KNOWLEDGE BASE INVENTORY REPORT",
            "=" * 60,
            f"Report ID: {report.report_id}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "DISCOVERY SUMMARY",
            "-" * 20,
            f"Total PDFs Found: {report.total_pdfs_found:,}",
            f"Total Files Scanned: {report.total_files_scanned:,}",
            f"Folders Scanned: {len(report.folders_scanned)}",
            f"Total Size: {report.total_size_mb:.1f} MB",
            f"Average File Size: {report.avg_file_size_mb:.1f} MB",
            "",
            "ACCESSIBILITY ANALYSIS",
            "-" * 20,
            f"Accessibility Rate: {report.accessibility_stats.accessibility_rate:.1f}%",
            f"Accessible Files: {report.accessibility_stats.accessible_count:,}",
            f"Restricted Files: {report.accessibility_stats.restricted_count:,}",
            f"Not Found Files: {report.accessibility_stats.not_found_count:,}",
            f"Error Files: {report.accessibility_stats.error_count:,}",
            "",
            "DOMAIN DISTRIBUTION",
            "-" * 20
        ]
        
        for stat in report.domain_stats:
            summary_lines.append(
                f"{stat.category.value}: {stat.count:,} files ({stat.percentage:.1f}%)"
            )
        
        if report.potential_issues:
            summary_lines.extend([
                "",
                "POTENTIAL ISSUES",
                "-" * 20
            ])
            for issue in report.potential_issues:
                summary_lines.append(f"• {issue}")
        
        summary_lines.extend([
            "",
            "PROCESSING RECOMMENDATIONS",
            "-" * 20,
            f"Estimated Processing Time: {report.estimated_processing_time_hours:.1f} hours",
            f"Recommended Batch Size: {report.recommended_batch_size}",
            "",
            "=" * 60
        ])
        
        return "\n".join(summary_lines)
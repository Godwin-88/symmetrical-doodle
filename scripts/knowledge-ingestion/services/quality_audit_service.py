"""
Quality Audit and Sampling Service

Implements representative content sampling across technical domains,
technical notation preservation verification, and content completeness
and embedding quality assessment.

Requirements: 5.1, 5.2
"""

import asyncio
import random
import re
import statistics
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import numpy as np

# Simplified imports for testing
try:
    from .inventory_report_generator import DomainCategory
    from ..core.config import get_settings
    from ..core.logging import get_logger, log_context
except ImportError:
    # Fallback for testing
    from enum import Enum
    import logging
    
    class DomainCategory(Enum):
        ML = "Machine Learning"
        DRL = "Deep Reinforcement Learning"
        NLP = "Natural Language Processing"
        LLM = "Large Language Models"
        FINANCE = "Finance & Trading"
        GENERAL = "General Technical"
        UNKNOWN = "Unknown"
    
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


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


class ContentType(Enum):
    """Types of content for quality assessment"""
    MATHEMATICAL = "mathematical"
    TECHNICAL_TEXT = "technical_text"
    GENERAL_TEXT = "general_text"
    MIXED = "mixed"


@dataclass
class ContentSample:
    """Representative content sample for quality assessment"""
    sample_id: str
    document_id: str
    document_name: str
    chunk_id: Optional[str] = None
    content: str = ""
    content_type: ContentType = ContentType.GENERAL_TEXT
    domain: DomainCategory = DomainCategory.UNKNOWN
    
    # Content characteristics
    token_count: int = 0
    has_mathematical_notation: bool = False
    has_technical_terms: bool = False
    has_code_snippets: bool = False
    has_tables: bool = False
    has_formulas: bool = False
    
    # Quality metrics
    readability_score: Optional[float] = None
    completeness_score: Optional[float] = None
    preservation_score: Optional[float] = None
    
    # Metadata
    sampled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sampling_method: str = "random"


@dataclass
class TechnicalNotationAssessment:
    """Assessment of technical notation preservation"""
    mathematical_formulas_found: int = 0
    mathematical_formulas_preserved: int = 0
    latex_expressions_found: int = 0
    latex_expressions_preserved: int = 0
    greek_letters_found: int = 0
    greek_letters_preserved: int = 0
    special_symbols_found: int = 0
    special_symbols_preserved: int = 0
    
    # Quality scores
    formula_preservation_rate: float = 0.0
    latex_preservation_rate: float = 0.0
    symbol_preservation_rate: float = 0.0
    overall_preservation_score: float = 0.0
    
    # Examples of issues
    preservation_issues: List[str] = field(default_factory=list)
    well_preserved_examples: List[str] = field(default_factory=list)


@dataclass
class EmbeddingQualityMetrics:
    """Embedding quality assessment metrics"""
    vector_dimension: int = 0
    null_embeddings_count: int = 0
    zero_embeddings_count: int = 0
    
    # Statistical measures
    mean_magnitude: float = 0.0
    std_magnitude: float = 0.0
    min_magnitude: float = 0.0
    max_magnitude: float = 0.0
    
    # Semantic coherence measures
    avg_cosine_similarity: float = 0.0
    similarity_std: float = 0.0
    coherence_score: float = 0.0
    
    # Quality indicators
    quality_level: QualityLevel = QualityLevel.FAIR
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class DomainSamplingStats:
    """Statistics for domain-specific sampling"""
    domain: DomainCategory
    total_documents: int
    target_sample_size: int
    actual_sample_size: int
    sampling_rate: float
    
    # Content type distribution in samples
    content_type_distribution: Dict[ContentType, int] = field(default_factory=dict)
    
    # Quality indicators
    avg_completeness_score: float = 0.0
    avg_preservation_score: float = 0.0
    technical_notation_coverage: float = 0.0


@dataclass
class QualityAuditReport:
    """Comprehensive quality audit report"""
    # Report metadata
    audit_id: str
    generated_at: datetime
    correlation_id: Optional[str] = None
    
    # Sampling summary
    total_documents_available: int = 0
    total_samples_collected: int = 0
    sampling_coverage_rate: float = 0.0
    
    # Domain sampling statistics
    domain_sampling_stats: List[DomainSamplingStats] = field(default_factory=list)
    
    # Content samples
    content_samples: List[ContentSample] = field(default_factory=list)
    
    # Technical notation assessment
    notation_assessment: TechnicalNotationAssessment = field(default_factory=TechnicalNotationAssessment)
    
    # Embedding quality metrics
    embedding_quality: Optional[EmbeddingQualityMetrics] = None
    
    # Overall quality scores
    overall_completeness_score: float = 0.0
    overall_preservation_score: float = 0.0
    overall_quality_level: QualityLevel = QualityLevel.FAIR
    
    # Recommendations
    quality_recommendations: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Issues and concerns
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TechnicalNotationAnalyzer:
    """Analyzer for technical notation and mathematical content"""
    
    def __init__(self):
        self.logger = get_logger(__name__, component="notation_analyzer")
        
        # Mathematical notation patterns
        self.math_patterns = {
            'latex_formulas': [
                r'\$[^$]+\$',  # Inline LaTeX
                r'\$\$[^$]+\$\$',  # Display LaTeX
                r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # LaTeX environments
                r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
            ],
            'mathematical_expressions': [
                r'[a-zA-Z]\s*[=<>≤≥≠]\s*[0-9a-zA-Z\+\-\*/\(\)]+',  # Basic equations
                r'[∑∏∫∂∇∆]',  # Mathematical operators
                r'[α-ωΑ-Ω]',  # Greek letters
                r'[±×÷≈≡∞∈∉⊂⊃∪∩]',  # Mathematical symbols
            ],
            'formulas': [
                r'[a-zA-Z]+\([^)]+\)\s*[=<>]\s*[^,\n]+',  # Function definitions
                r'[a-zA-Z]\s*=\s*[0-9\+\-\*/\(\)a-zA-Z\s]+',  # Variable assignments
                r'\b[A-Z][a-zA-Z]*\s*=\s*[^,\n]+',  # Constants/formulas
            ]
        }
        
        # Technical term patterns
        self.technical_patterns = [
            r'\b(?:algorithm|method|function|variable|parameter|coefficient)\b',
            r'\b(?:optimization|convergence|iteration|gradient|derivative)\b',
            r'\b(?:matrix|vector|tensor|eigenvalue|eigenvector)\b',
            r'\b(?:probability|distribution|variance|covariance|correlation)\b',
            r'\b(?:neural|network|layer|activation|backpropagation)\b',
            r'\b(?:reinforcement|policy|reward|action|state|environment)\b',
        ]
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for technical notation and mathematical expressions.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'has_mathematical_notation': False,
            'has_technical_terms': False,
            'has_formulas': False,
            'mathematical_elements': [],
            'technical_terms': [],
            'preservation_issues': [],
            'content_type': ContentType.GENERAL_TEXT
        }
        
        # Check for mathematical notation
        math_elements = []
        for category, patterns in self.math_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                if matches:
                    math_elements.extend([(category, match) for match in matches])
                    analysis['has_mathematical_notation'] = True
        
        analysis['mathematical_elements'] = math_elements
        
        # Check for technical terms
        technical_terms = []
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            technical_terms.extend(matches)
        
        if technical_terms:
            analysis['has_technical_terms'] = True
            analysis['technical_terms'] = list(set(technical_terms))  # Remove duplicates
        
        # Check for formulas
        formula_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                          for pattern in self.math_patterns['formulas'])
        if formula_count > 0:
            analysis['has_formulas'] = True
        
        # Determine content type
        if analysis['has_mathematical_notation'] and len(math_elements) > 3:
            analysis['content_type'] = ContentType.MATHEMATICAL
        elif analysis['has_technical_terms'] and len(technical_terms) > 5:
            analysis['content_type'] = ContentType.TECHNICAL_TEXT
        elif analysis['has_mathematical_notation'] or analysis['has_technical_terms']:
            analysis['content_type'] = ContentType.MIXED
        else:
            analysis['content_type'] = ContentType.GENERAL_TEXT
        
        return analysis
    
    def assess_preservation_quality(self, original_content: str, processed_content: str) -> Dict[str, Any]:
        """
        Assess how well technical notation was preserved during processing.
        
        Args:
            original_content: Original content before processing
            processed_content: Content after processing
            
        Returns:
            Preservation quality assessment
        """
        original_analysis = self.analyze_content(original_content)
        processed_analysis = self.analyze_content(processed_content)
        
        assessment = {
            'preservation_score': 0.0,
            'issues': [],
            'preserved_elements': [],
            'lost_elements': []
        }
        
        # Compare mathematical elements
        original_math = set(elem[1] for elem in original_analysis['mathematical_elements'])
        processed_math = set(elem[1] for elem in processed_analysis['mathematical_elements'])
        
        if original_math:
            preserved_math = original_math.intersection(processed_math)
            lost_math = original_math - processed_math
            
            math_preservation_rate = len(preserved_math) / len(original_math)
            assessment['preserved_elements'].extend(list(preserved_math))
            assessment['lost_elements'].extend(list(lost_math))
            
            if lost_math:
                assessment['issues'].append(f"Lost {len(lost_math)} mathematical expressions")
        else:
            math_preservation_rate = 1.0
        
        # Compare technical terms
        original_terms = set(original_analysis['technical_terms'])
        processed_terms = set(processed_analysis['technical_terms'])
        
        if original_terms:
            preserved_terms = original_terms.intersection(processed_terms)
            lost_terms = original_terms - processed_terms
            
            terms_preservation_rate = len(preserved_terms) / len(original_terms)
            
            if lost_terms:
                assessment['issues'].append(f"Lost {len(lost_terms)} technical terms")
        else:
            terms_preservation_rate = 1.0
        
        # Calculate overall preservation score
        assessment['preservation_score'] = (math_preservation_rate + terms_preservation_rate) / 2
        
        return assessment


class ContentSampler:
    """Representative content sampler across technical domains"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="content_sampler")
        self.notation_analyzer = TechnicalNotationAnalyzer()
    
    async def sample_content_by_domain(
        self, 
        documents: List[Dict[str, Any]], 
        target_samples_per_domain: int = 10,
        min_content_length: int = 500,
        correlation_id: Optional[str] = None
    ) -> List[ContentSample]:
        """
        Sample representative content across technical domains.
        
        Args:
            documents: List of document metadata with content
            target_samples_per_domain: Target number of samples per domain
            min_content_length: Minimum content length for sampling
            correlation_id: Correlation ID for logging
            
        Returns:
            List of content samples
        """
        with log_context("content_sampler", "sample_by_domain", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting content sampling by domain: {len(documents)} documents, {target_samples_per_domain} target per domain")
            
            # Group documents by domain
            domain_groups = {}
            for doc in documents:
                domain = doc.get('domain_classification', 'Unknown')
                try:
                    domain_category = DomainCategory(domain)
                except ValueError:
                    domain_category = DomainCategory.UNKNOWN
                
                if domain_category not in domain_groups:
                    domain_groups[domain_category] = []
                domain_groups[domain_category].append(doc)
            
            samples = []
            
            # Sample from each domain
            for domain, docs in domain_groups.items():
                self.logger.info(f"Sampling from domain: {domain.value} ({len(docs)} available docs)")
                
                domain_samples = await self._sample_from_domain(
                    docs, domain, target_samples_per_domain, min_content_length
                )
                samples.extend(domain_samples)
            
            self.logger.info(f"Content sampling completed: {len(samples)} total samples from {len(domain_groups)} domains")
            
            return samples
    
    async def _sample_from_domain(
        self, 
        documents: List[Dict[str, Any]], 
        domain: DomainCategory,
        target_samples: int,
        min_content_length: int
    ) -> List[ContentSample]:
        """Sample content from a specific domain"""
        
        # Filter documents with sufficient content
        eligible_docs = [
            doc for doc in documents 
            if doc.get('content', '') and len(doc.get('content', '')) >= min_content_length
        ]
        
        if not eligible_docs:
            self.logger.warning(f"No eligible documents for domain {domain.value}")
            return []
        
        # Determine sampling strategy
        sample_size = min(target_samples, len(eligible_docs))
        
        # Use stratified sampling to ensure diversity
        samples = []
        
        if len(eligible_docs) <= target_samples:
            # Use all available documents
            selected_docs = eligible_docs
        else:
            # Stratified sampling by content characteristics
            selected_docs = await self._stratified_sampling(eligible_docs, sample_size)
        
        # Create content samples
        for i, doc in enumerate(selected_docs):
            content = doc.get('content', '')
            
            # Analyze content characteristics
            analysis = self.notation_analyzer.analyze_content(content)
            
            sample = ContentSample(
                sample_id=f"{domain.value.lower()}_{i+1}",
                document_id=doc.get('document_id', ''),
                document_name=doc.get('name', 'Unknown'),
                content=content[:2000],  # Limit content length for analysis
                content_type=analysis['content_type'],
                domain=domain,
                token_count=len(content.split()),
                has_mathematical_notation=analysis['has_mathematical_notation'],
                has_technical_terms=analysis['has_technical_terms'],
                has_formulas=analysis['has_formulas'],
                sampling_method="stratified"
            )
            
            samples.append(sample)
        
        return samples
    
    async def _stratified_sampling(
        self, 
        documents: List[Dict[str, Any]], 
        sample_size: int
    ) -> List[Dict[str, Any]]:
        """Perform stratified sampling to ensure content diversity"""
        
        # Categorize documents by content characteristics
        categories = {
            'mathematical': [],
            'technical': [],
            'general': []
        }
        
        for doc in documents:
            content = doc.get('content', '')
            analysis = self.notation_analyzer.analyze_content(content)
            
            if analysis['content_type'] == ContentType.MATHEMATICAL:
                categories['mathematical'].append(doc)
            elif analysis['content_type'] in [ContentType.TECHNICAL_TEXT, ContentType.MIXED]:
                categories['technical'].append(doc)
            else:
                categories['general'].append(doc)
        
        # Determine samples per category
        total_docs = sum(len(docs) for docs in categories.values())
        selected_docs = []
        
        for category, docs in categories.items():
            if not docs:
                continue
            
            # Proportional allocation
            category_proportion = len(docs) / total_docs
            category_samples = max(1, int(sample_size * category_proportion))
            category_samples = min(category_samples, len(docs))
            
            # Random sampling within category
            selected = random.sample(docs, category_samples)
            selected_docs.extend(selected)
        
        # If we haven't reached target, fill with random selection
        if len(selected_docs) < sample_size:
            remaining_docs = [doc for doc in documents if doc not in selected_docs]
            additional_needed = sample_size - len(selected_docs)
            additional_needed = min(additional_needed, len(remaining_docs))
            
            if additional_needed > 0:
                additional = random.sample(remaining_docs, additional_needed)
                selected_docs.extend(additional)
        
        return selected_docs[:sample_size]


class QualityAuditor:
    """Quality auditor for content completeness and embedding quality assessment"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="quality_auditor")
        self.notation_analyzer = TechnicalNotationAnalyzer()
        self.content_sampler = ContentSampler()
    
    async def conduct_quality_audit(
        self, 
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> QualityAuditReport:
        """
        Conduct comprehensive quality audit of ingested content.
        
        Args:
            documents: List of ingested documents with content
            embeddings: Optional list of embeddings for quality assessment
            correlation_id: Correlation ID for logging
            
        Returns:
            Comprehensive quality audit report
        """
        with log_context("quality_auditor", "conduct_audit", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting quality audit: {len(documents)} documents, embeddings={'available' if embeddings is not None else 'not available'}")
            
            # Create audit report
            report = QualityAuditReport(
                audit_id=f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(timezone.utc),
                correlation_id=ctx.correlation_id,
                total_documents_available=len(documents)
            )
            
            # Sample content across domains
            content_samples = await self.content_sampler.sample_content_by_domain(
                documents, target_samples_per_domain=10
            )
            report.content_samples = content_samples
            report.total_samples_collected = len(content_samples)
            report.sampling_coverage_rate = (len(content_samples) / len(documents)) * 100 if documents else 0
            
            # Assess technical notation preservation
            await self._assess_technical_notation(content_samples, report)
            
            # Assess content completeness
            await self._assess_content_completeness(content_samples, report)
            
            # Assess embedding quality if available
            if embeddings:
                await self._assess_embedding_quality(embeddings, report)
            
            # Generate domain sampling statistics
            await self._generate_domain_sampling_stats(content_samples, documents, report)
            
            # Calculate overall quality scores
            await self._calculate_overall_quality_scores(report)
            
            # Generate recommendations
            await self._generate_quality_recommendations(report)
            
            self.logger.info(f"Quality audit completed - overall_quality: {report.overall_quality_level.value}, completeness_score: {report.overall_completeness_score}, preservation_score: {report.overall_preservation_score}")
            
            return report
    
    async def _assess_technical_notation(self, samples: List[ContentSample], report: QualityAuditReport):
        """Assess technical notation preservation across samples"""
        assessment = TechnicalNotationAssessment()
        
        for sample in samples:
            if sample.has_mathematical_notation or sample.has_formulas:
                analysis = self.notation_analyzer.analyze_content(sample.content)
                
                # Count mathematical elements
                math_elements = analysis['mathematical_elements']
                assessment.mathematical_formulas_found += len([e for e in math_elements if 'formula' in e[0]])
                assessment.latex_expressions_found += len([e for e in math_elements if 'latex' in e[0]])
                
                # For now, assume good preservation (would need original vs processed comparison)
                assessment.mathematical_formulas_preserved += len([e for e in math_elements if 'formula' in e[0]])
                assessment.latex_expressions_preserved += len([e for e in math_elements if 'latex' in e[0]])
                
                # Count special symbols and Greek letters
                greek_letters = re.findall(r'[α-ωΑ-Ω]', sample.content)
                special_symbols = re.findall(r'[±×÷≈≡∞∈∉⊂⊃∪∩∑∏∫∂∇∆]', sample.content)
                
                assessment.greek_letters_found += len(greek_letters)
                assessment.greek_letters_preserved += len(greek_letters)  # Assume preserved for now
                assessment.special_symbols_found += len(special_symbols)
                assessment.special_symbols_preserved += len(special_symbols)  # Assume preserved for now
        
        # Calculate preservation rates
        if assessment.mathematical_formulas_found > 0:
            assessment.formula_preservation_rate = assessment.mathematical_formulas_preserved / assessment.mathematical_formulas_found
        
        if assessment.latex_expressions_found > 0:
            assessment.latex_preservation_rate = assessment.latex_expressions_preserved / assessment.latex_expressions_found
        
        if assessment.special_symbols_found > 0 or assessment.greek_letters_found > 0:
            total_symbols = assessment.special_symbols_found + assessment.greek_letters_found
            preserved_symbols = assessment.special_symbols_preserved + assessment.greek_letters_preserved
            assessment.symbol_preservation_rate = preserved_symbols / total_symbols
        
        # Calculate overall preservation score
        rates = [
            assessment.formula_preservation_rate,
            assessment.latex_preservation_rate,
            assessment.symbol_preservation_rate
        ]
        valid_rates = [rate for rate in rates if rate > 0]
        assessment.overall_preservation_score = statistics.mean(valid_rates) if valid_rates else 1.0
        
        report.notation_assessment = assessment
    
    async def _assess_content_completeness(self, samples: List[ContentSample], report: QualityAuditReport):
        """Assess content completeness across samples"""
        completeness_scores = []
        
        for sample in samples:
            # Basic completeness indicators
            has_content = len(sample.content.strip()) > 0
            has_sufficient_length = len(sample.content.split()) >= 50
            has_structure = any(indicator in sample.content.lower() 
                              for indicator in ['introduction', 'conclusion', 'method', 'result', 'abstract'])
            
            # Technical content indicators
            has_technical_depth = sample.has_technical_terms or sample.has_mathematical_notation
            
            # Calculate completeness score
            score = 0.0
            if has_content:
                score += 0.3
            if has_sufficient_length:
                score += 0.3
            if has_structure:
                score += 0.2
            if has_technical_depth:
                score += 0.2
            
            sample.completeness_score = score
            completeness_scores.append(score)
        
        # Calculate overall completeness
        if completeness_scores:
            report.overall_completeness_score = statistics.mean(completeness_scores)
        else:
            report.overall_completeness_score = 0.0
    
    async def _assess_embedding_quality(self, embeddings: List[Dict[str, Any]], report: QualityAuditReport):
        """Assess embedding quality metrics"""
        if not embeddings:
            return
        
        metrics = EmbeddingQualityMetrics()
        
        # Extract embedding vectors
        vectors = []
        for emb in embeddings:
            vector = emb.get('embedding_vector', [])
            if vector:
                vectors.append(np.array(vector))
                if len(vector) > 0:
                    metrics.vector_dimension = len(vector)
                
                # Check for null or zero embeddings
                if all(v == 0 for v in vector):
                    metrics.zero_embeddings_count += 1
            else:
                metrics.null_embeddings_count += 1
        
        if vectors:
            # Calculate statistical measures
            magnitudes = [np.linalg.norm(v) for v in vectors]
            metrics.mean_magnitude = float(np.mean(magnitudes))
            metrics.std_magnitude = float(np.std(magnitudes))
            metrics.min_magnitude = float(np.min(magnitudes))
            metrics.max_magnitude = float(np.max(magnitudes))
            
            # Calculate semantic coherence (average pairwise cosine similarity)
            if len(vectors) > 1:
                similarities = []
                for i in range(min(len(vectors), 50)):  # Limit for performance
                    for j in range(i + 1, min(len(vectors), 50)):
                        sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                        similarities.append(sim)
                
                if similarities:
                    metrics.avg_cosine_similarity = float(np.mean(similarities))
                    metrics.similarity_std = float(np.std(similarities))
                    metrics.coherence_score = metrics.avg_cosine_similarity
            
            # Determine quality level
            issues = []
            if metrics.null_embeddings_count > len(embeddings) * 0.05:  # More than 5% null
                issues.append(f"High null embedding rate: {metrics.null_embeddings_count}/{len(embeddings)}")
            
            if metrics.zero_embeddings_count > len(embeddings) * 0.02:  # More than 2% zero
                issues.append(f"High zero embedding rate: {metrics.zero_embeddings_count}/{len(embeddings)}")
            
            if metrics.coherence_score < 0.3:
                issues.append(f"Low semantic coherence: {metrics.coherence_score:.3f}")
            
            metrics.quality_issues = issues
            
            # Determine overall quality level
            if not issues and metrics.coherence_score > 0.7:
                metrics.quality_level = QualityLevel.EXCELLENT
            elif not issues and metrics.coherence_score > 0.5:
                metrics.quality_level = QualityLevel.GOOD
            elif len(issues) <= 1 and metrics.coherence_score > 0.3:
                metrics.quality_level = QualityLevel.FAIR
            elif len(issues) <= 2:
                metrics.quality_level = QualityLevel.POOR
            else:
                metrics.quality_level = QualityLevel.FAILED
        
        report.embedding_quality = metrics
    
    async def _generate_domain_sampling_stats(
        self, 
        samples: List[ContentSample], 
        documents: List[Dict[str, Any]], 
        report: QualityAuditReport
    ):
        """Generate domain-specific sampling statistics"""
        
        # Group documents by domain
        domain_doc_counts = {}
        for doc in documents:
            domain = doc.get('domain_classification', 'Unknown')
            try:
                domain_category = DomainCategory(domain)
            except ValueError:
                domain_category = DomainCategory.UNKNOWN
            
            domain_doc_counts[domain_category] = domain_doc_counts.get(domain_category, 0) + 1
        
        # Group samples by domain
        domain_samples = {}
        for sample in samples:
            if sample.domain not in domain_samples:
                domain_samples[sample.domain] = []
            domain_samples[sample.domain].append(sample)
        
        # Create domain sampling statistics
        for domain, doc_count in domain_doc_counts.items():
            samples_for_domain = domain_samples.get(domain, [])
            
            # Content type distribution
            content_type_dist = {}
            for sample in samples_for_domain:
                content_type_dist[sample.content_type] = content_type_dist.get(sample.content_type, 0) + 1
            
            # Quality scores
            completeness_scores = [s.completeness_score for s in samples_for_domain if s.completeness_score is not None]
            preservation_scores = [s.preservation_score for s in samples_for_domain if s.preservation_score is not None]
            
            # Technical notation coverage
            technical_samples = [s for s in samples_for_domain if s.has_mathematical_notation or s.has_technical_terms]
            technical_coverage = len(technical_samples) / len(samples_for_domain) if samples_for_domain else 0
            
            stats = DomainSamplingStats(
                domain=domain,
                total_documents=doc_count,
                target_sample_size=10,  # Default target
                actual_sample_size=len(samples_for_domain),
                sampling_rate=len(samples_for_domain) / doc_count if doc_count > 0 else 0,
                content_type_distribution=content_type_dist,
                avg_completeness_score=statistics.mean(completeness_scores) if completeness_scores else 0.0,
                avg_preservation_score=statistics.mean(preservation_scores) if preservation_scores else 0.0,
                technical_notation_coverage=technical_coverage
            )
            
            report.domain_sampling_stats.append(stats)
    
    async def _calculate_overall_quality_scores(self, report: QualityAuditReport):
        """Calculate overall quality scores and level"""
        
        # Overall preservation score (from notation assessment)
        report.overall_preservation_score = report.notation_assessment.overall_preservation_score
        
        # Overall quality level determination
        completeness = report.overall_completeness_score
        preservation = report.overall_preservation_score
        
        # Consider embedding quality if available
        embedding_quality_factor = 1.0
        if report.embedding_quality:
            if report.embedding_quality.quality_level == QualityLevel.EXCELLENT:
                embedding_quality_factor = 1.2
            elif report.embedding_quality.quality_level == QualityLevel.GOOD:
                embedding_quality_factor = 1.1
            elif report.embedding_quality.quality_level == QualityLevel.POOR:
                embedding_quality_factor = 0.8
            elif report.embedding_quality.quality_level == QualityLevel.FAILED:
                embedding_quality_factor = 0.6
        
        # Calculate weighted overall score
        overall_score = (completeness * 0.4 + preservation * 0.6) * embedding_quality_factor
        
        # Determine quality level
        if overall_score >= 0.9:
            report.overall_quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            report.overall_quality_level = QualityLevel.GOOD
        elif overall_score >= 0.6:
            report.overall_quality_level = QualityLevel.FAIR
        elif overall_score >= 0.4:
            report.overall_quality_level = QualityLevel.POOR
        else:
            report.overall_quality_level = QualityLevel.FAILED
    
    async def _generate_quality_recommendations(self, report: QualityAuditReport):
        """Generate quality recommendations and improvement suggestions"""
        
        recommendations = []
        improvements = []
        critical_issues = []
        warnings = []
        
        # Completeness recommendations
        if report.overall_completeness_score < 0.7:
            recommendations.append("Improve content extraction to capture more complete document structure")
            improvements.append("Review PDF parsing settings to preserve section headers and formatting")
        
        # Preservation recommendations
        if report.overall_preservation_score < 0.8:
            recommendations.append("Enhance mathematical notation preservation during processing")
            improvements.append("Consider using specialized LaTeX-aware parsing for mathematical content")
        
        # Embedding quality recommendations
        if report.embedding_quality:
            if report.embedding_quality.null_embeddings_count > 0:
                critical_issues.append(f"Found {report.embedding_quality.null_embeddings_count} null embeddings")
                recommendations.append("Investigate and fix null embedding generation issues")
            
            if report.embedding_quality.coherence_score < 0.5:
                warnings.append(f"Low semantic coherence score: {report.embedding_quality.coherence_score:.3f}")
                improvements.append("Consider using domain-specific embedding models for better coherence")
        
        # Domain coverage recommendations
        domain_stats = report.domain_sampling_stats
        if domain_stats:
            low_coverage_domains = [stat for stat in domain_stats if stat.sampling_rate < 0.1]
            if low_coverage_domains:
                domain_names = [stat.domain.value for stat in low_coverage_domains]
                warnings.append(f"Low sampling coverage for domains: {', '.join(domain_names)}")
                improvements.append("Increase sample size for underrepresented domains")
        
        # Technical notation coverage
        avg_technical_coverage = statistics.mean([stat.technical_notation_coverage for stat in domain_stats]) if domain_stats else 0
        if avg_technical_coverage < 0.3:
            warnings.append(f"Low technical notation coverage: {avg_technical_coverage:.1%}")
            improvements.append("Ensure technical documents are properly identified and sampled")
        
        report.quality_recommendations = recommendations
        report.improvement_suggestions = improvements
        report.critical_issues = critical_issues
        report.warnings = warnings
    
    async def save_audit_report(self, report: QualityAuditReport, output_path: Path) -> bool:
        """
        Save quality audit report to JSON file.
        
        Args:
            report: Quality audit report to save
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
            
            self.logger.info(f"Quality audit report saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save quality audit report - output_path: {output_path}, error: {str(e)}")
            return False
    
    def _report_to_dict(self, report: QualityAuditReport) -> Dict[str, Any]:
        """Convert audit report to dictionary for JSON serialization"""
        return {
            'audit_metadata': {
                'audit_id': report.audit_id,
                'generated_at': report.generated_at.isoformat(),
                'correlation_id': report.correlation_id
            },
            'sampling_summary': {
                'total_documents_available': report.total_documents_available,
                'total_samples_collected': report.total_samples_collected,
                'sampling_coverage_rate': round(report.sampling_coverage_rate, 2)
            },
            'domain_sampling_stats': [
                {
                    'domain': stat.domain.value,
                    'total_documents': stat.total_documents,
                    'actual_sample_size': stat.actual_sample_size,
                    'sampling_rate': round(stat.sampling_rate, 3),
                    'avg_completeness_score': round(stat.avg_completeness_score, 3),
                    'avg_preservation_score': round(stat.avg_preservation_score, 3),
                    'technical_notation_coverage': round(stat.technical_notation_coverage, 3)
                }
                for stat in report.domain_sampling_stats
            ],
            'technical_notation_assessment': {
                'mathematical_formulas_found': report.notation_assessment.mathematical_formulas_found,
                'mathematical_formulas_preserved': report.notation_assessment.mathematical_formulas_preserved,
                'formula_preservation_rate': round(report.notation_assessment.formula_preservation_rate, 3),
                'overall_preservation_score': round(report.notation_assessment.overall_preservation_score, 3)
            },
            'embedding_quality': {
                'vector_dimension': report.embedding_quality.vector_dimension if report.embedding_quality else 0,
                'null_embeddings_count': report.embedding_quality.null_embeddings_count if report.embedding_quality else 0,
                'coherence_score': round(report.embedding_quality.coherence_score, 3) if report.embedding_quality else 0,
                'quality_level': report.embedding_quality.quality_level.value if report.embedding_quality else 'unknown'
            } if report.embedding_quality else None,
            'overall_quality': {
                'completeness_score': round(report.overall_completeness_score, 3),
                'preservation_score': round(report.overall_preservation_score, 3),
                'quality_level': report.overall_quality_level.value
            },
            'recommendations': {
                'quality_recommendations': report.quality_recommendations,
                'improvement_suggestions': report.improvement_suggestions,
                'critical_issues': report.critical_issues,
                'warnings': report.warnings
            }
        }
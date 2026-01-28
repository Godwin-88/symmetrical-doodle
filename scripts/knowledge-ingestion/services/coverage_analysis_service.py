"""
Coverage Analysis Service

Implements research thesis scope cross-referencing, missing domain identification,
and coverage scoring methodology for knowledge base readiness assessment.

Requirements: 5.3, 5.5
"""

import asyncio
import statistics
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re

# Simplified imports for testing
try:
    from .inventory_report_generator import DomainCategory, KnowledgeInventoryReport
    from .quality_audit_service import QualityAuditReport, ContentSample
    from ..core.config import get_settings
    from ..core.logging import get_logger, log_context
except ImportError:
    # Fallback for testing
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Dict, Any, Optional
    import logging
    
    class DomainCategory(Enum):
        ML = "Machine Learning"
        DRL = "Deep Reinforcement Learning"
        NLP = "Natural Language Processing"
        LLM = "Large Language Models"
        FINANCE = "Finance & Trading"
        GENERAL = "General Technical"
        UNKNOWN = "Unknown"
    
    @dataclass
    class KnowledgeInventoryReport:
        domain_distribution: Dict[str, int] = field(default_factory=dict)
        total_pdfs_found: int = 0
    
    @dataclass
    class QualityAuditReport:
        overall_completeness_score: float = 0.0
        overall_preservation_score: float = 0.0
    
    @dataclass
    class ContentSample:
        domain: DomainCategory = DomainCategory.UNKNOWN
        has_mathematical_notation: bool = False
        has_technical_terms: bool = False
    
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


class CoverageLevel(Enum):
    """Coverage assessment levels"""
    COMPREHENSIVE = "comprehensive"
    ADEQUATE = "adequate"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    MISSING = "missing"


class ResearchThesisScope(Enum):
    """Research thesis scope areas for algorithmic trading system"""
    # Core ML/AI domains
    MACHINE_LEARNING_FOUNDATIONS = "Machine Learning Foundations"
    DEEP_REINFORCEMENT_LEARNING = "Deep Reinforcement Learning"
    NATURAL_LANGUAGE_PROCESSING = "Natural Language Processing"
    LARGE_LANGUAGE_MODELS = "Large Language Models"
    
    # Financial domains
    QUANTITATIVE_FINANCE = "Quantitative Finance"
    ALGORITHMIC_TRADING = "Algorithmic Trading"
    RISK_MANAGEMENT = "Risk Management"
    PORTFOLIO_OPTIMIZATION = "Portfolio Optimization"
    
    # Technical domains
    TIME_SERIES_ANALYSIS = "Time Series Analysis"
    STATISTICAL_MODELING = "Statistical Modeling"
    OPTIMIZATION_METHODS = "Optimization Methods"
    GRAPH_ANALYTICS = "Graph Analytics"
    
    # Applied domains
    MARKET_MICROSTRUCTURE = "Market Microstructure"
    BEHAVIORAL_FINANCE = "Behavioral Finance"
    ALTERNATIVE_DATA = "Alternative Data"
    REGULATORY_COMPLIANCE = "Regulatory Compliance"


@dataclass
class DomainCoverageScore:
    """Coverage score for a specific domain"""
    domain: ResearchThesisScope
    mapped_categories: List[DomainCategory] = field(default_factory=list)
    
    # Content metrics
    document_count: int = 0
    total_content_volume: float = 0.0  # In MB or token count
    quality_score: float = 0.0
    
    # Coverage assessment
    coverage_level: CoverageLevel = CoverageLevel.MISSING
    coverage_score: float = 0.0  # 0.0 to 1.0
    
    # Gap analysis
    missing_subtopics: List[str] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    strength_areas: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_additions: List[str] = field(default_factory=list)
    priority_level: str = "medium"  # low, medium, high, critical


@dataclass
class CoverageGap:
    """Identified gap in knowledge coverage"""
    gap_id: str
    domain: ResearchThesisScope
    gap_type: str  # "missing_domain", "insufficient_depth", "quality_issues", "outdated_content"
    severity: str  # "critical", "high", "medium", "low"
    
    description: str
    impact_assessment: str
    recommended_actions: List[str] = field(default_factory=list)
    
    # Quantitative measures
    coverage_deficit: float = 0.0  # How much coverage is missing (0.0 to 1.0)
    quality_deficit: float = 0.0   # How much quality is lacking (0.0 to 1.0)
    
    # Context
    related_domains: List[ResearchThesisScope] = field(default_factory=list)
    current_alternatives: List[str] = field(default_factory=list)


@dataclass
class CoverageAnalysisReport:
    """Comprehensive coverage analysis report"""
    # Report metadata
    analysis_id: str
    generated_at: datetime
    correlation_id: Optional[str] = None
    
    # Input data summary
    total_documents_analyzed: int = 0
    total_domains_found: int = 0
    analysis_scope: List[ResearchThesisScope] = field(default_factory=list)
    
    # Coverage scores by domain
    domain_coverage_scores: List[DomainCoverageScore] = field(default_factory=list)
    
    # Overall coverage metrics
    overall_coverage_score: float = 0.0
    coverage_completeness: float = 0.0  # Percentage of required domains covered
    coverage_depth_score: float = 0.0   # Average depth across covered domains
    coverage_quality_score: float = 0.0 # Average quality across covered domains
    
    # Gap analysis
    identified_gaps: List[CoverageGap] = field(default_factory=list)
    critical_gaps: List[CoverageGap] = field(default_factory=list)
    
    # Domain mapping analysis
    well_covered_domains: List[ResearchThesisScope] = field(default_factory=list)
    partially_covered_domains: List[ResearchThesisScope] = field(default_factory=list)
    missing_domains: List[ResearchThesisScope] = field(default_factory=list)
    
    # Recommendations
    priority_recommendations: List[str] = field(default_factory=list)
    improvement_roadmap: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality indicators
    research_readiness_level: str = "insufficient"  # insufficient, partial, adequate, comprehensive
    estimated_research_capability: float = 0.0  # 0.0 to 1.0


class ResearchThesisScopeMapper:
    """Maps ingested content domains to research thesis scope areas"""
    
    def __init__(self):
        self.logger = get_logger(__name__, component="thesis_scope_mapper")
        
        # Define mapping from ingested domains to research thesis scope
        self.domain_mapping = {
            # Direct mappings
            DomainCategory.ML: [
                ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS,
                ResearchThesisScope.STATISTICAL_MODELING,
                ResearchThesisScope.OPTIMIZATION_METHODS
            ],
            DomainCategory.DRL: [
                ResearchThesisScope.DEEP_REINFORCEMENT_LEARNING,
                ResearchThesisScope.ALGORITHMIC_TRADING,
                ResearchThesisScope.OPTIMIZATION_METHODS
            ],
            DomainCategory.NLP: [
                ResearchThesisScope.NATURAL_LANGUAGE_PROCESSING,
                ResearchThesisScope.ALTERNATIVE_DATA
            ],
            DomainCategory.LLM: [
                ResearchThesisScope.LARGE_LANGUAGE_MODELS,
                ResearchThesisScope.NATURAL_LANGUAGE_PROCESSING,
                ResearchThesisScope.ALTERNATIVE_DATA
            ],
            DomainCategory.FINANCE: [
                ResearchThesisScope.QUANTITATIVE_FINANCE,
                ResearchThesisScope.ALGORITHMIC_TRADING,
                ResearchThesisScope.RISK_MANAGEMENT,
                ResearchThesisScope.PORTFOLIO_OPTIMIZATION,
                ResearchThesisScope.MARKET_MICROSTRUCTURE,
                ResearchThesisScope.BEHAVIORAL_FINANCE
            ],
            DomainCategory.GENERAL: [
                ResearchThesisScope.TIME_SERIES_ANALYSIS,
                ResearchThesisScope.STATISTICAL_MODELING,
                ResearchThesisScope.GRAPH_ANALYTICS
            ]
        }
        
        # Define minimum coverage thresholds for each research area
        self.coverage_thresholds = {
            # Core areas - higher thresholds
            ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS: {
                'min_documents': 15,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.7
            },
            ResearchThesisScope.DEEP_REINFORCEMENT_LEARNING: {
                'min_documents': 12,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.7
            },
            ResearchThesisScope.QUANTITATIVE_FINANCE: {
                'min_documents': 20,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.8
            },
            ResearchThesisScope.ALGORITHMIC_TRADING: {
                'min_documents': 15,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.8
            },
            
            # Supporting areas - moderate thresholds
            ResearchThesisScope.NATURAL_LANGUAGE_PROCESSING: {
                'min_documents': 8,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.LARGE_LANGUAGE_MODELS: {
                'min_documents': 6,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.RISK_MANAGEMENT: {
                'min_documents': 10,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.7
            },
            ResearchThesisScope.PORTFOLIO_OPTIMIZATION: {
                'min_documents': 8,
                'min_quality_score': 0.8,
                'min_coverage_score': 0.7
            },
            
            # Specialized areas - lower thresholds
            ResearchThesisScope.TIME_SERIES_ANALYSIS: {
                'min_documents': 5,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.STATISTICAL_MODELING: {
                'min_documents': 6,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.OPTIMIZATION_METHODS: {
                'min_documents': 5,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.GRAPH_ANALYTICS: {
                'min_documents': 4,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.MARKET_MICROSTRUCTURE: {
                'min_documents': 4,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.BEHAVIORAL_FINANCE: {
                'min_documents': 3,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.ALTERNATIVE_DATA: {
                'min_documents': 3,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.6
            },
            ResearchThesisScope.REGULATORY_COMPLIANCE: {
                'min_documents': 2,
                'min_quality_score': 0.7,
                'min_coverage_score': 0.5
            }
        }
    
    def map_domains_to_thesis_scope(
        self, 
        domain_distribution: Dict[str, int]
    ) -> Dict[ResearchThesisScope, List[DomainCategory]]:
        """
        Map ingested content domains to research thesis scope areas.
        
        Args:
            domain_distribution: Distribution of ingested content by domain
            
        Returns:
            Mapping from research thesis scope to contributing domain categories
        """
        thesis_mapping = {}
        
        for domain_name, count in domain_distribution.items():
            if count == 0:
                continue
                
            try:
                domain_category = DomainCategory(domain_name)
            except ValueError:
                self.logger.warning(f"Unknown domain category: {domain_name}")
                continue
            
            # Map to thesis scope areas
            if domain_category in self.domain_mapping:
                for thesis_area in self.domain_mapping[domain_category]:
                    if thesis_area not in thesis_mapping:
                        thesis_mapping[thesis_area] = []
                    thesis_mapping[thesis_area].append(domain_category)
        
        return thesis_mapping
    
    def get_coverage_requirements(self, thesis_area: ResearchThesisScope) -> Dict[str, Any]:
        """Get coverage requirements for a specific thesis area"""
        return self.coverage_thresholds.get(thesis_area, {
            'min_documents': 3,
            'min_quality_score': 0.7,
            'min_coverage_score': 0.6
        })


class CoverageAnalyzer:
    """
    Coverage analyzer for research thesis scope cross-referencing
    and missing domain identification.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="coverage_analyzer")
        self.thesis_mapper = ResearchThesisScopeMapper()
    
    async def analyze_coverage(
        self,
        inventory_report: KnowledgeInventoryReport,
        quality_report: QualityAuditReport,
        correlation_id: Optional[str] = None
    ) -> CoverageAnalysisReport:
        """
        Analyze coverage against research thesis scope.
        
        Args:
            inventory_report: Knowledge inventory report
            quality_report: Quality audit report
            correlation_id: Correlation ID for logging
            
        Returns:
            Comprehensive coverage analysis report
        """
        with log_context("coverage_analyzer", "analyze_coverage", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info(f"Starting coverage analysis: {inventory_report.total_pdfs_found} documents across {len(inventory_report.domain_distribution)} domains")
            
            # Create analysis report
            report = CoverageAnalysisReport(
                analysis_id=f"coverage_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(timezone.utc),
                correlation_id=ctx.correlation_id,
                total_documents_analyzed=inventory_report.total_pdfs_found,
                total_domains_found=len(inventory_report.domain_distribution),
                analysis_scope=list(ResearchThesisScope)
            )
            
            # Map domains to thesis scope
            thesis_mapping = self.thesis_mapper.map_domains_to_thesis_scope(
                inventory_report.domain_distribution
            )
            
            # Analyze coverage for each thesis area
            await self._analyze_domain_coverage(
                thesis_mapping, inventory_report, quality_report, report
            )
            
            # Identify gaps
            await self._identify_coverage_gaps(report)
            
            # Calculate overall scores
            await self._calculate_overall_coverage_scores(report)
            
            # Generate recommendations
            await self._generate_coverage_recommendations(report)
            
            self.logger.info(f"Coverage analysis completed - overall_score: {report.overall_coverage_score:.3f}, readiness_level: {report.research_readiness_level}")
            
            return report
    
    async def _analyze_domain_coverage(
        self,
        thesis_mapping: Dict[ResearchThesisScope, List[DomainCategory]],
        inventory_report: KnowledgeInventoryReport,
        quality_report: QualityAuditReport,
        report: CoverageAnalysisReport
    ):
        """Analyze coverage for each domain in the thesis scope"""
        
        for thesis_area in ResearchThesisScope:
            # Get contributing domains
            contributing_domains = thesis_mapping.get(thesis_area, [])
            
            # Calculate document count
            document_count = 0
            for domain_cat in contributing_domains:
                domain_name = domain_cat.value
                document_count += inventory_report.domain_distribution.get(domain_name, 0)
            
            # Get coverage requirements
            requirements = self.thesis_mapper.get_coverage_requirements(thesis_area)
            
            # Calculate coverage score
            coverage_score = await self._calculate_domain_coverage_score(
                thesis_area, document_count, quality_report, requirements
            )
            
            # Determine coverage level
            coverage_level = self._determine_coverage_level(coverage_score, requirements)
            
            # Identify gaps and strengths
            missing_subtopics, weak_areas, strength_areas = await self._analyze_domain_gaps(
                thesis_area, contributing_domains, document_count
            )
            
            # Generate recommendations
            recommendations = await self._generate_domain_recommendations(
                thesis_area, coverage_level, document_count, requirements
            )
            
            # Create domain coverage score
            domain_score = DomainCoverageScore(
                domain=thesis_area,
                mapped_categories=contributing_domains,
                document_count=document_count,
                quality_score=quality_report.overall_completeness_score,  # Use overall quality as proxy
                coverage_level=coverage_level,
                coverage_score=coverage_score,
                missing_subtopics=missing_subtopics,
                weak_areas=weak_areas,
                strength_areas=strength_areas,
                recommended_additions=recommendations,
                priority_level=self._determine_priority_level(coverage_level, thesis_area)
            )
            
            report.domain_coverage_scores.append(domain_score)
            
            # Categorize domains
            if coverage_level in [CoverageLevel.COMPREHENSIVE, CoverageLevel.ADEQUATE]:
                report.well_covered_domains.append(thesis_area)
            elif coverage_level == CoverageLevel.PARTIAL:
                report.partially_covered_domains.append(thesis_area)
            else:
                report.missing_domains.append(thesis_area)
    
    async def _calculate_domain_coverage_score(
        self,
        thesis_area: ResearchThesisScope,
        document_count: int,
        quality_report: QualityAuditReport,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate coverage score for a specific domain"""
        
        # Document count score (0.0 to 1.0)
        min_docs = requirements.get('min_documents', 3)
        doc_score = min(document_count / min_docs, 1.0) if min_docs > 0 else 0.0
        
        # Quality score (use overall quality from audit report)
        quality_score = (quality_report.overall_completeness_score + 
                        quality_report.overall_preservation_score) / 2
        
        # Weighted combination
        coverage_score = (doc_score * 0.6) + (quality_score * 0.4)
        
        return min(coverage_score, 1.0)
    
    def _determine_coverage_level(
        self, 
        coverage_score: float, 
        requirements: Dict[str, Any]
    ) -> CoverageLevel:
        """Determine coverage level based on score and requirements"""
        
        min_coverage = requirements.get('min_coverage_score', 0.6)
        
        if coverage_score >= min_coverage * 1.2:  # 20% above minimum
            return CoverageLevel.COMPREHENSIVE
        elif coverage_score >= min_coverage:
            return CoverageLevel.ADEQUATE
        elif coverage_score >= min_coverage * 0.7:  # 70% of minimum
            return CoverageLevel.PARTIAL
        elif coverage_score >= min_coverage * 0.3:  # 30% of minimum
            return CoverageLevel.INSUFFICIENT
        else:
            return CoverageLevel.MISSING
    
    async def _analyze_domain_gaps(
        self,
        thesis_area: ResearchThesisScope,
        contributing_domains: List[DomainCategory],
        document_count: int
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze gaps and strengths for a specific domain"""
        
        missing_subtopics = []
        weak_areas = []
        strength_areas = []
        
        # Define expected subtopics for each thesis area
        expected_subtopics = self._get_expected_subtopics(thesis_area)
        
        # For now, use simple heuristics based on document count and contributing domains
        if document_count == 0:
            missing_subtopics = expected_subtopics
        elif document_count < 5:
            missing_subtopics = expected_subtopics[len(expected_subtopics)//2:]
            weak_areas = expected_subtopics[:len(expected_subtopics)//2]
        else:
            strength_areas = expected_subtopics[:len(expected_subtopics)//2]
            if document_count < 10:
                weak_areas = expected_subtopics[len(expected_subtopics)//2:]
        
        return missing_subtopics, weak_areas, strength_areas
    
    def _get_expected_subtopics(self, thesis_area: ResearchThesisScope) -> List[str]:
        """Get expected subtopics for a thesis area"""
        
        subtopics_map = {
            ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS: [
                "Supervised Learning", "Unsupervised Learning", "Feature Engineering",
                "Model Selection", "Cross-Validation", "Ensemble Methods"
            ],
            ResearchThesisScope.DEEP_REINFORCEMENT_LEARNING: [
                "Q-Learning", "Policy Gradients", "Actor-Critic Methods",
                "Multi-Agent Systems", "Continuous Control", "Model-Based RL"
            ],
            ResearchThesisScope.QUANTITATIVE_FINANCE: [
                "Asset Pricing Models", "Risk Models", "Portfolio Theory",
                "Derivatives Pricing", "Market Microstructure", "Volatility Modeling"
            ],
            ResearchThesisScope.ALGORITHMIC_TRADING: [
                "Trading Strategies", "Execution Algorithms", "Market Making",
                "High-Frequency Trading", "Alternative Data", "Backtesting"
            ],
            ResearchThesisScope.NATURAL_LANGUAGE_PROCESSING: [
                "Text Processing", "Sentiment Analysis", "Named Entity Recognition",
                "Language Models", "Information Extraction", "Text Classification"
            ],
            ResearchThesisScope.LARGE_LANGUAGE_MODELS: [
                "Transformer Architecture", "Pre-training", "Fine-tuning",
                "Prompt Engineering", "In-Context Learning", "Reasoning"
            ]
        }
        
        return subtopics_map.get(thesis_area, ["General Topics", "Advanced Methods", "Applications"])
    
    async def _generate_domain_recommendations(
        self,
        thesis_area: ResearchThesisScope,
        coverage_level: CoverageLevel,
        document_count: int,
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving domain coverage"""
        
        recommendations = []
        min_docs = requirements.get('min_documents', 3)
        
        if coverage_level == CoverageLevel.MISSING:
            recommendations.append(f"Add foundational resources for {thesis_area.value}")
            recommendations.append(f"Target minimum {min_docs} high-quality documents")
        elif coverage_level == CoverageLevel.INSUFFICIENT:
            recommendations.append(f"Increase document count from {document_count} to {min_docs}")
            recommendations.append(f"Focus on core concepts in {thesis_area.value}")
        elif coverage_level == CoverageLevel.PARTIAL:
            recommendations.append(f"Add specialized resources to reach comprehensive coverage")
            recommendations.append(f"Focus on advanced topics and recent developments")
        
        return recommendations
    
    def _determine_priority_level(
        self, 
        coverage_level: CoverageLevel, 
        thesis_area: ResearchThesisScope
    ) -> str:
        """Determine priority level for addressing coverage gaps"""
        
        # Core areas get higher priority
        core_areas = [
            ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS,
            ResearchThesisScope.DEEP_REINFORCEMENT_LEARNING,
            ResearchThesisScope.QUANTITATIVE_FINANCE,
            ResearchThesisScope.ALGORITHMIC_TRADING
        ]
        
        is_core = thesis_area in core_areas
        
        if coverage_level == CoverageLevel.MISSING:
            return "critical" if is_core else "high"
        elif coverage_level == CoverageLevel.INSUFFICIENT:
            return "high" if is_core else "medium"
        elif coverage_level == CoverageLevel.PARTIAL:
            return "medium" if is_core else "low"
        else:
            return "low"
    
    async def _identify_coverage_gaps(self, report: CoverageAnalysisReport):
        """Identify specific coverage gaps"""
        
        gaps = []
        
        for domain_score in report.domain_coverage_scores:
            if domain_score.coverage_level in [CoverageLevel.MISSING, CoverageLevel.INSUFFICIENT]:
                gap = CoverageGap(
                    gap_id=f"gap_{domain_score.domain.name.lower()}",
                    domain=domain_score.domain,
                    gap_type="missing_domain" if domain_score.coverage_level == CoverageLevel.MISSING else "insufficient_depth",
                    severity=domain_score.priority_level,
                    description=f"Insufficient coverage in {domain_score.domain.value}",
                    impact_assessment=f"Limited research capability in {domain_score.domain.value} area",
                    recommended_actions=domain_score.recommended_additions,
                    coverage_deficit=1.0 - domain_score.coverage_score,
                    quality_deficit=max(0, 0.8 - domain_score.quality_score)
                )
                gaps.append(gap)
                
                if gap.severity in ["critical", "high"]:
                    report.critical_gaps.append(gap)
        
        report.identified_gaps = gaps
    
    async def _calculate_overall_coverage_scores(self, report: CoverageAnalysisReport):
        """Calculate overall coverage scores"""
        
        if not report.domain_coverage_scores:
            return
        
        # Coverage completeness (percentage of domains with adequate coverage)
        adequate_domains = [
            score for score in report.domain_coverage_scores
            if score.coverage_level in [CoverageLevel.COMPREHENSIVE, CoverageLevel.ADEQUATE]
        ]
        report.coverage_completeness = len(adequate_domains) / len(report.domain_coverage_scores) * 100
        
        # Coverage depth (average coverage score)
        coverage_scores = [score.coverage_score for score in report.domain_coverage_scores]
        report.coverage_depth_score = statistics.mean(coverage_scores)
        
        # Coverage quality (average quality score)
        quality_scores = [score.quality_score for score in report.domain_coverage_scores]
        report.coverage_quality_score = statistics.mean(quality_scores)
        
        # Overall coverage score (weighted combination)
        report.overall_coverage_score = (
            (report.coverage_completeness / 100) * 0.4 +
            report.coverage_depth_score * 0.4 +
            report.coverage_quality_score * 0.2
        )
        
        # Research readiness level
        if report.overall_coverage_score >= 0.8:
            report.research_readiness_level = "comprehensive"
        elif report.overall_coverage_score >= 0.65:
            report.research_readiness_level = "adequate"
        elif report.overall_coverage_score >= 0.4:
            report.research_readiness_level = "partial"
        else:
            report.research_readiness_level = "insufficient"
        
        # Estimated research capability
        report.estimated_research_capability = min(report.overall_coverage_score, 1.0)
    
    async def _generate_coverage_recommendations(self, report: CoverageAnalysisReport):
        """Generate overall coverage recommendations"""
        
        recommendations = []
        roadmap = []
        
        # Priority recommendations based on critical gaps
        if report.critical_gaps:
            recommendations.append(f"Address {len(report.critical_gaps)} critical coverage gaps immediately")
            
            for gap in report.critical_gaps[:3]:  # Top 3 critical gaps
                recommendations.append(f"Priority: Add resources for {gap.domain.value}")
        
        # Coverage completeness recommendations
        if report.coverage_completeness < 70:
            recommendations.append(f"Improve coverage completeness from {report.coverage_completeness:.1f}% to at least 70%")
        
        # Quality recommendations
        if report.coverage_quality_score < 0.7:
            recommendations.append(f"Improve content quality from {report.coverage_quality_score:.2f} to at least 0.70")
        
        # Generate improvement roadmap
        if report.missing_domains:
            roadmap.append({
                "phase": "Phase 1: Foundation Building",
                "timeline": "1-2 months",
                "focus": "Address missing core domains",
                "domains": [domain.value for domain in report.missing_domains[:3]],
                "target_documents": 30
            })
        
        if report.partially_covered_domains:
            roadmap.append({
                "phase": "Phase 2: Depth Enhancement",
                "timeline": "2-3 months",
                "focus": "Improve partially covered domains",
                "domains": [domain.value for domain in report.partially_covered_domains[:3]],
                "target_documents": 20
            })
        
        if report.well_covered_domains:
            roadmap.append({
                "phase": "Phase 3: Specialization",
                "timeline": "3-4 months",
                "focus": "Add specialized and advanced topics",
                "domains": [domain.value for domain in report.well_covered_domains[:2]],
                "target_documents": 15
            })
        
        report.priority_recommendations = recommendations
        report.improvement_roadmap = roadmap
    
    async def save_coverage_report(self, report: CoverageAnalysisReport, output_path: Path) -> bool:
        """
        Save coverage analysis report to JSON file.
        
        Args:
            report: Coverage analysis report to save
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
            
            self.logger.info(f"Coverage analysis report saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save coverage analysis report - output_path: {output_path}, error: {str(e)}")
            return False
    
    def _report_to_dict(self, report: CoverageAnalysisReport) -> Dict[str, Any]:
        """Convert coverage report to dictionary for JSON serialization"""
        return {
            'analysis_metadata': {
                'analysis_id': report.analysis_id,
                'generated_at': report.generated_at.isoformat(),
                'correlation_id': report.correlation_id
            },
            'input_summary': {
                'total_documents_analyzed': report.total_documents_analyzed,
                'total_domains_found': report.total_domains_found,
                'analysis_scope_count': len(report.analysis_scope)
            },
            'coverage_scores': {
                'overall_coverage_score': round(report.overall_coverage_score, 3),
                'coverage_completeness': round(report.coverage_completeness, 2),
                'coverage_depth_score': round(report.coverage_depth_score, 3),
                'coverage_quality_score': round(report.coverage_quality_score, 3),
                'research_readiness_level': report.research_readiness_level,
                'estimated_research_capability': round(report.estimated_research_capability, 3)
            },
            'domain_coverage': [
                {
                    'domain': score.domain.value,
                    'document_count': score.document_count,
                    'coverage_level': score.coverage_level.value,
                    'coverage_score': round(score.coverage_score, 3),
                    'quality_score': round(score.quality_score, 3),
                    'priority_level': score.priority_level,
                    'missing_subtopics': score.missing_subtopics,
                    'weak_areas': score.weak_areas,
                    'strength_areas': score.strength_areas,
                    'recommended_additions': score.recommended_additions
                }
                for score in report.domain_coverage_scores
            ],
            'gap_analysis': {
                'total_gaps_identified': len(report.identified_gaps),
                'critical_gaps_count': len(report.critical_gaps),
                'well_covered_domains': [domain.value for domain in report.well_covered_domains],
                'partially_covered_domains': [domain.value for domain in report.partially_covered_domains],
                'missing_domains': [domain.value for domain in report.missing_domains],
                'critical_gaps': [
                    {
                        'gap_id': gap.gap_id,
                        'domain': gap.domain.value,
                        'gap_type': gap.gap_type,
                        'severity': gap.severity,
                        'description': gap.description,
                        'coverage_deficit': round(gap.coverage_deficit, 3),
                        'recommended_actions': gap.recommended_actions
                    }
                    for gap in report.critical_gaps
                ]
            },
            'recommendations': {
                'priority_recommendations': report.priority_recommendations,
                'improvement_roadmap': report.improvement_roadmap
            }
        }
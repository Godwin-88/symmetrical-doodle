"""
Knowledge Readiness Memo Generation Service

Creates comprehensive readiness assessment reports with coverage scores,
improvement recommendations, and gap analysis with remediation suggestions.

Requirements: 5.4
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import statistics

# Simplified imports for testing
try:
    from .coverage_analysis_service import (
        CoverageAnalysisReport, CoverageLevel, ResearchThesisScope,
        DomainCoverageScore, CoverageGap
    )
    from .quality_audit_service import QualityAuditReport, QualityLevel
    from .inventory_report_generator import KnowledgeInventoryReport
    from ..core.config import get_settings
    from ..core.logging import get_logger, log_context
except ImportError:
    # Fallback for testing
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Dict, Any, Optional
    import logging
    
    class CoverageLevel(Enum):
        COMPREHENSIVE = "comprehensive"
        ADEQUATE = "adequate"
        PARTIAL = "partial"
        INSUFFICIENT = "insufficient"
        MISSING = "missing"
    
    class QualityLevel(Enum):
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"
        FAILED = "failed"
    
    class ResearchThesisScope(Enum):
        MACHINE_LEARNING_FOUNDATIONS = "Machine Learning Foundations"
        DEEP_REINFORCEMENT_LEARNING = "Deep Reinforcement Learning"
        QUANTITATIVE_FINANCE = "Quantitative Finance"
        ALGORITHMIC_TRADING = "Algorithmic Trading"
    
    @dataclass
    class CoverageAnalysisReport:
        overall_coverage_score: float = 0.0
        research_readiness_level: str = "insufficient"
        domain_coverage_scores: List = field(default_factory=list)
        critical_gaps: List = field(default_factory=list)
        priority_recommendations: List[str] = field(default_factory=list)
        improvement_roadmap: List[Dict[str, Any]] = field(default_factory=list)
    
    @dataclass
    class QualityAuditReport:
        overall_quality_level: QualityLevel = QualityLevel.FAIR
        overall_completeness_score: float = 0.0
        overall_preservation_score: float = 0.0
    
    @dataclass
    class KnowledgeInventoryReport:
        total_pdfs_found: int = 0
        total_size_mb: float = 0.0
        accessibility_stats: Any = None
    
    @dataclass
    class DomainCoverageScore:
        domain: ResearchThesisScope = ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS
        coverage_level: CoverageLevel = CoverageLevel.MISSING
        coverage_score: float = 0.0
        document_count: int = 0
        priority_level: str = "medium"
    
    @dataclass
    class CoverageGap:
        domain: ResearchThesisScope = ResearchThesisScope.MACHINE_LEARNING_FOUNDATIONS
        severity: str = "medium"
        description: str = ""
        recommended_actions: List[str] = field(default_factory=list)
    
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


class ReadinessLevel(Enum):
    """Knowledge base readiness levels"""
    PRODUCTION_READY = "production_ready"
    RESEARCH_READY = "research_ready"
    DEVELOPMENT_READY = "development_ready"
    PROTOTYPE_READY = "prototype_ready"
    NOT_READY = "not_ready"


class RecommendationPriority(Enum):
    """Recommendation priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


@dataclass
class ReadinessMetric:
    """Individual readiness metric"""
    metric_name: str
    current_value: float
    target_value: float
    unit: str
    status: str  # "meets_target", "below_target", "exceeds_target"
    gap: float
    impact_level: str  # "critical", "high", "medium", "low"


@dataclass
class RemediationAction:
    """Specific remediation action"""
    action_id: str
    title: str
    description: str
    priority: RecommendationPriority
    estimated_effort: str  # "low", "medium", "high"
    estimated_timeline: str  # "1-2 weeks", "1 month", etc.
    expected_impact: str  # "high", "medium", "low"
    
    # Dependencies and prerequisites
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    measurable_outcomes: List[str] = field(default_factory=list)
    
    # Resource requirements
    required_resources: List[str] = field(default_factory=list)
    estimated_cost: Optional[str] = None


@dataclass
class ImprovementPlan:
    """Structured improvement plan"""
    plan_id: str
    title: str
    objective: str
    timeline: str
    
    # Phases
    phases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Actions
    immediate_actions: List[RemediationAction] = field(default_factory=list)
    short_term_actions: List[RemediationAction] = field(default_factory=list)
    long_term_actions: List[RemediationAction] = field(default_factory=list)
    
    # Success metrics
    success_metrics: List[ReadinessMetric] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk assessment
    risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class KnowledgeReadinessMemo:
    """Comprehensive knowledge readiness memo"""
    # Memo metadata
    memo_id: str
    generated_at: datetime
    correlation_id: Optional[str] = None
    
    # Executive summary
    overall_readiness_level: ReadinessLevel = ReadinessLevel.NOT_READY
    readiness_score: float = 0.0  # 0.0 to 1.0
    executive_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    
    # Assessment results
    readiness_metrics: List[ReadinessMetric] = field(default_factory=list)
    
    # Coverage assessment
    coverage_summary: Dict[str, Any] = field(default_factory=dict)
    domain_readiness: Dict[str, str] = field(default_factory=dict)  # domain -> readiness level
    
    # Quality assessment
    quality_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Gap analysis
    critical_gaps: List[str] = field(default_factory=list)
    major_gaps: List[str] = field(default_factory=list)
    minor_gaps: List[str] = field(default_factory=list)
    
    # Recommendations
    priority_recommendations: List[str] = field(default_factory=list)
    remediation_actions: List[RemediationAction] = field(default_factory=list)
    improvement_plan: Optional[ImprovementPlan] = None
    
    # Risk assessment
    readiness_risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Timeline and next steps
    recommended_timeline: str = ""
    immediate_next_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Supporting data references
    inventory_report_id: Optional[str] = None
    quality_audit_id: Optional[str] = None
    coverage_analysis_id: Optional[str] = None


class ReadinessAssessmentEngine:
    """Engine for assessing knowledge base readiness"""
    
    def __init__(self):
        self.logger = get_logger(__name__, component="readiness_assessment")
        
        # Define readiness thresholds
        self.readiness_thresholds = {
            ReadinessLevel.PRODUCTION_READY: {
                'overall_coverage': 0.9,
                'quality_score': 0.85,
                'completeness': 0.9,
                'critical_gaps': 0,
                'accessibility_rate': 95.0
            },
            ReadinessLevel.RESEARCH_READY: {
                'overall_coverage': 0.75,
                'quality_score': 0.75,
                'completeness': 0.8,
                'critical_gaps': 2,
                'accessibility_rate': 90.0
            },
            ReadinessLevel.DEVELOPMENT_READY: {
                'overall_coverage': 0.6,
                'quality_score': 0.65,
                'completeness': 0.7,
                'critical_gaps': 5,
                'accessibility_rate': 85.0
            },
            ReadinessLevel.PROTOTYPE_READY: {
                'overall_coverage': 0.4,
                'quality_score': 0.5,
                'completeness': 0.6,
                'critical_gaps': 10,
                'accessibility_rate': 75.0
            }
        }
    
    def assess_readiness_level(
        self,
        coverage_report: CoverageAnalysisReport,
        quality_report: QualityAuditReport,
        inventory_report: KnowledgeInventoryReport
    ) -> Tuple[ReadinessLevel, float, List[ReadinessMetric]]:
        """
        Assess overall readiness level based on multiple reports.
        
        Args:
            coverage_report: Coverage analysis report
            quality_report: Quality audit report
            inventory_report: Inventory report
            
        Returns:
            Tuple of (readiness_level, readiness_score, metrics)
        """
        
        # Calculate individual metrics
        metrics = []
        
        # Coverage metrics
        coverage_metric = ReadinessMetric(
            metric_name="Overall Coverage Score",
            current_value=coverage_report.overall_coverage_score,
            target_value=0.75,
            unit="score",
            status="below_target" if coverage_report.overall_coverage_score < 0.75 else "meets_target",
            gap=max(0, 0.75 - coverage_report.overall_coverage_score),
            impact_level="critical" if coverage_report.overall_coverage_score < 0.5 else "high"
        )
        metrics.append(coverage_metric)
        
        # Quality metrics
        quality_score = (quality_report.overall_completeness_score + 
                        quality_report.overall_preservation_score) / 2
        quality_metric = ReadinessMetric(
            metric_name="Content Quality Score",
            current_value=quality_score,
            target_value=0.75,
            unit="score",
            status="below_target" if quality_score < 0.75 else "meets_target",
            gap=max(0, 0.75 - quality_score),
            impact_level="high" if quality_score < 0.6 else "medium"
        )
        metrics.append(quality_metric)
        
        # Critical gaps metric
        critical_gaps_count = len(coverage_report.critical_gaps)
        gaps_metric = ReadinessMetric(
            metric_name="Critical Gaps Count",
            current_value=critical_gaps_count,
            target_value=2,
            unit="count",
            status="exceeds_target" if critical_gaps_count > 2 else "meets_target",
            gap=max(0, critical_gaps_count - 2),
            impact_level="critical" if critical_gaps_count > 5 else "high"
        )
        metrics.append(gaps_metric)
        
        # Accessibility metric
        accessibility_rate = (inventory_report.accessibility_stats.accessibility_rate 
                            if inventory_report.accessibility_stats else 0.0)
        access_metric = ReadinessMetric(
            metric_name="Content Accessibility Rate",
            current_value=accessibility_rate,
            target_value=90.0,
            unit="percentage",
            status="below_target" if accessibility_rate < 90.0 else "meets_target",
            gap=max(0, 90.0 - accessibility_rate),
            impact_level="medium" if accessibility_rate < 80.0 else "low"
        )
        metrics.append(access_metric)
        
        # Determine readiness level
        readiness_level = self._determine_readiness_level(
            coverage_report.overall_coverage_score,
            quality_score,
            critical_gaps_count,
            accessibility_rate
        )
        
        # Calculate overall readiness score
        readiness_score = self._calculate_readiness_score(metrics)
        
        return readiness_level, readiness_score, metrics
    
    def _determine_readiness_level(
        self,
        coverage_score: float,
        quality_score: float,
        critical_gaps: int,
        accessibility_rate: float
    ) -> ReadinessLevel:
        """Determine readiness level based on key metrics"""
        
        # Check each level from highest to lowest
        for level, thresholds in self.readiness_thresholds.items():
            if (coverage_score >= thresholds['overall_coverage'] and
                quality_score >= thresholds['quality_score'] and
                critical_gaps <= thresholds['critical_gaps'] and
                accessibility_rate >= thresholds['accessibility_rate']):
                return level
        
        return ReadinessLevel.NOT_READY
    
    def _calculate_readiness_score(self, metrics: List[ReadinessMetric]) -> float:
        """Calculate overall readiness score from individual metrics"""
        
        if not metrics:
            return 0.0
        
        # Weight metrics by impact level
        weights = {
            "critical": 0.4,
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1
        }
        
        weighted_scores = []
        total_weight = 0
        
        for metric in metrics:
            # Calculate metric score (1.0 if meets target, proportional if below)
            if metric.status == "meets_target" or metric.status == "exceeds_target":
                metric_score = 1.0
            else:
                # Proportional score based on gap
                if metric.target_value > 0:
                    metric_score = max(0, metric.current_value / metric.target_value)
                else:
                    metric_score = 0.0
            
            weight = weights.get(metric.impact_level, 0.1)
            weighted_scores.append(metric_score * weight)
            total_weight += weight
        
        return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0


class KnowledgeReadinessMemoGenerator:
    """
    Knowledge Readiness Memo Generator
    
    Creates comprehensive readiness assessment reports with coverage scores,
    improvement recommendations, and gap analysis with remediation suggestions.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__, component="readiness_memo_generator")
        self.assessment_engine = ReadinessAssessmentEngine()
    
    async def generate_readiness_memo(
        self,
        inventory_report: KnowledgeInventoryReport,
        quality_report: QualityAuditReport,
        coverage_report: CoverageAnalysisReport,
        correlation_id: Optional[str] = None
    ) -> KnowledgeReadinessMemo:
        """
        Generate comprehensive knowledge readiness memo.
        
        Args:
            inventory_report: Knowledge inventory report
            quality_report: Quality audit report
            coverage_report: Coverage analysis report
            correlation_id: Correlation ID for logging
            
        Returns:
            Comprehensive readiness memo
        """
        with log_context("readiness_memo_generator", "generate_memo", 
                        correlation_id=correlation_id) as ctx:
            
            self.logger.info("Starting knowledge readiness memo generation")
            
            # Create memo structure
            memo = KnowledgeReadinessMemo(
                memo_id=f"readiness_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(timezone.utc),
                correlation_id=ctx.correlation_id,
                inventory_report_id=getattr(inventory_report, 'report_id', None),
                quality_audit_id=getattr(quality_report, 'audit_id', None),
                coverage_analysis_id=getattr(coverage_report, 'analysis_id', None)
            )
            
            # Assess readiness level
            readiness_level, readiness_score, metrics = self.assessment_engine.assess_readiness_level(
                coverage_report, quality_report, inventory_report
            )
            
            memo.overall_readiness_level = readiness_level
            memo.readiness_score = readiness_score
            memo.readiness_metrics = metrics
            
            # Generate executive summary
            await self._generate_executive_summary(memo, inventory_report, quality_report, coverage_report)
            
            # Analyze coverage and quality
            await self._analyze_coverage_and_quality(memo, coverage_report, quality_report)
            
            # Identify gaps
            await self._identify_and_categorize_gaps(memo, coverage_report)
            
            # Generate recommendations and remediation actions
            await self._generate_recommendations_and_actions(memo, coverage_report, quality_report)
            
            # Create improvement plan
            await self._create_improvement_plan(memo, coverage_report)
            
            # Assess risks and mitigation strategies
            await self._assess_risks_and_mitigation(memo, readiness_level)
            
            # Define timeline and next steps
            await self._define_timeline_and_next_steps(memo, readiness_level)
            
            self.logger.info(f"Knowledge readiness memo generated - readiness_level: {readiness_level.value}, score: {readiness_score:.3f}")
            
            return memo
    
    async def _generate_executive_summary(
        self,
        memo: KnowledgeReadinessMemo,
        inventory_report: KnowledgeInventoryReport,
        quality_report: QualityAuditReport,
        coverage_report: CoverageAnalysisReport
    ):
        """Generate executive summary and key findings"""
        
        # Executive summary
        summary_parts = [
            f"Knowledge base readiness assessment reveals a {memo.overall_readiness_level.value.replace('_', ' ')} status",
            f"with an overall readiness score of {memo.readiness_score:.2f}/1.0.",
            f"The analysis covers {inventory_report.total_pdfs_found} documents",
            f"across {len(coverage_report.domain_coverage_scores)} research domains,",
            f"with {len(coverage_report.critical_gaps)} critical gaps identified."
        ]
        
        if memo.overall_readiness_level in [ReadinessLevel.RESEARCH_READY, ReadinessLevel.PRODUCTION_READY]:
            summary_parts.append("The knowledge base demonstrates sufficient coverage and quality for research activities.")
        elif memo.overall_readiness_level == ReadinessLevel.DEVELOPMENT_READY:
            summary_parts.append("The knowledge base supports development activities but requires enhancement for full research capability.")
        else:
            summary_parts.append("Significant improvements are required before the knowledge base can support research activities.")
        
        memo.executive_summary = " ".join(summary_parts)
        
        # Key findings
        key_findings = []
        
        # Coverage findings
        if coverage_report.overall_coverage_score >= 0.8:
            key_findings.append(f"Excellent coverage across research domains ({coverage_report.overall_coverage_score:.2f}/1.0)")
        elif coverage_report.overall_coverage_score >= 0.6:
            key_findings.append(f"Adequate coverage with room for improvement ({coverage_report.overall_coverage_score:.2f}/1.0)")
        else:
            key_findings.append(f"Insufficient coverage requiring significant expansion ({coverage_report.overall_coverage_score:.2f}/1.0)")
        
        # Quality findings
        quality_score = (quality_report.overall_completeness_score + quality_report.overall_preservation_score) / 2
        if quality_score >= 0.8:
            key_findings.append(f"High-quality content with excellent preservation ({quality_score:.2f}/1.0)")
        elif quality_score >= 0.6:
            key_findings.append(f"Good content quality with minor issues ({quality_score:.2f}/1.0)")
        else:
            key_findings.append(f"Content quality issues requiring attention ({quality_score:.2f}/1.0)")
        
        # Gap findings
        if len(coverage_report.critical_gaps) == 0:
            key_findings.append("No critical gaps identified")
        elif len(coverage_report.critical_gaps) <= 3:
            key_findings.append(f"{len(coverage_report.critical_gaps)} critical gaps requiring immediate attention")
        else:
            key_findings.append(f"{len(coverage_report.critical_gaps)} critical gaps indicating significant deficiencies")
        
        # Domain findings
        well_covered = len(coverage_report.well_covered_domains)
        missing = len(coverage_report.missing_domains)
        if well_covered > missing:
            key_findings.append(f"Strong foundation with {well_covered} well-covered domains")
        else:
            key_findings.append(f"Foundation gaps with {missing} missing domains")
        
        memo.key_findings = key_findings
    
    async def _analyze_coverage_and_quality(
        self,
        memo: KnowledgeReadinessMemo,
        coverage_report: CoverageAnalysisReport,
        quality_report: QualityAuditReport
    ):
        """Analyze coverage and quality for the memo"""
        
        # Coverage summary
        memo.coverage_summary = {
            'overall_score': round(coverage_report.overall_coverage_score, 3),
            'completeness_percentage': round(coverage_report.coverage_completeness, 1),
            'depth_score': round(coverage_report.coverage_depth_score, 3),
            'well_covered_count': len(coverage_report.well_covered_domains),
            'partially_covered_count': len(coverage_report.partially_covered_domains),
            'missing_count': len(coverage_report.missing_domains),
            'research_readiness': coverage_report.research_readiness_level
        }
        
        # Domain readiness mapping
        for domain_score in coverage_report.domain_coverage_scores:
            domain_name = domain_score.domain.value
            if domain_score.coverage_level == CoverageLevel.COMPREHENSIVE:
                memo.domain_readiness[domain_name] = "ready"
            elif domain_score.coverage_level == CoverageLevel.ADEQUATE:
                memo.domain_readiness[domain_name] = "adequate"
            elif domain_score.coverage_level == CoverageLevel.PARTIAL:
                memo.domain_readiness[domain_name] = "partial"
            else:
                memo.domain_readiness[domain_name] = "not_ready"
        
        # Quality summary
        memo.quality_summary = {
            'overall_level': quality_report.overall_quality_level.value,
            'completeness_score': round(quality_report.overall_completeness_score, 3),
            'preservation_score': round(quality_report.overall_preservation_score, 3),
            'combined_score': round((quality_report.overall_completeness_score + 
                                   quality_report.overall_preservation_score) / 2, 3)
        }
    
    async def _identify_and_categorize_gaps(
        self,
        memo: KnowledgeReadinessMemo,
        coverage_report: CoverageAnalysisReport
    ):
        """Identify and categorize gaps by severity"""
        
        critical_gaps = []
        major_gaps = []
        minor_gaps = []
        
        for gap in coverage_report.identified_gaps:
            gap_description = f"{gap.domain.value}: {gap.description}"
            
            if gap.severity == "critical":
                critical_gaps.append(gap_description)
            elif gap.severity == "high":
                major_gaps.append(gap_description)
            else:
                minor_gaps.append(gap_description)
        
        # Add domain-level gaps
        for domain in coverage_report.missing_domains:
            critical_gaps.append(f"Missing domain: {domain.value}")
        
        for domain in coverage_report.partially_covered_domains:
            major_gaps.append(f"Partial coverage: {domain.value}")
        
        memo.critical_gaps = critical_gaps
        memo.major_gaps = major_gaps
        memo.minor_gaps = minor_gaps
    
    async def _generate_recommendations_and_actions(
        self,
        memo: KnowledgeReadinessMemo,
        coverage_report: CoverageAnalysisReport,
        quality_report: QualityAuditReport
    ):
        """Generate recommendations and specific remediation actions"""
        
        # Priority recommendations from coverage report
        memo.priority_recommendations = coverage_report.priority_recommendations.copy()
        
        # Add quality-based recommendations
        if quality_report.overall_completeness_score < 0.7:
            memo.priority_recommendations.append("Improve content extraction and processing quality")
        
        if quality_report.overall_preservation_score < 0.8:
            memo.priority_recommendations.append("Enhance mathematical notation and technical content preservation")
        
        # Generate specific remediation actions
        actions = []
        
        # Critical gap actions
        for i, gap in enumerate(coverage_report.critical_gaps[:3]):  # Top 3 critical gaps
            action = RemediationAction(
                action_id=f"critical_gap_{i+1}",
                title=f"Address Critical Gap: {gap.domain.value}",
                description=gap.description,
                priority=RecommendationPriority.CRITICAL,
                estimated_effort="high",
                estimated_timeline="2-4 weeks",
                expected_impact="high",
                success_criteria=[
                    f"Add minimum 5 high-quality documents for {gap.domain.value}",
                    f"Achieve coverage score > 0.6 for {gap.domain.value}",
                    "Verify content quality and technical notation preservation"
                ],
                measurable_outcomes=[
                    f"Document count for {gap.domain.value} increases by 5+",
                    f"Coverage score for {gap.domain.value} > 0.6",
                    "Quality audit shows improved preservation scores"
                ],
                required_resources=[
                    "Subject matter expert review",
                    "High-quality source materials",
                    "Processing and validation time"
                ]
            )
            actions.append(action)
        
        # Quality improvement actions
        if quality_report.overall_quality_level in [QualityLevel.POOR, QualityLevel.FAILED]:
            quality_action = RemediationAction(
                action_id="quality_improvement",
                title="Improve Content Quality and Preservation",
                description="Enhance content extraction, processing, and technical notation preservation",
                priority=RecommendationPriority.HIGH,
                estimated_effort="medium",
                estimated_timeline="2-3 weeks",
                expected_impact="high",
                success_criteria=[
                    "Overall quality score > 0.75",
                    "Technical notation preservation > 80%",
                    "Content completeness > 75%"
                ],
                measurable_outcomes=[
                    "Quality audit shows improved scores",
                    "Reduced content processing errors",
                    "Better mathematical notation preservation"
                ],
                required_resources=[
                    "PDF processing optimization",
                    "Quality validation tools",
                    "Technical review process"
                ]
            )
            actions.append(quality_action)
        
        # Coverage expansion actions
        if coverage_report.overall_coverage_score < 0.6:
            coverage_action = RemediationAction(
                action_id="coverage_expansion",
                title="Expand Domain Coverage",
                description="Add content for missing and partially covered domains",
                priority=RecommendationPriority.HIGH,
                estimated_effort="high",
                estimated_timeline="4-6 weeks",
                expected_impact="high",
                success_criteria=[
                    "Overall coverage score > 0.65",
                    "Reduce missing domains by 50%",
                    "Improve partial domains to adequate level"
                ],
                measurable_outcomes=[
                    "Coverage analysis shows improved scores",
                    "Reduced number of critical gaps",
                    "Better domain distribution"
                ],
                required_resources=[
                    "Additional source materials",
                    "Domain expertise",
                    "Processing capacity"
                ]
            )
            actions.append(coverage_action)
        
        memo.remediation_actions = actions
    
    async def _create_improvement_plan(
        self,
        memo: KnowledgeReadinessMemo,
        coverage_report: CoverageAnalysisReport
    ):
        """Create structured improvement plan"""
        
        plan = ImprovementPlan(
            plan_id=f"improvement_{memo.memo_id}",
            title="Knowledge Base Readiness Improvement Plan",
            objective="Achieve research-ready knowledge base with comprehensive coverage and high quality",
            timeline="3-6 months"
        )
        
        # Define phases based on roadmap
        if coverage_report.improvement_roadmap:
            plan.phases = coverage_report.improvement_roadmap
        else:
            # Default phases
            plan.phases = [
                {
                    "phase": "Phase 1: Critical Gap Resolution",
                    "timeline": "1-2 months",
                    "focus": "Address critical gaps and quality issues",
                    "target_outcome": "Achieve development-ready status"
                },
                {
                    "phase": "Phase 2: Coverage Enhancement",
                    "timeline": "2-3 months",
                    "focus": "Expand domain coverage and depth",
                    "target_outcome": "Achieve research-ready status"
                },
                {
                    "phase": "Phase 3: Quality Optimization",
                    "timeline": "3-4 months",
                    "focus": "Optimize quality and specialized content",
                    "target_outcome": "Achieve production-ready status"
                }
            ]
        
        # Categorize actions by timeline
        for action in memo.remediation_actions:
            if action.priority == RecommendationPriority.CRITICAL:
                plan.immediate_actions.append(action)
            elif action.priority == RecommendationPriority.HIGH:
                plan.short_term_actions.append(action)
            else:
                plan.long_term_actions.append(action)
        
        # Define success metrics
        plan.success_metrics = [
            ReadinessMetric(
                metric_name="Overall Coverage Score",
                current_value=coverage_report.overall_coverage_score,
                target_value=0.75,
                unit="score",
                status="target",
                gap=0,
                impact_level="critical"
            ),
            ReadinessMetric(
                metric_name="Critical Gaps Count",
                current_value=len(coverage_report.critical_gaps),
                target_value=2,
                unit="count",
                status="target",
                gap=0,
                impact_level="high"
            )
        ]
        
        # Define milestones
        plan.milestones = [
            {
                "milestone": "Critical Gaps Resolved",
                "timeline": "1 month",
                "criteria": "All critical gaps addressed, coverage score > 0.5"
            },
            {
                "milestone": "Research Ready",
                "timeline": "3 months",
                "criteria": "Coverage score > 0.75, quality score > 0.75"
            },
            {
                "milestone": "Production Ready",
                "timeline": "6 months",
                "criteria": "Coverage score > 0.9, quality score > 0.85"
            }
        ]
        
        memo.improvement_plan = plan
    
    async def _assess_risks_and_mitigation(
        self,
        memo: KnowledgeReadinessMemo,
        readiness_level: ReadinessLevel
    ):
        """Assess risks and mitigation strategies"""
        
        risks = []
        mitigations = []
        
        # Readiness-specific risks
        if readiness_level == ReadinessLevel.NOT_READY:
            risks.extend([
                "Knowledge base insufficient for any research activities",
                "High risk of research gaps and incomplete analysis",
                "Potential for incorrect conclusions due to missing information"
            ])
            mitigations.extend([
                "Prioritize critical domain coverage immediately",
                "Implement quality validation processes",
                "Establish content review and verification procedures"
            ])
        
        elif readiness_level == ReadinessLevel.PROTOTYPE_READY:
            risks.extend([
                "Limited research scope due to coverage gaps",
                "Quality issues may affect research reliability",
                "Missing domains may lead to incomplete analysis"
            ])
            mitigations.extend([
                "Focus on core domain expansion",
                "Implement quality improvement processes",
                "Establish domain expertise review"
            ])
        
        # General risks
        risks.extend([
            "Content quality degradation over time",
            "New research areas not covered",
            "Technical notation preservation issues"
        ])
        
        mitigations.extend([
            "Implement regular quality audits",
            "Establish content update and expansion processes",
            "Maintain technical processing capabilities"
        ])
        
        memo.readiness_risks = risks
        memo.mitigation_strategies = mitigations
    
    async def _define_timeline_and_next_steps(
        self,
        memo: KnowledgeReadinessMemo,
        readiness_level: ReadinessLevel
    ):
        """Define timeline and immediate next steps"""
        
        # Timeline based on readiness level
        timeline_map = {
            ReadinessLevel.NOT_READY: "6-12 months to achieve research readiness",
            ReadinessLevel.PROTOTYPE_READY: "3-6 months to achieve research readiness",
            ReadinessLevel.DEVELOPMENT_READY: "2-4 months to achieve research readiness",
            ReadinessLevel.RESEARCH_READY: "1-3 months to achieve production readiness",
            ReadinessLevel.PRODUCTION_READY: "Ongoing maintenance and enhancement"
        }
        
        memo.recommended_timeline = timeline_map.get(readiness_level, "Timeline assessment needed")
        
        # Immediate next steps
        next_steps = []
        
        if readiness_level in [ReadinessLevel.NOT_READY, ReadinessLevel.PROTOTYPE_READY]:
            next_steps.extend([
                "Address critical gaps immediately",
                "Implement quality improvement processes",
                "Establish content acquisition strategy"
            ])
        elif readiness_level == ReadinessLevel.DEVELOPMENT_READY:
            next_steps.extend([
                "Expand coverage in partially covered domains",
                "Enhance content quality and preservation",
                "Implement regular quality monitoring"
            ])
        else:
            next_steps.extend([
                "Maintain current quality levels",
                "Monitor for new research areas",
                "Implement continuous improvement processes"
            ])
        
        # Add specific next steps from remediation actions
        for action in memo.remediation_actions[:3]:  # Top 3 actions
            if action.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]:
                next_steps.append(f"Execute: {action.title}")
        
        memo.immediate_next_steps = next_steps
        
        # Success criteria
        memo.success_criteria = [
            "Achieve target readiness level within recommended timeline",
            "Resolve all critical gaps identified in this assessment",
            "Maintain quality scores above minimum thresholds",
            "Establish sustainable content management processes"
        ]
    
    async def save_readiness_memo(self, memo: KnowledgeReadinessMemo, output_path: Path) -> bool:
        """
        Save knowledge readiness memo to JSON file.
        
        Args:
            memo: Knowledge readiness memo to save
            output_path: Path to save the memo
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert memo to dictionary for JSON serialization
            memo_dict = self._memo_to_dict(memo)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(memo_dict, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Knowledge readiness memo saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save readiness memo - output_path: {output_path}, error: {str(e)}")
            return False
    
    async def generate_memo_summary(self, memo: KnowledgeReadinessMemo) -> str:
        """
        Generate a human-readable summary of the readiness memo.
        
        Args:
            memo: Knowledge readiness memo
            
        Returns:
            Formatted summary string
        """
        summary_lines = [
            "=" * 70,
            "KNOWLEDGE BASE READINESS MEMO",
            "=" * 70,
            f"Memo ID: {memo.memo_id}",
            f"Generated: {memo.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            memo.executive_summary,
            "",
            "READINESS ASSESSMENT",
            "-" * 20,
            f"Overall Readiness Level: {memo.overall_readiness_level.value.replace('_', ' ').title()}",
            f"Readiness Score: {memo.readiness_score:.2f}/1.0",
            "",
            "KEY FINDINGS",
            "-" * 20
        ]
        
        for finding in memo.key_findings:
            summary_lines.append(f"• {finding}")
        
        summary_lines.extend([
            "",
            "CRITICAL GAPS",
            "-" * 20
        ])
        
        if memo.critical_gaps:
            for gap in memo.critical_gaps:
                summary_lines.append(f"• {gap}")
        else:
            summary_lines.append("• No critical gaps identified")
        
        summary_lines.extend([
            "",
            "PRIORITY RECOMMENDATIONS",
            "-" * 20
        ])
        
        for rec in memo.priority_recommendations[:5]:  # Top 5 recommendations
            summary_lines.append(f"• {rec}")
        
        summary_lines.extend([
            "",
            "IMMEDIATE NEXT STEPS",
            "-" * 20
        ])
        
        for step in memo.immediate_next_steps:
            summary_lines.append(f"• {step}")
        
        summary_lines.extend([
            "",
            "RECOMMENDED TIMELINE",
            "-" * 20,
            memo.recommended_timeline,
            "",
            "=" * 70
        ])
        
        return "\n".join(summary_lines)
    
    def _memo_to_dict(self, memo: KnowledgeReadinessMemo) -> Dict[str, Any]:
        """Convert readiness memo to dictionary for JSON serialization"""
        return {
            'memo_metadata': {
                'memo_id': memo.memo_id,
                'generated_at': memo.generated_at.isoformat(),
                'correlation_id': memo.correlation_id,
                'source_reports': {
                    'inventory_report_id': memo.inventory_report_id,
                    'quality_audit_id': memo.quality_audit_id,
                    'coverage_analysis_id': memo.coverage_analysis_id
                }
            },
            'executive_summary': {
                'overall_readiness_level': memo.overall_readiness_level.value,
                'readiness_score': round(memo.readiness_score, 3),
                'summary': memo.executive_summary,
                'key_findings': memo.key_findings
            },
            'readiness_metrics': [
                {
                    'metric_name': metric.metric_name,
                    'current_value': round(metric.current_value, 3),
                    'target_value': round(metric.target_value, 3),
                    'unit': metric.unit,
                    'status': metric.status,
                    'gap': round(metric.gap, 3),
                    'impact_level': metric.impact_level
                }
                for metric in memo.readiness_metrics
            ],
            'assessment_results': {
                'coverage_summary': memo.coverage_summary,
                'quality_summary': memo.quality_summary,
                'domain_readiness': memo.domain_readiness
            },
            'gap_analysis': {
                'critical_gaps': memo.critical_gaps,
                'major_gaps': memo.major_gaps,
                'minor_gaps': memo.minor_gaps
            },
            'recommendations': {
                'priority_recommendations': memo.priority_recommendations,
                'remediation_actions': [
                    {
                        'action_id': action.action_id,
                        'title': action.title,
                        'description': action.description,
                        'priority': action.priority.value,
                        'estimated_effort': action.estimated_effort,
                        'estimated_timeline': action.estimated_timeline,
                        'expected_impact': action.expected_impact,
                        'success_criteria': action.success_criteria,
                        'required_resources': action.required_resources
                    }
                    for action in memo.remediation_actions
                ]
            },
            'improvement_plan': {
                'plan_id': memo.improvement_plan.plan_id if memo.improvement_plan else None,
                'title': memo.improvement_plan.title if memo.improvement_plan else None,
                'objective': memo.improvement_plan.objective if memo.improvement_plan else None,
                'timeline': memo.improvement_plan.timeline if memo.improvement_plan else None,
                'phases': memo.improvement_plan.phases if memo.improvement_plan else [],
                'milestones': memo.improvement_plan.milestones if memo.improvement_plan else []
            } if memo.improvement_plan else None,
            'risk_assessment': {
                'readiness_risks': memo.readiness_risks,
                'mitigation_strategies': memo.mitigation_strategies
            },
            'timeline_and_next_steps': {
                'recommended_timeline': memo.recommended_timeline,
                'immediate_next_steps': memo.immediate_next_steps,
                'success_criteria': memo.success_criteria
            }
        }
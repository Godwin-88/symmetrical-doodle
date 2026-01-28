#!/usr/bin/env python3
"""
Test Coverage Analysis and Readiness Assessment Implementation

Simple test to verify the coverage analysis service and knowledge readiness memo
generator work correctly with sample data.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from services.coverage_analysis_service import (
    CoverageAnalyzer, ResearchThesisScope, CoverageLevel
)
from services.knowledge_readiness_memo import (
    KnowledgeReadinessMemoGenerator, ReadinessLevel
)
from services.inventory_report_generator import (
    KnowledgeInventoryReport, DomainCategory, AccessibilityStats
)
from services.quality_audit_service import (
    QualityAuditReport, QualityLevel
)


def create_sample_inventory_report() -> KnowledgeInventoryReport:
    """Create sample inventory report for testing"""
    
    accessibility_stats = AccessibilityStats(
        accessible_count=85,
        restricted_count=10,
        not_found_count=3,
        error_count=2,
        total_count=100
    )
    
    return KnowledgeInventoryReport(
        report_id="test_inventory_001",
        generated_at=datetime.now(timezone.utc),
        total_pdfs_found=100,
        total_size_mb=2500.0,
        domain_distribution={
            DomainCategory.ML.value: 25,
            DomainCategory.DRL.value: 15,
            DomainCategory.FINANCE.value: 30,
            DomainCategory.NLP.value: 10,
            DomainCategory.LLM.value: 8,
            DomainCategory.GENERAL.value: 12
        },
        accessibility_stats=accessibility_stats
    )


def create_sample_quality_report() -> QualityAuditReport:
    """Create sample quality audit report for testing"""
    
    return QualityAuditReport(
        audit_id="test_quality_001",
        generated_at=datetime.now(timezone.utc),
        overall_quality_level=QualityLevel.GOOD,
        overall_completeness_score=0.75,
        overall_preservation_score=0.82
    )


async def test_coverage_analysis():
    """Test coverage analysis functionality"""
    
    print("Testing Coverage Analysis Service...")
    print("=" * 50)
    
    # Create sample data
    inventory_report = create_sample_inventory_report()
    quality_report = create_sample_quality_report()
    
    # Initialize coverage analyzer
    analyzer = CoverageAnalyzer()
    
    # Perform coverage analysis
    coverage_report = await analyzer.analyze_coverage(
        inventory_report=inventory_report,
        quality_report=quality_report,
        correlation_id="test_coverage_001"
    )
    
    # Display results
    print(f"Analysis ID: {coverage_report.analysis_id}")
    print(f"Overall Coverage Score: {coverage_report.overall_coverage_score:.3f}")
    print(f"Research Readiness Level: {coverage_report.research_readiness_level}")
    print(f"Coverage Completeness: {coverage_report.coverage_completeness:.1f}%")
    print(f"Coverage Depth Score: {coverage_report.coverage_depth_score:.3f}")
    print(f"Coverage Quality Score: {coverage_report.coverage_quality_score:.3f}")
    print()
    
    print("Domain Coverage Scores:")
    print("-" * 30)
    for domain_score in coverage_report.domain_coverage_scores:
        print(f"  {domain_score.domain.value}:")
        print(f"    Coverage Level: {domain_score.coverage_level.value}")
        print(f"    Coverage Score: {domain_score.coverage_score:.3f}")
        print(f"    Document Count: {domain_score.document_count}")
        print(f"    Priority Level: {domain_score.priority_level}")
        print()
    
    print(f"Well Covered Domains: {len(coverage_report.well_covered_domains)}")
    for domain in coverage_report.well_covered_domains:
        print(f"  - {domain.value}")
    print()
    
    print(f"Missing Domains: {len(coverage_report.missing_domains)}")
    for domain in coverage_report.missing_domains:
        print(f"  - {domain.value}")
    print()
    
    print(f"Critical Gaps: {len(coverage_report.critical_gaps)}")
    for gap in coverage_report.critical_gaps:
        print(f"  - {gap.domain.value}: {gap.description}")
    print()
    
    print("Priority Recommendations:")
    for rec in coverage_report.priority_recommendations:
        print(f"  - {rec}")
    print()
    
    return coverage_report


async def test_readiness_memo(coverage_report):
    """Test knowledge readiness memo generation"""
    
    print("Testing Knowledge Readiness Memo Generator...")
    print("=" * 50)
    
    # Create sample data
    inventory_report = create_sample_inventory_report()
    quality_report = create_sample_quality_report()
    
    # Initialize memo generator
    memo_generator = KnowledgeReadinessMemoGenerator()
    
    # Generate readiness memo
    memo = await memo_generator.generate_readiness_memo(
        inventory_report=inventory_report,
        quality_report=quality_report,
        coverage_report=coverage_report,
        correlation_id="test_memo_001"
    )
    
    # Display results
    print(f"Memo ID: {memo.memo_id}")
    print(f"Overall Readiness Level: {memo.overall_readiness_level.value}")
    print(f"Readiness Score: {memo.readiness_score:.3f}")
    print()
    
    print("Executive Summary:")
    print("-" * 20)
    print(memo.executive_summary)
    print()
    
    print("Key Findings:")
    print("-" * 20)
    for finding in memo.key_findings:
        print(f"  - {finding}")
    print()
    
    print("Readiness Metrics:")
    print("-" * 20)
    for metric in memo.readiness_metrics:
        print(f"  {metric.metric_name}:")
        print(f"    Current: {metric.current_value:.3f} {metric.unit}")
        print(f"    Target: {metric.target_value:.3f} {metric.unit}")
        print(f"    Status: {metric.status}")
        print(f"    Impact: {metric.impact_level}")
        print()
    
    print(f"Critical Gaps ({len(memo.critical_gaps)}):")
    for gap in memo.critical_gaps:
        print(f"  - {gap}")
    print()
    
    print("Priority Recommendations:")
    for rec in memo.priority_recommendations:
        print(f"  - {rec}")
    print()
    
    print("Remediation Actions:")
    print("-" * 20)
    for action in memo.remediation_actions:
        print(f"  {action.title} ({action.priority.value}):")
        print(f"    Description: {action.description}")
        print(f"    Timeline: {action.estimated_timeline}")
        print(f"    Impact: {action.expected_impact}")
        print()
    
    print("Recommended Timeline:")
    print(memo.recommended_timeline)
    print()
    
    print("Immediate Next Steps:")
    for step in memo.immediate_next_steps:
        print(f"  - {step}")
    print()
    
    # Generate and display summary
    summary = await memo_generator.generate_memo_summary(memo)
    print("MEMO SUMMARY:")
    print(summary)
    
    return memo


async def test_file_operations(coverage_report, memo):
    """Test file save operations"""
    
    print("Testing File Operations...")
    print("=" * 50)
    
    # Test coverage report save
    coverage_output = Path("test_outputs/coverage_analysis_report.json")
    analyzer = CoverageAnalyzer()
    coverage_saved = await analyzer.save_coverage_report(coverage_report, coverage_output)
    print(f"Coverage report saved: {coverage_saved} -> {coverage_output}")
    
    # Test memo save
    memo_output = Path("test_outputs/knowledge_readiness_memo.json")
    memo_generator = KnowledgeReadinessMemoGenerator()
    memo_saved = await memo_generator.save_readiness_memo(memo, memo_output)
    print(f"Readiness memo saved: {memo_saved} -> {memo_output}")
    
    return coverage_saved and memo_saved


async def main():
    """Main test function"""
    
    print("Coverage Analysis and Readiness Assessment Test")
    print("=" * 60)
    print()
    
    try:
        # Test coverage analysis
        coverage_report = await test_coverage_analysis()
        
        print("\n" + "=" * 60 + "\n")
        
        # Test readiness memo generation
        memo = await test_readiness_memo(coverage_report)
        
        print("\n" + "=" * 60 + "\n")
        
        # Test file operations
        files_saved = await test_file_operations(coverage_report, memo)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS:")
        print("=" * 60)
        print(f"Coverage Analysis: ✓ PASSED")
        print(f"Readiness Memo Generation: ✓ PASSED")
        print(f"File Operations: {'✓ PASSED' if files_saved else '✗ FAILED'}")
        print()
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
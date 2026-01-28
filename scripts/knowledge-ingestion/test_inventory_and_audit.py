"""
Test script for inventory report generation and quality audit services.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from services.google_drive_discovery import PDFMetadata, DiscoveryResult, AccessStatus
from services.inventory_report_generator import (
    InventoryReportGenerator, DomainCategory
)
from services.quality_audit_service import (
    QualityAuditor, ContentSample, ContentType
)
from core.logging import configure_logging, get_logger


async def test_inventory_report_generation():
    """Test inventory report generation with sample data"""
    logger = get_logger(__name__)
    logger.info("Testing inventory report generation")
    
    # Create sample PDF metadata
    sample_pdfs = [
        PDFMetadata(
            file_id="1",
            name="Deep Learning for Trading Strategies.pdf",
            mime_type="application/pdf",
            modified_time=datetime.now(timezone.utc),
            size=5 * 1024 * 1024,  # 5MB
            web_view_link="https://drive.google.com/file/d/1/view",
            access_status=AccessStatus.ACCESSIBLE,
            domain_classification="Machine Learning"
        ),
        PDFMetadata(
            file_id="2",
            name="Reinforcement Learning in Finance.pdf",
            mime_type="application/pdf",
            modified_time=datetime.now(timezone.utc),
            size=3 * 1024 * 1024,  # 3MB
            web_view_link="https://drive.google.com/file/d/2/view",
            access_status=AccessStatus.ACCESSIBLE,
            domain_classification="Deep Reinforcement Learning"
        ),
        PDFMetadata(
            file_id="3",
            name="Natural Language Processing for Market Analysis.pdf",
            mime_type="application/pdf",
            modified_time=datetime.now(timezone.utc),
            size=7 * 1024 * 1024,  # 7MB
            web_view_link="https://drive.google.com/file/d/3/view",
            access_status=AccessStatus.RESTRICTED,
            domain_classification="Natural Language Processing"
        ),
        PDFMetadata(
            file_id="4",
            name="Portfolio Optimization Methods.pdf",
            mime_type="application/pdf",
            modified_time=datetime.now(timezone.utc),
            size=2 * 1024 * 1024,  # 2MB
            web_view_link="https://drive.google.com/file/d/4/view",
            access_status=AccessStatus.ACCESSIBLE,
            domain_classification="Finance & Trading"
        )
    ]
    
    # Create discovery result
    discovery_result = DiscoveryResult(
        success=True,
        pdfs_found=sample_pdfs,
        total_files_scanned=10,
        inaccessible_files=[
            {
                'file_id': '5',
                'name': 'Restricted Document.pdf',
                'reason': 'Access denied - insufficient permissions'
            }
        ],
        folders_scanned=['folder1', 'folder2']
    )
    
    # Generate inventory report
    generator = InventoryReportGenerator()
    report = await generator.generate_inventory_report(
        discovery_result, 
        include_raw_data=True
    )
    
    # Verify report contents
    assert report.total_pdfs_found == 4
    assert report.total_files_scanned == 10
    assert len(report.domain_stats) > 0
    assert report.accessibility_stats.accessible_count == 3
    assert report.accessibility_stats.restricted_count == 1
    assert len(report.inaccessible_files) == 1
    
    logger.info("Inventory report generated successfully",
               total_pdfs=report.total_pdfs_found,
               domains=len(report.domain_stats),
               accessibility_rate=report.accessibility_stats.accessibility_rate)
    
    # Save report to file
    output_path = Path("test_inventory_report.json")
    success = await generator.save_report_to_file(report, output_path)
    assert success
    
    # Generate summary
    summary = await generator.generate_summary_report(report)
    print("\n" + "="*60)
    print("INVENTORY REPORT SUMMARY")
    print("="*60)
    print(summary)
    
    return report


async def test_quality_audit():
    """Test quality audit with sample data"""
    logger = get_logger(__name__)
    logger.info("Testing quality audit")
    
    # Create sample documents with content
    sample_documents = [
        {
            'document_id': '1',
            'name': 'Deep Learning Paper.pdf',
            'domain_classification': 'Machine Learning',
            'content': '''
            This paper presents a novel approach to deep learning for financial markets.
            The method uses neural networks with multiple hidden layers to learn complex patterns.
            We define the loss function as L(θ) = Σ(y_i - f(x_i; θ))² where θ represents the parameters.
            The gradient descent algorithm updates parameters using ∇L(θ) = ∂L/∂θ.
            Our experiments show significant improvement in prediction accuracy.
            '''
        },
        {
            'document_id': '2',
            'name': 'Reinforcement Learning Study.pdf',
            'domain_classification': 'Deep Reinforcement Learning',
            'content': '''
            Reinforcement learning in trading environments requires careful consideration of the MDP formulation.
            The state space S includes market conditions, portfolio positions, and technical indicators.
            The action space A consists of buy, sell, and hold decisions for each asset.
            The reward function R(s,a,s') captures profit/loss and risk-adjusted returns.
            The Bellman equation V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')] defines optimal value.
            Policy gradient methods like PPO and A3C have shown promising results.
            '''
        },
        {
            'document_id': '3',
            'name': 'NLP for Finance.pdf',
            'domain_classification': 'Natural Language Processing',
            'content': '''
            Natural language processing techniques can extract valuable insights from financial texts.
            We use transformer models like BERT for sentiment analysis of news articles.
            The attention mechanism allows the model to focus on relevant parts of the text.
            Named entity recognition (NER) identifies companies, currencies, and financial instruments.
            Text classification categorizes documents into topics like earnings, mergers, or regulatory changes.
            '''
        }
    ]
    
    # Create sample embeddings
    sample_embeddings = [
        {
            'document_id': '1',
            'chunk_id': '1_1',
            'embedding_vector': [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector
        },
        {
            'document_id': '2',
            'chunk_id': '2_1',
            'embedding_vector': [0.2, 0.3, 0.4, 0.5, 0.6] * 100  # 500-dim vector
        },
        {
            'document_id': '3',
            'chunk_id': '3_1',
            'embedding_vector': [0.0] * 500  # Zero vector (quality issue)
        }
    ]
    
    # Conduct quality audit
    auditor = QualityAuditor()
    audit_report = await auditor.conduct_quality_audit(
        sample_documents,
        sample_embeddings
    )
    
    # Verify audit results
    assert audit_report.total_documents_available == 3
    assert audit_report.total_samples_collected > 0
    assert len(audit_report.domain_sampling_stats) > 0
    assert audit_report.overall_completeness_score > 0
    assert audit_report.embedding_quality is not None
    
    logger.info("Quality audit completed successfully",
               samples_collected=audit_report.total_samples_collected,
               overall_quality=audit_report.overall_quality_level.value,
               completeness_score=audit_report.overall_completeness_score)
    
    # Save audit report
    output_path = Path("test_quality_audit_report.json")
    success = await auditor.save_audit_report(audit_report, output_path)
    assert success
    
    print("\n" + "="*60)
    print("QUALITY AUDIT SUMMARY")
    print("="*60)
    print(f"Overall Quality Level: {audit_report.overall_quality_level.value}")
    print(f"Completeness Score: {audit_report.overall_completeness_score:.3f}")
    print(f"Preservation Score: {audit_report.overall_preservation_score:.3f}")
    print(f"Samples Collected: {audit_report.total_samples_collected}")
    
    if audit_report.quality_recommendations:
        print("\nRecommendations:")
        for rec in audit_report.quality_recommendations:
            print(f"• {rec}")
    
    if audit_report.warnings:
        print("\nWarnings:")
        for warning in audit_report.warnings:
            print(f"• {warning}")
    
    return audit_report


async def main():
    """Main test function"""
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting inventory and audit system tests")
        
        # Test inventory report generation
        inventory_report = await test_inventory_report_generation()
        
        # Test quality audit
        audit_report = await test_quality_audit()
        
        logger.info("All tests completed successfully")
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"✓ Inventory Report: {inventory_report.total_pdfs_found} PDFs analyzed")
        print(f"✓ Quality Audit: {audit_report.total_samples_collected} samples audited")
        print(f"✓ Overall Quality: {audit_report.overall_quality_level.value}")
        print("✓ All tests passed!")
        
    except Exception as e:
        logger.error("Test failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
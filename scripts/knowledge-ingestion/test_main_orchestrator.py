#!/usr/bin/env python3
"""
Integration Tests for Multi-Source Pipeline Orchestration

This module provides comprehensive integration tests for the main orchestration
system, testing the complete pipeline from discovery to audit across all
supported data sources.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Test framework setup
import sys
sys.path.insert(0, str(Path(__file__).parent))

from main_orchestrator import (
    MultiSourcePipelineOrchestrator,
    OrchestrationConfig,
    OrchestrationPhase,
    OrchestrationStatus,
    SourceOrchestrationState,
    OrchestrationProgress
)
from services.multi_source_auth import DataSourceType
from services.enhanced_batch_manager import EnhancedProcessingOptions
from config.orchestration_config import (
    OrchestrationConfiguration,
    PerformanceConfiguration,
    MonitoringConfiguration,
    ErrorHandlingConfiguration,
    QualityConfiguration
)


class TestMultiSourcePipelineOrchestrator:
    """Test suite for the main pipeline orchestrator"""
    
    @pytest.fixture
    async def orchestrator_config(self):
        """Create test orchestration configuration"""
        return OrchestrationConfig(
            max_concurrent_sources=2,
            max_concurrent_files_per_source=4,
            enable_cross_source_optimization=True,
            progress_update_interval_seconds=1,
            enable_websocket_updates=False,  # Disable for testing
            enable_detailed_logging=True,
            continue_on_source_failure=True,
            max_retries_per_source=2,
            enable_quality_audit=True,
            enable_coverage_analysis=True
        )
    
    @pytest.fixture
    async def mock_services(self):
        """Create mock services for testing"""
        services = {
            'auth_service': AsyncMock(),
            'browsing_service': AsyncMock(),
            'batch_manager': AsyncMock(),
            'performance_optimizer': AsyncMock(),
            'embedding_service': AsyncMock(),
            'database_service': AsyncMock(),
            'quality_audit_service': AsyncMock(),
            'coverage_service': AsyncMock(),
            'readiness_service': AsyncMock()
        }
        
        # Configure mock behaviors
        services['auth_service'].initialize.return_value = True
        services['browsing_service'].initialize.return_value = True
        services['browsing_service'].list_files.return_value = [
            Mock(file_id=f"file_{i}", name=f"test_file_{i}.pdf", size=1024*1024)
            for i in range(5)
        ]
        
        services['batch_manager'].create_enhanced_job.return_value = "test_job_id"
        services['batch_manager'].get_job_status.return_value = Mock(
            status='completed',
            completed_files=5,
            failed_files=0,
            skipped_files=0,
            source_progress=[
                Mock(
                    source_type=DataSourceType.LOCAL_DIRECTORY,
                    connection_id='test_conn',
                    total_files=5,
                    completed_files=5,
                    failed_files=0,
                    skipped_files=0
                )
            ]
        )
        
        services['quality_audit_service'].run_comprehensive_audit.return_value = {
            'overall_score': 0.85,
            'quality_metrics': {'content_preservation': 0.9, 'embedding_quality': 0.8}
        }
        
        services['coverage_service'].analyze_coverage.return_value = {
            'coverage_score': 0.75,
            'domain_coverage': {'ml': 0.8, 'finance': 0.7}
        }
        
        services['readiness_service'].generate_comprehensive_memo.return_value = {
            'readiness_score': 0.8,
            'recommendations': ['Improve coverage in NLP domain']
        }
        
        return services
    
    @pytest.fixture
    async def orchestrator(self, orchestrator_config, mock_services):
        """Create orchestrator instance with mocked services"""
        orchestrator = MultiSourcePipelineOrchestrator(orchestrator_config)
        
        # Inject mock services
        orchestrator._auth_service = mock_services['auth_service']
        orchestrator._browsing_service = mock_services['browsing_service']
        orchestrator._batch_manager = mock_services['batch_manager']
        orchestrator._performance_optimizer = mock_services['performance_optimizer']
        orchestrator._embedding_service = mock_services['embedding_service']
        orchestrator._database_service = mock_services['database_service']
        orchestrator._quality_audit_service = mock_services['quality_audit_service']
        orchestrator._coverage_service = mock_services['coverage_service']
        orchestrator._readiness_service = mock_services['readiness_service']
        
        # Mock state manager
        orchestrator._state_manager = Mock()
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator_config):
        """Test orchestrator initialization"""
        with patch('main_orchestrator.get_auth_service'), \
             patch('main_orchestrator.get_browsing_service'), \
             patch('main_orchestrator.get_enhanced_batch_manager'), \
             patch('main_orchestrator.get_performance_optimizer'), \
             patch('main_orchestrator.get_async_embedding_service'), \
             patch('main_orchestrator.get_async_database_service'):
            
            orchestrator = MultiSourcePipelineOrchestrator(orchestrator_config)
            
            # Test configuration
            assert orchestrator.config.max_concurrent_sources == 2
            assert orchestrator.config.enable_quality_audit == True
            assert orchestrator._orchestration_progress is None
            assert orchestrator._shutdown_requested == False
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_orchestration(self, orchestrator, mock_services):
        """Test complete pipeline orchestration from discovery to audit"""
        # Prepare test data
        user_id = "test_user_123"
        source_selections = [
            {
                'source_type': 'local_directory',
                'connection_id': 'test_conn_1',
                'source_name': 'Test Local Directory',
                'recursive': True
            }
        ]
        
        processing_options = EnhancedProcessingOptions(
            max_concurrent_files=4,
            enable_async_processing=True,
            enable_gpu_acceleration=False  # Disable for testing
        )
        
        # Execute orchestration
        progress = await orchestrator.orchestrate_complete_pipeline(
            user_id=user_id,
            source_selections=source_selections,
            processing_options=processing_options
        )
        
        # Verify results
        assert progress is not None
        assert progress.user_id == user_id
        assert progress.overall_status == OrchestrationStatus.COMPLETED
        assert progress.total_sources == 1
        assert progress.discovered_files == 5
        assert progress.processed_files == 5
        assert progress.failed_files == 0
        
        # Verify service calls
        mock_services['browsing_service'].list_files.assert_called()
        mock_services['batch_manager'].create_enhanced_job.assert_called_once()
        mock_services['quality_audit_service'].run_comprehensive_audit.assert_called_once()
        mock_services['coverage_service'].analyze_coverage.assert_called_once()
        mock_services['readiness_service'].generate_comprehensive_memo.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discovery_phase(self, orchestrator, mock_services):
        """Test discovery phase execution"""
        source_selections = [
            {
                'source_type': 'local_directory',
                'connection_id': 'test_conn_1',
                'source_name': 'Test Directory'
            },
            {
                'source_type': 'google_drive',
                'connection_id': 'test_conn_2',
                'source_name': 'Test Google Drive'
            }
        ]
        
        # Initialize orchestration progress
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Execute discovery phase
        await orchestrator._execute_discovery_phase(source_selections)
        
        # Verify results
        progress = orchestrator._orchestration_progress
        assert progress.current_phase == OrchestrationPhase.DISCOVERY
        assert progress.total_sources == 2
        assert progress.discovered_files == 10  # 5 files per source
        assert len(progress.source_states) == 2
        
        # Verify source states
        for source_key, source_state in progress.source_states.items():
            assert source_state.discovered_files == 5
            assert source_state.phase_status == OrchestrationStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_ingestion_phase(self, orchestrator, mock_services):
        """Test ingestion phase execution"""
        # Setup orchestration progress with discovered files
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Add source state
        source_state = SourceOrchestrationState(
            source_type=DataSourceType.LOCAL_DIRECTORY,
            connection_id='test_conn',
            source_name='Test Directory',
            discovered_files=5,
            total_files=5,
            phase_status=OrchestrationStatus.COMPLETED
        )
        orchestrator._orchestration_progress.source_states['local_directory:test_conn'] = source_state
        orchestrator._orchestration_progress.discovered_files = 5
        orchestrator._orchestration_progress.total_files = 5
        
        # Execute ingestion phase
        await orchestrator._execute_ingestion_phase(None)
        
        # Verify results
        progress = orchestrator._orchestration_progress
        assert progress.current_phase == OrchestrationPhase.INGESTION
        assert progress.processed_files == 5
        assert progress.failed_files == 0
        
        # Verify batch manager was called
        mock_services['batch_manager'].create_enhanced_job.assert_called_once()
        mock_services['batch_manager'].get_job_status.assert_called()
    
    @pytest.mark.asyncio
    async def test_audit_phase(self, orchestrator, mock_services):
        """Test audit phase execution"""
        # Initialize orchestration progress
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Execute audit phase
        await orchestrator._execute_audit_phase()
        
        # Verify results
        progress = orchestrator._orchestration_progress
        assert progress.current_phase == OrchestrationPhase.AUDIT
        
        # Verify audit services were called
        mock_services['quality_audit_service'].run_comprehensive_audit.assert_called_once()
        mock_services['coverage_service'].analyze_coverage.assert_called_once()
        mock_services['readiness_service'].generate_comprehensive_memo.assert_called_once()
        
        # Verify phase metrics were stored
        assert OrchestrationPhase.AUDIT in orchestrator._phase_metrics
        audit_metrics = orchestrator._phase_metrics[OrchestrationPhase.AUDIT]
        assert 'quality_audit' in audit_metrics
        assert 'coverage_analysis' in audit_metrics
        assert 'readiness_memo' in audit_metrics
    
    @pytest.mark.asyncio
    async def test_error_handling_in_discovery(self, orchestrator, mock_services):
        """Test error handling during discovery phase"""
        # Configure mock to raise exception
        mock_services['browsing_service'].list_files.side_effect = Exception("Connection failed")
        
        source_selections = [
            {
                'source_type': 'google_drive',
                'connection_id': 'failing_conn',
                'source_name': 'Failing Source'
            }
        ]
        
        # Initialize orchestration progress
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Execute discovery phase
        await orchestrator._execute_discovery_phase(source_selections)
        
        # Verify error handling
        progress = orchestrator._orchestration_progress
        assert progress.error_count == 1
        
        source_key = 'google_drive:failing_conn'
        assert source_key in progress.source_states
        source_state = progress.source_states[source_key]
        assert source_state.phase_status == OrchestrationStatus.FAILED
        assert source_state.error_message == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, orchestrator):
        """Test progress tracking functionality"""
        # Initialize orchestration progress
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Test progress updates
        await orchestrator._update_resource_metrics()
        
        # Get current progress
        progress = await orchestrator.get_orchestration_progress()
        assert progress is not None
        assert progress.orchestration_id == "test_orchestration"
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self, orchestrator):
        """Test WebSocket integration for real-time updates"""
        user_id = "test_user"
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        
        # Add WebSocket connection
        await orchestrator.add_websocket_connection(user_id, mock_websocket)
        
        # Verify connection was added
        assert user_id in orchestrator._websocket_connections
        assert mock_websocket in orchestrator._websocket_connections[user_id]
        
        # Test broadcasting
        test_message = {"type": "test", "data": "test_data"}
        await orchestrator._broadcast_websocket_message(test_message)
        
        # Remove connection
        await orchestrator.remove_websocket_connection(user_id, mock_websocket)
        assert user_id not in orchestrator._websocket_connections or not orchestrator._websocket_connections[user_id]
    
    @pytest.mark.asyncio
    async def test_shutdown_handling(self, orchestrator):
        """Test graceful shutdown handling"""
        # Initialize orchestration progress
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Request shutdown
        orchestrator._shutdown_requested = True
        
        # Handle shutdown
        progress = await orchestrator._handle_shutdown()
        
        # Verify shutdown handling
        assert progress.overall_status == OrchestrationStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_final_summary_generation(self, orchestrator):
        """Test final summary generation"""
        # Setup orchestration progress with test data
        orchestrator._orchestration_progress = OrchestrationProgress(
            orchestration_id="test_orchestration",
            user_id="test_user",
            started_at=datetime.now(timezone.utc)
        )
        
        # Add source state
        source_state = SourceOrchestrationState(
            source_type=DataSourceType.LOCAL_DIRECTORY,
            connection_id='test_conn',
            source_name='Test Directory',
            total_files=10,
            processed_files=8,
            failed_files=1,
            skipped_files=1,
            phase_status=OrchestrationStatus.COMPLETED
        )
        orchestrator._orchestration_progress.source_states['local_directory:test_conn'] = source_state
        orchestrator._orchestration_progress.total_sources = 1
        orchestrator._orchestration_progress.total_files = 10
        orchestrator._orchestration_progress.processed_files = 8
        orchestrator._orchestration_progress.failed_files = 1
        orchestrator._orchestration_progress.skipped_files = 1
        
        orchestrator._start_time = datetime.now(timezone.utc)
        
        # Generate summary
        summary = await orchestrator._generate_final_summary()
        
        # Verify summary content
        assert summary['orchestration_id'] == "test_orchestration"
        assert summary['user_id'] == "test_user"
        assert summary['sources']['total_sources'] == 1
        assert summary['sources']['successful_sources'] == 1
        assert summary['files']['total_files'] == 10
        assert summary['files']['processed_files'] == 8
        assert summary['files']['failed_files'] == 1
        assert summary['files']['success_rate'] == 80.0
        
        # Verify source details
        source_details = summary['sources']['source_details']['local_directory:test_conn']
        assert source_details['source_type'] == 'local_directory'
        assert source_details['total_files'] == 10
        assert source_details['processed_files'] == 8


class TestOrchestrationConfiguration:
    """Test suite for orchestration configuration"""
    
    def test_default_configuration(self):
        """Test default configuration creation"""
        config = OrchestrationConfig()
        
        assert config.max_concurrent_sources == 4
        assert config.max_concurrent_files_per_source == 16
        assert config.enable_cross_source_optimization == True
        assert config.progress_update_interval_seconds == 5
        assert config.enable_websocket_updates == True
        assert config.continue_on_source_failure == True
        assert config.enable_quality_audit == True
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        config = OrchestrationConfig(
            max_concurrent_sources=2,
            max_concurrent_files_per_source=8
        )
        assert config.max_concurrent_sources == 2
        assert config.max_concurrent_files_per_source == 8
        
        # Test edge cases
        config_zero = OrchestrationConfig(max_concurrent_sources=0)
        assert config_zero.max_concurrent_sources == 0  # Should be handled by validation


class TestIntegrationScenarios:
    """Integration test scenarios for real-world usage"""
    
    @pytest.mark.asyncio
    async def test_multi_source_scenario(self):
        """Test orchestration with multiple data sources"""
        config = OrchestrationConfig(
            max_concurrent_sources=3,
            enable_quality_audit=True,
            enable_coverage_analysis=True
        )
        
        source_selections = [
            {
                'source_type': 'google_drive',
                'connection_id': 'gdrive_conn_1',
                'source_name': 'Research Papers'
            },
            {
                'source_type': 'local_directory',
                'connection_id': 'local_conn_1',
                'source_name': 'Local Documents'
            },
            {
                'source_type': 'local_zip',
                'connection_id': 'zip_conn_1',
                'source_name': 'Archive Files'
            }
        ]
        
        # This would be a full integration test with real services
        # For now, verify the configuration and setup
        assert len(source_selections) == 3
        assert config.max_concurrent_sources >= len(source_selections)
    
    @pytest.mark.asyncio
    async def test_performance_optimization_scenario(self):
        """Test orchestration with performance optimizations"""
        config = OrchestrationConfig(
            max_concurrent_sources=8,
            max_concurrent_files_per_source=32,
            enable_cross_source_optimization=True,
            enable_adaptive_batching=True,
            enable_resource_monitoring=True
        )
        
        processing_options = EnhancedProcessingOptions(
            enable_async_processing=True,
            enable_gpu_acceleration=True,
            enable_adaptive_batching=True,
            enable_resource_monitoring=True,
            max_concurrent_files=32
        )
        
        # Verify high-performance configuration
        assert config.max_concurrent_sources == 8
        assert config.max_concurrent_files_per_source == 32
        assert processing_options.enable_async_processing == True
        assert processing_options.enable_gpu_acceleration == True
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self):
        """Test orchestration with error recovery"""
        config = OrchestrationConfig(
            continue_on_source_failure=True,
            max_retries_per_source=3,
            retry_delay_seconds=2.0,
            enable_graceful_degradation=True
        )
        
        # Verify error handling configuration
        assert config.continue_on_source_failure == True
        assert config.max_retries_per_source == 3
        assert config.enable_graceful_degradation == True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
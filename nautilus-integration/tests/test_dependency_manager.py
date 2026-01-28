"""
Tests for Dependency Manager.

This module tests the comprehensive dependency management system.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from nautilus_integration.services.dependency_manager import (
    DependencyManager,
    DependencyInfo,
    DependencyType,
    DependencyStatus,
    CompatibilityMatrix,
    CompatibilityLevel,
    DependencySnapshot,
    VulnerabilityInfo,
    DependencyHealthReport
)


@pytest.fixture
async def dependency_manager():
    """Create a dependency manager for testing."""
    manager = DependencyManager(
        database_url="sqlite+aiosqlite:///:memory:",
        environments=["development", "testing", "production"],
        check_interval=60,
        vulnerability_sources=["https://test-vuln-source.com"]
    )
    
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def sample_python_dependency():
    """Create a sample Python dependency."""
    return DependencyInfo(
        name="nautilus_trader",
        type=DependencyType.PYTHON,
        current_version="1.190.0",
        latest_version="1.195.0",
        status=DependencyStatus.WARNING,
        environment="development",
        metadata={
            "venv_path": "/path/to/venv",
            "editable": False
        }
    )


@pytest.fixture
def sample_rust_dependency():
    """Create a sample Rust dependency."""
    return DependencyInfo(
        name="serde",
        type=DependencyType.RUST,
        current_version="1.0.190",
        latest_version="1.0.195",
        status=DependencyStatus.WARNING,
        environment="development",
        metadata={
            "cargo_file": "./Cargo.toml",
            "workspace": "execution-core"
        }
    )


@pytest.fixture
def sample_nodejs_dependency():
    """Create a sample Node.js dependency."""
    return DependencyInfo(
        name="react",
        type=DependencyType.NODEJS,
        current_version="18.2.0",
        latest_version="18.3.1",
        status=DependencyStatus.WARNING,
        environment="development",
        metadata={
            "package_file": "./frontend/package.json",
            "is_dev_dependency": False,
            "package_name": "trading-frontend"
        }
    )


@pytest.fixture
def sample_vulnerability():
    """Create a sample vulnerability."""
    return VulnerabilityInfo(
        dependency_name="requests",
        affected_versions="<2.31.0",
        severity="high",
        cve_id="CVE-2023-32681",
        description="Requests library vulnerable to proxy authentication bypass",
        fix_version="2.31.0",
        discovered_at=datetime.utcnow(),
        status="open"
    )


class TestDependencyManager:
    """Test cases for DependencyManager."""
    
    async def test_initialization(self, dependency_manager):
        """Test dependency manager initialization."""
        assert dependency_manager.engine is not None
        assert dependency_manager.session_factory is not None
        assert len(dependency_manager.environments) == 3
        assert "development" in dependency_manager.environments
    
    @patch('subprocess.run')
    async def test_scan_python_dependencies(self, mock_subprocess, dependency_manager):
        """Test scanning Python dependencies."""
        # Mock pip list output
        mock_subprocess.return_value.stdout = json.dumps([
            {"name": "nautilus_trader", "version": "1.190.0"},
            {"name": "pandas", "version": "2.0.3"},
            {"name": "numpy", "version": "1.24.3"}
        ])
        mock_subprocess.return_value.returncode = 0
        
        # Mock PyPI API responses
        with patch.object(dependency_manager, '_get_latest_pypi_version') as mock_pypi:
            mock_pypi.side_effect = ["1.195.0", "2.1.0", "1.25.0"]
            
            dependencies = await dependency_manager._scan_python_dependencies()
        
        assert len(dependencies) == 3
        assert all(dep.type == DependencyType.PYTHON for dep in dependencies)
        
        nautilus_dep = next(dep for dep in dependencies if dep.name == "nautilus_trader")
        assert nautilus_dep.current_version == "1.190.0"
        assert nautilus_dep.latest_version == "1.195.0"
        assert nautilus_dep.status == DependencyStatus.WARNING
    
    async def test_scan_rust_dependencies(self, dependency_manager, tmp_path):
        """Test scanning Rust dependencies."""
        # Create temporary Cargo.toml
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_content = """
[package]
name = "test-package"
version = "0.1.0"

[dependencies]
serde = "1.0.190"
tokio = { version = "1.32.0", features = ["full"] }
"""
        cargo_toml.write_text(cargo_content)
        
        # Mock crates.io API responses
        with patch.object(dependency_manager, '_get_latest_crates_version') as mock_crates:
            mock_crates.side_effect = ["1.0.195", "1.33.0"]
            
            with patch('pathlib.Path.rglob', return_value=[cargo_toml]):
                dependencies = await dependency_manager._scan_rust_dependencies()
        
        assert len(dependencies) == 2
        assert all(dep.type == DependencyType.RUST for dep in dependencies)
        
        serde_dep = next(dep for dep in dependencies if dep.name == "serde")
        assert serde_dep.current_version == "1.0.190"
        assert serde_dep.latest_version == "1.0.195"
    
    async def test_scan_nodejs_dependencies(self, dependency_manager, tmp_path):
        """Test scanning Node.js dependencies."""
        # Create temporary package.json
        package_json = tmp_path / "package.json"
        package_content = {
            "name": "test-frontend",
            "dependencies": {
                "react": "18.2.0",
                "axios": "1.4.0"
            },
            "devDependencies": {
                "typescript": "5.1.6"
            }
        }
        package_json.write_text(json.dumps(package_content))
        
        # Mock npm API responses
        with patch.object(dependency_manager, '_get_latest_npm_version') as mock_npm:
            mock_npm.side_effect = ["18.3.1", "1.5.0", "5.2.2"]
            
            with patch('pathlib.Path.rglob', return_value=[package_json]):
                dependencies = await dependency_manager._scan_nodejs_dependencies()
        
        assert len(dependencies) == 3
        assert all(dep.type == DependencyType.NODEJS for dep in dependencies)
        
        react_dep = next(dep for dep in dependencies if dep.name == "react")
        assert react_dep.current_version == "18.2.0"
        assert react_dep.latest_version == "18.3.1"
        assert react_dep.metadata["is_dev_dependency"] is False
        
        typescript_dep = next(dep for dep in dependencies if dep.name == "typescript")
        assert typescript_dep.metadata["is_dev_dependency"] is True
    
    async def test_scan_dependencies_integration(self, dependency_manager):
        """Test full dependency scanning integration."""
        # Mock all scanning methods
        python_deps = [
            DependencyInfo(name="pandas", type=DependencyType.PYTHON, current_version="2.0.3", status=DependencyStatus.HEALTHY),
            DependencyInfo(name="numpy", type=DependencyType.PYTHON, current_version="1.24.3", status=DependencyStatus.WARNING)
        ]
        
        rust_deps = [
            DependencyInfo(name="serde", type=DependencyType.RUST, current_version="1.0.190", status=DependencyStatus.WARNING)
        ]
        
        nodejs_deps = [
            DependencyInfo(name="react", type=DependencyType.NODEJS, current_version="18.2.0", status=DependencyStatus.HEALTHY)
        ]
        
        with patch.object(dependency_manager, '_scan_python_dependencies', return_value=python_deps), \
             patch.object(dependency_manager, '_scan_rust_dependencies', return_value=rust_deps), \
             patch.object(dependency_manager, '_scan_nodejs_dependencies', return_value=nodejs_deps):
            
            report = await dependency_manager.scan_dependencies("development")
        
        assert report.environment == "development"
        assert report.total_dependencies == 4
        assert report.healthy_count == 2
        assert report.warning_count == 2
        assert report.critical_count == 0
    
    async def test_check_compatibility(self, dependency_manager):
        """Test compatibility checking between dependencies."""
        # Test compatibility check
        matrix = await dependency_manager.check_compatibility(
            "pandas", "2.0.3",
            "numpy", "1.24.3",
            "development"
        )
        
        assert matrix.dependency_a == "pandas"
        assert matrix.version_a == "2.0.3"
        assert matrix.dependency_b == "numpy"
        assert matrix.version_b == "1.24.3"
        assert matrix.compatibility_level in [CompatibilityLevel.COMPATIBLE, CompatibilityLevel.UNKNOWN]
    
    async def test_create_snapshot(self, dependency_manager, sample_python_dependency):
        """Test creating a dependency snapshot."""
        # Mock current dependencies
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=[sample_python_dependency]):
            snapshot = await dependency_manager.create_snapshot(
                name="test_snapshot",
                environment="development",
                user_id="admin",
                notes="Test snapshot creation",
                is_rollback_point=True
            )
        
        assert snapshot.name == "test_snapshot"
        assert snapshot.environment == "development"
        assert snapshot.created_by == "admin"
        assert snapshot.is_rollback_point is True
        assert len(snapshot.dependencies) == 1
        assert snapshot.dependencies[0].name == "nautilus_trader"
    
    async def test_rollback_to_snapshot(self, dependency_manager, sample_python_dependency):
        """Test rolling back to a dependency snapshot."""
        # Create a snapshot first
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=[sample_python_dependency]):
            await dependency_manager.create_snapshot(
                name="rollback_test",
                environment="development",
                is_rollback_point=True
            )
        
        # Mock current dependencies (different from snapshot)
        current_deps = [
            DependencyInfo(
                name="nautilus_trader",
                type=DependencyType.PYTHON,
                current_version="1.200.0",  # Different version
                status=DependencyStatus.HEALTHY
            )
        ]
        
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=current_deps), \
             patch.object(dependency_manager, '_apply_dependency_change') as mock_apply:
            
            result = await dependency_manager.rollback_to_snapshot(
                "rollback_test",
                "development",
                dry_run=True
            )
        
        assert result["snapshot_name"] == "rollback_test"
        assert result["dry_run"] is True
        assert len(result["changes"]) == 1
        
        change = result["changes"][0]
        assert change["action"] == "update"
        assert change["dependency"] == "nautilus_trader"
        assert change["from_version"] == "1.200.0"
        assert change["to_version"] == "1.190.0"
        
        # Verify no actual changes were applied in dry run
        mock_apply.assert_not_called()
    
    async def test_scan_vulnerabilities(self, dependency_manager, sample_python_dependency, sample_vulnerability):
        """Test vulnerability scanning."""
        # Mock current dependencies
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=[sample_python_dependency]):
            # Mock vulnerability scanning
            with patch.object(dependency_manager, '_scan_dependency_vulnerabilities', return_value=[sample_vulnerability]):
                vulnerabilities = await dependency_manager.scan_vulnerabilities("development")
        
        assert len(vulnerabilities) == 1
        assert vulnerabilities[0].dependency_name == "requests"
        assert vulnerabilities[0].severity == "high"
        assert vulnerabilities[0].cve_id == "CVE-2023-32681"
    
    async def test_dependency_status_determination(self, dependency_manager):
        """Test dependency status determination logic."""
        # Test healthy dependency (current == latest)
        status = dependency_manager._determine_status("1.0.0", "1.0.0")
        assert status == DependencyStatus.HEALTHY
        
        # Test warning (minor version behind)
        status = dependency_manager._determine_status("1.0.0", "1.1.0")
        assert status == DependencyStatus.WARNING
        
        # Test critical (major version behind)
        status = dependency_manager._determine_status("1.0.0", "2.0.0")
        assert status == DependencyStatus.CRITICAL
        
        # Test unknown (invalid version)
        status = dependency_manager._determine_status("invalid", "1.0.0")
        assert status == DependencyStatus.UNKNOWN
        
        # Test unknown (no latest version)
        status = dependency_manager._determine_status("1.0.0", None)
        assert status == DependencyStatus.UNKNOWN
    
    async def test_alert_callback_registration(self, dependency_manager):
        """Test registering and receiving alert callbacks."""
        callback_calls = []
        
        async def test_callback(environment, alerts, report):
            callback_calls.append((environment, alerts, report))
        
        # Register callback
        await dependency_manager.register_alert_callback(test_callback)
        
        # Create a report with critical issues
        critical_dep = DependencyInfo(
            name="critical_package",
            type=DependencyType.PYTHON,
            current_version="1.0.0",
            latest_version="3.0.0",
            status=DependencyStatus.CRITICAL
        )
        
        with patch.object(dependency_manager, '_scan_python_dependencies', return_value=[critical_dep]), \
             patch.object(dependency_manager, '_scan_rust_dependencies', return_value=[]), \
             patch.object(dependency_manager, '_scan_nodejs_dependencies', return_value=[]):
            
            await dependency_manager.scan_dependencies("development")
        
        # Verify callback was called
        assert len(callback_calls) == 1
        environment, alerts, report = callback_calls[0]
        assert environment == "development"
        assert "Critical dependencies found: 1" in alerts
        assert report.critical_count == 1
    
    async def test_health_report_generation(self, dependency_manager):
        """Test health report generation."""
        # Create mixed dependency statuses
        dependencies = [
            DependencyInfo(name="healthy1", type=DependencyType.PYTHON, status=DependencyStatus.HEALTHY),
            DependencyInfo(name="healthy2", type=DependencyType.RUST, status=DependencyStatus.HEALTHY),
            DependencyInfo(name="warning1", type=DependencyType.NODEJS, status=DependencyStatus.WARNING),
            DependencyInfo(name="critical1", type=DependencyType.PYTHON, status=DependencyStatus.CRITICAL),
            DependencyInfo(name="unknown1", type=DependencyType.RUST, status=DependencyStatus.UNKNOWN)
        ]
        
        report = await dependency_manager._generate_health_report(dependencies, "development")
        
        assert report.environment == "development"
        assert report.total_dependencies == 5
        assert report.healthy_count == 2
        assert report.warning_count == 1
        assert report.critical_count == 1
        assert report.unknown_count == 1
        assert report.deprecated_count == 0
    
    async def test_compatibility_matrix_caching(self, dependency_manager):
        """Test compatibility matrix caching."""
        # First check should perform actual test
        matrix1 = await dependency_manager.check_compatibility(
            "pandas", "2.0.0", "numpy", "1.24.0", "development"
        )
        
        # Second check should use cache
        matrix2 = await dependency_manager.check_compatibility(
            "pandas", "2.0.0", "numpy", "1.24.0", "development"
        )
        
        assert matrix1.dependency_a == matrix2.dependency_a
        assert matrix1.version_a == matrix2.version_a
        assert matrix1.compatibility_level == matrix2.compatibility_level
    
    async def test_background_tasks(self, dependency_manager):
        """Test background tasks are started and stopped properly."""
        # Verify background tasks are running
        assert len(dependency_manager._background_tasks) > 0
        
        # Shutdown should stop all tasks
        await dependency_manager.shutdown()
        
        # Verify all tasks are cancelled
        for task in dependency_manager._background_tasks:
            assert task.cancelled() or task.done()
    
    async def test_error_handling_in_scanning(self, dependency_manager):
        """Test error handling during dependency scanning."""
        # Mock scanning methods to raise exceptions
        with patch.object(dependency_manager, '_scan_python_dependencies', side_effect=Exception("Python scan error")), \
             patch.object(dependency_manager, '_scan_rust_dependencies', return_value=[]), \
             patch.object(dependency_manager, '_scan_nodejs_dependencies', return_value=[]):
            
            # Should not raise exception, but handle gracefully
            report = await dependency_manager.scan_dependencies("development")
            
            # Should still return a report, even with errors
            assert report.environment == "development"
            assert report.total_dependencies == 0  # No dependencies due to error


class TestDependencyIntegration:
    """Integration tests for dependency manager."""
    
    async def test_nautilus_trader_dependency_scenario(self, dependency_manager):
        """Test a realistic NautilusTrader dependency scenario."""
        # Mock NautilusTrader-related dependencies
        nautilus_deps = [
            DependencyInfo(
                name="nautilus_trader",
                type=DependencyType.PYTHON,
                current_version="1.190.0",
                latest_version="1.195.0",
                status=DependencyStatus.WARNING,
                metadata={"critical_for_trading": True}
            ),
            DependencyInfo(
                name="pandas",
                type=DependencyType.PYTHON,
                current_version="2.0.3",
                latest_version="2.1.0",
                status=DependencyStatus.WARNING
            ),
            DependencyInfo(
                name="numpy",
                type=DependencyType.PYTHON,
                current_version="1.24.3",
                latest_version="1.25.0",
                status=DependencyStatus.WARNING
            )
        ]
        
        rust_deps = [
            DependencyInfo(
                name="serde",
                type=DependencyType.RUST,
                current_version="1.0.190",
                latest_version="1.0.195",
                status=DependencyStatus.WARNING
            ),
            DependencyInfo(
                name="tokio",
                type=DependencyType.RUST,
                current_version="1.32.0",
                latest_version="1.33.0",
                status=DependencyStatus.WARNING
            )
        ]
        
        frontend_deps = [
            DependencyInfo(
                name="react",
                type=DependencyType.NODEJS,
                current_version="18.2.0",
                latest_version="18.3.1",
                status=DependencyStatus.WARNING
            )
        ]
        
        with patch.object(dependency_manager, '_scan_python_dependencies', return_value=nautilus_deps), \
             patch.object(dependency_manager, '_scan_rust_dependencies', return_value=rust_deps), \
             patch.object(dependency_manager, '_scan_nodejs_dependencies', return_value=frontend_deps):
            
            report = await dependency_manager.scan_dependencies("production")
        
        # Verify comprehensive report
        assert report.environment == "production"
        assert report.total_dependencies == 6
        assert report.warning_count == 6
        assert report.critical_count == 0
        
        # Verify all dependency types are represented
        python_deps = [dep for dep in report.dependencies if dep.type == DependencyType.PYTHON]
        rust_deps = [dep for dep in report.dependencies if dep.type == DependencyType.RUST]
        nodejs_deps = [dep for dep in report.dependencies if dep.type == DependencyType.NODEJS]
        
        assert len(python_deps) == 3
        assert len(rust_deps) == 2
        assert len(nodejs_deps) == 1
        
        # Verify NautilusTrader is included
        nautilus_dep = next(dep for dep in python_deps if dep.name == "nautilus_trader")
        assert nautilus_dep.current_version == "1.190.0"
        assert nautilus_dep.metadata.get("critical_for_trading") is True
    
    async def test_vulnerability_management_scenario(self, dependency_manager):
        """Test vulnerability management scenario."""
        # Create dependencies with vulnerabilities
        vulnerable_deps = [
            DependencyInfo(
                name="requests",
                type=DependencyType.PYTHON,
                current_version="2.30.0",
                latest_version="2.31.0",
                status=DependencyStatus.CRITICAL
            ),
            DependencyInfo(
                name="urllib3",
                type=DependencyType.PYTHON,
                current_version="1.26.15",
                latest_version="2.0.4",
                status=DependencyStatus.CRITICAL
            )
        ]
        
        vulnerabilities = [
            VulnerabilityInfo(
                dependency_name="requests",
                affected_versions="<2.31.0",
                severity="high",
                cve_id="CVE-2023-32681",
                description="Proxy authentication bypass vulnerability",
                fix_version="2.31.0"
            ),
            VulnerabilityInfo(
                dependency_name="urllib3",
                affected_versions="<2.0.0",
                severity="critical",
                cve_id="CVE-2023-43804",
                description="Cookie parsing vulnerability",
                fix_version="2.0.4"
            )
        ]
        
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=vulnerable_deps), \
             patch.object(dependency_manager, '_scan_dependency_vulnerabilities', return_value=vulnerabilities):
            
            found_vulns = await dependency_manager.scan_vulnerabilities("production")
        
        assert len(found_vulns) == 2
        
        # Verify high severity vulnerability
        high_vuln = next(v for v in found_vulns if v.severity == "high")
        assert high_vuln.dependency_name == "requests"
        assert high_vuln.cve_id == "CVE-2023-32681"
        
        # Verify critical severity vulnerability
        critical_vuln = next(v for v in found_vulns if v.severity == "critical")
        assert critical_vuln.dependency_name == "urllib3"
        assert critical_vuln.cve_id == "CVE-2023-43804"
    
    async def test_rollback_scenario_with_multiple_changes(self, dependency_manager):
        """Test complex rollback scenario with multiple dependency changes."""
        # Create snapshot with specific versions
        snapshot_deps = [
            DependencyInfo(
                name="nautilus_trader",
                type=DependencyType.PYTHON,
                current_version="1.190.0",
                status=DependencyStatus.HEALTHY
            ),
            DependencyInfo(
                name="pandas",
                type=DependencyType.PYTHON,
                current_version="2.0.3",
                status=DependencyStatus.HEALTHY
            )
        ]
        
        # Create snapshot
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=snapshot_deps):
            await dependency_manager.create_snapshot(
                name="pre_upgrade_snapshot",
                environment="production",
                user_id="admin",
                notes="Before major upgrade",
                is_rollback_point=True
            )
        
        # Simulate current state after upgrade (with issues)
        current_deps = [
            DependencyInfo(
                name="nautilus_trader",
                type=DependencyType.PYTHON,
                current_version="1.195.0",  # Upgraded but has issues
                status=DependencyStatus.CRITICAL
            ),
            DependencyInfo(
                name="pandas",
                type=DependencyType.PYTHON,
                current_version="2.1.0",  # Upgraded
                status=DependencyStatus.WARNING
            ),
            DependencyInfo(
                name="new_dependency",
                type=DependencyType.PYTHON,
                current_version="1.0.0",  # New dependency added
                status=DependencyStatus.HEALTHY
            )
        ]
        
        with patch.object(dependency_manager, '_get_current_dependencies', return_value=current_deps):
            # Perform rollback (dry run first)
            dry_run_result = await dependency_manager.rollback_to_snapshot(
                "pre_upgrade_snapshot",
                "production",
                user_id="admin",
                dry_run=True
            )
        
        # Verify rollback plan
        assert dry_run_result["dry_run"] is True
        assert len(dry_run_result["changes"]) == 3
        
        changes_by_action = {}
        for change in dry_run_result["changes"]:
            action = change["action"]
            if action not in changes_by_action:
                changes_by_action[action] = []
            changes_by_action[action].append(change)
        
        # Should have 2 updates and 1 removal
        assert len(changes_by_action["update"]) == 2
        assert len(changes_by_action["remove"]) == 1
        
        # Verify specific changes
        nautilus_update = next(c for c in changes_by_action["update"] if c["dependency"] == "nautilus_trader")
        assert nautilus_update["from_version"] == "1.195.0"
        assert nautilus_update["to_version"] == "1.190.0"
        
        pandas_update = next(c for c in changes_by_action["update"] if c["dependency"] == "pandas")
        assert pandas_update["from_version"] == "2.1.0"
        assert pandas_update["to_version"] == "2.0.3"
        
        new_dep_removal = changes_by_action["remove"][0]
        assert new_dep_removal["dependency"] == "new_dependency"
        assert new_dep_removal["version"] == "1.0.0"
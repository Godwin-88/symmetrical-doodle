"""
Comprehensive Dependency Management System for NautilusTrader Integration.

This system provides dependency tracking, compatibility matrices, health monitoring,
and rollback capabilities for Python, Rust, and Node.js components.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    import packaging.version
except ImportError:
    packaging = None
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..core.logging import get_logger
from ..core.error_handling_simple import ErrorRecoveryManager, CircuitBreaker

logger = get_logger(__name__)

Base = declarative_base()


class DependencyType(str, Enum):
    """Dependency type enumeration."""
    PYTHON = "python"
    RUST = "rust"
    NODEJS = "nodejs"
    SYSTEM = "system"


class DependencyStatus(str, Enum):
    """Dependency status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEPRECATED = "deprecated"


class CompatibilityLevel(str, Enum):
    """Compatibility level enumeration."""
    COMPATIBLE = "compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class DependencyModel(Base):
    """SQLAlchemy model for dependencies."""
    
    __tablename__ = "dependencies"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    current_version = Column(String)
    required_version = Column(String)
    latest_version = Column(String)
    status = Column(String, default=DependencyStatus.UNKNOWN.value)
    environment = Column(String)
    model_metadata = Column(Text)  # JSON metadata
    last_checked = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CompatibilityMatrixModel(Base):
    """SQLAlchemy model for compatibility matrices."""
    
    __tablename__ = "compatibility_matrices"
    
    id = Column(String, primary_key=True)
    dependency_a = Column(String, nullable=False)
    version_a = Column(String, nullable=False)
    dependency_b = Column(String, nullable=False)
    version_b = Column(String, nullable=False)
    compatibility_level = Column(String, nullable=False)
    environment = Column(String)
    test_results = Column(Text)  # JSON test results
    notes = Column(Text)
    verified_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class DependencySnapshotModel(Base):
    """SQLAlchemy model for dependency snapshots."""
    
    __tablename__ = "dependency_snapshots"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    environment = Column(String, nullable=False)
    dependencies = Column(Text, nullable=False)  # JSON dependency list
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)
    notes = Column(Text)
    is_rollback_point = Column(Boolean, default=False)


class VulnerabilityModel(Base):
    """SQLAlchemy model for vulnerability tracking."""
    
    __tablename__ = "vulnerabilities"
    
    id = Column(String, primary_key=True)
    dependency_name = Column(String, nullable=False)
    affected_versions = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    cve_id = Column(String)
    description = Column(Text)
    fix_version = Column(String)
    discovered_at = Column(DateTime)
    resolved_at = Column(DateTime)
    status = Column(String, default="open")


class DependencyInfo(BaseModel):
    """Dependency information model."""
    
    name: str
    type: DependencyType
    current_version: Optional[str] = None
    required_version: Optional[str] = None
    latest_version: Optional[str] = None
    status: DependencyStatus = DependencyStatus.UNKNOWN
    environment: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_checked: Optional[datetime] = None
    vulnerabilities: List[str] = Field(default_factory=list)


class CompatibilityMatrix(BaseModel):
    """Compatibility matrix model."""
    
    dependency_a: str
    version_a: str
    dependency_b: str
    version_b: str
    compatibility_level: CompatibilityLevel
    environment: str = "default"
    test_results: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    verified_at: Optional[datetime] = None


class DependencySnapshot(BaseModel):
    """Dependency snapshot model."""
    
    name: str
    environment: str
    dependencies: List[DependencyInfo]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    notes: Optional[str] = None
    is_rollback_point: bool = False


class VulnerabilityInfo(BaseModel):
    """Vulnerability information model."""
    
    dependency_name: str
    affected_versions: str
    severity: str
    cve_id: Optional[str] = None
    description: Optional[str] = None
    fix_version: Optional[str] = None
    discovered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    status: str = "open"


class DependencyHealthReport(BaseModel):
    """Dependency health report model."""
    
    environment: str
    total_dependencies: int
    healthy_count: int
    warning_count: int
    critical_count: int
    unknown_count: int
    deprecated_count: int
    vulnerabilities_count: int
    last_updated: datetime
    dependencies: List[DependencyInfo]
    compatibility_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DependencyManager:
    """
    Comprehensive dependency management system.
    
    Provides:
    - Dependency tracking for Python, Rust, and Node.js components
    - Version validation and compatibility checking
    - Compatibility matrices for supported environments
    - Health monitoring with automatic alerts
    - Vulnerability scanning and tracking
    - Rollback capabilities for dependencies and configurations
    """
    
    def __init__(
        self,
        database_url: str,
        environments: List[str] = None,
        check_interval: int = 3600,
        vulnerability_sources: List[str] = None
    ):
        """
        Initialize the dependency manager.
        
        Args:
            database_url: Database connection URL
            environments: List of environments to manage
            check_interval: Health check interval in seconds
            vulnerability_sources: List of vulnerability data sources
        """
        self.database_url = database_url
        self.environments = environments or ["development", "testing", "staging", "production"]
        self.check_interval = check_interval
        self.vulnerability_sources = vulnerability_sources or [
            "https://pyup.io/api/v1/safety/",
            "https://rustsec.org/advisories/",
            "https://registry.npmjs.org/-/npm/v1/security/audits/"
        ]
        
        # Initialize components
        self.engine = None
        self.session_factory = None
        self.error_recovery = ErrorRecoveryManager()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        # Dependency tracking
        self._dependency_cache: Dict[str, Dict[str, DependencyInfo]] = {}
        self._compatibility_cache: Dict[str, List[CompatibilityMatrix]] = {}
        self._vulnerability_cache: Dict[str, List[VulnerabilityInfo]] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Alert callbacks
        self._alert_callbacks: List[callable] = []
    
    async def initialize(self) -> None:
        """Initialize the dependency manager."""
        try:
            # Initialize database
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True
            )
            
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # Load initial data
            await self._load_dependencies()
            await self._load_compatibility_matrices()
            await self._load_vulnerabilities()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Dependency manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dependency manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the dependency manager."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close database connection
            if self.engine:
                await self.engine.dispose()
            
            logger.info("Dependency manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during dependency manager shutdown: {e}")
    
    async def scan_dependencies(self, environment: str = "default") -> DependencyHealthReport:
        """
        Scan and analyze dependencies for an environment.
        
        Args:
            environment: Environment to scan
            
        Returns:
            Dependency health report
        """
        try:
            dependencies = []
            
            # Scan Python dependencies
            python_deps = await self._scan_python_dependencies()
            dependencies.extend(python_deps)
            
            # Scan Rust dependencies
            rust_deps = await self._scan_rust_dependencies()
            dependencies.extend(rust_deps)
            
            # Scan Node.js dependencies
            nodejs_deps = await self._scan_nodejs_dependencies()
            dependencies.extend(nodejs_deps)
            
            # Update database
            await self._update_dependencies(dependencies, environment)
            
            # Generate health report
            report = await self._generate_health_report(dependencies, environment)
            
            # Check for alerts
            await self._check_and_send_alerts(report)
            
            logger.info(f"Scanned {len(dependencies)} dependencies for environment: {environment}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to scan dependencies for {environment}: {e}")
            raise
    
    async def check_compatibility(
        self,
        dependency_a: str,
        version_a: str,
        dependency_b: str,
        version_b: str,
        environment: str = "default"
    ) -> CompatibilityMatrix:
        """
        Check compatibility between two dependencies.
        
        Args:
            dependency_a: First dependency name
            version_a: First dependency version
            dependency_b: Second dependency name
            version_b: Second dependency version
            environment: Environment to check
            
        Returns:
            Compatibility matrix entry
        """
        try:
            # Check cache first
            cache_key = f"{dependency_a}:{version_a}:{dependency_b}:{version_b}:{environment}"
            if cache_key in self._compatibility_cache:
                return self._compatibility_cache[cache_key][0]
            
            # Check database
            async with self.session_factory() as session:
                result = await session.execute(
                    """
                    SELECT * FROM compatibility_matrices 
                    WHERE dependency_a = ? AND version_a = ? 
                    AND dependency_b = ? AND version_b = ? 
                    AND environment = ?
                    """,
                    (dependency_a, version_a, dependency_b, version_b, environment)
                )
                row = result.fetchone()
                
                if row:
                    matrix = CompatibilityMatrix(
                        dependency_a=row.dependency_a,
                        version_a=row.version_a,
                        dependency_b=row.dependency_b,
                        version_b=row.version_b,
                        compatibility_level=CompatibilityLevel(row.compatibility_level),
                        environment=row.environment,
                        test_results=json.loads(row.test_results or "{}"),
                        notes=row.notes,
                        verified_at=row.verified_at
                    )
                    
                    # Update cache
                    self._compatibility_cache[cache_key] = [matrix]
                    return matrix
            
            # Perform compatibility test
            matrix = await self._test_compatibility(
                dependency_a, version_a, dependency_b, version_b, environment
            )
            
            # Store result
            await self._store_compatibility_result(matrix)
            
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to check compatibility: {e}")
            # Return unknown compatibility on error
            return CompatibilityMatrix(
                dependency_a=dependency_a,
                version_a=version_a,
                dependency_b=dependency_b,
                version_b=version_b,
                compatibility_level=CompatibilityLevel.UNKNOWN,
                environment=environment,
                notes=f"Error during compatibility check: {str(e)}"
            )
    
    async def create_snapshot(
        self,
        name: str,
        environment: str,
        user_id: Optional[str] = None,
        notes: Optional[str] = None,
        is_rollback_point: bool = False
    ) -> DependencySnapshot:
        """
        Create a dependency snapshot.
        
        Args:
            name: Snapshot name
            environment: Environment to snapshot
            user_id: User creating the snapshot
            notes: Optional notes
            is_rollback_point: Whether this is a rollback point
            
        Returns:
            Created dependency snapshot
        """
        try:
            # Get current dependencies
            dependencies = await self._get_current_dependencies(environment)
            
            # Create snapshot
            snapshot = DependencySnapshot(
                name=name,
                environment=environment,
                dependencies=dependencies,
                created_by=user_id,
                notes=notes,
                is_rollback_point=is_rollback_point
            )
            
            # Store in database
            async with self.session_factory() as session:
                snapshot_model = DependencySnapshotModel(
                    id=f"{name}:{environment}:{int(time.time())}",
                    name=name,
                    environment=environment,
                    dependencies=json.dumps([dep.model_dump() for dep in dependencies]),
                    created_by=user_id,
                    notes=notes,
                    is_rollback_point=is_rollback_point
                )
                
                session.add(snapshot_model)
                await session.commit()
            
            logger.info(f"Created dependency snapshot: {name} for {environment}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot {name}: {e}")
            raise
    
    async def rollback_to_snapshot(
        self,
        snapshot_name: str,
        environment: str,
        user_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Rollback dependencies to a snapshot.
        
        Args:
            snapshot_name: Name of the snapshot to rollback to
            environment: Environment to rollback
            user_id: User performing the rollback
            dry_run: Whether to perform a dry run
            
        Returns:
            Rollback result information
        """
        try:
            # Get snapshot
            async with self.session_factory() as session:
                result = await session.execute(
                    """
                    SELECT * FROM dependency_snapshots 
                    WHERE name = ? AND environment = ? 
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (snapshot_name, environment)
                )
                row = result.fetchone()
                
                if not row:
                    raise ValueError(f"Snapshot '{snapshot_name}' not found for environment '{environment}'")
                
                snapshot_deps = json.loads(row.dependencies)
            
            # Get current dependencies
            current_deps = await self._get_current_dependencies(environment)
            current_deps_dict = {dep.name: dep for dep in current_deps}
            
            # Calculate changes
            changes = []
            for snap_dep_data in snapshot_deps:
                snap_dep = DependencyInfo(**snap_dep_data)
                current_dep = current_deps_dict.get(snap_dep.name)
                
                if not current_dep:
                    changes.append({
                        "action": "install",
                        "dependency": snap_dep.name,
                        "version": snap_dep.current_version,
                        "type": snap_dep.type
                    })
                elif current_dep.current_version != snap_dep.current_version:
                    changes.append({
                        "action": "update",
                        "dependency": snap_dep.name,
                        "from_version": current_dep.current_version,
                        "to_version": snap_dep.current_version,
                        "type": snap_dep.type
                    })
            
            # Check for dependencies to remove
            snapshot_names = {dep_data["name"] for dep_data in snapshot_deps}
            for current_dep in current_deps:
                if current_dep.name not in snapshot_names:
                    changes.append({
                        "action": "remove",
                        "dependency": current_dep.name,
                        "version": current_dep.current_version,
                        "type": current_dep.type
                    })
            
            result = {
                "snapshot_name": snapshot_name,
                "environment": environment,
                "changes": changes,
                "dry_run": dry_run,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if dry_run:
                logger.info(f"Dry run rollback to {snapshot_name}: {len(changes)} changes")
                return result
            
            # Perform actual rollback
            success_count = 0
            failed_changes = []
            
            for change in changes:
                try:
                    await self._apply_dependency_change(change, environment)
                    success_count += 1
                except Exception as e:
                    failed_changes.append({
                        "change": change,
                        "error": str(e)
                    })
                    logger.error(f"Failed to apply change {change}: {e}")
            
            result.update({
                "success_count": success_count,
                "failed_count": len(failed_changes),
                "failed_changes": failed_changes,
                "completed": len(failed_changes) == 0
            })
            
            logger.info(f"Rollback to {snapshot_name} completed: {success_count}/{len(changes)} successful")
            return result
            
        except Exception as e:
            logger.error(f"Failed to rollback to snapshot {snapshot_name}: {e}")
            raise
    
    async def scan_vulnerabilities(self, environment: str = "default") -> List[VulnerabilityInfo]:
        """
        Scan for vulnerabilities in dependencies.
        
        Args:
            environment: Environment to scan
            
        Returns:
            List of vulnerability information
        """
        try:
            vulnerabilities = []
            
            # Get current dependencies
            dependencies = await self._get_current_dependencies(environment)
            
            # Scan each dependency type
            for dep in dependencies:
                dep_vulns = await self._scan_dependency_vulnerabilities(dep)
                vulnerabilities.extend(dep_vulns)
            
            # Update vulnerability cache and database
            await self._update_vulnerabilities(vulnerabilities, environment)
            
            logger.info(f"Found {len(vulnerabilities)} vulnerabilities in {environment}")
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Failed to scan vulnerabilities for {environment}: {e}")
            raise
    
    async def register_alert_callback(self, callback: callable) -> None:
        """Register a callback for dependency alerts."""
        self._alert_callbacks.append(callback)
    
    async def _scan_python_dependencies(self) -> List[DependencyInfo]:
        """Scan Python dependencies."""
        dependencies = []
        
        try:
            # Check if we're in a virtual environment
            venv_path = sys.prefix
            
            # Use pip list to get installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages = json.loads(result.stdout)
            
            for package in packages:
                # Get latest version from PyPI
                latest_version = await self._get_latest_pypi_version(package["name"])
                
                dep = DependencyInfo(
                    name=package["name"],
                    type=DependencyType.PYTHON,
                    current_version=package["version"],
                    latest_version=latest_version,
                    status=self._determine_status(package["version"], latest_version),
                    metadata={
                        "venv_path": venv_path,
                        "editable": package.get("editable", False)
                    },
                    last_checked=datetime.utcnow()
                )
                
                dependencies.append(dep)
            
        except Exception as e:
            logger.error(f"Failed to scan Python dependencies: {e}")
        
        return dependencies
    
    async def _scan_rust_dependencies(self) -> List[DependencyInfo]:
        """Scan Rust dependencies."""
        dependencies = []
        
        try:
            # Look for Cargo.toml files
            cargo_files = list(Path(".").rglob("Cargo.toml"))
            
            for cargo_file in cargo_files:
                try:
                    # Parse Cargo.toml
                    import toml
                    with open(cargo_file, 'r') as f:
                        cargo_data = toml.load(f)
                    
                    # Get dependencies
                    deps = cargo_data.get("dependencies", {})
                    
                    for dep_name, dep_info in deps.items():
                        if isinstance(dep_info, str):
                            version = dep_info
                        elif isinstance(dep_info, dict):
                            version = dep_info.get("version", "unknown")
                        else:
                            version = "unknown"
                        
                        # Get latest version from crates.io
                        latest_version = await self._get_latest_crates_version(dep_name)
                        
                        dep = DependencyInfo(
                            name=dep_name,
                            type=DependencyType.RUST,
                            current_version=version,
                            latest_version=latest_version,
                            status=self._determine_status(version, latest_version),
                            metadata={
                                "cargo_file": str(cargo_file),
                                "workspace": cargo_data.get("package", {}).get("name", "unknown")
                            },
                            last_checked=datetime.utcnow()
                        )
                        
                        dependencies.append(dep)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {cargo_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to scan Rust dependencies: {e}")
        
        return dependencies
    
    async def _scan_nodejs_dependencies(self) -> List[DependencyInfo]:
        """Scan Node.js dependencies."""
        dependencies = []
        
        try:
            # Look for package.json files
            package_files = list(Path(".").rglob("package.json"))
            
            for package_file in package_files:
                try:
                    with open(package_file, 'r') as f:
                        package_data = json.load(f)
                    
                    # Get dependencies
                    deps = package_data.get("dependencies", {})
                    dev_deps = package_data.get("devDependencies", {})
                    all_deps = {**deps, **dev_deps}
                    
                    for dep_name, version in all_deps.items():
                        # Get latest version from npm
                        latest_version = await self._get_latest_npm_version(dep_name)
                        
                        dep = DependencyInfo(
                            name=dep_name,
                            type=DependencyType.NODEJS,
                            current_version=version,
                            latest_version=latest_version,
                            status=self._determine_status(version, latest_version),
                            metadata={
                                "package_file": str(package_file),
                                "is_dev_dependency": dep_name in dev_deps,
                                "package_name": package_data.get("name", "unknown")
                            },
                            last_checked=datetime.utcnow()
                        )
                        
                        dependencies.append(dep)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {package_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to scan Node.js dependencies: {e}")
        
        return dependencies
    
    async def _get_latest_pypi_version(self, package_name: str) -> Optional[str]:
        """Get latest version from PyPI."""
        if not aiohttp:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://pypi.org/pypi/{package_name}/json") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["info"]["version"]
        except Exception as e:
            logger.debug(f"Failed to get PyPI version for {package_name}: {e}")
        return None
    
    async def _get_latest_crates_version(self, crate_name: str) -> Optional[str]:
        """Get latest version from crates.io."""
        if not aiohttp:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://crates.io/api/v1/crates/{crate_name}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["crate"]["max_version"]
        except Exception as e:
            logger.debug(f"Failed to get crates.io version for {crate_name}: {e}")
        return None
    
    async def _get_latest_npm_version(self, package_name: str) -> Optional[str]:
        """Get latest version from npm."""
        if not aiohttp:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://registry.npmjs.org/{package_name}/latest") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["version"]
        except Exception as e:
            logger.debug(f"Failed to get npm version for {package_name}: {e}")
        return None
    
    def _determine_status(self, current_version: str, latest_version: Optional[str]) -> DependencyStatus:
        """Determine dependency status based on versions."""
        if not latest_version or current_version == "unknown" or not packaging:
            return DependencyStatus.UNKNOWN
        
        try:
            current = packaging.version.parse(current_version.lstrip("^~>=<"))
            latest = packaging.version.parse(latest_version)
            
            if current == latest:
                return DependencyStatus.HEALTHY
            elif current < latest:
                # Check how far behind
                if latest.major > current.major:
                    return DependencyStatus.CRITICAL
                elif latest.minor > current.minor:
                    return DependencyStatus.WARNING
                else:
                    return DependencyStatus.HEALTHY
            else:
                # Current version is newer than latest (pre-release?)
                return DependencyStatus.WARNING
                
        except Exception:
            return DependencyStatus.UNKNOWN
    
    async def _update_dependencies(self, dependencies: List[DependencyInfo], environment: str) -> None:
        """Update dependencies in database."""
        try:
            async with self.session_factory() as session:
                for dep in dependencies:
                    # Check if dependency exists
                    result = await session.execute(
                        "SELECT * FROM dependencies WHERE name = ? AND environment = ?",
                        (dep.name, environment)
                    )
                    existing = result.fetchone()
                    
                    if existing:
                        # Update existing
                        await session.execute(
                            """
                            UPDATE dependencies 
                            SET current_version = ?, latest_version = ?, status = ?, 
                                model_metadata = ?, last_checked = ?, updated_at = ?
                            WHERE name = ? AND environment = ?
                            """,
                            (
                                dep.current_version, dep.latest_version, dep.status.value,
                                json.dumps(dep.metadata), dep.last_checked, datetime.utcnow(),
                                dep.name, environment
                            )
                        )
                    else:
                        # Insert new
                        dep_model = DependencyModel(
                            id=f"{dep.name}:{environment}",
                            name=dep.name,
                            type=dep.type.value,
                            current_version=dep.current_version,
                            latest_version=dep.latest_version,
                            status=dep.status.value,
                            environment=environment,
                            model_metadata=json.dumps(dep.metadata),
                            last_checked=dep.last_checked
                        )
                        session.add(dep_model)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update dependencies in database: {e}")
    
    async def _generate_health_report(
        self,
        dependencies: List[DependencyInfo],
        environment: str
    ) -> DependencyHealthReport:
        """Generate dependency health report."""
        status_counts = {
            DependencyStatus.HEALTHY: 0,
            DependencyStatus.WARNING: 0,
            DependencyStatus.CRITICAL: 0,
            DependencyStatus.UNKNOWN: 0,
            DependencyStatus.DEPRECATED: 0
        }
        
        for dep in dependencies:
            status_counts[dep.status] += 1
        
        # Get vulnerability count
        vulnerabilities = self._vulnerability_cache.get(environment, [])
        vuln_count = len(vulnerabilities)
        
        # Generate compatibility issues
        compatibility_issues = await self._check_compatibility_issues(dependencies, environment)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(dependencies, vulnerabilities)
        
        return DependencyHealthReport(
            environment=environment,
            total_dependencies=len(dependencies),
            healthy_count=status_counts[DependencyStatus.HEALTHY],
            warning_count=status_counts[DependencyStatus.WARNING],
            critical_count=status_counts[DependencyStatus.CRITICAL],
            unknown_count=status_counts[DependencyStatus.UNKNOWN],
            deprecated_count=status_counts[DependencyStatus.DEPRECATED],
            vulnerabilities_count=vuln_count,
            last_updated=datetime.utcnow(),
            dependencies=dependencies,
            compatibility_issues=compatibility_issues,
            recommendations=recommendations
        )
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health monitoring task
        task = asyncio.create_task(self._health_monitoring_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        # Vulnerability scanning task
        task = asyncio.create_task(self._vulnerability_scanning_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                for environment in self.environments:
                    await self.scan_dependencies(environment)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _vulnerability_scanning_task(self) -> None:
        """Background task for vulnerability scanning."""
        while not self._shutdown_event.is_set():
            try:
                for environment in self.environments:
                    await self.scan_vulnerabilities(environment)
                
                await asyncio.sleep(self.check_interval * 2)  # Less frequent than health checks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in vulnerability scanning task: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _load_dependencies(self) -> None:
        """Load dependencies from database."""
        try:
            async with self.session_factory() as session:
                result = await session.execute("SELECT * FROM dependencies")
                for row in result:
                    dep = DependencyInfo(
                        name=row.name,
                        type=DependencyType(row.type),
                        current_version=row.current_version,
                        latest_version=row.latest_version,
                        status=DependencyStatus(row.status),
                        environment=row.environment,
                        metadata=json.loads(row.model_metadata or "{}"),
                        last_checked=row.last_checked
                    )
                    
                    if row.environment not in self._dependency_cache:
                        self._dependency_cache[row.environment] = {}
                    
                    self._dependency_cache[row.environment][row.name] = dep
                    
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
    
    async def _load_compatibility_matrices(self) -> None:
        """Load compatibility matrices from database."""
        try:
            async with self.session_factory() as session:
                result = await session.execute("SELECT * FROM compatibility_matrices")
                for row in result:
                    matrix = CompatibilityMatrix(
                        dependency_a=row.dependency_a,
                        version_a=row.version_a,
                        dependency_b=row.dependency_b,
                        version_b=row.version_b,
                        compatibility_level=CompatibilityLevel(row.compatibility_level),
                        environment=row.environment,
                        test_results=json.loads(row.test_results or "{}"),
                        notes=row.notes,
                        verified_at=row.verified_at
                    )
                    
                    cache_key = f"{row.dependency_a}:{row.version_a}:{row.dependency_b}:{row.version_b}:{row.environment}"
                    self._compatibility_cache[cache_key] = [matrix]
                    
        except Exception as e:
            logger.error(f"Failed to load compatibility matrices: {e}")
    
    async def _load_vulnerabilities(self) -> None:
        """Load vulnerabilities from database."""
        try:
            async with self.session_factory() as session:
                result = await session.execute("SELECT * FROM vulnerabilities WHERE status = 'open'")
                for row in result:
                    vuln = VulnerabilityInfo(
                        dependency_name=row.dependency_name,
                        affected_versions=row.affected_versions,
                        severity=row.severity,
                        cve_id=row.cve_id,
                        description=row.description,
                        fix_version=row.fix_version,
                        discovered_at=row.discovered_at,
                        resolved_at=row.resolved_at,
                        status=row.status
                    )
                    
                    # Group by environment (simplified - in reality would need more complex logic)
                    for env in self.environments:
                        if env not in self._vulnerability_cache:
                            self._vulnerability_cache[env] = []
                        self._vulnerability_cache[env].append(vuln)
                    
        except Exception as e:
            logger.error(f"Failed to load vulnerabilities: {e}")
    
    async def _get_current_dependencies(self, environment: str) -> List[DependencyInfo]:
        """Get current dependencies for an environment."""
        return list(self._dependency_cache.get(environment, {}).values())
    
    async def _check_and_send_alerts(self, report: DependencyHealthReport) -> None:
        """Check for alert conditions and send notifications."""
        alerts = []
        
        # Check for critical dependencies
        if report.critical_count > 0:
            alerts.append(f"Critical dependencies found: {report.critical_count}")
        
        # Check for vulnerabilities
        if report.vulnerabilities_count > 0:
            alerts.append(f"Vulnerabilities found: {report.vulnerabilities_count}")
        
        # Check for compatibility issues
        if report.compatibility_issues:
            alerts.append(f"Compatibility issues: {len(report.compatibility_issues)}")
        
        # Send alerts
        if alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(report.environment, alerts, report)
                    else:
                        callback(report.environment, alerts, report)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    async def _test_compatibility(
        self,
        dependency_a: str,
        version_a: str,
        dependency_b: str,
        version_b: str,
        environment: str
    ) -> CompatibilityMatrix:
        """Test compatibility between two dependencies."""
        # This is a simplified implementation
        # In practice, this would involve creating test environments,
        # installing dependencies, and running compatibility tests
        
        try:
            # For now, use version-based heuristics
            compatibility_level = CompatibilityLevel.COMPATIBLE
            test_results = {"method": "heuristic", "timestamp": datetime.utcnow().isoformat()}
            
            # Simple version compatibility check
            if packaging:
                try:
                    ver_a = packaging.version.parse(version_a.lstrip("^~>=<"))
                    ver_b = packaging.version.parse(version_b.lstrip("^~>=<"))
                    
                    # Major version differences might indicate incompatibility
                    if abs(ver_a.major - ver_b.major) > 1:
                        compatibility_level = CompatibilityLevel.INCOMPATIBLE
                    elif abs(ver_a.major - ver_b.major) == 1:
                        compatibility_level = CompatibilityLevel.PARTIALLY_COMPATIBLE
                        
                except Exception:
                    compatibility_level = CompatibilityLevel.UNKNOWN
            
            return CompatibilityMatrix(
                dependency_a=dependency_a,
                version_a=version_a,
                dependency_b=dependency_b,
                version_b=version_b,
                compatibility_level=compatibility_level,
                environment=environment,
                test_results=test_results,
                verified_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to test compatibility: {e}")
            return CompatibilityMatrix(
                dependency_a=dependency_a,
                version_a=version_a,
                dependency_b=dependency_b,
                version_b=version_b,
                compatibility_level=CompatibilityLevel.UNKNOWN,
                environment=environment,
                notes=f"Test failed: {str(e)}"
            )
    
    async def _store_compatibility_result(self, matrix: CompatibilityMatrix) -> None:
        """Store compatibility test result in database."""
        try:
            async with self.session_factory() as session:
                matrix_model = CompatibilityMatrixModel(
                    id=f"{matrix.dependency_a}:{matrix.version_a}:{matrix.dependency_b}:{matrix.version_b}:{matrix.environment}",
                    dependency_a=matrix.dependency_a,
                    version_a=matrix.version_a,
                    dependency_b=matrix.dependency_b,
                    version_b=matrix.version_b,
                    compatibility_level=matrix.compatibility_level.value,
                    environment=matrix.environment,
                    test_results=json.dumps(matrix.test_results),
                    notes=matrix.notes,
                    verified_at=matrix.verified_at
                )
                
                session.add(matrix_model)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store compatibility result: {e}")
    
    async def _apply_dependency_change(self, change: Dict[str, Any], environment: str) -> None:
        """Apply a dependency change."""
        # This is a simplified implementation
        # In practice, this would involve calling package managers
        # and handling environment-specific installations
        
        action = change["action"]
        dependency = change["dependency"]
        dep_type = change["type"]
        
        if action == "install":
            version = change["version"]
            logger.info(f"Installing {dependency}=={version} ({dep_type})")
            # Implementation would call appropriate package manager
            
        elif action == "update":
            from_version = change["from_version"]
            to_version = change["to_version"]
            logger.info(f"Updating {dependency} from {from_version} to {to_version} ({dep_type})")
            # Implementation would call appropriate package manager
            
        elif action == "remove":
            version = change["version"]
            logger.info(f"Removing {dependency}=={version} ({dep_type})")
            # Implementation would call appropriate package manager
    
    async def _scan_dependency_vulnerabilities(self, dependency: DependencyInfo) -> List[VulnerabilityInfo]:
        """Scan a specific dependency for vulnerabilities."""
        vulnerabilities = []
        
        # This is a simplified implementation
        # In practice, this would query vulnerability databases
        # based on the dependency type and version
        
        try:
            if dependency.type == DependencyType.PYTHON:
                # Query PyUp.io Safety database or similar
                pass
            elif dependency.type == DependencyType.RUST:
                # Query RustSec advisory database
                pass
            elif dependency.type == DependencyType.NODEJS:
                # Query npm audit or similar
                pass
                
        except Exception as e:
            logger.error(f"Failed to scan vulnerabilities for {dependency.name}: {e}")
        
        return vulnerabilities
    
    async def _update_vulnerabilities(self, vulnerabilities: List[VulnerabilityInfo], environment: str) -> None:
        """Update vulnerabilities in cache and database."""
        self._vulnerability_cache[environment] = vulnerabilities
        
        # Store in database
        try:
            async with self.session_factory() as session:
                for vuln in vulnerabilities:
                    # Check if vulnerability exists
                    result = await session.execute(
                        "SELECT * FROM vulnerabilities WHERE dependency_name = ? AND cve_id = ?",
                        (vuln.dependency_name, vuln.cve_id)
                    )
                    existing = result.fetchone()
                    
                    if not existing:
                        vuln_model = VulnerabilityModel(
                            id=f"{vuln.dependency_name}:{vuln.cve_id or 'unknown'}",
                            dependency_name=vuln.dependency_name,
                            affected_versions=vuln.affected_versions,
                            severity=vuln.severity,
                            cve_id=vuln.cve_id,
                            description=vuln.description,
                            fix_version=vuln.fix_version,
                            discovered_at=vuln.discovered_at,
                            status=vuln.status
                        )
                        session.add(vuln_model)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update vulnerabilities in database: {e}")
    
    async def _check_compatibility_issues(
        self,
        dependencies: List[DependencyInfo],
        environment: str
    ) -> List[str]:
        """Check for compatibility issues between dependencies."""
        issues = []
        
        # This is a simplified implementation
        # In practice, this would perform comprehensive compatibility analysis
        
        for i, dep_a in enumerate(dependencies):
            for dep_b in dependencies[i+1:]:
                try:
                    matrix = await self.check_compatibility(
                        dep_a.name, dep_a.current_version or "unknown",
                        dep_b.name, dep_b.current_version or "unknown",
                        environment
                    )
                    
                    if matrix.compatibility_level == CompatibilityLevel.INCOMPATIBLE:
                        issues.append(f"{dep_a.name} {dep_a.current_version} incompatible with {dep_b.name} {dep_b.current_version}")
                    elif matrix.compatibility_level == CompatibilityLevel.PARTIALLY_COMPATIBLE:
                        issues.append(f"{dep_a.name} {dep_a.current_version} partially compatible with {dep_b.name} {dep_b.current_version}")
                        
                except Exception as e:
                    logger.debug(f"Failed to check compatibility between {dep_a.name} and {dep_b.name}: {e}")
        
        return issues
    
    async def _generate_recommendations(
        self,
        dependencies: List[DependencyInfo],
        vulnerabilities: List[VulnerabilityInfo]
    ) -> List[str]:
        """Generate recommendations based on dependency analysis."""
        recommendations = []
        
        # Recommend updates for outdated dependencies
        outdated = [dep for dep in dependencies if dep.status in [DependencyStatus.WARNING, DependencyStatus.CRITICAL]]
        if outdated:
            recommendations.append(f"Update {len(outdated)} outdated dependencies")
        
        # Recommend vulnerability fixes
        if vulnerabilities:
            high_severity = [v for v in vulnerabilities if v.severity in ["high", "critical"]]
            if high_severity:
                recommendations.append(f"Fix {len(high_severity)} high/critical severity vulnerabilities")
        
        # Recommend dependency cleanup
        unknown_deps = [dep for dep in dependencies if dep.status == DependencyStatus.UNKNOWN]
        if unknown_deps:
            recommendations.append(f"Review {len(unknown_deps)} dependencies with unknown status")
        
        return recommendations
"""
Feature Flag and Dependency Management Orchestrator.

This service coordinates feature flags and dependency management to provide
comprehensive system configuration and rollback capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..core.config import NautilusConfig
from ..core.logging import get_logger
from ..core.error_handling_simple import ErrorRecoveryManager, CircuitBreaker
from .feature_flag_service import (
    FeatureFlagService,
    FeatureFlagConfig,
    FeatureFlagEvaluationContext,
    FeatureFlagResult,
    ABTestConfig,
    UserGroup
)
from .dependency_manager import (
    DependencyManager,
    DependencyHealthReport,
    DependencySnapshot,
    VulnerabilityInfo
)

logger = get_logger(__name__)


class FeatureDependencyOrchestrator:
    """
    Orchestrator for feature flags and dependency management.
    
    Provides unified management of feature flags and dependencies with
    coordinated rollback capabilities and health monitoring.
    """
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Nautilus configuration
        """
        self.config = config
        
        # Initialize services
        self.feature_flag_service = None
        self.dependency_manager = None
        
        # Error handling
        self.error_recovery = ErrorRecoveryManager()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Health monitoring
        self._last_health_check = None
        self._health_status = "unknown"
        
        # Rollback coordination
        self._rollback_in_progress = False
        self._rollback_correlation_id = None
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and its services."""
        try:
            # Initialize feature flag service
            if self.config.feature_flags.enabled:
                self.feature_flag_service = FeatureFlagService(
                    database_url=self.config.feature_flags.database_url,
                    redis_url=self.config.feature_flags.redis_url,
                    config_file_path=self.config.feature_flags.config_file_path,
                    cache_ttl=self.config.feature_flags.cache_ttl,
                    enable_audit_logging=self.config.feature_flags.enable_audit_logging
                )
                await self.feature_flag_service.initialize()
                
                # Register for feature flag updates
                await self.feature_flag_service.register_update_callback(
                    self._on_feature_flag_update
                )
            
            # Initialize dependency manager
            if self.config.dependency_management.enabled:
                self.dependency_manager = DependencyManager(
                    database_url=self.config.dependency_management.database_url,
                    environments=self.config.dependency_management.environments,
                    check_interval=self.config.dependency_management.check_interval,
                    vulnerability_sources=self.config.dependency_management.vulnerability_sources
                )
                await self.dependency_manager.initialize()
                
                # Register for dependency alerts
                await self.dependency_manager.register_alert_callback(
                    self._on_dependency_alert
                )
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Feature and dependency orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and its services."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown services
            if self.feature_flag_service:
                await self.feature_flag_service.shutdown()
            
            if self.dependency_manager:
                await self.dependency_manager.shutdown()
            
            logger.info("Feature and dependency orchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")
    
    async def evaluate_feature_flag(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        user_groups: List[UserGroup] = None,
        session_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> FeatureFlagResult:
        """
        Evaluate a feature flag with dependency health consideration.
        
        Args:
            flag_name: Name of the feature flag
            user_id: User ID for evaluation
            user_groups: User groups for targeting
            session_id: Session ID for tracking
            metadata: Additional metadata
            
        Returns:
            Feature flag evaluation result
        """
        if not self.feature_flag_service:
            return FeatureFlagResult(
                flag_name=flag_name,
                enabled=False,
                reason="Feature flag service not enabled",
                evaluation_context=FeatureFlagEvaluationContext(
                    user_id=user_id,
                    user_groups=user_groups or [],
                    session_id=session_id,
                    metadata=metadata or {}
                )
            )
        
        try:
            # Create evaluation context
            context = FeatureFlagEvaluationContext(
                user_id=user_id,
                user_groups=user_groups or [],
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Check if flag depends on healthy dependencies
            if await self._flag_requires_healthy_dependencies(flag_name):
                health_status = await self._get_dependency_health_status()
                if health_status != "healthy":
                    return FeatureFlagResult(
                        flag_name=flag_name,
                        enabled=False,
                        reason=f"Dependencies not healthy: {health_status}",
                        evaluation_context=context
                    )
            
            # Evaluate flag
            result = await self.feature_flag_service.evaluate_flag(flag_name, context)
            
            # Log evaluation for audit
            logger.debug(f"Feature flag {flag_name} evaluated: {result.enabled} ({result.reason})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate feature flag {flag_name}: {e}")
            return FeatureFlagResult(
                flag_name=flag_name,
                enabled=False,
                reason=f"Evaluation error: {str(e)}",
                evaluation_context=FeatureFlagEvaluationContext(
                    user_id=user_id,
                    user_groups=user_groups or [],
                    session_id=session_id,
                    metadata=metadata or {}
                )
            )
    
    async def create_feature_flag(
        self,
        name: str,
        description: Optional[str] = None,
        rollout_percentage: float = 0.0,
        target_groups: List[UserGroup] = None,
        config_data: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        require_healthy_dependencies: bool = False
    ) -> FeatureFlagConfig:
        """
        Create a new feature flag with dependency health requirements.
        
        Args:
            name: Feature flag name
            description: Feature flag description
            rollout_percentage: Initial rollout percentage
            target_groups: Target user groups
            config_data: Configuration data
            user_id: User creating the flag
            require_healthy_dependencies: Whether flag requires healthy dependencies
            
        Returns:
            Created feature flag configuration
        """
        if not self.feature_flag_service:
            raise ValueError("Feature flag service not enabled")
        
        # Add dependency health requirement to config
        config_data = config_data or {}
        if require_healthy_dependencies:
            config_data["require_healthy_dependencies"] = True
        
        # Create flag configuration
        flag_config = FeatureFlagConfig(
            name=name,
            description=description,
            rollout_percentage=rollout_percentage,
            target_groups=target_groups or [],
            config_data=config_data
        )
        
        return await self.feature_flag_service.create_feature_flag(flag_config, user_id)
    
    async def create_ab_test(
        self,
        feature_flag_name: str,
        test_name: str,
        description: Optional[str] = None,
        variant_a_config: Dict[str, Any] = None,
        variant_b_config: Dict[str, Any] = None,
        traffic_split: float = 0.5,
        user_id: Optional[str] = None
    ) -> ABTestConfig:
        """
        Create an A/B test for a feature flag.
        
        Args:
            feature_flag_name: Name of the feature flag
            test_name: Name of the A/B test
            description: Test description
            variant_a_config: Configuration for variant A
            variant_b_config: Configuration for variant B
            traffic_split: Traffic split for variant B
            user_id: User creating the test
            
        Returns:
            Created A/B test configuration
        """
        if not self.feature_flag_service:
            raise ValueError("Feature flag service not enabled")
        
        # Get feature flag
        flags = await self.feature_flag_service.get_all_flags()
        flag = next((f for f in flags if f.name == feature_flag_name), None)
        if not flag:
            raise ValueError(f"Feature flag '{feature_flag_name}' not found")
        
        # Create A/B test configuration
        ab_test_config = ABTestConfig(
            feature_flag_id=flag.id,
            name=test_name,
            description=description,
            variant_a_config=variant_a_config or {},
            variant_b_config=variant_b_config or {},
            traffic_split=traffic_split
        )
        
        return await self.feature_flag_service.create_ab_test(ab_test_config, user_id)
    
    async def get_dependency_health_report(self, environment: str = "default") -> DependencyHealthReport:
        """
        Get dependency health report for an environment.
        
        Args:
            environment: Environment to check
            
        Returns:
            Dependency health report
        """
        if not self.dependency_manager:
            raise ValueError("Dependency management not enabled")
        
        return await self.dependency_manager.scan_dependencies(environment)
    
    async def create_system_snapshot(
        self,
        name: str,
        environment: str,
        user_id: Optional[str] = None,
        notes: Optional[str] = None,
        include_feature_flags: bool = True,
        include_dependencies: bool = True
    ) -> Dict[str, Any]:
        """
        Create a comprehensive system snapshot including feature flags and dependencies.
        
        Args:
            name: Snapshot name
            environment: Environment to snapshot
            user_id: User creating the snapshot
            notes: Optional notes
            include_feature_flags: Whether to include feature flags
            include_dependencies: Whether to include dependencies
            
        Returns:
            Snapshot information
        """
        snapshot_data = {
            "name": name,
            "environment": environment,
            "created_by": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "notes": notes,
            "components": {}
        }
        
        try:
            # Snapshot feature flags
            if include_feature_flags and self.feature_flag_service:
                flags = await self.feature_flag_service.get_all_flags()
                snapshot_data["components"]["feature_flags"] = [
                    flag.model_dump() for flag in flags
                ]
                logger.info(f"Included {len(flags)} feature flags in snapshot")
            
            # Snapshot dependencies
            if include_dependencies and self.dependency_manager:
                dep_snapshot = await self.dependency_manager.create_snapshot(
                    name=f"{name}_dependencies",
                    environment=environment,
                    user_id=user_id,
                    notes=f"Dependencies for system snapshot: {name}",
                    is_rollback_point=True
                )
                snapshot_data["components"]["dependencies"] = dep_snapshot.model_dump()
                logger.info(f"Included {len(dep_snapshot.dependencies)} dependencies in snapshot")
            
            # Store snapshot metadata (in practice, this would go to a database)
            snapshot_file = f"./snapshots/{name}_{environment}_{int(datetime.utcnow().timestamp())}.json"
            import os
            os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            snapshot_data["snapshot_file"] = snapshot_file
            
            logger.info(f"Created system snapshot: {name} for {environment}")
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Failed to create system snapshot {name}: {e}")
            raise
    
    async def rollback_system(
        self,
        snapshot_name: str,
        environment: str,
        user_id: Optional[str] = None,
        rollback_feature_flags: bool = True,
        rollback_dependencies: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Rollback system to a previous snapshot.
        
        Args:
            snapshot_name: Name of the snapshot to rollback to
            environment: Environment to rollback
            user_id: User performing the rollback
            rollback_feature_flags: Whether to rollback feature flags
            rollback_dependencies: Whether to rollback dependencies
            dry_run: Whether to perform a dry run
            
        Returns:
            Rollback result information
        """
        if self._rollback_in_progress:
            raise ValueError("Another rollback operation is already in progress")
        
        try:
            self._rollback_in_progress = True
            self._rollback_correlation_id = f"rollback_{int(datetime.utcnow().timestamp())}"
            
            rollback_result = {
                "snapshot_name": snapshot_name,
                "environment": environment,
                "user_id": user_id,
                "correlation_id": self._rollback_correlation_id,
                "dry_run": dry_run,
                "started_at": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            # Load snapshot data
            snapshot_data = await self._load_snapshot_data(snapshot_name, environment)
            if not snapshot_data:
                raise ValueError(f"Snapshot '{snapshot_name}' not found")
            
            # Rollback dependencies first (they may affect feature flag evaluation)
            if rollback_dependencies and self.dependency_manager and "dependencies" in snapshot_data["components"]:
                try:
                    dep_result = await self.dependency_manager.rollback_to_snapshot(
                        snapshot_name=f"{snapshot_name}_dependencies",
                        environment=environment,
                        user_id=user_id,
                        dry_run=dry_run
                    )
                    rollback_result["components"]["dependencies"] = dep_result
                    logger.info(f"Dependencies rollback result: {dep_result['success_count']}/{len(dep_result['changes'])} successful")
                    
                except Exception as e:
                    logger.error(f"Failed to rollback dependencies: {e}")
                    rollback_result["components"]["dependencies"] = {"error": str(e)}
            
            # Rollback feature flags
            if rollback_feature_flags and self.feature_flag_service and "feature_flags" in snapshot_data["components"]:
                try:
                    flag_result = await self._rollback_feature_flags(
                        snapshot_data["components"]["feature_flags"],
                        user_id,
                        dry_run
                    )
                    rollback_result["components"]["feature_flags"] = flag_result
                    logger.info(f"Feature flags rollback result: {flag_result['success_count']}/{len(flag_result['changes'])} successful")
                    
                except Exception as e:
                    logger.error(f"Failed to rollback feature flags: {e}")
                    rollback_result["components"]["feature_flags"] = {"error": str(e)}
            
            rollback_result["completed_at"] = datetime.utcnow().isoformat()
            rollback_result["success"] = all(
                comp.get("success", True) for comp in rollback_result["components"].values()
                if isinstance(comp, dict) and "error" not in comp
            )
            
            logger.info(f"System rollback to {snapshot_name} completed: {'success' if rollback_result['success'] else 'partial failure'}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Failed to rollback system to {snapshot_name}: {e}")
            raise
        finally:
            self._rollback_in_progress = False
            self._rollback_correlation_id = None
    
    async def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health status information
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "components": {}
        }
        
        component_statuses = []
        
        # Check feature flag service health
        if self.feature_flag_service:
            try:
                flags = await self.feature_flag_service.get_all_flags()
                health_status["components"]["feature_flags"] = {
                    "status": "healthy",
                    "flag_count": len(flags),
                    "last_checked": datetime.utcnow().isoformat()
                }
                component_statuses.append("healthy")
            except Exception as e:
                health_status["components"]["feature_flags"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }
                component_statuses.append("unhealthy")
        
        # Check dependency manager health
        if self.dependency_manager:
            try:
                for env in self.config.dependency_management.environments:
                    report = await self.dependency_manager.scan_dependencies(env)
                    env_status = "healthy"
                    if report.critical_count > 0 or report.vulnerabilities_count > 0:
                        env_status = "critical"
                    elif report.warning_count > 0:
                        env_status = "warning"
                    
                    health_status["components"][f"dependencies_{env}"] = {
                        "status": env_status,
                        "total_dependencies": report.total_dependencies,
                        "critical_count": report.critical_count,
                        "warning_count": report.warning_count,
                        "vulnerabilities_count": report.vulnerabilities_count,
                        "last_checked": report.last_updated.isoformat()
                    }
                    component_statuses.append(env_status)
                    
            except Exception as e:
                health_status["components"]["dependencies"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }
                component_statuses.append("unhealthy")
        
        # Determine overall status
        if "unhealthy" in component_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "critical" in component_statuses:
            health_status["overall_status"] = "critical"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        elif component_statuses:
            health_status["overall_status"] = "healthy"
        
        self._last_health_check = datetime.utcnow()
        self._health_status = health_status["overall_status"]
        
        return health_status
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health monitoring task
        task = asyncio.create_task(self._health_monitoring_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self.get_system_health_status()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(60)
    
    async def _on_feature_flag_update(
        self,
        flag_name: str,
        config: Optional[FeatureFlagConfig]
    ) -> None:
        """Handle feature flag updates."""
        try:
            if config:
                logger.info(f"Feature flag updated: {flag_name}")
                
                # Check if flag affects system dependencies
                if config.config_data.get("affects_dependencies"):
                    logger.info(f"Feature flag {flag_name} affects dependencies, triggering health check")
                    await self.get_system_health_status()
            else:
                logger.info(f"Feature flag deleted: {flag_name}")
                
        except Exception as e:
            logger.error(f"Error handling feature flag update for {flag_name}: {e}")
    
    async def _on_dependency_alert(
        self,
        environment: str,
        alerts: List[str],
        report: DependencyHealthReport
    ) -> None:
        """Handle dependency alerts."""
        try:
            logger.warning(f"Dependency alerts for {environment}: {alerts}")
            
            # Check if any feature flags should be disabled due to dependency issues
            if self.feature_flag_service and report.critical_count > 0:
                flags = await self.feature_flag_service.get_all_flags()
                for flag in flags:
                    if flag.config_data.get("require_healthy_dependencies"):
                        logger.warning(f"Feature flag {flag.name} requires healthy dependencies but critical issues found")
                        # In practice, you might want to automatically disable such flags
                        
        except Exception as e:
            logger.error(f"Error handling dependency alert: {e}")
    
    async def _flag_requires_healthy_dependencies(self, flag_name: str) -> bool:
        """Check if a flag requires healthy dependencies."""
        if not self.feature_flag_service:
            return False
        
        try:
            flags = await self.feature_flag_service.get_all_flags()
            flag = next((f for f in flags if f.name == flag_name), None)
            if flag:
                return flag.config_data.get("require_healthy_dependencies", False)
        except Exception as e:
            logger.error(f"Error checking dependency requirement for flag {flag_name}: {e}")
        
        return False
    
    async def _get_dependency_health_status(self) -> str:
        """Get overall dependency health status."""
        if not self.dependency_manager:
            return "unknown"
        
        try:
            # Check all environments
            for env in self.config.dependency_management.environments:
                report = await self.dependency_manager.scan_dependencies(env)
                if report.critical_count > 0:
                    return "critical"
                elif report.warning_count > 0:
                    return "warning"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Error getting dependency health status: {e}")
            return "unknown"
    
    async def _load_snapshot_data(self, snapshot_name: str, environment: str) -> Optional[Dict[str, Any]]:
        """Load snapshot data from storage."""
        try:
            # In practice, this would query a database
            # For now, look for snapshot files
            import glob
            pattern = f"./snapshots/{snapshot_name}_{environment}_*.json"
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            # Get the most recent snapshot
            latest_file = max(files)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load snapshot data for {snapshot_name}: {e}")
            return None
    
    async def _rollback_feature_flags(
        self,
        snapshot_flags: List[Dict[str, Any]],
        user_id: Optional[str],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Rollback feature flags to snapshot state."""
        if not self.feature_flag_service:
            return {"error": "Feature flag service not available"}
        
        try:
            # Get current flags
            current_flags = await self.feature_flag_service.get_all_flags()
            current_flags_dict = {flag.name: flag for flag in current_flags}
            
            changes = []
            
            # Check for flags to update or create
            for snap_flag_data in snapshot_flags:
                snap_flag = FeatureFlagConfig(**snap_flag_data)
                current_flag = current_flags_dict.get(snap_flag.name)
                
                if not current_flag:
                    changes.append({
                        "action": "create",
                        "flag_name": snap_flag.name,
                        "config": snap_flag_data
                    })
                elif current_flag.model_dump() != snap_flag_data:
                    changes.append({
                        "action": "update",
                        "flag_name": snap_flag.name,
                        "old_config": current_flag.model_dump(),
                        "new_config": snap_flag_data
                    })
            
            # Check for flags to delete
            snapshot_names = {flag_data["name"] for flag_data in snapshot_flags}
            for current_flag in current_flags:
                if current_flag.name not in snapshot_names:
                    changes.append({
                        "action": "delete",
                        "flag_name": current_flag.name,
                        "config": current_flag.model_dump()
                    })
            
            result = {
                "changes": changes,
                "dry_run": dry_run,
                "success_count": 0,
                "failed_changes": []
            }
            
            if dry_run:
                return result
            
            # Apply changes
            for change in changes:
                try:
                    if change["action"] == "create":
                        config = FeatureFlagConfig(**change["config"])
                        await self.feature_flag_service.create_feature_flag(config, user_id)
                    elif change["action"] == "update":
                        await self.feature_flag_service.update_feature_flag(
                            change["flag_name"],
                            change["new_config"],
                            user_id
                        )
                    elif change["action"] == "delete":
                        await self.feature_flag_service.delete_feature_flag(
                            change["flag_name"],
                            user_id
                        )
                    
                    result["success_count"] += 1
                    
                except Exception as e:
                    result["failed_changes"].append({
                        "change": change,
                        "error": str(e)
                    })
                    logger.error(f"Failed to apply feature flag change {change}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to rollback feature flags: {e}")
            return {"error": str(e)}
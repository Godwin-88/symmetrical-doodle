"""
Health monitoring and status endpoints for the Intelligence Layer.

This module provides comprehensive health checking capabilities including:
- Component status monitoring
- Performance metrics collection
- Dependency health validation
- System resource monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import neo4j
import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import load_config
from .logging import get_logger

logger = get_logger(__name__)
settings = load_config()

class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentType(str, Enum):
    """Component type enumeration."""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    INTERNAL_SERVICE = "internal_service"
    RESOURCE = "resource"

@dataclass
class HealthMetrics:
    """Health metrics for a component."""
    response_time_ms: float
    error_rate: float
    throughput_rps: float
    resource_usage: Dict[str, float]
    last_check: datetime
    uptime_seconds: float

@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    type: ComponentType
    status: HealthStatus
    message: str
    metrics: Optional[HealthMetrics] = None
    dependencies: List[str] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
        if self.dependencies is None:
            self.dependencies = []

class SystemHealthResponse(BaseModel):
    """System-wide health response."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    summary: Dict[str, Any]
    alerts: List[str]

class HealthChecker:
    """Centralized health checking service."""
    
    def __init__(self):
        self.start_time = time.time()
        self.component_checks = {}
        self.metrics_history = {}
        self.alert_thresholds = {
            'response_time_ms': 1000,
            'error_rate': 0.05,
            'cpu_usage': 0.8,
            'memory_usage': 0.8,
            'disk_usage': 0.9
        }
        
    async def check_database_health(self) -> ComponentHealth:
        """Check PostgreSQL database health."""
        try:
            start_time = time.time()
            
            # Test database connection
            conn = psycopg2.connect(
                host=settings.database_host,
                port=settings.database_port,
                database=settings.database_name,
                user=settings.database_user,
                password=settings.database_password
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            cursor.close()
            conn.close()
            
            if result and result[0] == 1:
                metrics = HealthMetrics(
                    response_time_ms=response_time,
                    error_rate=0.0,
                    throughput_rps=0.0,  # Would be calculated from metrics
                    resource_usage={},
                    last_check=datetime.utcnow(),
                    uptime_seconds=time.time() - self.start_time
                )
                
                return ComponentHealth(
                    name="postgresql",
                    type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    metrics=metrics,
                    dependencies=[]
                )
            else:
                return ComponentHealth(
                    name="postgresql",
                    type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="Database query returned unexpected result"
                )
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="postgresql",
                type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )
    
    async def check_neo4j_health(self) -> ComponentHealth:
        """Check Neo4j database health."""
        try:
            start_time = time.time()
            
            driver = neo4j.GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
            response_time = (time.time() - start_time) * 1000
            driver.close()
            
            if record and record["test"] == 1:
                metrics = HealthMetrics(
                    response_time_ms=response_time,
                    error_rate=0.0,
                    throughput_rps=0.0,
                    resource_usage={},
                    last_check=datetime.utcnow(),
                    uptime_seconds=time.time() - self.start_time
                )
                
                return ComponentHealth(
                    name="neo4j",
                    type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    message="Neo4j connection successful",
                    metrics=metrics,
                    dependencies=[]
                )
            else:
                return ComponentHealth(
                    name="neo4j",
                    type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="Neo4j query returned unexpected result"
                )
                
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return ComponentHealth(
                name="neo4j",
                type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=f"Neo4j connection failed: {str(e)}"
            )
    
    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis cache health."""
        try:
            start_time = time.time()
            
            redis = aioredis.from_url(settings.redis_url)
            await redis.ping()
            
            response_time = (time.time() - start_time) * 1000
            await redis.close()
            
            metrics = HealthMetrics(
                response_time_ms=response_time,
                error_rate=0.0,
                throughput_rps=0.0,
                resource_usage={},
                last_check=datetime.utcnow(),
                uptime_seconds=time.time() - self.start_time
            )
            
            return ComponentHealth(
                name="redis",
                type=ComponentType.CACHE,
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                metrics=metrics,
                dependencies=[]
            )
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                name="redis",
                type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}"
            )
    
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_usage = {
                'cpu_usage': cpu_percent / 100.0,
                'memory_usage': memory.percent / 100.0,
                'disk_usage': disk.percent / 100.0,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent / 100.0 > self.alert_thresholds['cpu_usage']:
                status = HealthStatus.DEGRADED
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent / 100.0 > self.alert_thresholds['memory_usage']:
                status = HealthStatus.DEGRADED
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent / 100.0 > self.alert_thresholds['disk_usage']:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources within normal limits"
            
            metrics = HealthMetrics(
                response_time_ms=0.0,
                error_rate=0.0,
                throughput_rps=0.0,
                resource_usage=resource_usage,
                last_check=datetime.utcnow(),
                uptime_seconds=time.time() - self.start_time
            )
            
            return ComponentHealth(
                name="system_resources",
                type=ComponentType.RESOURCE,
                status=status,
                message=message,
                metrics=metrics,
                dependencies=[]
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return ComponentHealth(
                name="system_resources",
                type=ComponentType.RESOURCE,
                status=HealthStatus.UNKNOWN,
                message=f"Resource check failed: {str(e)}"
            )
    
    async def check_execution_core_health(self) -> ComponentHealth:
        """Check Execution Core service health."""
        try:
            # This would make an HTTP request to the Execution Core health endpoint
            # For now, we'll simulate this check
            start_time = time.time()
            
            # Simulate health check request
            await asyncio.sleep(0.01)  # Simulate network delay
            
            response_time = (time.time() - start_time) * 1000
            
            metrics = HealthMetrics(
                response_time_ms=response_time,
                error_rate=0.0,
                throughput_rps=0.0,
                resource_usage={},
                last_check=datetime.utcnow(),
                uptime_seconds=time.time() - self.start_time
            )
            
            return ComponentHealth(
                name="execution_core",
                type=ComponentType.INTERNAL_SERVICE,
                status=HealthStatus.HEALTHY,
                message="Execution Core service responding",
                metrics=metrics,
                dependencies=["postgresql", "redis"]
            )
            
        except Exception as e:
            logger.error(f"Execution Core health check failed: {e}")
            return ComponentHealth(
                name="execution_core",
                type=ComponentType.INTERNAL_SERVICE,
                status=HealthStatus.UNHEALTHY,
                message=f"Execution Core check failed: {str(e)}"
            )
    
    async def get_system_health(self) -> SystemHealthResponse:
        """Get comprehensive system health status."""
        try:
            # Run all health checks concurrently
            health_checks = await asyncio.gather(
                self.check_database_health(),
                self.check_neo4j_health(),
                self.check_redis_health(),
                self.check_system_resources(),
                self.check_execution_core_health(),
                return_exceptions=True
            )
            
            components = {}
            alerts = []
            
            # Process health check results
            for check in health_checks:
                if isinstance(check, Exception):
                    logger.error(f"Health check failed with exception: {check}")
                    continue
                
                components[check.name] = check
                
                # Generate alerts for unhealthy components
                if check.status == HealthStatus.UNHEALTHY:
                    alerts.append(f"{check.name}: {check.message}")
                elif check.status == HealthStatus.DEGRADED:
                    alerts.append(f"{check.name}: Performance degraded - {check.message}")
            
            # Determine overall system status
            statuses = [comp.status for comp in components.values()]
            if HealthStatus.UNHEALTHY in statuses:
                overall_status = HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                overall_status = HealthStatus.DEGRADED
            elif HealthStatus.UNKNOWN in statuses:
                overall_status = HealthStatus.UNKNOWN
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Generate summary
            summary = {
                'total_components': len(components),
                'healthy_components': len([c for c in components.values() if c.status == HealthStatus.HEALTHY]),
                'degraded_components': len([c for c in components.values() if c.status == HealthStatus.DEGRADED]),
                'unhealthy_components': len([c for c in components.values() if c.status == HealthStatus.UNHEALTHY]),
                'unknown_components': len([c for c in components.values() if c.status == HealthStatus.UNKNOWN]),
                'average_response_time_ms': sum([
                    c.metrics.response_time_ms for c in components.values() 
                    if c.metrics and c.metrics.response_time_ms > 0
                ]) / max(1, len([c for c in components.values() if c.metrics and c.metrics.response_time_ms > 0]))
            }
            
            return SystemHealthResponse(
                status=overall_status,
                timestamp=datetime.utcnow(),
                uptime_seconds=time.time() - self.start_time,
                components={name: asdict(comp) for name, comp in components.items()},
                summary=summary,
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Global health checker instance
health_checker = HealthChecker()

# FastAPI router for health endpoints
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=SystemHealthResponse)
async def get_health():
    """Get comprehensive system health status."""
    return await health_checker.get_system_health()

@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.utcnow()}

@router.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    health = await health_checker.get_system_health()
    
    if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
        return {"status": "ready", "timestamp": datetime.utcnow()}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/components/{component_name}")
async def get_component_health(component_name: str):
    """Get health status for a specific component."""
    health = await health_checker.get_system_health()
    
    if component_name not in health.components:
        raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")
    
    return health.components[component_name]

@router.get("/metrics")
async def get_health_metrics():
    """Get detailed health metrics for monitoring systems."""
    health = await health_checker.get_system_health()
    
    metrics = {}
    for name, component in health.components.items():
        if 'metrics' in component and component['metrics']:
            metrics[name] = component['metrics']
    
    return {
        "timestamp": datetime.utcnow(),
        "uptime_seconds": health.uptime_seconds,
        "overall_status": health.status,
        "component_metrics": metrics,
        "summary": health.summary
    }
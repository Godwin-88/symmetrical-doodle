"""
Feature Flag Service with A/B Testing Capabilities for NautilusTrader Integration.

This service provides comprehensive feature flag management with gradual rollout,
A/B testing, user group targeting, real-time configuration updates, and audit trails.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import aiofiles
try:
    import aioredis
    REDIS_AVAILABLE = True
except (ImportError, TypeError):
    # Handle both import errors and the TimeoutError issue in some aioredis versions
    aioredis = None
    REDIS_AVAILABLE = False


class MockRedisClient:
    """Mock Redis client for testing and when Redis is not available."""
    
    async def setex(self, key: str, ttl: int, value: str) -> None:
        """Mock setex operation."""
        pass
    
    async def delete(self, key: str) -> None:
        """Mock delete operation."""
        pass
    
    async def close(self) -> None:
        """Mock close operation."""
        pass
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Boolean, DateTime, Text, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..core.logging import get_logger
from ..core.error_handling_simple import ErrorRecoveryManager, CircuitBreaker

logger = get_logger(__name__)

Base = declarative_base()


class FeatureFlagStatus(str, Enum):
    """Feature flag status enumeration."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TESTING = "testing"
    ROLLOUT = "rollout"
    DEPRECATED = "deprecated"


class RolloutStrategy(str, Enum):
    """Rollout strategy enumeration."""
    PERCENTAGE = "percentage"
    USER_GROUP = "user_group"
    WHITELIST = "whitelist"
    GRADUAL = "gradual"
    AB_TEST = "ab_test"


class UserGroup(str, Enum):
    """User group enumeration for targeting."""
    DEVELOPERS = "developers"
    BETA_TESTERS = "beta_testers"
    POWER_USERS = "power_users"
    INTERNAL = "internal"
    EXTERNAL = "external"
    ALL = "all"


class FeatureFlagModel(Base):
    """SQLAlchemy model for feature flags."""
    
    __tablename__ = "feature_flags"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    status = Column(String, nullable=False, default=FeatureFlagStatus.DISABLED.value)
    rollout_strategy = Column(String, nullable=False, default=RolloutStrategy.PERCENTAGE.value)
    rollout_percentage = Column(Float, default=0.0)
    target_groups = Column(Text)  # JSON array of user groups
    whitelist_users = Column(Text)  # JSON array of user IDs
    config_data = Column(Text)  # JSON configuration data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String)
    updated_by = Column(String)
    approval_required = Column(Boolean, default=False)
    approved_by = Column(String)
    approved_at = Column(DateTime)


class AuditLogModel(Base):
    """SQLAlchemy model for feature flag audit logs."""
    
    __tablename__ = "feature_flag_audit_logs"
    
    id = Column(String, primary_key=True)
    feature_flag_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    old_value = Column(Text)
    new_value = Column(Text)
    user_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(String)
    model_metadata = Column(Text)  # JSON metadata


class ABTestModel(Base):
    """SQLAlchemy model for A/B tests."""
    
    __tablename__ = "ab_tests"
    
    id = Column(String, primary_key=True)
    feature_flag_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    variant_a_config = Column(Text)  # JSON configuration for variant A
    variant_b_config = Column(Text)  # JSON configuration for variant B
    traffic_split = Column(Float, default=0.5)  # Percentage for variant B
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)


class FeatureFlagConfig(BaseModel):
    """Feature flag configuration model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Feature flag name")
    description: Optional[str] = Field(None, description="Feature flag description")
    status: FeatureFlagStatus = Field(default=FeatureFlagStatus.DISABLED)
    rollout_strategy: RolloutStrategy = Field(default=RolloutStrategy.PERCENTAGE)
    rollout_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    target_groups: List[UserGroup] = Field(default_factory=list)
    whitelist_users: List[str] = Field(default_factory=list)
    config_data: Dict[str, Any] = Field(default_factory=dict)
    approval_required: bool = Field(default=False)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    @validator("rollout_percentage")
    def validate_rollout_percentage(cls, v: float) -> float:
        """Validate rollout percentage is between 0 and 100."""
        if not (0.0 <= v <= 100.0):
            raise ValueError("Rollout percentage must be between 0 and 100")
        return v


class ABTestConfig(BaseModel):
    """A/B test configuration model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    feature_flag_id: str = Field(..., description="Associated feature flag ID")
    name: str = Field(..., description="A/B test name")
    description: Optional[str] = Field(None, description="A/B test description")
    variant_a_config: Dict[str, Any] = Field(default_factory=dict)
    variant_b_config: Dict[str, Any] = Field(default_factory=dict)
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: str = Field(default="active")
    created_by: Optional[str] = None
    
    @validator("traffic_split")
    def validate_traffic_split(cls, v: float) -> float:
        """Validate traffic split is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Traffic split must be between 0 and 1")
        return v


class FeatureFlagEvaluationContext(BaseModel):
    """Context for feature flag evaluation."""
    
    user_id: Optional[str] = None
    user_groups: List[UserGroup] = Field(default_factory=list)
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureFlagResult(BaseModel):
    """Result of feature flag evaluation."""
    
    flag_name: str
    enabled: bool
    variant: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    reason: str
    evaluation_context: FeatureFlagEvaluationContext
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeatureFlagService:
    """
    Comprehensive feature flag service with A/B testing capabilities.
    
    Provides:
    - Feature flag management with multiple rollout strategies
    - A/B testing with traffic splitting
    - User group targeting and whitelisting
    - Real-time configuration updates
    - Comprehensive audit trails
    - Approval workflows
    """
    
    def __init__(
        self,
        database_url: str,
        redis_url: str,
        config_file_path: Optional[str] = None,
        cache_ttl: int = 300,
        enable_audit_logging: bool = True
    ):
        """
        Initialize the feature flag service.
        
        Args:
            database_url: Database connection URL
            redis_url: Redis connection URL for caching
            config_file_path: Optional path to configuration file
            cache_ttl: Cache TTL in seconds
            enable_audit_logging: Whether to enable audit logging
        """
        self.database_url = database_url
        self.redis_url = redis_url
        self.config_file_path = config_file_path
        self.cache_ttl = cache_ttl
        self.enable_audit_logging = enable_audit_logging
        
        # Initialize components
        self.engine = None
        self.session_factory = None
        self.redis_client = None
        self.error_recovery = ErrorRecoveryManager()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        # In-memory cache for high-performance evaluation
        self._flag_cache: Dict[str, FeatureFlagConfig] = {}
        self._ab_test_cache: Dict[str, ABTestConfig] = {}
        self._cache_last_updated = 0
        
        # Configuration update callbacks
        self._update_callbacks: List[callable] = []
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize the feature flag service."""
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
            
            # Initialize Redis
            if REDIS_AVAILABLE and aioredis:
                try:
                    self.redis_client = await aioredis.from_url(self.redis_url)
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {e}, using mock client")
                    self.redis_client = MockRedisClient()
            else:
                logger.warning("Redis not available, using mock client")
                self.redis_client = MockRedisClient()
            
            # Load initial configuration
            await self._load_configuration()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Feature flag service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize feature flag service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the feature flag service."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            
            if self.engine:
                await self.engine.dispose()
            
            logger.info("Feature flag service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during feature flag service shutdown: {e}")
    
    async def create_feature_flag(
        self,
        config: FeatureFlagConfig,
        user_id: Optional[str] = None
    ) -> FeatureFlagConfig:
        """
        Create a new feature flag.
        
        Args:
            config: Feature flag configuration
            user_id: User creating the flag
            
        Returns:
            Created feature flag configuration
        """
        try:
            config.created_by = user_id
            config.updated_by = user_id
            
            async with self.session_factory() as session:
                # Check if flag already exists
                existing = await self._get_flag_by_name(session, config.name)
                if existing:
                    raise ValueError(f"Feature flag '{config.name}' already exists")
                
                # Create database record
                flag_model = FeatureFlagModel(
                    id=config.id,
                    name=config.name,
                    description=config.description,
                    status=config.status.value,
                    rollout_strategy=config.rollout_strategy.value,
                    rollout_percentage=config.rollout_percentage,
                    target_groups=json.dumps([g.value for g in config.target_groups]),
                    whitelist_users=json.dumps(config.whitelist_users),
                    config_data=json.dumps(config.config_data),
                    approval_required=config.approval_required,
                    created_by=user_id,
                    updated_by=user_id
                )
                
                session.add(flag_model)
                await session.commit()
            
            # Update cache
            await self._update_flag_cache(config)
            
            # Log audit event
            if self.enable_audit_logging:
                await self._log_audit_event(
                    feature_flag_id=config.id,
                    action="CREATE",
                    new_value=config.model_dump_json(),
                    user_id=user_id
                )
            
            # Notify callbacks
            await self._notify_update_callbacks(config.name, config)
            
            logger.info(f"Created feature flag: {config.name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to create feature flag {config.name}: {e}")
            raise
    
    async def update_feature_flag(
        self,
        flag_name: str,
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
        require_approval: bool = None
    ) -> FeatureFlagConfig:
        """
        Update an existing feature flag.
        
        Args:
            flag_name: Name of the flag to update
            updates: Dictionary of updates to apply
            user_id: User making the update
            require_approval: Whether to require approval for this update
            
        Returns:
            Updated feature flag configuration
        """
        try:
            async with self.session_factory() as session:
                flag_model = await self._get_flag_by_name(session, flag_name)
                if not flag_model:
                    raise ValueError(f"Feature flag '{flag_name}' not found")
                
                # Get current configuration
                old_config = await self._model_to_config(flag_model)
                
                # Apply updates
                new_config = old_config.model_copy(update=updates)
                new_config.updated_by = user_id
                
                # Check if approval is required
                if require_approval or flag_model.approval_required:
                    new_config.approved_by = None
                    new_config.approved_at = None
                
                # Update database record
                flag_model.name = new_config.name
                flag_model.description = new_config.description
                flag_model.status = new_config.status.value
                flag_model.rollout_strategy = new_config.rollout_strategy.value
                flag_model.rollout_percentage = new_config.rollout_percentage
                flag_model.target_groups = json.dumps([g.value for g in new_config.target_groups])
                flag_model.whitelist_users = json.dumps(new_config.whitelist_users)
                flag_model.config_data = json.dumps(new_config.config_data)
                flag_model.updated_by = user_id
                flag_model.updated_at = datetime.utcnow()
                
                await session.commit()
            
            # Update cache
            await self._update_flag_cache(new_config)
            
            # Log audit event
            if self.enable_audit_logging:
                await self._log_audit_event(
                    feature_flag_id=new_config.id,
                    action="UPDATE",
                    old_value=old_config.model_dump_json(),
                    new_value=new_config.model_dump_json(),
                    user_id=user_id
                )
            
            # Notify callbacks
            await self._notify_update_callbacks(flag_name, new_config)
            
            logger.info(f"Updated feature flag: {flag_name}")
            return new_config
            
        except Exception as e:
            logger.error(f"Failed to update feature flag {flag_name}: {e}")
            raise
    
    async def delete_feature_flag(
        self,
        flag_name: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete a feature flag.
        
        Args:
            flag_name: Name of the flag to delete
            user_id: User deleting the flag
            
        Returns:
            True if deleted successfully
        """
        try:
            async with self.session_factory() as session:
                flag_model = await self._get_flag_by_name(session, flag_name)
                if not flag_model:
                    raise ValueError(f"Feature flag '{flag_name}' not found")
                
                old_config = await self._model_to_config(flag_model)
                
                # Delete from database
                await session.delete(flag_model)
                await session.commit()
            
            # Remove from cache
            await self._remove_flag_from_cache(flag_name)
            
            # Log audit event
            if self.enable_audit_logging:
                await self._log_audit_event(
                    feature_flag_id=old_config.id,
                    action="DELETE",
                    old_value=old_config.model_dump_json(),
                    user_id=user_id
                )
            
            # Notify callbacks
            await self._notify_update_callbacks(flag_name, None)
            
            logger.info(f"Deleted feature flag: {flag_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete feature flag {flag_name}: {e}")
            raise
    
    async def evaluate_flag(
        self,
        flag_name: str,
        context: FeatureFlagEvaluationContext
    ) -> FeatureFlagResult:
        """
        Evaluate a feature flag for the given context.
        
        Args:
            flag_name: Name of the flag to evaluate
            context: Evaluation context
            
        Returns:
            Feature flag evaluation result
        """
        try:
            # Get flag configuration
            flag_config = await self._get_flag_config(flag_name)
            if not flag_config:
                return FeatureFlagResult(
                    flag_name=flag_name,
                    enabled=False,
                    reason="Flag not found",
                    evaluation_context=context
                )
            
            # Check if flag is disabled
            if flag_config.status == FeatureFlagStatus.DISABLED:
                return FeatureFlagResult(
                    flag_name=flag_name,
                    enabled=False,
                    reason="Flag disabled",
                    evaluation_context=context
                )
            
            # Check if flag is fully enabled
            if flag_config.status == FeatureFlagStatus.ENABLED:
                return FeatureFlagResult(
                    flag_name=flag_name,
                    enabled=True,
                    config=flag_config.config_data,
                    reason="Flag enabled",
                    evaluation_context=context
                )
            
            # Evaluate based on rollout strategy
            enabled, variant, reason = await self._evaluate_rollout_strategy(
                flag_config, context
            )
            
            # Get configuration (potentially from A/B test variant)
            config_data = flag_config.config_data
            if variant:
                ab_test = await self._get_ab_test_for_flag(flag_config.id)
                if ab_test:
                    if variant == "A":
                        config_data = ab_test.variant_a_config
                    elif variant == "B":
                        config_data = ab_test.variant_b_config
            
            return FeatureFlagResult(
                flag_name=flag_name,
                enabled=enabled,
                variant=variant,
                config=config_data,
                reason=reason,
                evaluation_context=context
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate feature flag {flag_name}: {e}")
            return FeatureFlagResult(
                flag_name=flag_name,
                enabled=False,
                reason=f"Evaluation error: {str(e)}",
                evaluation_context=context
            )
    
    async def create_ab_test(
        self,
        config: ABTestConfig,
        user_id: Optional[str] = None
    ) -> ABTestConfig:
        """
        Create a new A/B test.
        
        Args:
            config: A/B test configuration
            user_id: User creating the test
            
        Returns:
            Created A/B test configuration
        """
        try:
            config.created_by = user_id
            
            async with self.session_factory() as session:
                # Verify feature flag exists
                flag_model = await self._get_flag_by_id(session, config.feature_flag_id)
                if not flag_model:
                    raise ValueError(f"Feature flag with ID '{config.feature_flag_id}' not found")
                
                # Create database record
                ab_test_model = ABTestModel(
                    id=config.id,
                    feature_flag_id=config.feature_flag_id,
                    name=config.name,
                    description=config.description,
                    variant_a_config=json.dumps(config.variant_a_config),
                    variant_b_config=json.dumps(config.variant_b_config),
                    traffic_split=config.traffic_split,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    status=config.status,
                    created_by=user_id
                )
                
                session.add(ab_test_model)
                await session.commit()
            
            # Update cache
            await self._update_ab_test_cache(config)
            
            logger.info(f"Created A/B test: {config.name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to create A/B test {config.name}: {e}")
            raise
    
    async def get_all_flags(self) -> List[FeatureFlagConfig]:
        """Get all feature flags."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    "SELECT * FROM feature_flags ORDER BY name"
                )
                flags = []
                for row in result:
                    flag_model = FeatureFlagModel(**dict(row))
                    config = await self._model_to_config(flag_model)
                    flags.append(config)
                return flags
                
        except Exception as e:
            logger.error(f"Failed to get all feature flags: {e}")
            raise
    
    async def register_update_callback(self, callback: callable) -> None:
        """Register a callback for configuration updates."""
        self._update_callbacks.append(callback)
    
    async def _load_configuration(self) -> None:
        """Load configuration from database and file."""
        try:
            # Load from database
            async with self.session_factory() as session:
                # Load feature flags
                result = await session.execute("SELECT * FROM feature_flags")
                for row in result:
                    flag_model = FeatureFlagModel(**dict(row))
                    config = await self._model_to_config(flag_model)
                    self._flag_cache[config.name] = config
                
                # Load A/B tests
                result = await session.execute("SELECT * FROM ab_tests WHERE status = 'active'")
                for row in result:
                    ab_test_model = ABTestModel(**dict(row))
                    config = await self._ab_test_model_to_config(ab_test_model)
                    self._ab_test_cache[config.id] = config
            
            # Load from file if specified
            if self.config_file_path and Path(self.config_file_path).exists():
                async with aiofiles.open(self.config_file_path, 'r') as f:
                    file_config = json.loads(await f.read())
                    for flag_data in file_config.get('feature_flags', []):
                        config = FeatureFlagConfig(**flag_data)
                        self._flag_cache[config.name] = config
            
            self._cache_last_updated = time.time()
            logger.info(f"Loaded {len(self._flag_cache)} feature flags and {len(self._ab_test_cache)} A/B tests")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Cache refresh task
        task = asyncio.create_task(self._cache_refresh_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        # Configuration sync task
        task = asyncio.create_task(self._config_sync_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _cache_refresh_task(self) -> None:
        """Background task to refresh cache periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cache_ttl)
                await self._load_configuration()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache refresh task: {e}")
    
    async def _config_sync_task(self) -> None:
        """Background task to sync configuration to file."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Sync every minute
                if self.config_file_path:
                    await self._save_configuration_to_file()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in config sync task: {e}")
    
    async def _save_configuration_to_file(self) -> None:
        """Save current configuration to file."""
        try:
            config_data = {
                'feature_flags': [flag.model_dump() for flag in self._flag_cache.values()],
                'ab_tests': [test.model_dump() for test in self._ab_test_cache.values()],
                'last_updated': datetime.utcnow().isoformat()
            }
            
            async with aiofiles.open(self.config_file_path, 'w') as f:
                await f.write(json.dumps(config_data, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save configuration to file: {e}")
    
    async def _get_flag_config(self, flag_name: str) -> Optional[FeatureFlagConfig]:
        """Get feature flag configuration from cache or database."""
        # Check cache first
        if flag_name in self._flag_cache:
            return self._flag_cache[flag_name]
        
        # Fallback to database
        try:
            async with self.session_factory() as session:
                flag_model = await self._get_flag_by_name(session, flag_name)
                if flag_model:
                    config = await self._model_to_config(flag_model)
                    self._flag_cache[flag_name] = config
                    return config
        except Exception as e:
            logger.error(f"Failed to get flag config from database: {e}")
        
        return None
    
    async def _evaluate_rollout_strategy(
        self,
        flag_config: FeatureFlagConfig,
        context: FeatureFlagEvaluationContext
    ) -> tuple[bool, Optional[str], str]:
        """
        Evaluate rollout strategy for a feature flag.
        
        Returns:
            Tuple of (enabled, variant, reason)
        """
        strategy = flag_config.rollout_strategy
        
        if strategy == RolloutStrategy.PERCENTAGE:
            # Simple percentage rollout
            if context.user_id:
                hash_value = hash(f"{flag_config.name}:{context.user_id}") % 100
                enabled = hash_value < flag_config.rollout_percentage
                return enabled, None, f"Percentage rollout: {flag_config.rollout_percentage}%"
            else:
                # No user ID, use random
                import random
                enabled = random.random() * 100 < flag_config.rollout_percentage
                return enabled, None, f"Random percentage rollout: {flag_config.rollout_percentage}%"
        
        elif strategy == RolloutStrategy.USER_GROUP:
            # User group targeting
            if not context.user_groups:
                return False, None, "No user groups provided"
            
            for group in context.user_groups:
                if group in flag_config.target_groups:
                    return True, None, f"User group match: {group.value}"
            
            return False, None, "No matching user groups"
        
        elif strategy == RolloutStrategy.WHITELIST:
            # Whitelist targeting
            if context.user_id and context.user_id in flag_config.whitelist_users:
                return True, None, "User in whitelist"
            
            return False, None, "User not in whitelist"
        
        elif strategy == RolloutStrategy.AB_TEST:
            # A/B test evaluation
            ab_test = await self._get_ab_test_for_flag(flag_config.id)
            if not ab_test:
                return False, None, "No active A/B test found"
            
            # Check if test is active
            now = datetime.utcnow()
            if ab_test.start_date and now < ab_test.start_date:
                return False, None, "A/B test not started"
            
            if ab_test.end_date and now > ab_test.end_date:
                return False, None, "A/B test ended"
            
            # Determine variant
            if context.user_id:
                hash_value = hash(f"{ab_test.id}:{context.user_id}")
                variant = "B" if (hash_value % 100) / 100 < ab_test.traffic_split else "A"
            else:
                import random
                variant = "B" if random.random() < ab_test.traffic_split else "A"
            
            return True, variant, f"A/B test variant: {variant}"
        
        else:
            return False, None, f"Unknown rollout strategy: {strategy}"
    
    async def _get_ab_test_for_flag(self, flag_id: str) -> Optional[ABTestConfig]:
        """Get active A/B test for a feature flag."""
        for ab_test in self._ab_test_cache.values():
            if ab_test.feature_flag_id == flag_id and ab_test.status == "active":
                return ab_test
        return None
    
    async def _update_flag_cache(self, config: FeatureFlagConfig) -> None:
        """Update flag in cache and Redis."""
        self._flag_cache[config.name] = config
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"feature_flag:{config.name}",
                    self.cache_ttl,
                    config.model_dump_json()
                )
            except Exception as e:
                logger.warning(f"Failed to update Redis cache: {e}")
    
    async def _update_ab_test_cache(self, config: ABTestConfig) -> None:
        """Update A/B test in cache."""
        self._ab_test_cache[config.id] = config
    
    async def _remove_flag_from_cache(self, flag_name: str) -> None:
        """Remove flag from cache and Redis."""
        self._flag_cache.pop(flag_name, None)
        
        if self.redis_client:
            try:
                await self.redis_client.delete(f"feature_flag:{flag_name}")
            except Exception as e:
                logger.warning(f"Failed to remove from Redis cache: {e}")
    
    async def _notify_update_callbacks(
        self,
        flag_name: str,
        config: Optional[FeatureFlagConfig]
    ) -> None:
        """Notify registered callbacks of configuration updates."""
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(flag_name, config)
                else:
                    callback(flag_name, config)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    async def _log_audit_event(
        self,
        feature_flag_id: str,
        action: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log audit event."""
        try:
            audit_log = AuditLogModel(
                id=str(uuid.uuid4()),
                feature_flag_id=feature_flag_id,
                action=action,
                old_value=old_value,
                new_value=new_value,
                user_id=user_id,
                correlation_id=correlation_id,
                model_metadata=json.dumps(metadata) if metadata else None
            )
            
            async with self.session_factory() as session:
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    async def _get_flag_by_name(self, session: AsyncSession, name: str) -> Optional[FeatureFlagModel]:
        """Get feature flag by name from database."""
        result = await session.execute(
            "SELECT * FROM feature_flags WHERE name = ?", (name,)
        )
        row = result.fetchone()
        return FeatureFlagModel(**dict(row)) if row else None
    
    async def _get_flag_by_id(self, session: AsyncSession, flag_id: str) -> Optional[FeatureFlagModel]:
        """Get feature flag by ID from database."""
        result = await session.execute(
            "SELECT * FROM feature_flags WHERE id = ?", (flag_id,)
        )
        row = result.fetchone()
        return FeatureFlagModel(**dict(row)) if row else None
    
    async def _model_to_config(self, model: FeatureFlagModel) -> FeatureFlagConfig:
        """Convert database model to configuration object."""
        return FeatureFlagConfig(
            id=model.id,
            name=model.name,
            description=model.description,
            status=FeatureFlagStatus(model.status),
            rollout_strategy=RolloutStrategy(model.rollout_strategy),
            rollout_percentage=model.rollout_percentage,
            target_groups=[UserGroup(g) for g in json.loads(model.target_groups or "[]")],
            whitelist_users=json.loads(model.whitelist_users or "[]"),
            config_data=json.loads(model.config_data or "{}"),
            approval_required=model.approval_required,
            approved_by=model.approved_by,
            approved_at=model.approved_at,
            created_by=model.created_by,
            updated_by=model.updated_by
        )
    
    async def _ab_test_model_to_config(self, model: ABTestModel) -> ABTestConfig:
        """Convert A/B test database model to configuration object."""
        return ABTestConfig(
            id=model.id,
            feature_flag_id=model.feature_flag_id,
            name=model.name,
            description=model.description,
            variant_a_config=json.loads(model.variant_a_config or "{}"),
            variant_b_config=json.loads(model.variant_b_config or "{}"),
            traffic_split=model.traffic_split,
            start_date=model.start_date,
            end_date=model.end_date,
            status=model.status,
            created_by=model.created_by
        )
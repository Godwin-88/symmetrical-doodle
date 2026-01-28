"""
Signal Validation and Format Checking

This module provides comprehensive validation for AI signals from F5 Intelligence
Layer, including format validation, content validation, and quality checks.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, ValidationError

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.logging import (
    get_correlation_id,
    get_logger,
    log_error_with_context,
    with_correlation_id,
)
from nautilus_integration.services.signal_router import AISignal, SignalType, SignalConfidence


class ValidationRule(BaseModel):
    """Signal validation rule."""
    
    rule_id: str
    name: str
    description: str
    signal_types: List[SignalType] = Field(default_factory=list)  # Empty = all types
    severity: str = "error"  # error, warning, info
    enabled: bool = True


class ValidationResult(BaseModel):
    """Result of signal validation."""
    
    validation_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_id: str
    rule_id: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SignalValidationReport(BaseModel):
    """Comprehensive signal validation report."""
    
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_id: str
    overall_valid: bool = False
    
    # Validation results
    format_results: List[ValidationResult] = Field(default_factory=list)
    content_results: List[ValidationResult] = Field(default_factory=list)
    quality_results: List[ValidationResult] = Field(default_factory=list)
    
    # Summary
    total_rules_applied: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    info_count: int = 0
    
    # Metadata
    validation_time: datetime = Field(default_factory=datetime.now)
    validation_duration_ms: float = 0.0


class SignalQualityMetrics(BaseModel):
    """Signal quality metrics."""
    
    signal_id: str
    
    # Timeliness metrics
    generation_latency_ms: float = 0.0
    delivery_latency_ms: float = 0.0
    age_seconds: float = 0.0
    
    # Confidence metrics
    confidence_score: float = 0.0
    confidence_consistency: float = 0.0  # Consistency with historical signals
    
    # Content quality
    value_validity: float = 0.0  # 0-1 score for value validity
    metadata_completeness: float = 0.0  # 0-1 score for metadata completeness
    
    # Source quality
    source_reliability: float = 0.0  # Historical reliability of source model
    model_version_currency: float = 0.0  # How current is the model version
    
    # Overall quality score
    overall_quality_score: float = 0.0


class SignalValidationService:
    """
    Comprehensive signal validation service.
    
    This service provides multi-layered validation including:
    - Format validation (schema, types, required fields)
    - Content validation (value ranges, logical consistency)
    - Quality assessment (timeliness, reliability, completeness)
    - Historical consistency checking
    - Source model validation
    """
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize signal validation service.
        
        Args:
            config: NautilusTrader integration configuration
        """
        self.config = config
        self.logger = get_logger("nautilus_integration.signal_validator")
        
        # Validation rules
        self._validation_rules = self._initialize_validation_rules()
        self._enabled_rules = {rule.rule_id: rule for rule in self._validation_rules if rule.enabled}
        
        # Historical data for consistency checking
        self._signal_history: Dict[str, List[AISignal]] = {}  # instrument_id -> signals
        self._source_reliability: Dict[str, float] = {}  # source_model -> reliability score
        
        # Quality thresholds
        self._quality_thresholds = {
            "min_confidence": config.signal_validation.min_confidence,
            "max_age_seconds": config.signal_validation.max_age_seconds,
            "min_metadata_completeness": config.signal_validation.min_metadata_completeness,
        }
        
        self.logger.info(
            "Signal Validation Service initialized",
            rules_count=len(self._validation_rules),
            enabled_rules_count=len(self._enabled_rules),
        )
    
    async def validate_signal(self, signal: AISignal) -> SignalValidationReport:
        """
        Perform comprehensive validation of AI signal.
        
        Args:
            signal: AI signal to validate
            
        Returns:
            Comprehensive validation report
        """
        start_time = datetime.now()
        
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Starting signal validation",
                signal_id=signal.signal_id,
                signal_type=signal.signal_type.value,
                instrument_id=signal.instrument_id,
            )
            
            try:
                # Initialize validation report
                report = SignalValidationReport(signal_id=signal.signal_id)
                
                # Step 1: Format validation
                format_results = await self._validate_format(signal)
                report.format_results.extend(format_results)
                
                # Step 2: Content validation
                content_results = await self._validate_content(signal)
                report.content_results.extend(content_results)
                
                # Step 3: Quality assessment
                quality_results = await self._validate_quality(signal)
                report.quality_results.extend(quality_results)
                
                # Calculate summary statistics
                all_results = report.format_results + report.content_results + report.quality_results
                report.total_rules_applied = len(all_results)
                report.errors_count = sum(1 for r in all_results if r.severity == "error" and not r.passed)
                report.warnings_count = sum(1 for r in all_results if r.severity == "warning" and not r.passed)
                report.info_count = sum(1 for r in all_results if r.severity == "info" and not r.passed)
                
                # Determine overall validity
                report.overall_valid = report.errors_count == 0
                
                # Calculate validation duration
                end_time = datetime.now()
                report.validation_duration_ms = (end_time - start_time).total_seconds() * 1000
                
                # Update historical data
                await self._update_signal_history(signal)
                
                self.logger.debug(
                    "Signal validation completed",
                    signal_id=signal.signal_id,
                    overall_valid=report.overall_valid,
                    errors_count=report.errors_count,
                    warnings_count=report.warnings_count,
                    validation_duration_ms=report.validation_duration_ms,
                )
                
                return report
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_signal",
                        "signal_id": signal.signal_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to validate signal"
                )
                
                # Return error report
                error_report = SignalValidationReport(
                    signal_id=signal.signal_id,
                    overall_valid=False,
                    errors_count=1,
                )
                
                error_result = ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="validation_error",
                    passed=False,
                    severity="error",
                    message=f"Validation failed: {error}",
                )
                error_report.format_results.append(error_result)
                
                return error_report
    
    async def calculate_quality_metrics(self, signal: AISignal) -> SignalQualityMetrics:
        """
        Calculate comprehensive quality metrics for signal.
        
        Args:
            signal: AI signal to analyze
            
        Returns:
            Signal quality metrics
        """
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Calculating signal quality metrics",
                signal_id=signal.signal_id,
            )
            
            try:
                metrics = SignalQualityMetrics(signal_id=signal.signal_id)
                
                # Timeliness metrics
                now = datetime.now()
                metrics.age_seconds = (now - signal.timestamp).total_seconds()
                metrics.generation_latency_ms = (
                    signal.timestamp - signal.generation_time
                ).total_seconds() * 1000
                
                # Confidence metrics
                metrics.confidence_score = signal.confidence
                metrics.confidence_consistency = await self._calculate_confidence_consistency(signal)
                
                # Content quality
                metrics.value_validity = await self._assess_value_validity(signal)
                metrics.metadata_completeness = await self._assess_metadata_completeness(signal)
                
                # Source quality
                metrics.source_reliability = self._source_reliability.get(signal.source_model, 0.5)
                metrics.model_version_currency = await self._assess_model_version_currency(signal)
                
                # Calculate overall quality score
                metrics.overall_quality_score = await self._calculate_overall_quality_score(metrics)
                
                self.logger.debug(
                    "Signal quality metrics calculated",
                    signal_id=signal.signal_id,
                    overall_quality_score=metrics.overall_quality_score,
                )
                
                return metrics
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "calculate_quality_metrics",
                        "signal_id": signal.signal_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to calculate quality metrics"
                )
                
                # Return default metrics
                return SignalQualityMetrics(
                    signal_id=signal.signal_id,
                    overall_quality_score=0.0,
                )
    
    async def validate_signal_batch(self, signals: List[AISignal]) -> List[SignalValidationReport]:
        """
        Validate a batch of signals efficiently.
        
        Args:
            signals: List of signals to validate
            
        Returns:
            List of validation reports
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Starting batch signal validation",
                signals_count=len(signals),
            )
            
            try:
                reports = []
                
                for signal in signals:
                    report = await self.validate_signal(signal)
                    reports.append(report)
                
                # Calculate batch statistics
                total_valid = sum(1 for r in reports if r.overall_valid)
                total_errors = sum(r.errors_count for r in reports)
                total_warnings = sum(r.warnings_count for r in reports)
                
                self.logger.info(
                    "Batch signal validation completed",
                    signals_count=len(signals),
                    valid_signals=total_valid,
                    total_errors=total_errors,
                    total_warnings=total_warnings,
                    correlation_id=correlation_id,
                )
                
                return reports
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_signal_batch",
                        "signals_count": len(signals),
                        "correlation_id": correlation_id,
                    },
                    "Failed to validate signal batch"
                )
                raise
    
    async def get_validation_statistics(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get validation statistics and metrics.
        
        Args:
            time_range: Optional time range filter
            
        Returns:
            Validation statistics
        """
        try:
            # TODO: Implement validation statistics collection
            # This would track validation results over time
            
            stats = {
                "total_validations": 0,
                "validation_success_rate": 0.0,
                "average_validation_time_ms": 0.0,
                "error_distribution": {},
                "quality_score_distribution": {},
                "source_reliability_scores": self._source_reliability.copy(),
            }
            
            return stats
            
        except Exception as error:
            self.logger.error(
                "Failed to get validation statistics",
                error=str(error),
            )
            return {}
    
    # Private validation methods
    
    async def _validate_format(self, signal: AISignal) -> List[ValidationResult]:
        """Validate signal format and schema."""
        results = []
        
        try:
            # Schema validation (already done by Pydantic, but we can add custom checks)
            
            # Required fields validation
            required_fields = ["signal_id", "signal_type", "instrument_id", "timestamp", "confidence", "value", "source_model"]
            for field in required_fields:
                if not hasattr(signal, field) or getattr(signal, field) is None:
                    results.append(ValidationResult(
                        signal_id=signal.signal_id,
                        rule_id="missing_required_field",
                        passed=False,
                        severity="error",
                        message=f"Required field missing: {field}",
                        details={"field": field}
                    ))
            
            # Signal ID format validation
            if not re.match(r'^[a-f0-9-]{36}$', signal.signal_id):
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="invalid_signal_id_format",
                    passed=False,
                    severity="warning",
                    message="Signal ID should be a valid UUID",
                    details={"signal_id": signal.signal_id}
                ))
            
            # Instrument ID format validation
            if not re.match(r'^[A-Z0-9._-]+$', signal.instrument_id):
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="invalid_instrument_id_format",
                    passed=False,
                    severity="warning",
                    message="Instrument ID format may be invalid",
                    details={"instrument_id": signal.instrument_id}
                ))
            
            # Timestamp validation
            now = datetime.now()
            if signal.timestamp > now + timedelta(minutes=5):
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="future_timestamp",
                    passed=False,
                    severity="warning",
                    message="Signal timestamp is in the future",
                    details={"timestamp": signal.timestamp.isoformat()}
                ))
            
            # Confidence range validation
            if not 0.0 <= signal.confidence <= 1.0:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="invalid_confidence_range",
                    passed=False,
                    severity="error",
                    message=f"Confidence must be between 0.0 and 1.0: {signal.confidence}",
                    details={"confidence": signal.confidence}
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="format_validation_error",
                passed=False,
                severity="error",
                message=f"Format validation failed: {error}",
            ))
        
        return results
    
    async def _validate_content(self, signal: AISignal) -> List[ValidationResult]:
        """Validate signal content and logical consistency."""
        results = []
        
        try:
            # Signal type specific validation
            if signal.signal_type == SignalType.REGIME_PREDICTION:
                results.extend(await self._validate_regime_prediction(signal))
            
            elif signal.signal_type == SignalType.VOLATILITY_FORECAST:
                results.extend(await self._validate_volatility_forecast(signal))
            
            elif signal.signal_type == SignalType.SENTIMENT_SCORE:
                results.extend(await self._validate_sentiment_score(signal))
            
            elif signal.signal_type == SignalType.CORRELATION_SHIFT:
                results.extend(await self._validate_correlation_shift(signal))
            
            # Value consistency validation
            value_validation = await self._validate_value_consistency(signal)
            results.extend(value_validation)
            
            # Metadata validation
            metadata_validation = await self._validate_metadata(signal)
            results.extend(metadata_validation)
            
        except Exception as error:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="content_validation_error",
                passed=False,
                severity="error",
                message=f"Content validation failed: {error}",
            ))
        
        return results
    
    async def _validate_quality(self, signal: AISignal) -> List[ValidationResult]:
        """Validate signal quality characteristics."""
        results = []
        
        try:
            # Age validation
            age_seconds = (datetime.now() - signal.timestamp).total_seconds()
            if age_seconds > self._quality_thresholds["max_age_seconds"]:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="signal_too_old",
                    passed=False,
                    severity="warning",
                    message=f"Signal is too old: {age_seconds:.1f} seconds",
                    details={"age_seconds": age_seconds}
                ))
            
            # Confidence threshold validation
            if signal.confidence < self._quality_thresholds["min_confidence"]:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="low_confidence",
                    passed=False,
                    severity="warning",
                    message=f"Signal confidence below threshold: {signal.confidence}",
                    details={"confidence": signal.confidence, "threshold": self._quality_thresholds["min_confidence"]}
                ))
            
            # Source model reliability
            source_reliability = self._source_reliability.get(signal.source_model, 0.5)
            if source_reliability < 0.7:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="unreliable_source",
                    passed=False,
                    severity="info",
                    message=f"Signal from unreliable source: {signal.source_model}",
                    details={"source_model": signal.source_model, "reliability": source_reliability}
                ))
            
            # Metadata completeness
            completeness_score = await self._assess_metadata_completeness(signal)
            if completeness_score < self._quality_thresholds["min_metadata_completeness"]:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="incomplete_metadata",
                    passed=False,
                    severity="info",
                    message=f"Incomplete metadata: {completeness_score:.2f}",
                    details={"completeness_score": completeness_score}
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="quality_validation_error",
                passed=False,
                severity="error",
                message=f"Quality validation failed: {error}",
            ))
        
        return results
    
    # Signal type specific validation methods
    
    async def _validate_regime_prediction(self, signal: AISignal) -> List[ValidationResult]:
        """Validate regime prediction signal."""
        results = []
        
        valid_regimes = ["LOW_VOL_TRENDING", "MEDIUM_VOL_TRENDING", "HIGH_VOL_RANGING", "CRISIS"]
        
        if isinstance(signal.value, str):
            if signal.value not in valid_regimes:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="invalid_regime_value",
                    passed=False,
                    severity="warning",
                    message=f"Unknown regime value: {signal.value}",
                    details={"value": signal.value, "valid_regimes": valid_regimes}
                ))
        else:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="invalid_regime_type",
                passed=False,
                severity="error",
                message="Regime prediction value must be string",
                details={"value_type": type(signal.value).__name__}
            ))
        
        return results
    
    async def _validate_volatility_forecast(self, signal: AISignal) -> List[ValidationResult]:
        """Validate volatility forecast signal."""
        results = []
        
        if isinstance(signal.value, (int, float)):
            if signal.value < 0:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="negative_volatility",
                    passed=False,
                    severity="error",
                    message="Volatility cannot be negative",
                    details={"value": signal.value}
                ))
            elif signal.value > 1.0:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="extreme_volatility",
                    passed=False,
                    severity="warning",
                    message="Extremely high volatility forecast",
                    details={"value": signal.value}
                ))
        else:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="invalid_volatility_type",
                passed=False,
                severity="error",
                message="Volatility forecast value must be numeric",
                details={"value_type": type(signal.value).__name__}
            ))
        
        return results
    
    async def _validate_sentiment_score(self, signal: AISignal) -> List[ValidationResult]:
        """Validate sentiment score signal."""
        results = []
        
        if isinstance(signal.value, (int, float)):
            if not -1.0 <= signal.value <= 1.0:
                results.append(ValidationResult(
                    signal_id=signal.signal_id,
                    rule_id="invalid_sentiment_range",
                    passed=False,
                    severity="error",
                    message="Sentiment score must be between -1.0 and 1.0",
                    details={"value": signal.value}
                ))
        else:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="invalid_sentiment_type",
                passed=False,
                severity="error",
                message="Sentiment score value must be numeric",
                details={"value_type": type(signal.value).__name__}
            ))
        
        return results
    
    async def _validate_correlation_shift(self, signal: AISignal) -> List[ValidationResult]:
        """Validate correlation shift signal."""
        results = []
        
        if isinstance(signal.value, dict):
            required_keys = ["from_correlation", "to_correlation", "shift_magnitude"]
            for key in required_keys:
                if key not in signal.value:
                    results.append(ValidationResult(
                        signal_id=signal.signal_id,
                        rule_id="missing_correlation_field",
                        passed=False,
                        severity="error",
                        message=f"Missing correlation field: {key}",
                        details={"missing_field": key}
                    ))
        else:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="invalid_correlation_type",
                passed=False,
                severity="error",
                message="Correlation shift value must be dictionary",
                details={"value_type": type(signal.value).__name__}
            ))
        
        return results
    
    # Helper methods
    
    async def _validate_value_consistency(self, signal: AISignal) -> List[ValidationResult]:
        """Validate value consistency with historical signals."""
        results = []
        
        try:
            # Get historical signals for instrument
            historical_signals = self._signal_history.get(signal.instrument_id, [])
            
            # Find recent signals of same type
            recent_signals = [
                s for s in historical_signals
                if s.signal_type == signal.signal_type
                and (signal.timestamp - s.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if len(recent_signals) >= 3:
                # Check for extreme deviations
                if isinstance(signal.value, (int, float)):
                    recent_values = [s.value for s in recent_signals if isinstance(s.value, (int, float))]
                    if recent_values:
                        avg_value = sum(recent_values) / len(recent_values)
                        std_dev = (sum((v - avg_value) ** 2 for v in recent_values) / len(recent_values)) ** 0.5
                        
                        if std_dev > 0 and abs(signal.value - avg_value) > 3 * std_dev:
                            results.append(ValidationResult(
                                signal_id=signal.signal_id,
                                rule_id="value_outlier",
                                passed=False,
                                severity="info",
                                message="Signal value is statistical outlier",
                                details={
                                    "value": signal.value,
                                    "average": avg_value,
                                    "std_dev": std_dev,
                                    "z_score": (signal.value - avg_value) / std_dev
                                }
                            ))
            
        except Exception as error:
            self.logger.warning(
                "Failed to validate value consistency",
                signal_id=signal.signal_id,
                error=str(error),
            )
        
        return results
    
    async def _validate_metadata(self, signal: AISignal) -> List[ValidationResult]:
        """Validate signal metadata."""
        results = []
        
        # Check for recommended metadata fields
        recommended_fields = ["model_confidence", "data_sources", "computation_time", "feature_importance"]
        missing_fields = [field for field in recommended_fields if field not in signal.metadata]
        
        if missing_fields:
            results.append(ValidationResult(
                signal_id=signal.signal_id,
                rule_id="missing_recommended_metadata",
                passed=False,
                severity="info",
                message=f"Missing recommended metadata fields: {missing_fields}",
                details={"missing_fields": missing_fields}
            ))
        
        return results
    
    async def _calculate_confidence_consistency(self, signal: AISignal) -> float:
        """Calculate confidence consistency with historical signals."""
        try:
            historical_signals = self._signal_history.get(signal.instrument_id, [])
            
            # Find recent signals of same type
            recent_signals = [
                s for s in historical_signals
                if s.signal_type == signal.signal_type
                and (signal.timestamp - s.timestamp).total_seconds() < 3600
            ]
            
            if len(recent_signals) < 2:
                return 1.0  # No comparison possible
            
            # Calculate confidence variance
            confidences = [s.confidence for s in recent_signals] + [signal.confidence]
            avg_confidence = sum(confidences) / len(confidences)
            variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            
            # Convert variance to consistency score (lower variance = higher consistency)
            consistency = max(0.0, 1.0 - variance * 4)  # Scale factor
            
            return consistency
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def _assess_value_validity(self, signal: AISignal) -> float:
        """Assess validity of signal value."""
        try:
            # Basic type and range checks
            if signal.signal_type == SignalType.VOLATILITY_FORECAST:
                if isinstance(signal.value, (int, float)) and signal.value >= 0:
                    return 1.0 if signal.value <= 1.0 else 0.7
                return 0.0
            
            elif signal.signal_type == SignalType.SENTIMENT_SCORE:
                if isinstance(signal.value, (int, float)) and -1.0 <= signal.value <= 1.0:
                    return 1.0
                return 0.0
            
            elif signal.signal_type == SignalType.REGIME_PREDICTION:
                valid_regimes = ["LOW_VOL_TRENDING", "MEDIUM_VOL_TRENDING", "HIGH_VOL_RANGING", "CRISIS"]
                if isinstance(signal.value, str) and signal.value in valid_regimes:
                    return 1.0
                return 0.0
            
            # Default validity for other types
            return 0.8
            
        except Exception:
            return 0.0
    
    async def _assess_metadata_completeness(self, signal: AISignal) -> float:
        """Assess completeness of signal metadata."""
        try:
            total_fields = 10  # Expected number of metadata fields
            present_fields = len(signal.metadata)
            
            # Bonus for important fields
            important_fields = ["model_confidence", "data_sources", "computation_time"]
            bonus = sum(0.1 for field in important_fields if field in signal.metadata)
            
            completeness = min(1.0, (present_fields / total_fields) + bonus)
            return completeness
            
        except Exception:
            return 0.0
    
    async def _assess_model_version_currency(self, signal: AISignal) -> float:
        """Assess how current the model version is."""
        try:
            # Simple version currency assessment
            # In a real implementation, this would check against a model registry
            
            if signal.model_version == "1.0":
                return 0.5  # Older version
            elif signal.model_version.startswith("2."):
                return 0.8  # Recent version
            elif signal.model_version.startswith("3."):
                return 1.0  # Latest version
            
            return 0.6  # Unknown version
            
        except Exception:
            return 0.5
    
    async def _calculate_overall_quality_score(self, metrics: SignalQualityMetrics) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            # Weighted combination of quality factors
            weights = {
                "confidence": 0.25,
                "timeliness": 0.20,
                "validity": 0.20,
                "completeness": 0.15,
                "reliability": 0.15,
                "currency": 0.05,
            }
            
            # Timeliness score (fresher is better)
            timeliness_score = max(0.0, 1.0 - metrics.age_seconds / 3600)  # Decay over 1 hour
            
            # Calculate weighted score
            overall_score = (
                weights["confidence"] * metrics.confidence_score +
                weights["timeliness"] * timeliness_score +
                weights["validity"] * metrics.value_validity +
                weights["completeness"] * metrics.metadata_completeness +
                weights["reliability"] * metrics.source_reliability +
                weights["currency"] * metrics.model_version_currency
            )
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception:
            return 0.0
    
    async def _update_signal_history(self, signal: AISignal) -> None:
        """Update signal history for consistency checking."""
        try:
            if signal.instrument_id not in self._signal_history:
                self._signal_history[signal.instrument_id] = []
            
            # Add signal to history
            self._signal_history[signal.instrument_id].append(signal)
            
            # Keep only recent signals (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self._signal_history[signal.instrument_id] = [
                s for s in self._signal_history[signal.instrument_id]
                if s.timestamp > cutoff_time
            ]
            
            # Update source reliability based on signal quality
            quality_metrics = await self.calculate_quality_metrics(signal)
            current_reliability = self._source_reliability.get(signal.source_model, 0.5)
            
            # Exponential moving average update
            alpha = 0.1
            new_reliability = (1 - alpha) * current_reliability + alpha * quality_metrics.overall_quality_score
            self._source_reliability[signal.source_model] = new_reliability
            
        except Exception as error:
            self.logger.warning(
                "Failed to update signal history",
                signal_id=signal.signal_id,
                error=str(error),
            )
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules."""
        return [
            ValidationRule(
                rule_id="missing_required_field",
                name="Required Field Check",
                description="Validate presence of required fields",
                severity="error"
            ),
            ValidationRule(
                rule_id="invalid_confidence_range",
                name="Confidence Range Check",
                description="Validate confidence is between 0.0 and 1.0",
                severity="error"
            ),
            ValidationRule(
                rule_id="signal_too_old",
                name="Signal Age Check",
                description="Validate signal is not too old",
                severity="warning"
            ),
            ValidationRule(
                rule_id="low_confidence",
                name="Confidence Threshold Check",
                description="Validate signal meets minimum confidence threshold",
                severity="warning"
            ),
            ValidationRule(
                rule_id="value_outlier",
                name="Statistical Outlier Check",
                description="Detect statistical outliers in signal values",
                severity="info"
            ),
        ]
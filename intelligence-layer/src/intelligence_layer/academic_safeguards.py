"""
Academic safeguards and validation for algorithmic trading research.

This module provides:
- Data leakage prevention mechanisms
- Lookahead bias detection and testing
- Negative findings documentation framework
- Research integrity validation

Requirements: 7.7
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from pathlib import Path
import json
import hashlib

from .config import load_config
from .logging import get_logger

logger = get_logger(__name__)
config = load_config()


class BiasType(str, Enum):
    """Types of research biases to detect."""
    LOOKAHEAD = "lookahead"
    SURVIVORSHIP = "survivorship"
    SELECTION = "selection"
    CONFIRMATION = "confirmation"
    DATA_SNOOPING = "data_snooping"
    OVERFITTING = "overfitting"


class SeverityLevel(str, Enum):
    """Severity levels for bias detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasDetectionResult:
    """Result from bias detection analysis."""
    
    bias_type: BiasType
    severity: SeverityLevel
    detected: bool
    confidence: float  # 0.0 to 1.0
    
    description: str
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Technical details
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    
    # Metadata
    detection_timestamp: datetime = field(default_factory=datetime.now)
    data_hash: Optional[str] = None


@dataclass
class NegativeFinding:
    """Documentation of negative research findings."""
    
    finding_id: str
    hypothesis: str
    methodology: str
    result_summary: str
    
    # Statistical details
    test_statistics: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    
    # Context
    data_period: Tuple[datetime, datetime]
    sample_size: int
    market_conditions: List[str] = field(default_factory=list)
    
    # Documentation
    researcher: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    # Reproducibility
    code_hash: Optional[str] = None
    data_hash: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for research integrity."""
    
    report_id: str
    experiment_id: str
    
    # Bias detection results
    bias_results: List[BiasDetectionResult] = field(default_factory=list)
    
    # Data integrity
    data_quality_score: float = 0.0
    temporal_consistency: bool = True
    missing_data_percentage: float = 0.0
    
    # Methodology validation
    statistical_power: float = 0.0
    multiple_testing_correction: bool = False
    cross_validation_used: bool = False
    
    # Reproducibility
    code_reproducible: bool = False
    data_reproducible: bool = False
    results_reproducible: bool = False
    
    # Overall assessment
    integrity_score: float = 0.0
    approved_for_publication: bool = False
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validator: str = ""


class LookaheadBiasDetector:
    """Detects lookahead bias in trading strategies and data processing."""
    
    def __init__(self):
        """Initialize lookahead bias detector."""
        self.detected_violations: List[Dict] = []
    
    def validate_temporal_ordering(
        self, 
        data: pd.DataFrame, 
        timestamp_column: str = 'timestamp',
        feature_columns: Optional[List[str]] = None
    ) -> BiasDetectionResult:
        """Validate that data maintains proper temporal ordering.
        
        Args:
            data: DataFrame with time series data
            timestamp_column: Name of timestamp column
            feature_columns: Optional list of feature columns to check
            
        Returns:
            Bias detection result
        """
        violations = []
        
        # Check timestamp ordering
        timestamps = pd.to_datetime(data[timestamp_column])
        if not timestamps.is_monotonic_increasing:
            violations.append("Timestamps are not monotonically increasing")
        
        # Check for duplicate timestamps
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            violations.append(f"Found {duplicates} duplicate timestamps")
        
        # Check for future data usage (simplified heuristic)
        if feature_columns:
            for col in feature_columns:
                if col in data.columns:
                    # Check for suspiciously perfect correlations that might indicate lookahead
                    if len(data) > 1:
                        future_corr = self._check_future_correlation(data, col, timestamp_column)
                        if future_corr > 0.95:
                            violations.append(f"Suspiciously high future correlation in {col}: {future_corr:.3f}")
        
        # Determine severity
        severity = SeverityLevel.LOW
        if len(violations) > 0:
            severity = SeverityLevel.HIGH if any("future" in v.lower() for v in violations) else SeverityLevel.MEDIUM
        
        return BiasDetectionResult(
            bias_type=BiasType.LOOKAHEAD,
            severity=severity,
            detected=len(violations) > 0,
            confidence=min(1.0, len(violations) * 0.3),
            description="Temporal ordering and lookahead bias validation",
            evidence=violations,
            recommendations=self._get_lookahead_recommendations(violations),
            data_hash=self._calculate_data_hash(data)
        )
    
    def validate_feature_construction(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series,
        timestamp_column: str = 'timestamp'
    ) -> BiasDetectionResult:
        """Validate that features don't contain future information.
        
        Args:
            features: Feature matrix
            targets: Target values
            timestamp_column: Name of timestamp column
            
        Returns:
            Bias detection result
        """
        violations = []
        
        # Check for perfect correlations (potential data leakage)
        for col in features.columns:
            if col != timestamp_column:
                corr = np.corrcoef(features[col].fillna(0), targets.fillna(0))[0, 1]
                if abs(corr) > 0.99:
                    violations.append(f"Perfect correlation detected in feature {col}: {corr:.4f}")
        
        # Check for forward-looking statistics
        for col in features.columns:
            if any(keyword in col.lower() for keyword in ['future', 'forward', 'ahead', 'next']):
                violations.append(f"Feature name suggests forward-looking data: {col}")
        
        # Check for information leakage through rolling statistics
        rolling_violations = self._check_rolling_statistics(features, timestamp_column)
        violations.extend(rolling_violations)
        
        severity = SeverityLevel.CRITICAL if any("perfect" in v.lower() for v in violations) else SeverityLevel.MEDIUM
        
        return BiasDetectionResult(
            bias_type=BiasType.LOOKAHEAD,
            severity=severity,
            detected=len(violations) > 0,
            confidence=min(1.0, len(violations) * 0.4),
            description="Feature construction lookahead bias validation",
            evidence=violations,
            recommendations=self._get_feature_recommendations(violations)
        )
    
    def _check_future_correlation(self, data: pd.DataFrame, column: str, timestamp_col: str) -> float:
        """Check correlation with future values."""
        try:
            values = data[column].fillna(method='ffill').fillna(0)
            future_values = values.shift(-1).fillna(0)
            
            if len(values) > 1 and np.std(values) > 0 and np.std(future_values) > 0:
                return abs(np.corrcoef(values[:-1], future_values[:-1])[0, 1])
            return 0.0
        except Exception:
            return 0.0
    
    def _check_rolling_statistics(self, features: pd.DataFrame, timestamp_col: str) -> List[str]:
        """Check for improper rolling statistics that might include future data."""
        violations = []
        
        for col in features.columns:
            if col == timestamp_col:
                continue
                
            # Check for suspiciously smooth rolling statistics
            if 'rolling' in col.lower() or 'ma' in col.lower() or 'sma' in col.lower():
                values = features[col].dropna()
                if len(values) > 10:
                    # Check if rolling statistic has unrealistic smoothness
                    volatility = np.std(np.diff(values))
                    mean_level = np.mean(np.abs(values))
                    
                    if mean_level > 0 and volatility / mean_level < 0.001:
                        violations.append(f"Rolling statistic {col} appears unrealistically smooth")
        
        return violations
    
    def _get_lookahead_recommendations(self, violations: List[str]) -> List[str]:
        """Get recommendations for fixing lookahead bias."""
        recommendations = []
        
        if any("timestamp" in v.lower() for v in violations):
            recommendations.append("Ensure data is sorted by timestamp before processing")
            recommendations.append("Implement strict temporal validation in data pipeline")
        
        if any("correlation" in v.lower() for v in violations):
            recommendations.append("Investigate high correlations for potential data leakage")
            recommendations.append("Implement feature-target independence tests")
        
        recommendations.append("Use walk-forward validation instead of random splits")
        recommendations.append("Implement temporal cross-validation")
        
        return recommendations
    
    def _get_feature_recommendations(self, violations: List[str]) -> List[str]:
        """Get recommendations for fixing feature construction issues."""
        recommendations = []
        
        if any("perfect" in v.lower() for v in violations):
            recommendations.append("Remove features with perfect correlations")
            recommendations.append("Investigate data construction pipeline for leakage")
        
        if any("forward" in v.lower() for v in violations):
            recommendations.append("Rename features to avoid forward-looking terminology")
            recommendations.append("Verify feature construction uses only past data")
        
        recommendations.append("Implement feature validation framework")
        recommendations.append("Use causal feature engineering principles")
        
        return recommendations
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for reproducibility."""
        try:
            data_str = data.to_string()
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "hash_error"


class DataSnoopingDetector:
    """Detects data snooping and multiple testing issues."""
    
    def __init__(self):
        """Initialize data snooping detector."""
        self.test_count = 0
        self.significant_results = 0
    
    def validate_multiple_testing(
        self, 
        p_values: List[float], 
        alpha: float = 0.05,
        correction_method: str = "bonferroni"
    ) -> BiasDetectionResult:
        """Validate multiple testing correction.
        
        Args:
            p_values: List of p-values from multiple tests
            alpha: Significance level
            correction_method: Multiple testing correction method
            
        Returns:
            Bias detection result
        """
        violations = []
        
        # Count significant results without correction
        uncorrected_significant = sum(1 for p in p_values if p < alpha)
        
        # Apply correction
        if correction_method == "bonferroni":
            corrected_alpha = alpha / len(p_values)
            corrected_significant = sum(1 for p in p_values if p < corrected_alpha)
        else:
            corrected_significant = uncorrected_significant  # Placeholder for other methods
        
        # Check for excessive significant results
        expected_false_positives = len(p_values) * alpha
        if uncorrected_significant > expected_false_positives * 2:
            violations.append(f"Excessive significant results: {uncorrected_significant} vs expected {expected_false_positives:.1f}")
        
        # Check if correction was applied
        if uncorrected_significant != corrected_significant and corrected_significant == 0:
            violations.append("All significant results disappear after multiple testing correction")
        
        severity = SeverityLevel.HIGH if len(violations) > 0 else SeverityLevel.LOW
        
        return BiasDetectionResult(
            bias_type=BiasType.DATA_SNOOPING,
            severity=severity,
            detected=len(violations) > 0,
            confidence=min(1.0, len(violations) * 0.5),
            description="Multiple testing and data snooping validation",
            evidence=violations,
            recommendations=self._get_snooping_recommendations(violations),
            test_statistic=float(uncorrected_significant),
            threshold=expected_false_positives
        )
    
    def _get_snooping_recommendations(self, violations: List[str]) -> List[str]:
        """Get recommendations for fixing data snooping issues."""
        recommendations = []
        
        recommendations.append("Apply appropriate multiple testing correction (Bonferroni, FDR)")
        recommendations.append("Pre-register hypotheses and analysis plan")
        recommendations.append("Use holdout validation sets")
        recommendations.append("Implement cross-validation with proper temporal splits")
        recommendations.append("Document all analyses performed, including negative results")
        
        return recommendations


class NegativeFindingsRegistry:
    """Registry for documenting negative research findings."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize negative findings registry.
        
        Args:
            registry_path: Path to store registry files
        """
        if registry_path is None:
            registry_path = Path(config.get('negative_findings_dir', './negative_findings'))
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.findings: List[NegativeFinding] = []
        self._load_existing_findings()
    
    def register_negative_finding(
        self,
        hypothesis: str,
        methodology: str,
        result_summary: str,
        test_statistics: Dict[str, float],
        p_values: Dict[str, float],
        data_period: Tuple[datetime, datetime],
        sample_size: int,
        researcher: str = "",
        notes: str = "",
        market_conditions: Optional[List[str]] = None
    ) -> str:
        """Register a negative finding.
        
        Args:
            hypothesis: The hypothesis that was tested
            methodology: Description of methodology used
            result_summary: Summary of negative results
            test_statistics: Dictionary of test statistics
            p_values: Dictionary of p-values
            data_period: Start and end dates of data used
            sample_size: Sample size used in analysis
            researcher: Name of researcher
            notes: Additional notes
            market_conditions: List of market conditions during study period
            
        Returns:
            Finding ID
        """
        finding_id = f"NF_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.findings):03d}"
        
        finding = NegativeFinding(
            finding_id=finding_id,
            hypothesis=hypothesis,
            methodology=methodology,
            result_summary=result_summary,
            test_statistics=test_statistics,
            p_values=p_values,
            data_period=data_period,
            sample_size=sample_size,
            market_conditions=market_conditions or [],
            researcher=researcher,
            notes=notes
        )
        
        self.findings.append(finding)
        self._save_finding(finding)
        
        logger.info(f"Registered negative finding: {finding_id}")
        return finding_id
    
    def search_findings(
        self,
        hypothesis_keywords: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        researcher: Optional[str] = None
    ) -> List[NegativeFinding]:
        """Search for negative findings.
        
        Args:
            hypothesis_keywords: Keywords to search in hypothesis
            date_range: Date range for finding timestamp
            researcher: Researcher name to filter by
            
        Returns:
            List of matching findings
        """
        results = self.findings.copy()
        
        if hypothesis_keywords:
            results = [
                f for f in results 
                if any(keyword.lower() in f.hypothesis.lower() for keyword in hypothesis_keywords)
            ]
        
        if date_range:
            start_date, end_date = date_range
            results = [
                f for f in results 
                if start_date <= f.timestamp <= end_date
            ]
        
        if researcher:
            results = [
                f for f in results 
                if researcher.lower() in f.researcher.lower()
            ]
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of negative findings."""
        if not self.findings:
            return {"message": "No negative findings registered"}
        
        # Summary statistics
        total_findings = len(self.findings)
        researchers = set(f.researcher for f in self.findings if f.researcher)
        
        # Temporal distribution
        timestamps = [f.timestamp for f in self.findings]
        earliest = min(timestamps)
        latest = max(timestamps)
        
        # Common themes
        hypothesis_words = []
        for finding in self.findings:
            hypothesis_words.extend(finding.hypothesis.lower().split())
        
        from collections import Counter
        common_themes = Counter(hypothesis_words).most_common(10)
        
        return {
            "summary": {
                "total_negative_findings": total_findings,
                "unique_researchers": len(researchers),
                "date_range": {
                    "earliest": earliest.isoformat(),
                    "latest": latest.isoformat()
                }
            },
            "common_themes": [{"word": word, "count": count} for word, count in common_themes],
            "researchers": list(researchers),
            "recent_findings": [
                {
                    "finding_id": f.finding_id,
                    "hypothesis": f.hypothesis[:100] + "..." if len(f.hypothesis) > 100 else f.hypothesis,
                    "timestamp": f.timestamp.isoformat(),
                    "researcher": f.researcher
                }
                for f in sorted(self.findings, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def _load_existing_findings(self):
        """Load existing findings from registry."""
        try:
            for finding_file in self.registry_path.glob("*.json"):
                with open(finding_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert datetime strings back to datetime objects
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                data['data_period'] = tuple(datetime.fromisoformat(d) for d in data['data_period'])
                
                finding = NegativeFinding(**data)
                self.findings.append(finding)
                
        except Exception as e:
            logger.warning(f"Failed to load existing findings: {e}")
    
    def _save_finding(self, finding: NegativeFinding):
        """Save finding to registry."""
        try:
            finding_file = self.registry_path / f"{finding.finding_id}.json"
            
            # Convert to serializable format
            data = {
                'finding_id': finding.finding_id,
                'hypothesis': finding.hypothesis,
                'methodology': finding.methodology,
                'result_summary': finding.result_summary,
                'test_statistics': finding.test_statistics,
                'p_values': finding.p_values,
                'data_period': [d.isoformat() for d in finding.data_period],
                'sample_size': finding.sample_size,
                'market_conditions': finding.market_conditions,
                'researcher': finding.researcher,
                'timestamp': finding.timestamp.isoformat(),
                'notes': finding.notes,
                'code_hash': finding.code_hash,
                'data_hash': finding.data_hash,
                'config_hash': finding.config_hash
            }
            
            with open(finding_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save finding {finding.finding_id}: {e}")


class AcademicValidator:
    """Comprehensive academic validation framework."""
    
    def __init__(self):
        """Initialize academic validator."""
        self.lookahead_detector = LookaheadBiasDetector()
        self.snooping_detector = DataSnoopingDetector()
        self.negative_registry = NegativeFindingsRegistry()
    
    def validate_research_integrity(
        self,
        experiment_id: str,
        data: pd.DataFrame,
        features: pd.DataFrame,
        targets: pd.Series,
        p_values: List[float],
        validator_name: str = ""
    ) -> ValidationReport:
        """Perform comprehensive research integrity validation.
        
        Args:
            experiment_id: Unique experiment identifier
            data: Raw data used in experiment
            features: Feature matrix
            targets: Target values
            p_values: List of p-values from statistical tests
            validator_name: Name of person performing validation
            
        Returns:
            Comprehensive validation report
        """
        report_id = f"VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_id}"
        
        # Bias detection
        bias_results = []
        
        # Lookahead bias detection
        temporal_result = self.lookahead_detector.validate_temporal_ordering(data)
        bias_results.append(temporal_result)
        
        feature_result = self.lookahead_detector.validate_feature_construction(features, targets)
        bias_results.append(feature_result)
        
        # Data snooping detection
        if p_values:
            snooping_result = self.snooping_detector.validate_multiple_testing(p_values)
            bias_results.append(snooping_result)
        
        # Data quality assessment
        data_quality_score = self._assess_data_quality(data)
        temporal_consistency = self._check_temporal_consistency(data)
        missing_data_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        
        # Methodology validation
        statistical_power = self._estimate_statistical_power(len(data))
        
        # Reproducibility checks
        code_reproducible = True  # Placeholder - would check code determinism
        data_reproducible = True  # Placeholder - would check data consistency
        results_reproducible = True  # Placeholder - would check result consistency
        
        # Overall integrity score
        integrity_score = self._calculate_integrity_score(bias_results, data_quality_score)
        
        # Approval decision
        critical_issues = any(r.severity == SeverityLevel.CRITICAL for r in bias_results)
        approved_for_publication = integrity_score > 0.7 and not critical_issues
        
        report = ValidationReport(
            report_id=report_id,
            experiment_id=experiment_id,
            bias_results=bias_results,
            data_quality_score=data_quality_score,
            temporal_consistency=temporal_consistency,
            missing_data_percentage=missing_data_pct,
            statistical_power=statistical_power,
            multiple_testing_correction=len(p_values) > 1 if p_values else False,
            cross_validation_used=True,  # Placeholder
            code_reproducible=code_reproducible,
            data_reproducible=data_reproducible,
            results_reproducible=results_reproducible,
            integrity_score=integrity_score,
            approved_for_publication=approved_for_publication,
            validator=validator_name
        )
        
        logger.info(f"Generated validation report {report_id} with integrity score {integrity_score:.3f}")
        
        return report
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess overall data quality."""
        quality_factors = []
        
        # Missing data penalty
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_factors.append(1.0 - missing_pct)
        
        # Duplicate data penalty
        duplicate_pct = data.duplicated().sum() / len(data)
        quality_factors.append(1.0 - duplicate_pct)
        
        # Data variance (avoid constant columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variances = data[numeric_cols].var()
            non_zero_var_pct = (variances > 0).sum() / len(variances)
            quality_factors.append(non_zero_var_pct)
        
        return np.mean(quality_factors)
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> bool:
        """Check temporal consistency of data."""
        timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if not timestamp_cols:
            return True  # No timestamp columns to check
        
        try:
            for col in timestamp_cols:
                timestamps = pd.to_datetime(data[col])
                if not timestamps.is_monotonic_increasing:
                    return False
            return True
        except Exception:
            return False
    
    def _estimate_statistical_power(self, sample_size: int, effect_size: float = 0.5, alpha: float = 0.05) -> float:
        """Estimate statistical power for given sample size."""
        # Simplified power calculation for t-test
        # In practice, would use more sophisticated power analysis
        
        from scipy.stats import norm
        
        # Cohen's d effect size
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(0.8)  # 80% power
        
        required_n = ((z_alpha + z_beta) / effect_size) ** 2 * 2
        
        if sample_size >= required_n:
            return 0.8
        else:
            return sample_size / required_n * 0.8
    
    def _calculate_integrity_score(self, bias_results: List[BiasDetectionResult], data_quality: float) -> float:
        """Calculate overall research integrity score."""
        # Start with data quality
        score = data_quality * 0.3
        
        # Penalty for detected biases
        bias_penalty = 0.0
        for result in bias_results:
            if result.detected:
                severity_weights = {
                    SeverityLevel.LOW: 0.1,
                    SeverityLevel.MEDIUM: 0.2,
                    SeverityLevel.HIGH: 0.4,
                    SeverityLevel.CRITICAL: 0.8
                }
                penalty = severity_weights.get(result.severity, 0.2) * result.confidence
                bias_penalty += penalty
        
        # Apply bias penalty
        score += (0.7 - bias_penalty)
        
        return max(0.0, min(1.0, score))


# Global instances
_academic_validator = None
_negative_registry = None


def get_academic_validator() -> AcademicValidator:
    """Get global academic validator instance."""
    global _academic_validator
    if _academic_validator is None:
        _academic_validator = AcademicValidator()
    return _academic_validator


def get_negative_findings_registry() -> NegativeFindingsRegistry:
    """Get global negative findings registry instance."""
    global _negative_registry
    if _negative_registry is None:
        _negative_registry = NegativeFindingsRegistry()
    return _negative_registry
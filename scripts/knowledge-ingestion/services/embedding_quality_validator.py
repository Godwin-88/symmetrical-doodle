"""
Embedding Quality Validation Service

This module provides comprehensive quality validation for generated embeddings,
including dimension verification, null detection, semantic coherence measurement,
and automatic regeneration for failed quality checks.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.distance import cosine
import statistics

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """Types of quality issues that can be detected"""
    NULL_EMBEDDING = "null_embedding"
    WRONG_DIMENSION = "wrong_dimension"
    ZERO_VECTOR = "zero_vector"
    LOW_MAGNITUDE = "low_magnitude"
    HIGH_MAGNITUDE = "high_magnitude"
    LOW_COHERENCE = "low_coherence"
    SUSPICIOUS_VALUES = "suspicious_values"


@dataclass
class QualityMetrics:
    """Quality metrics for an embedding"""
    dimension: int
    magnitude: float
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    zero_count: int
    nan_count: int
    inf_count: int
    coherence_score: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of embedding quality validation"""
    is_valid: bool
    quality_score: float
    metrics: QualityMetrics
    issues: List[QualityIssue]
    recommendations: List[str]
    should_regenerate: bool


@dataclass
class CoherenceTestCase:
    """Test case for semantic coherence validation"""
    text1: str
    text2: str
    expected_similarity: float  # 0.0 to 1.0
    tolerance: float = 0.2


class EmbeddingQualityValidator:
    """
    Comprehensive embedding quality validation system that ensures
    generated embeddings meet quality standards for semantic search.
    """
    
    def __init__(self):
        """Initialize the quality validator with thresholds and test cases"""
        self.dimension_thresholds = self._initialize_dimension_thresholds()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.coherence_test_cases = self._initialize_coherence_tests()
        
        logger.info("EmbeddingQualityValidator initialized")
    
    def _initialize_dimension_thresholds(self) -> Dict[str, int]:
        """Initialize expected dimensions for different models"""
        return {
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds for validation"""
        return {
            "min_magnitude": 0.1,
            "max_magnitude": 10.0,
            "min_std_deviation": 0.01,
            "max_zero_ratio": 0.5,
            "min_coherence_score": 0.3,
            "suspicious_value_threshold": 100.0,
        }
    
    def _initialize_coherence_tests(self) -> List[CoherenceTestCase]:
        """Initialize semantic coherence test cases"""
        return [
            # Similar concepts should have high similarity
            CoherenceTestCase(
                "machine learning algorithm",
                "artificial intelligence model",
                expected_similarity=0.7,
                tolerance=0.2
            ),
            CoherenceTestCase(
                "financial portfolio optimization",
                "investment strategy management",
                expected_similarity=0.6,
                tolerance=0.2
            ),
            CoherenceTestCase(
                "neural network training",
                "deep learning model optimization",
                expected_similarity=0.8,
                tolerance=0.2
            ),
            
            # Dissimilar concepts should have low similarity
            CoherenceTestCase(
                "machine learning",
                "cooking recipes",
                expected_similarity=0.1,
                tolerance=0.2
            ),
            CoherenceTestCase(
                "financial markets",
                "weather patterns",
                expected_similarity=0.1,
                tolerance=0.2
            ),
            
            # Mathematical concepts
            CoherenceTestCase(
                "gradient descent optimization",
                "mathematical optimization algorithm",
                expected_similarity=0.7,
                tolerance=0.2
            ),
        ]
    
    async def validate_embedding(self, embedding: List[float], 
                               model_name: str = "",
                               text_content: str = "") -> ValidationResult:
        """
        Validate a single embedding for quality issues.
        
        Args:
            embedding: The embedding vector to validate
            model_name: Name of the model that generated the embedding
            text_content: Original text content (for context)
            
        Returns:
            ValidationResult with quality assessment
        """
        try:
            # Convert to numpy array for easier processing
            emb_array = np.array(embedding, dtype=np.float32)
            
            # Calculate quality metrics
            metrics = self._calculate_metrics(emb_array)
            
            # Detect quality issues
            issues = self._detect_issues(emb_array, metrics, model_name)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(metrics, issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, metrics)
            
            # Determine if regeneration is needed
            should_regenerate = self._should_regenerate(issues, quality_score)
            
            # Overall validity
            is_valid = len(issues) == 0 or quality_score > 0.5
            
            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                metrics=metrics,
                issues=issues,
                recommendations=recommendations,
                should_regenerate=should_regenerate
            )
            
        except Exception as e:
            logger.error(f"Error validating embedding: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                metrics=QualityMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
                issues=[QualityIssue.SUSPICIOUS_VALUES],
                recommendations=[f"Validation failed: {str(e)}"],
                should_regenerate=True
            )
    
    def _calculate_metrics(self, embedding: np.ndarray) -> QualityMetrics:
        """Calculate comprehensive quality metrics for an embedding"""
        # Basic statistics
        magnitude = float(np.linalg.norm(embedding))
        mean_value = float(np.mean(embedding))
        std_deviation = float(np.std(embedding))
        min_value = float(np.min(embedding))
        max_value = float(np.max(embedding))
        
        # Count special values
        zero_count = int(np.sum(embedding == 0.0))
        nan_count = int(np.sum(np.isnan(embedding)))
        inf_count = int(np.sum(np.isinf(embedding)))
        
        return QualityMetrics(
            dimension=len(embedding),
            magnitude=magnitude,
            mean_value=mean_value,
            std_deviation=std_deviation,
            min_value=min_value,
            max_value=max_value,
            zero_count=zero_count,
            nan_count=nan_count,
            inf_count=inf_count
        )
    
    def _detect_issues(self, embedding: np.ndarray, metrics: QualityMetrics, 
                      model_name: str) -> List[QualityIssue]:
        """Detect quality issues in the embedding"""
        issues = []
        
        # Check for null/empty embedding
        if len(embedding) == 0:
            issues.append(QualityIssue.NULL_EMBEDDING)
            return issues
        
        # Check dimension
        if model_name in self.dimension_thresholds:
            expected_dim = self.dimension_thresholds[model_name]
            if metrics.dimension != expected_dim:
                issues.append(QualityIssue.WRONG_DIMENSION)
        
        # Check for zero vector
        if metrics.magnitude < 1e-10:
            issues.append(QualityIssue.ZERO_VECTOR)
        
        # Check magnitude bounds
        if metrics.magnitude < self.quality_thresholds["min_magnitude"]:
            issues.append(QualityIssue.LOW_MAGNITUDE)
        elif metrics.magnitude > self.quality_thresholds["max_magnitude"]:
            issues.append(QualityIssue.HIGH_MAGNITUDE)
        
        # Check for suspicious values
        if (metrics.nan_count > 0 or metrics.inf_count > 0 or
            abs(metrics.max_value) > self.quality_thresholds["suspicious_value_threshold"] or
            abs(metrics.min_value) > self.quality_thresholds["suspicious_value_threshold"]):
            issues.append(QualityIssue.SUSPICIOUS_VALUES)
        
        # Check zero ratio
        zero_ratio = metrics.zero_count / metrics.dimension
        if zero_ratio > self.quality_thresholds["max_zero_ratio"]:
            issues.append(QualityIssue.LOW_MAGNITUDE)
        
        return issues
    
    def _calculate_quality_score(self, metrics: QualityMetrics, 
                               issues: List[QualityIssue]) -> float:
        """Calculate overall quality score (0.0 to 1.0)"""
        base_score = 1.0
        
        # Penalize each issue type
        issue_penalties = {
            QualityIssue.NULL_EMBEDDING: 1.0,
            QualityIssue.ZERO_VECTOR: 1.0,
            QualityIssue.SUSPICIOUS_VALUES: 0.8,
            QualityIssue.WRONG_DIMENSION: 0.6,
            QualityIssue.LOW_MAGNITUDE: 0.3,
            QualityIssue.HIGH_MAGNITUDE: 0.2,
            QualityIssue.LOW_COHERENCE: 0.4,
        }
        
        for issue in issues:
            penalty = issue_penalties.get(issue, 0.1)
            base_score -= penalty
        
        # Bonus for good statistical properties
        if metrics.std_deviation > self.quality_thresholds["min_std_deviation"]:
            base_score += 0.1
        
        # Include coherence score if available
        if metrics.coherence_score is not None:
            coherence_weight = 0.3
            base_score = (base_score * (1 - coherence_weight) + 
                         metrics.coherence_score * coherence_weight)
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, issues: List[QualityIssue], 
                                metrics: QualityMetrics) -> List[str]:
        """Generate actionable recommendations for quality issues"""
        recommendations = []
        
        if QualityIssue.NULL_EMBEDDING in issues:
            recommendations.append("Regenerate embedding - null or empty vector detected")
        
        if QualityIssue.ZERO_VECTOR in issues:
            recommendations.append("Regenerate embedding - zero vector detected")
        
        if QualityIssue.WRONG_DIMENSION in issues:
            recommendations.append("Check model configuration - unexpected embedding dimension")
        
        if QualityIssue.SUSPICIOUS_VALUES in issues:
            recommendations.append("Regenerate embedding - suspicious values (NaN/Inf) detected")
        
        if QualityIssue.LOW_MAGNITUDE in issues:
            recommendations.append("Consider different model or preprocessing - low embedding magnitude")
        
        if QualityIssue.HIGH_MAGNITUDE in issues:
            recommendations.append("Check for normalization issues - unusually high magnitude")
        
        if QualityIssue.LOW_COHERENCE in issues:
            recommendations.append("Consider different model - low semantic coherence detected")
        
        if not recommendations:
            recommendations.append("Embedding quality is acceptable")
        
        return recommendations
    
    def _should_regenerate(self, issues: List[QualityIssue], quality_score: float) -> bool:
        """Determine if embedding should be regenerated"""
        critical_issues = {
            QualityIssue.NULL_EMBEDDING,
            QualityIssue.ZERO_VECTOR,
            QualityIssue.SUSPICIOUS_VALUES
        }
        
        # Regenerate for critical issues
        if any(issue in critical_issues for issue in issues):
            return True
        
        # Regenerate for very low quality score
        if quality_score < 0.3:
            return True
        
        return False
    
    async def measure_semantic_coherence(self, embedding_func, 
                                       test_cases: Optional[List[CoherenceTestCase]] = None) -> float:
        """
        Measure semantic coherence using test cases.
        
        Args:
            embedding_func: Function that generates embeddings for text
            test_cases: Optional custom test cases
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        test_cases = test_cases or self.coherence_test_cases
        scores = []
        
        try:
            for test_case in test_cases:
                # Generate embeddings for both texts
                emb1 = await embedding_func(test_case.text1)
                emb2 = await embedding_func(test_case.text2)
                
                if not emb1 or not emb2:
                    continue
                
                # Calculate cosine similarity
                similarity = 1 - cosine(emb1, emb2)
                
                # Compare with expected similarity
                expected = test_case.expected_similarity
                tolerance = test_case.tolerance
                
                # Score based on how close actual is to expected
                if abs(similarity - expected) <= tolerance:
                    score = 1.0
                else:
                    # Gradual penalty for deviation
                    deviation = abs(similarity - expected) - tolerance
                    score = max(0.0, 1.0 - deviation)
                
                scores.append(score)
                
                logger.debug(f"Coherence test: '{test_case.text1}' vs '{test_case.text2}' "
                           f"- Expected: {expected:.2f}, Actual: {similarity:.2f}, Score: {score:.2f}")
            
            if scores:
                coherence_score = statistics.mean(scores)
                logger.info(f"Overall coherence score: {coherence_score:.3f} ({len(scores)} tests)")
                return coherence_score
            else:
                logger.warning("No coherence tests completed successfully")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error measuring semantic coherence: {e}")
            return 0.0
    
    async def validate_batch(self, embeddings: List[Tuple[List[float], str, str]]) -> List[ValidationResult]:
        """
        Validate multiple embeddings concurrently.
        
        Args:
            embeddings: List of (embedding, model_name, text_content) tuples
            
        Returns:
            List of ValidationResult objects
        """
        tasks = [
            self.validate_embedding(emb, model, text)
            for emb, model, text in embeddings
        ]
        
        return await asyncio.gather(*tasks)
    
    async def regenerate_failed_embeddings(self, validation_results: List[ValidationResult],
                                         embedding_func, texts: List[str]) -> List[Tuple[int, List[float]]]:
        """
        Regenerate embeddings that failed quality validation.
        
        Args:
            validation_results: Results from previous validation
            embedding_func: Function to generate new embeddings
            texts: Original texts corresponding to validation results
            
        Returns:
            List of (index, new_embedding) tuples for regenerated embeddings
        """
        regenerated = []
        
        for i, result in enumerate(validation_results):
            if result.should_regenerate and i < len(texts):
                try:
                    logger.info(f"Regenerating embedding for text {i} due to quality issues: {result.issues}")
                    new_embedding = await embedding_func(texts[i])
                    
                    if new_embedding:
                        # Validate the new embedding
                        new_result = await self.validate_embedding(new_embedding)
                        
                        if new_result.quality_score > result.quality_score:
                            regenerated.append((i, new_embedding))
                            logger.info(f"Successfully regenerated embedding {i} "
                                      f"(quality improved from {result.quality_score:.3f} to {new_result.quality_score:.3f})")
                        else:
                            logger.warning(f"Regenerated embedding {i} did not improve quality")
                    
                except Exception as e:
                    logger.error(f"Failed to regenerate embedding {i}: {e}")
        
        return regenerated
    
    def get_quality_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of validation results"""
        if not validation_results:
            return {}
        
        valid_count = sum(1 for r in validation_results if r.is_valid)
        quality_scores = [r.quality_score for r in validation_results]
        
        # Count issues by type
        issue_counts = {}
        for result in validation_results:
            for issue in result.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        
        return {
            "total_embeddings": len(validation_results),
            "valid_embeddings": valid_count,
            "invalid_embeddings": len(validation_results) - valid_count,
            "validity_rate": valid_count / len(validation_results),
            "mean_quality_score": statistics.mean(quality_scores),
            "median_quality_score": statistics.median(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "issue_counts": issue_counts,
            "regeneration_needed": sum(1 for r in validation_results if r.should_regenerate)
        }
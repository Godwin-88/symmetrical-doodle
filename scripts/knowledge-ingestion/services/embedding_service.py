"""
Embedding Generation Service

This module provides the main embedding generation service that integrates
content classification, model routing, and quality validation for the
Google Drive Knowledge Base Ingestion system.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import local services
from .content_classifier import ContentClassifier, ClassificationResult
from .embedding_router import EmbeddingRouter, EmbeddingResult, EmbeddingModel
from .embedding_quality_validator import EmbeddingQualityValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingServiceResult:
    """Complete result from embedding service including all metadata"""
    embedding: List[float]
    model_used: EmbeddingModel
    classification: ClassificationResult
    quality_validation: ValidationResult
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing"""
    total_processed: int
    successful: int
    failed: int
    regenerated: int
    average_quality_score: float
    processing_time: float
    model_usage: Dict[str, int]


class EmbeddingService:
    """
    Main embedding generation service that orchestrates content classification,
    model selection, embedding generation, and quality validation.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 enable_quality_validation: bool = True,
                 enable_regeneration: bool = True):
        """
        Initialize the embedding service with all components.
        
        Args:
            openai_api_key: OpenAI API key for OpenAI models
            enable_quality_validation: Whether to perform quality validation
            enable_regeneration: Whether to regenerate failed embeddings
        """
        self.classifier = ContentClassifier()
        self.router = EmbeddingRouter(openai_api_key)
        self.validator = EmbeddingQualityValidator() if enable_quality_validation else None
        
        self.enable_quality_validation = enable_quality_validation
        self.enable_regeneration = enable_regeneration
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "regenerated_embeddings": 0,
            "model_usage": {},
            "average_processing_time": 0.0
        }
        
        logger.info(f"EmbeddingService initialized (validation: {enable_quality_validation}, "
                   f"regeneration: {enable_regeneration})")
    
    async def generate_embedding(self, text: str, title: str = "", 
                               max_retries: int = 2) -> EmbeddingServiceResult:
        """
        Generate a high-quality embedding for text content.
        
        Args:
            text: Text content to embed
            title: Optional title for additional context
            max_retries: Maximum number of regeneration attempts
            
        Returns:
            EmbeddingServiceResult with embedding and metadata
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Step 1: Classify content
            classification = await self.classifier.classify_content(text, title)
            logger.debug(f"Content classified as {classification.domain.value}/{classification.content_type.value} "
                        f"(confidence: {classification.confidence_score:.3f})")
            
            # Step 2: Generate embedding
            embedding_result = await self.router.generate_embedding(text, title)
            
            if not embedding_result.success:
                return self._create_error_result(
                    f"Embedding generation failed: {embedding_result.error_message}",
                    classification, start_time
                )
            
            # Update model usage stats
            model_name = embedding_result.model_used.value
            self.stats["model_usage"][model_name] = self.stats["model_usage"].get(model_name, 0) + 1
            
            # Step 3: Quality validation (if enabled)
            quality_validation = None
            if self.enable_quality_validation and self.validator:
                quality_validation = await self.validator.validate_embedding(
                    embedding_result.embedding, 
                    model_name, 
                    text
                )
                
                logger.debug(f"Quality validation: score={quality_validation.quality_score:.3f}, "
                           f"issues={[issue.value for issue in quality_validation.issues]}")
                
                # Step 4: Regeneration (if needed and enabled)
                if (quality_validation.should_regenerate and 
                    self.enable_regeneration and 
                    max_retries > 0):
                    
                    logger.info(f"Regenerating embedding due to quality issues: {quality_validation.issues}")
                    regenerated_result = await self.generate_embedding(text, title, max_retries - 1)
                    
                    if regenerated_result.success and regenerated_result.quality_validation:
                        if regenerated_result.quality_validation.quality_score > quality_validation.quality_score:
                            self.stats["regenerated_embeddings"] += 1
                            return regenerated_result
            
            # Create successful result
            processing_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self._update_average_processing_time(processing_time)
            
            return EmbeddingServiceResult(
                embedding=embedding_result.embedding,
                model_used=embedding_result.model_used,
                classification=classification,
                quality_validation=quality_validation,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in embedding service: {e}")
            return self._create_error_result(str(e), None, start_time)
    
    def _create_error_result(self, error_message: str, 
                           classification: Optional[ClassificationResult],
                           start_time: float) -> EmbeddingServiceResult:
        """Create an error result"""
        self.stats["failed_requests"] += 1
        processing_time = time.time() - start_time
        self._update_average_processing_time(processing_time)
        
        return EmbeddingServiceResult(
            embedding=[],
            model_used=EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002,  # Default
            classification=classification,
            quality_validation=None,
            processing_time=processing_time,
            success=False,
            error_message=error_message
        )
    
    def _update_average_processing_time(self, processing_time: float) -> None:
        """Update running average of processing time"""
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_processing_time"]
        
        # Calculate new average
        self.stats["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    async def generate_batch_embeddings(self, texts: List[Tuple[str, str]], 
                                      max_concurrent: int = 10) -> Tuple[List[EmbeddingServiceResult], BatchProcessingStats]:
        """
        Generate embeddings for multiple texts with batch optimization.
        
        Args:
            texts: List of (text, title) tuples
            max_concurrent: Maximum concurrent processing
            
        Returns:
            Tuple of (results, batch_stats)
        """
        start_time = time.time()
        
        # Process in batches to avoid overwhelming the system
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(text_title: Tuple[str, str]) -> EmbeddingServiceResult:
            async with semaphore:
                text, title = text_title
                return await self.generate_embedding(text, title)
        
        # Process all texts
        tasks = [process_single(text_title) for text_title in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing text {i}: {result}")
                processed_results.append(self._create_error_result(str(result), None, start_time))
            else:
                processed_results.append(result)
        
        # Calculate batch statistics
        batch_stats = self._calculate_batch_stats(processed_results, time.time() - start_time)
        
        logger.info(f"Batch processing completed: {batch_stats.successful}/{batch_stats.total_processed} successful "
                   f"({batch_stats.average_quality_score:.3f} avg quality, {batch_stats.processing_time:.2f}s)")
        
        return processed_results, batch_stats
    
    def _calculate_batch_stats(self, results: List[EmbeddingServiceResult], 
                             processing_time: float) -> BatchProcessingStats:
        """Calculate statistics for batch processing"""
        total_processed = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_processed - successful
        regenerated = self.stats["regenerated_embeddings"]  # This is cumulative
        
        # Calculate average quality score for successful results
        quality_scores = [
            r.quality_validation.quality_score 
            for r in results 
            if r.success and r.quality_validation
        ]
        average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Count model usage in this batch
        model_usage = {}
        for result in results:
            if result.success:
                model_name = result.model_used.value
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        return BatchProcessingStats(
            total_processed=total_processed,
            successful=successful,
            failed=failed,
            regenerated=regenerated,
            average_quality_score=average_quality_score,
            processing_time=processing_time,
            model_usage=model_usage
        )
    
    async def measure_coherence(self, test_cases: Optional[List] = None) -> float:
        """
        Measure semantic coherence of the embedding system.
        
        Args:
            test_cases: Optional custom test cases
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not self.validator:
            logger.warning("Quality validation disabled, cannot measure coherence")
            return 0.0
        
        async def embedding_func(text: str) -> List[float]:
            result = await self.generate_embedding(text)
            return result.embedding if result.success else []
        
        return await self.validator.measure_semantic_coherence(embedding_func, test_cases)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current service statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset service statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "regenerated_embeddings": 0,
            "model_usage": {},
            "average_processing_time": 0.0
        }
    
    async def get_available_models(self) -> List[EmbeddingModel]:
        """Get list of currently available embedding models"""
        return self.router.get_available_models()
    
    def get_model_info(self, model: EmbeddingModel) -> Dict[str, Any]:
        """Get information about a specific model"""
        config = self.router.get_model_info(model)
        return {
            "model": config.model.value,
            "max_tokens": config.max_tokens,
            "dimension": config.dimension,
            "batch_size": config.batch_size,
            "requires_gpu": config.requires_gpu,
            "api_based": config.api_based
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.router.cleanup()
        logger.info("EmbeddingService cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
"""
Multi-Model Embedding Router

This module provides intelligent embedding model selection and routing based on
content classification. It integrates multiple embedding models optimized for
different content types and domains.
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np

# Import embedding libraries
import openai
from sentence_transformers import SentenceTransformer
import torch

# Import local modules
from .content_classifier import ContentClassifier, ContentDomain, ContentType, ClassificationResult

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models"""
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: List[float]
    model_used: EmbeddingModel
    dimension: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for embedding models"""
    model: EmbeddingModel
    max_tokens: int
    dimension: int
    batch_size: int
    requires_gpu: bool
    api_based: bool


class EmbeddingRouter:
    """
    Intelligent embedding router that selects optimal models based on content
    classification and generates high-quality embeddings for semantic search.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the embedding router with model configurations.
        
        Args:
            openai_api_key: OpenAI API key for OpenAI models
        """
        self.classifier = ContentClassifier()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize model configurations
        self.model_configs = self._initialize_model_configs()
        
        # Initialize loaded models cache
        self._loaded_models: Dict[EmbeddingModel, Any] = {}
        self._model_selection_rules = self._initialize_selection_rules()
        
        # Set up OpenAI client if API key is available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        logger.info("EmbeddingRouter initialized")
    
    def _initialize_model_configs(self) -> Dict[EmbeddingModel, ModelConfig]:
        """Initialize configuration for each embedding model"""
        return {
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE: ModelConfig(
                model=EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
                max_tokens=8192,
                dimension=3072,
                batch_size=100,
                requires_gpu=False,
                api_based=True
            ),
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002: ModelConfig(
                model=EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002,
                max_tokens=8192,
                dimension=1536,
                batch_size=100,
                requires_gpu=False,
                api_based=True
            ),
            EmbeddingModel.BGE_LARGE_EN_V1_5: ModelConfig(
                model=EmbeddingModel.BGE_LARGE_EN_V1_5,
                max_tokens=512,
                dimension=1024,
                batch_size=32,
                requires_gpu=True,
                api_based=False
            ),
            EmbeddingModel.ALL_MPNET_BASE_V2: ModelConfig(
                model=EmbeddingModel.ALL_MPNET_BASE_V2,
                max_tokens=384,
                dimension=768,
                batch_size=32,
                requires_gpu=True,
                api_based=False
            )
        }
    
    def _initialize_selection_rules(self) -> Dict[Tuple[ContentDomain, ContentType], EmbeddingModel]:
        """Initialize model selection rules based on content classification"""
        return {
            # Financial content - use BGE for financial domain expertise
            (ContentDomain.FINANCE, ContentType.GENERAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            (ContentDomain.FINANCE, ContentType.MATHEMATICAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            (ContentDomain.FINANCE, ContentType.MIXED): EmbeddingModel.BGE_LARGE_EN_V1_5,
            
            # Mathematical content - use all-mpnet for math awareness
            (ContentDomain.MACHINE_LEARNING, ContentType.MATHEMATICAL): EmbeddingModel.ALL_MPNET_BASE_V2,
            (ContentDomain.DEEP_REINFORCEMENT_LEARNING, ContentType.MATHEMATICAL): EmbeddingModel.ALL_MPNET_BASE_V2,
            (ContentDomain.NATURAL_LANGUAGE_PROCESSING, ContentType.MATHEMATICAL): EmbeddingModel.ALL_MPNET_BASE_V2,
            (ContentDomain.LARGE_LANGUAGE_MODELS, ContentType.MATHEMATICAL): EmbeddingModel.ALL_MPNET_BASE_V2,
            (ContentDomain.GENERAL, ContentType.MATHEMATICAL): EmbeddingModel.ALL_MPNET_BASE_V2,
            
            # Technical content - use BGE for technical domains
            (ContentDomain.MACHINE_LEARNING, ContentType.GENERAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            (ContentDomain.DEEP_REINFORCEMENT_LEARNING, ContentType.GENERAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            (ContentDomain.NATURAL_LANGUAGE_PROCESSING, ContentType.GENERAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            (ContentDomain.LARGE_LANGUAGE_MODELS, ContentType.GENERAL): EmbeddingModel.BGE_LARGE_EN_V1_5,
            
            # Mixed content - use OpenAI for broad coverage
            (ContentDomain.MACHINE_LEARNING, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            (ContentDomain.DEEP_REINFORCEMENT_LEARNING, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            (ContentDomain.NATURAL_LANGUAGE_PROCESSING, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            (ContentDomain.LARGE_LANGUAGE_MODELS, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            (ContentDomain.FINANCE, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            
            # General content - use OpenAI for broad coverage
            (ContentDomain.GENERAL, ContentType.GENERAL): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
            (ContentDomain.GENERAL, ContentType.MIXED): EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE,
        }
    
    async def select_model(self, classification: ClassificationResult) -> EmbeddingModel:
        """
        Select the optimal embedding model based on content classification.
        
        Args:
            classification: Content classification result
            
        Returns:
            Selected embedding model
        """
        # Primary selection based on domain and content type
        key = (classification.domain, classification.content_type)
        selected_model = self._model_selection_rules.get(key)
        
        if selected_model:
            # Check if model is available
            if await self._is_model_available(selected_model):
                logger.info(f"Selected model {selected_model.value} for {classification.domain.value}/{classification.content_type.value}")
                return selected_model
        
        # Fallback logic based on confidence and availability
        fallback_model = await self._select_fallback_model(classification)
        logger.warning(f"Using fallback model {fallback_model.value} for {classification.domain.value}/{classification.content_type.value}")
        return fallback_model
    
    async def _is_model_available(self, model: EmbeddingModel) -> bool:
        """Check if a model is available for use"""
        config = self.model_configs[model]
        
        if config.api_based:
            # Check API key availability
            return self.openai_api_key is not None
        else:
            # Check GPU availability for local models
            if config.requires_gpu and not torch.cuda.is_available():
                return False
            return True
    
    async def _select_fallback_model(self, classification: ClassificationResult) -> EmbeddingModel:
        """Select fallback model when primary selection fails"""
        # Try OpenAI models first (most reliable)
        if await self._is_model_available(EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE):
            return EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE
        
        if await self._is_model_available(EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002):
            return EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002
        
        # Try local models
        if await self._is_model_available(EmbeddingModel.BGE_LARGE_EN_V1_5):
            return EmbeddingModel.BGE_LARGE_EN_V1_5
        
        if await self._is_model_available(EmbeddingModel.ALL_MPNET_BASE_V2):
            return EmbeddingModel.ALL_MPNET_BASE_V2
        
        # This should not happen in normal circumstances
        raise RuntimeError("No embedding models available")
    
    async def generate_embedding(self, text: str, title: str = "") -> EmbeddingResult:
        """
        Generate embedding for a single text using optimal model selection.
        
        Args:
            text: Text content to embed
            title: Optional title for context
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Classify content
            classification = await self.classifier.classify_content(text, title)
            
            # Select optimal model
            selected_model = await self.select_model(classification)
            
            # Generate embedding
            embedding = await self._generate_with_model(text, selected_model)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                model_used=selected_model,
                dimension=len(embedding),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error generating embedding: {e}")
            
            return EmbeddingResult(
                embedding=[],
                model_used=EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002,  # Default
                dimension=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _generate_with_model(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using specified model"""
        config = self.model_configs[model]
        
        if config.api_based:
            return await self._generate_openai_embedding(text, model)
        else:
            return await self._generate_local_embedding(text, model)
    
    async def _generate_openai_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            # Truncate text if too long
            config = self.model_configs[model]
            if len(text.split()) > config.max_tokens:
                words = text.split()[:config.max_tokens]
                text = " ".join(words)
                logger.warning(f"Text truncated to {config.max_tokens} tokens for {model.value}")
            
            response = await asyncio.to_thread(
                openai.embeddings.create,
                input=text,
                model=model.value
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    async def _generate_local_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using local sentence-transformers model"""
        try:
            # Load model if not already loaded
            if model not in self._loaded_models:
                await self._load_local_model(model)
            
            model_instance = self._loaded_models[model]
            
            # Generate embedding
            embedding = await asyncio.to_thread(
                model_instance.encode,
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    async def _load_local_model(self, model: EmbeddingModel) -> None:
        """Load local sentence-transformers model"""
        try:
            logger.info(f"Loading local model: {model.value}")
            
            model_instance = await asyncio.to_thread(
                SentenceTransformer,
                model.value
            )
            
            self._loaded_models[model] = model_instance
            logger.info(f"Successfully loaded model: {model.value}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model.value}: {e}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[Tuple[str, str]]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts concurrently.
        
        Args:
            texts: List of (text, title) tuples
            
        Returns:
            List of EmbeddingResult objects
        """
        # Group texts by optimal model for batch processing
        model_groups = await self._group_texts_by_model(texts)
        
        # Process each group concurrently
        tasks = []
        for model, text_indices in model_groups.items():
            batch_texts = [texts[i] for i in text_indices]
            task = self._process_model_batch(model, batch_texts, text_indices)
            tasks.append(task)
        
        # Collect results and maintain original order
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten and reorder results
        results = [None] * len(texts)
        for batch_result in batch_results:
            for result, original_index in batch_result:
                results[original_index] = result
        
        return results
    
    async def _group_texts_by_model(self, texts: List[Tuple[str, str]]) -> Dict[EmbeddingModel, List[int]]:
        """Group texts by their optimal embedding model"""
        model_groups = {}
        
        # Classify all texts
        classifications = await self.classifier.classify_batch(texts)
        
        # Group by selected model
        for i, classification in enumerate(classifications):
            model = await self.select_model(classification)
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(i)
        
        return model_groups
    
    async def _process_model_batch(self, model: EmbeddingModel, texts: List[Tuple[str, str]], 
                                 indices: List[int]) -> List[Tuple[EmbeddingResult, int]]:
        """Process a batch of texts with the same model"""
        results = []
        
        # Process in smaller batches based on model configuration
        config = self.model_configs[model]
        batch_size = config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_indices = indices[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_tasks = [
                self.generate_embedding(text, title) 
                for text, title in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Pair results with original indices
            for result, original_index in zip(batch_results, batch_indices):
                results.append((result, original_index))
        
        return results
    
    def get_model_info(self, model: EmbeddingModel) -> ModelConfig:
        """Get configuration information for a model"""
        return self.model_configs[model]
    
    def get_available_models(self) -> List[EmbeddingModel]:
        """Get list of currently available models"""
        available = []
        for model in EmbeddingModel:
            if asyncio.run(self._is_model_available(model)):
                available.append(model)
        return available
    
    async def cleanup(self) -> None:
        """Clean up loaded models and resources"""
        for model_instance in self._loaded_models.values():
            if hasattr(model_instance, 'cpu'):
                model_instance.cpu()
        
        self._loaded_models.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("EmbeddingRouter cleanup completed")
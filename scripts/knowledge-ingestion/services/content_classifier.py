"""
Content Classification Service

This module provides domain classification and content type detection for the
Google Drive Knowledge Base Ingestion system. It classifies content into
technical domains (ML, DRL, NLP, LLMs, finance, general) and detects
mathematical vs. general text content.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


class ContentDomain(Enum):
    """Content domain classifications"""
    MACHINE_LEARNING = "ML"
    DEEP_REINFORCEMENT_LEARNING = "DRL"
    NATURAL_LANGUAGE_PROCESSING = "NLP"
    LARGE_LANGUAGE_MODELS = "LLMs"
    FINANCE = "finance"
    GENERAL = "general"


class ContentType(Enum):
    """Content type classifications"""
    MATHEMATICAL = "mathematical"
    GENERAL = "general"
    MIXED = "mixed"


@dataclass
class ClassificationResult:
    """Result of content classification"""
    domain: ContentDomain
    content_type: ContentType
    confidence_score: float
    domain_scores: Dict[str, float]
    mathematical_indicators: List[str]
    reasoning: str


class ContentClassifier:
    """
    Intelligent content classification system that categorizes text content
    by domain and type for optimal embedding model selection.
    """
    
    def __init__(self):
        """Initialize the content classifier with domain keywords and patterns"""
        self._domain_keywords = self._initialize_domain_keywords()
        self._math_patterns = self._initialize_math_patterns()
        self._confidence_threshold = 0.3
        
    def _initialize_domain_keywords(self) -> Dict[ContentDomain, List[str]]:
        """Initialize domain-specific keywords for classification"""
        return {
            ContentDomain.MACHINE_LEARNING: [
                "machine learning", "ml", "supervised learning", "unsupervised learning",
                "neural network", "deep learning", "gradient descent", "backpropagation",
                "feature engineering", "cross validation", "overfitting", "regularization",
                "classification", "regression", "clustering", "dimensionality reduction",
                "support vector machine", "random forest", "decision tree", "ensemble",
                "scikit-learn", "tensorflow", "pytorch", "keras", "model training",
                "hyperparameter", "loss function", "activation function", "optimizer"
            ],
            ContentDomain.DEEP_REINFORCEMENT_LEARNING: [
                "reinforcement learning", "rl", "deep reinforcement learning", "drl",
                "q-learning", "policy gradient", "actor-critic", "dqn", "ddpg", "ppo",
                "markov decision process", "mdp", "reward function", "value function",
                "exploration", "exploitation", "epsilon-greedy", "temporal difference",
                "monte carlo", "bellman equation", "state space", "action space",
                "environment", "agent", "episode", "trajectory", "experience replay",
                "target network", "soft update", "advantage function", "gae"
            ],
            ContentDomain.NATURAL_LANGUAGE_PROCESSING: [
                "natural language processing", "nlp", "text processing", "tokenization",
                "part of speech", "pos tagging", "named entity recognition", "ner",
                "sentiment analysis", "text classification", "language model",
                "word embedding", "word2vec", "glove", "fasttext", "bert", "transformer",
                "attention mechanism", "sequence to sequence", "seq2seq", "rnn", "lstm",
                "gru", "text generation", "machine translation", "summarization",
                "question answering", "information extraction", "parsing", "syntax"
            ],
            ContentDomain.LARGE_LANGUAGE_MODELS: [
                "large language model", "llm", "gpt", "bert", "transformer", "attention",
                "pre-training", "fine-tuning", "prompt engineering", "in-context learning",
                "few-shot learning", "zero-shot learning", "chain of thought", "reasoning",
                "instruction following", "alignment", "rlhf", "constitutional ai",
                "emergent abilities", "scaling laws", "parameter count", "tokens",
                "context window", "temperature", "top-k", "top-p", "nucleus sampling",
                "beam search", "greedy decoding", "perplexity", "bleu score", "rouge"
            ],
            ContentDomain.FINANCE: [
                "finance", "financial", "trading", "investment", "portfolio", "risk",
                "return", "volatility", "sharpe ratio", "alpha", "beta", "correlation",
                "diversification", "asset allocation", "market", "stock", "bond",
                "derivative", "option", "future", "swap", "hedge", "arbitrage",
                "quantitative finance", "algorithmic trading", "high frequency trading",
                "market making", "liquidity", "spread", "slippage", "execution",
                "backtesting", "performance attribution", "var", "value at risk",
                "credit risk", "market risk", "operational risk", "compliance"
            ],
            ContentDomain.GENERAL: [
                "research", "analysis", "methodology", "experiment", "data", "statistics",
                "probability", "hypothesis", "theory", "framework", "approach", "technique",
                "algorithm", "optimization", "performance", "evaluation", "metrics",
                "benchmark", "comparison", "survey", "review", "literature", "study"
            ]
        }
    
    def _initialize_math_patterns(self) -> List[re.Pattern]:
        """Initialize regex patterns for mathematical content detection"""
        patterns = [
            # LaTeX mathematical expressions
            r'\$[^$]+\$',  # Inline math
            r'\$\$[^$]+\$\$',  # Display math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{matrix\}.*?\\end\{matrix\}',
            
            # Mathematical symbols and operators
            r'[∑∏∫∂∇∆∞±≤≥≠≈∈∉⊂⊃∪∩]',
            r'[αβγδεζηθικλμνξοπρστυφχψω]',
            r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',
            
            # Mathematical notation
            r'\b(?:lim|sup|inf|max|min|arg(?:max|min))\b',
            r'\b(?:sin|cos|tan|log|ln|exp|sqrt)\b',
            r'\b(?:theorem|lemma|proof|corollary|proposition)\b',
            
            # Equations and formulas
            r'[a-zA-Z]\s*[=≈≠]\s*[^a-zA-Z\s]',
            r'\b\d+\.\d+\s*[×·*]\s*\d+',
            r'\([^)]*\)\s*[²³⁴⁵⁶⁷⁸⁹]',
            
            # Fractions and exponents
            r'\b\d+/\d+\b',
            r'\b[a-zA-Z]\^\{[^}]+\}',
            r'\b[a-zA-Z]_\{[^}]+\}',
        ]
        
        return [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    
    async def classify_content(self, text: str, title: str = "") -> ClassificationResult:
        """
        Classify content by domain and type with confidence scoring.
        
        Args:
            text: The text content to classify
            title: Optional title for additional context
            
        Returns:
            ClassificationResult with domain, type, and confidence information
        """
        try:
            # Combine title and text for classification
            full_text = f"{title} {text}".lower()
            
            # Calculate domain scores
            domain_scores = await self._calculate_domain_scores(full_text)
            
            # Determine primary domain
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])
            domain = ContentDomain(primary_domain[0])
            confidence = primary_domain[1]
            
            # Detect mathematical content
            math_indicators = self._detect_mathematical_content(text)
            content_type = self._determine_content_type(math_indicators, text)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(domain_scores, math_indicators, confidence)
            
            return ClassificationResult(
                domain=domain,
                content_type=content_type,
                confidence_score=confidence,
                domain_scores={k: v for k, v in domain_scores.items()},
                mathematical_indicators=math_indicators,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error classifying content: {e}")
            # Return default classification on error
            return ClassificationResult(
                domain=ContentDomain.GENERAL,
                content_type=ContentType.GENERAL,
                confidence_score=0.0,
                domain_scores={},
                mathematical_indicators=[],
                reasoning=f"Classification failed: {str(e)}"
            )
    
    async def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate confidence scores for each domain"""
        scores = {}
        text_words = set(text.split())
        
        for domain, keywords in self._domain_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Calculate weighted score based on keyword frequency and rarity
            if matches > 0:
                # Base score from match count
                base_score = matches / len(keywords)
                
                # Boost score for exact phrase matches
                phrase_boost = sum(0.1 for keyword in keywords if keyword in text and len(keyword.split()) > 1)
                
                # Normalize by text length (longer texts get slight penalty)
                length_factor = min(1.0, 1000 / max(len(text), 100))
                
                scores[domain.value] = min(1.0, base_score + phrase_boost) * length_factor
            else:
                scores[domain.value] = 0.0
        
        return scores
    
    def _detect_mathematical_content(self, text: str) -> List[str]:
        """Detect mathematical indicators in text"""
        indicators = []
        
        for pattern in self._math_patterns:
            matches = pattern.findall(text)
            if matches:
                indicators.extend(matches[:5])  # Limit to first 5 matches per pattern
        
        return indicators
    
    def _determine_content_type(self, math_indicators: List[str], text: str) -> ContentType:
        """Determine if content is mathematical, general, or mixed"""
        if not math_indicators:
            return ContentType.GENERAL
        
        # Calculate mathematical density
        math_density = len(math_indicators) / max(len(text.split()), 1)
        
        if math_density > 0.05:  # More than 5% mathematical indicators
            return ContentType.MATHEMATICAL
        elif math_density > 0.01:  # 1-5% mathematical indicators
            return ContentType.MIXED
        else:
            return ContentType.GENERAL
    
    def _generate_reasoning(self, domain_scores: Dict[str, float], 
                          math_indicators: List[str], confidence: float) -> str:
        """Generate human-readable reasoning for the classification"""
        top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_domains_str = ", ".join([f"{domain}: {score:.2f}" for domain, score in top_domains])
        
        math_info = f"Mathematical indicators found: {len(math_indicators)}" if math_indicators else "No mathematical content detected"
        
        return f"Domain scores: {top_domains_str}. {math_info}. Confidence: {confidence:.2f}"
    
    async def classify_batch(self, texts: List[Tuple[str, str]]) -> List[ClassificationResult]:
        """
        Classify multiple texts concurrently.
        
        Args:
            texts: List of (text, title) tuples
            
        Returns:
            List of ClassificationResult objects
        """
        tasks = [self.classify_content(text, title) for text, title in texts]
        return await asyncio.gather(*tasks)
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold"""
        return self._confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for classification"""
        if 0.0 <= threshold <= 1.0:
            self._confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
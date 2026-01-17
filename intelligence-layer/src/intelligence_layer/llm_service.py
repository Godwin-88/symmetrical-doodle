"""LLM Service for financial intelligence and research."""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from datetime import datetime, timezone
import aiohttp
import openai
import anthropic
from groq import Groq
import together

from .llm_config import LLMIntelligenceConfig, load_llm_config
from .logging import get_logger

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """LLM response model."""
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class LocalLLMService:
    """Local LLM service using transformers/llama.cpp."""
    
    def __init__(self, config: LLMIntelligenceConfig):
        self.config = config.local_llm
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    async def initialize(self):
        """Initialize local LLM model."""
        if not self.config.enabled:
            logger.info("Local LLM disabled")
            return
        
        try:
            # Import here to avoid dependency issues if not using local LLM
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading local LLM: {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Configure device and quantization
            device_map = "auto" if self.config.device == "cuda" else None
            torch_dtype = torch.float16 if self.config.device == "cuda" else torch.float32
            
            # Load model with quantization if specified
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
            
            self.is_loaded = True
            logger.info("Local LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.is_loaded = False
    
    async def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Generate response using local LLM."""
        if not self.is_loaded:
            raise RuntimeError("Local LLM not loaded")
        
        start_time = time.time()
        
        try:
            # Format prompt based on model type
            if self.config.model_type == "llama":
                formatted_prompt = self._format_llama_prompt(prompt, system_prompt)
            else:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.context_length - self.config.max_tokens
            )
            
            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text.strip(),
                model=self.config.model_path,
                provider="local",
                tokens_used=len(response_tokens),
                response_time=response_time,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise
    
    def _format_llama_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """Format prompt for Llama models."""
        if system_prompt:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


class ExternalLLMService:
    """External LLM service supporting multiple providers."""
    
    def __init__(self, config: LLMIntelligenceConfig):
        self.config = config.external_llm
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients."""
        if self.config.openai_api_key:
            self.clients['openai'] = openai.AsyncOpenAI(api_key=self.config.openai_api_key)
        
        if self.config.anthropic_api_key:
            self.clients['anthropic'] = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
        
        if self.config.groq_api_key:
            self.clients['groq'] = Groq(api_key=self.config.groq_api_key)
        
        if self.config.together_api_key:
            self.clients['together'] = together.AsyncTogether(api_key=self.config.together_api_key)
    
    async def generate(self, prompt: str, system_prompt: str = None, provider: str = None) -> LLMResponse:
        """Generate response using external LLM."""
        if not self.config.enabled:
            raise RuntimeError("External LLM disabled")
        
        provider = provider or self.config.primary_provider
        
        # Try primary provider first, then fallbacks
        providers_to_try = [provider] + [p for p in self.config.fallback_providers if p != provider]
        
        for current_provider in providers_to_try:
            if current_provider not in self.clients:
                continue
            
            try:
                return await self._generate_with_provider(prompt, system_prompt, current_provider)
            except Exception as e:
                logger.warning(f"Provider {current_provider} failed: {e}")
                continue
        
        raise RuntimeError("All external LLM providers failed")
    
    async def _generate_with_provider(self, prompt: str, system_prompt: str, provider: str) -> LLMResponse:
        """Generate response with specific provider."""
        start_time = time.time()
        
        if provider == "openai":
            return await self._generate_openai(prompt, system_prompt, start_time)
        elif provider == "anthropic":
            return await self._generate_anthropic(prompt, system_prompt, start_time)
        elif provider == "groq":
            return await self._generate_groq(prompt, system_prompt, start_time)
        elif provider == "together":
            return await self._generate_together(prompt, system_prompt, start_time)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _generate_openai(self, prompt: str, system_prompt: str, start_time: float) -> LLMResponse:
        """Generate using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.clients['openai'].chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.config.openai_model,
            provider="openai",
            tokens_used=response.usage.total_tokens,
            response_time=time.time() - start_time,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _generate_anthropic(self, prompt: str, system_prompt: str, start_time: float) -> LLMResponse:
        """Generate using Anthropic."""
        response = await self.clients['anthropic'].messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.config.anthropic_model,
            provider="anthropic",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            response_time=time.time() - start_time,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _generate_groq(self, prompt: str, system_prompt: str, start_time: float) -> LLMResponse:
        """Generate using Groq."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.clients['groq'].chat.completions.create(
            model=self.config.groq_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.config.groq_model,
            provider="groq",
            tokens_used=response.usage.total_tokens,
            response_time=time.time() - start_time,
            timestamp=datetime.now(timezone.utc)
        )


class LLMService:
    """Main LLM service orchestrator."""
    
    def __init__(self, config: LLMIntelligenceConfig = None):
        self.config = config or load_llm_config()
        self.local_service = LocalLLMService(self.config) if self.config.local_llm.enabled else None
        self.external_service = ExternalLLMService(self.config) if self.config.external_llm.enabled else None
        self.request_cache = {}
    
    async def initialize(self):
        """Initialize LLM services."""
        if self.local_service:
            await self.local_service.initialize()
        
        logger.info("LLM Service initialized")
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: str = None,
        prefer_local: bool = True,
        provider: str = None
    ) -> LLMResponse:
        """Generate response using available LLM services."""
        
        # Check cache first
        cache_key = f"{prompt}:{system_prompt}:{provider}"
        if cache_key in self.request_cache:
            cached_response, timestamp = self.request_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return cached_response
        
        # Determine which service to use
        if prefer_local and self.local_service and self.local_service.is_loaded:
            try:
                response = await self.local_service.generate(prompt, system_prompt)
                self.request_cache[cache_key] = (response, time.time())
                return response
            except Exception as e:
                logger.warning(f"Local LLM failed, falling back to external: {e}")
        
        if self.external_service:
            response = await self.external_service.generate(prompt, system_prompt, provider)
            self.request_cache[cache_key] = (response, time.time())
            return response
        
        raise RuntimeError("No LLM services available")
    
    async def financial_analysis(self, query: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Perform financial analysis using specialized prompt."""
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        prompt = f"""Analyze the following financial query with professional expertise:

Query: {query}{context_str}

Please provide:
1. Key insights and analysis
2. Relevant market factors
3. Risk considerations
4. Data-driven recommendations
5. Confidence level and limitations

Format your response with clear sections and bullet points where appropriate."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt_financial,
            prefer_local=True
        )
    
    async def research_query(self, query: str, documents: List[str] = None) -> LLMResponse:
        """Perform research query with document context."""
        context_str = ""
        if documents:
            context_str = f"\n\nRelevant Documents:\n" + "\n\n".join(documents[:5])  # Limit context
        
        prompt = f"""Research Query: {query}{context_str}

Please provide comprehensive research analysis including:
1. Summary of key findings
2. Supporting evidence and sources
3. Market implications
4. Trend analysis
5. Future outlook

Be thorough but concise, and cite specific information from the provided documents."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt_research,
            prefer_local=False  # Use external for research to access latest information
        )
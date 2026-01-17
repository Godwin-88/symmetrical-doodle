"""Financial research service with web search and analysis capabilities."""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import json
import re
from bs4 import BeautifulSoup

from .llm_config import LLMIntelligenceConfig, load_llm_config
from .llm_service import LLMService
from .rag_service import RAGService
from .logging import get_logger

logger = get_logger(__name__)


class WebSearchService:
    """Web search service for financial research."""
    
    def __init__(self, config: LLMIntelligenceConfig):
        self.config = config.research
        self.session = None
    
    async def initialize(self):
        """Initialize web search service."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def close(self):
        """Close web search service."""
        if self.session:
            await self.session.close()
    
    async def search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                
                # Extract instant answer
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'DuckDuckGo Instant Answer'),
                        'snippet': data['Abstract'],
                        'url': data.get('AbstractURL', ''),
                        'source': 'duckduckgo_instant'
                    })
                
                # Extract related topics
                for topic in data.get('RelatedTopics', [])[:max_results]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                            'snippet': topic['Text'],
                            'url': topic.get('FirstURL', ''),
                            'source': 'duckduckgo_related'
                        })
                
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    async def search_financial_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search financial news sources."""
        results = []
        
        # Search multiple financial news sources
        sources = [
            "site:reuters.com/business",
            "site:bloomberg.com",
            "site:cnbc.com",
            "site:marketwatch.com",
            "site:wsj.com"
        ]
        
        for source in sources[:3]:  # Limit to avoid rate limiting
            search_query = f"{query} {source}"
            source_results = await self.search_duckduckgo(search_query, max_results // 3)
            results.extend(source_results)
        
        return results[:max_results]


class MarketDataService:
    """Market data service for financial analysis."""
    
    def __init__(self):
        pass
    
    async def get_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get stock data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period=period)
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Calculate volatility
            volatility = hist['Close'].pct_change().std() * (252 ** 0.5)  # Annualized
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "technical_indicators": {
                    "sma_20": float(sma_20) if pd.notna(sma_20) else None,
                    "sma_50": float(sma_50) if pd.notna(sma_50) else None,
                    "rsi": float(rsi) if pd.notna(rsi) else None,
                    "volatility": float(volatility) if pd.notna(volatility) else None
                },
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("longBusinessSummary", "")[:500]
            }
            
        except Exception as e:
            logger.error(f"Failed to get stock data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators."""
        try:
            # Get major indices
            indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
            index_data = {}
            
            for index in indices:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100
                    
                    index_data[index] = {
                        "current": float(current),
                        "change_percent": float(change)
                    }
            
            # Get currency data
            currencies = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
            currency_data = {}
            
            for currency in currencies:
                ticker = yf.Ticker(currency)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100
                    
                    currency_data[currency] = {
                        "current": float(current),
                        "change_percent": float(change)
                    }
            
            return {
                "indices": index_data,
                "currencies": currency_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get economic indicators: {e}")
            return {"error": str(e)}


class SentimentAnalysisService:
    """Sentiment analysis service for financial text."""
    
    def __init__(self):
        self.sentiment_model = None
    
    async def initialize(self):
        """Initialize sentiment analysis model."""
        try:
            # Try to load FinBERT for financial sentiment
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            model_name = "ProsusAI/finbert"
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("FinBERT sentiment model loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load FinBERT, using basic sentiment: {e}")
            self.sentiment_model = None
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of financial text."""
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}
        
        try:
            if self.sentiment_model:
                # Use FinBERT
                result = self.sentiment_model(text[:512])  # Limit text length
                return {
                    "sentiment": result[0]['label'].lower(),
                    "confidence": result[0]['score']
                }
            else:
                # Basic sentiment analysis
                positive_words = ["bullish", "positive", "growth", "profit", "gain", "up", "rise", "strong"]
                negative_words = ["bearish", "negative", "loss", "decline", "down", "fall", "weak", "risk"]
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                    confidence = min(0.8, positive_count / (positive_count + negative_count + 1))
                elif negative_count > positive_count:
                    sentiment = "negative"
                    confidence = min(0.8, negative_count / (positive_count + negative_count + 1))
                else:
                    sentiment = "neutral"
                    confidence = 0.5
                
                return {"sentiment": sentiment, "confidence": confidence}
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}


class FinancialResearchService:
    """Main financial research service."""
    
    def __init__(self, config: LLMIntelligenceConfig = None):
        self.config = config or load_llm_config()
        self.llm_service = LLMService(self.config)
        self.rag_service = RAGService(self.config)
        self.web_search = WebSearchService(self.config)
        self.market_data = MarketDataService()
        self.sentiment_analysis = SentimentAnalysisService()
    
    async def initialize(self):
        """Initialize research service."""
        await self.llm_service.initialize()
        await self.rag_service.initialize()
        await self.web_search.initialize()
        await self.sentiment_analysis.initialize()
        
        logger.info("Financial Research Service initialized")
    
    async def close(self):
        """Close research service."""
        await self.web_search.close()
    
    async def comprehensive_research(self, query: str, include_web: bool = True) -> Dict[str, Any]:
        """Perform comprehensive financial research."""
        try:
            research_results = {
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rag_analysis": None,
                "web_research": None,
                "market_data": None,
                "sentiment": None,
                "comprehensive_analysis": None
            }
            
            # 1. RAG-based analysis using internal documents
            try:
                rag_result = await self.rag_service.query(query)
                research_results["rag_analysis"] = rag_result
            except Exception as e:
                logger.error(f"RAG analysis failed: {e}")
                research_results["rag_analysis"] = {"error": str(e)}
            
            # 2. Web research if enabled
            if include_web and self.config.research.enable_web_search:
                try:
                    web_results = await self.web_search.search_financial_news(query)
                    
                    # Analyze sentiment of web results
                    sentiments = []
                    for result in web_results[:5]:
                        sentiment = await self.sentiment_analysis.analyze_sentiment(result['snippet'])
                        sentiments.append(sentiment)
                    
                    research_results["web_research"] = {
                        "results": web_results,
                        "sentiment_analysis": sentiments
                    }
                    
                except Exception as e:
                    logger.error(f"Web research failed: {e}")
                    research_results["web_research"] = {"error": str(e)}
            
            # 3. Extract and analyze any stock symbols mentioned
            symbols = self._extract_stock_symbols(query)
            if symbols:
                try:
                    market_data = {}
                    for symbol in symbols[:3]:  # Limit to 3 symbols
                        data = await self.market_data.get_stock_data(symbol)
                        market_data[symbol] = data
                    
                    research_results["market_data"] = market_data
                    
                except Exception as e:
                    logger.error(f"Market data analysis failed: {e}")
                    research_results["market_data"] = {"error": str(e)}
            
            # 4. Generate comprehensive analysis
            try:
                analysis_context = {
                    "rag_findings": research_results.get("rag_analysis", {}).get("answer", "No internal documents found"),
                    "web_findings": [r['snippet'] for r in research_results.get("web_research", {}).get("results", [])[:3]],
                    "market_data": research_results.get("market_data", {}),
                    "query": query
                }
                
                comprehensive_response = await self.llm_service.financial_analysis(
                    f"Provide a comprehensive financial analysis based on the following research:",
                    analysis_context
                )
                
                research_results["comprehensive_analysis"] = {
                    "analysis": comprehensive_response.content,
                    "model_info": {
                        "model": comprehensive_response.model,
                        "provider": comprehensive_response.provider,
                        "tokens_used": comprehensive_response.tokens_used
                    }
                }
                
            except Exception as e:
                logger.error(f"Comprehensive analysis failed: {e}")
                research_results["comprehensive_analysis"] = {"error": str(e)}
            
            return research_results
            
        except Exception as e:
            logger.error(f"Comprehensive research failed: {e}")
            return {"error": str(e), "query": query}
    
    async def stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform detailed stock analysis."""
        try:
            # Get market data
            market_data = await self.market_data.get_stock_data(symbol)
            
            # Search for recent news
            news_query = f"{symbol} stock analysis earnings financial"
            news_results = await self.web_search.search_financial_news(news_query, max_results=5)
            
            # Analyze news sentiment
            news_sentiments = []
            for news in news_results:
                sentiment = await self.sentiment_analysis.analyze_sentiment(news['snippet'])
                news_sentiments.append({
                    "title": news['title'],
                    "sentiment": sentiment['sentiment'],
                    "confidence": sentiment['confidence']
                })
            
            # Generate AI analysis
            analysis_prompt = f"""
            Analyze {symbol} stock based on the following data:
            
            Market Data: {json.dumps(market_data, indent=2)}
            
            Recent News Sentiment: {json.dumps(news_sentiments, indent=2)}
            
            Provide a comprehensive analysis including:
            1. Current valuation assessment
            2. Technical analysis summary
            3. Fundamental strengths and weaknesses
            4. Market sentiment overview
            5. Risk factors
            6. Investment recommendation with rationale
            """
            
            ai_analysis = await self.llm_service.financial_analysis(analysis_prompt)
            
            return {
                "symbol": symbol,
                "market_data": market_data,
                "news_analysis": {
                    "articles": news_results,
                    "sentiment_summary": news_sentiments
                },
                "ai_analysis": ai_analysis.content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stock analysis failed for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Simple regex to find potential stock symbols
        pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(pattern, text)
        
        # Filter out common words that aren't stock symbols
        common_words = {"THE", "AND", "OR", "BUT", "FOR", "WITH", "TO", "FROM", "BY", "AT", "IN", "ON", "UP", "DOWN"}
        symbols = [s for s in potential_symbols if s not in common_words and len(s) <= 5]
        
        return list(set(symbols))  # Remove duplicates
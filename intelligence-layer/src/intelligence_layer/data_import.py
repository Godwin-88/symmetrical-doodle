"""External data import service for fetching market data from various sources."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
from enum import Enum

import yfinance as yf
import pandas as pd
import numpy as np

from .logging import get_logger
from .models import MarketData

logger = get_logger(__name__)


class DataSource(str, Enum):
    """Supported external data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    FRED = "fred"
    CRYPTOCOMPARE = "cryptocompare"


class DataImporter:
    """Service for importing data from external sources."""
    
    def __init__(self):
        self.supported_sources = {
            DataSource.YAHOO_FINANCE: self._fetch_yahoo_finance,
            # Add more sources as needed
        }
    
    async def fetch_data(
        self,
        source: DataSource,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
    ) -> List[MarketData]:
        """
        Fetch market data from external source.
        
        Args:
            source: Data source to use
            symbol: Asset symbol (e.g., 'AAPL', 'EURUSD=X')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            List of MarketData objects
        """
        if source not in self.supported_sources:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Default date range: last 30 days
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.info(
            f"Fetching data from {source}",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval=interval,
        )
        
        try:
            fetch_func = self.supported_sources[source]
            data = await fetch_func(symbol, start_date, end_date, interval)
            
            logger.info(
                f"Successfully fetched {len(data)} data points",
                symbol=symbol,
                source=source,
            )
            
            return data
            
        except Exception as e:
            logger.error(
                f"Failed to fetch data from {source}",
                symbol=symbol,
                error=str(e),
            )
            raise
    
    async def _fetch_yahoo_finance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> List[MarketData]:
        """Fetch data from Yahoo Finance."""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            self._download_yahoo_data,
            symbol,
            start_date,
            end_date,
            interval,
        )
        
        # Convert to MarketData objects
        market_data = []
        for idx, row in df.iterrows():
            try:
                data = MarketData(
                    timestamp=idx.to_pydatetime(),
                    asset_id=symbol,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    metadata={
                        'source': 'yahoo_finance',
                        'interval': interval,
                    }
                )
                market_data.append(data)
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue
        
        return market_data
    
    def _download_yahoo_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance (blocking)."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )
        
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        return df
    
    async def search_symbols(
        self,
        query: str,
        source: DataSource = DataSource.YAHOO_FINANCE,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for symbols matching query.
        
        Args:
            query: Search query (e.g., 'Apple', 'EUR')
            source: Data source to search
            limit: Maximum number of results
            
        Returns:
            List of symbol information dictionaries
        """
        if source == DataSource.YAHOO_FINANCE:
            return await self._search_yahoo_symbols(query, limit)
        
        return []
    
    async def _search_yahoo_symbols(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Search Yahoo Finance symbols."""
        
        # Common forex pairs for Yahoo Finance
        forex_pairs = {
            'EUR': 'EURUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'USDJPY=X',
            'AUD': 'AUDUSD=X',
            'CHF': 'USDCHF=X',
            'CAD': 'USDCAD=X',
            'NZD': 'NZDUSD=X',
        }
        
        # Common crypto pairs
        crypto_pairs = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'USDT': 'USDT-USD',
            'BNB': 'BNB-USD',
            'XRP': 'XRP-USD',
        }
        
        results = []
        query_upper = query.upper()
        
        # Search forex
        for name, symbol in forex_pairs.items():
            if query_upper in name or query_upper in symbol:
                results.append({
                    'symbol': symbol,
                    'name': f'{name}/USD',
                    'type': 'forex',
                    'exchange': 'FX',
                })
        
        # Search crypto
        for name, symbol in crypto_pairs.items():
            if query_upper in name or query_upper in symbol:
                results.append({
                    'symbol': symbol,
                    'name': f'{name} USD',
                    'type': 'crypto',
                    'exchange': 'CCC',
                })
        
        # For stocks, try direct lookup
        if len(results) < limit:
            try:
                ticker = yf.Ticker(query_upper)
                info = ticker.info
                if info and 'symbol' in info:
                    results.append({
                        'symbol': info.get('symbol', query_upper),
                        'name': info.get('longName', query_upper),
                        'type': 'stock',
                        'exchange': info.get('exchange', 'Unknown'),
                    })
            except:
                pass
        
        return results[:limit]
    
    async def get_symbol_info(
        self,
        symbol: str,
        source: DataSource = DataSource.YAHOO_FINANCE,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Asset symbol
            source: Data source
            
        Returns:
            Symbol information dictionary
        """
        if source == DataSource.YAHOO_FINANCE:
            return await self._get_yahoo_symbol_info(symbol)
        
        return {}
    
    async def _get_yahoo_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get Yahoo Finance symbol information."""
        loop = asyncio.get_event_loop()
        
        def _get_info():
            ticker = yf.Ticker(symbol)
            return ticker.info
        
        try:
            info = await loop.run_in_executor(None, _get_info)
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'type': info.get('quoteType', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'description': info.get('longBusinessSummary'),
            }
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return {'symbol': symbol, 'error': str(e)}


# Global instance
data_importer = DataImporter()

"""
Multi-Source Real-Time Market Data Providers
Supports: Gold, Silver, Bitcoin, Forex Major Pairs, Crypto
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import json
import os

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    EQUITY = "equity"


class DataProvider(str, Enum):
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    BINANCE = "binance"
    COINBASE = "coinbase"
    METALS_API = "metals_api"
    TWELVE_DATA = "twelve_data"
    FIXER = "fixer"
    OANDA = "oanda"


@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    provider: Optional[str] = None

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        if self.mid == 0:
            return 0
        return float((self.ask - self.bid) / self.mid * 10000)


@dataclass
class OHLCV:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: float
    interval: str  # 1m, 5m, 15m, 1h, 4h, 1d
    provider: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
            "interval": self.interval,
            "provider": self.provider,
        }


@dataclass
class AssetInfo:
    """Asset metadata"""
    symbol: str
    name: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    exchange: Optional[str] = None
    min_trade_size: Optional[Decimal] = None
    tick_size: Optional[Decimal] = None
    margin_requirement: Optional[float] = None
    trading_hours: Optional[str] = None


# Standard asset definitions
STANDARD_ASSETS = {
    # Precious Metals
    "XAUUSD": AssetInfo("XAUUSD", "Gold", AssetClass.COMMODITY, "XAU", "USD", margin_requirement=5.0),
    "XAGUSD": AssetInfo("XAGUSD", "Silver", AssetClass.COMMODITY, "XAG", "USD", margin_requirement=5.0),
    "XPTUSD": AssetInfo("XPTUSD", "Platinum", AssetClass.COMMODITY, "XPT", "USD", margin_requirement=5.0),

    # Major Forex Pairs
    "EURUSD": AssetInfo("EURUSD", "Euro/US Dollar", AssetClass.FOREX, "EUR", "USD", margin_requirement=3.33),
    "GBPUSD": AssetInfo("GBPUSD", "British Pound/US Dollar", AssetClass.FOREX, "GBP", "USD", margin_requirement=3.33),
    "USDJPY": AssetInfo("USDJPY", "US Dollar/Japanese Yen", AssetClass.FOREX, "USD", "JPY", margin_requirement=3.33),
    "USDCHF": AssetInfo("USDCHF", "US Dollar/Swiss Franc", AssetClass.FOREX, "USD", "CHF", margin_requirement=3.33),
    "AUDUSD": AssetInfo("AUDUSD", "Australian Dollar/US Dollar", AssetClass.FOREX, "AUD", "USD", margin_requirement=3.33),
    "USDCAD": AssetInfo("USDCAD", "US Dollar/Canadian Dollar", AssetClass.FOREX, "USD", "CAD", margin_requirement=3.33),
    "NZDUSD": AssetInfo("NZDUSD", "New Zealand Dollar/US Dollar", AssetClass.FOREX, "NZD", "USD", margin_requirement=3.33),

    # Cross Pairs
    "EURGBP": AssetInfo("EURGBP", "Euro/British Pound", AssetClass.FOREX, "EUR", "GBP", margin_requirement=3.33),
    "EURJPY": AssetInfo("EURJPY", "Euro/Japanese Yen", AssetClass.FOREX, "EUR", "JPY", margin_requirement=3.33),
    "GBPJPY": AssetInfo("GBPJPY", "British Pound/Japanese Yen", AssetClass.FOREX, "GBP", "JPY", margin_requirement=3.33),

    # Cryptocurrencies
    "BTCUSD": AssetInfo("BTCUSD", "Bitcoin/US Dollar", AssetClass.CRYPTO, "BTC", "USD", margin_requirement=50.0),
    "ETHUSD": AssetInfo("ETHUSD", "Ethereum/US Dollar", AssetClass.CRYPTO, "ETH", "USD", margin_requirement=50.0),
    "BTCEUR": AssetInfo("BTCEUR", "Bitcoin/Euro", AssetClass.CRYPTO, "BTC", "EUR", margin_requirement=50.0),
    "ETHBTC": AssetInfo("ETHBTC", "Ethereum/Bitcoin", AssetClass.CRYPTO, "ETH", "BTC", margin_requirement=50.0),
    "SOLUSD": AssetInfo("SOLUSD", "Solana/US Dollar", AssetClass.CRYPTO, "SOL", "USD", margin_requirement=50.0),
    "XRPUSD": AssetInfo("XRPUSD", "Ripple/US Dollar", AssetClass.CRYPTO, "XRP", "USD", margin_requirement=50.0),
}


class BaseMarketDataProvider(ABC):
    """Abstract base class for market data providers"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self._rate_limit_remaining = 100
        self._last_request_time = datetime.now()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def supported_assets(self) -> List[AssetClass]:
        pass

    @abstractmethod
    async def get_tick(self, symbol: str) -> MarketTick:
        """Get current tick data"""
        pass

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        """Get historical OHLCV data"""
        pass

    async def close(self):
        await self.client.aclose()


class YahooFinanceProvider(BaseMarketDataProvider):
    """Yahoo Finance data provider (free, delayed)"""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance"

    SYMBOL_MAP = {
        # Metals
        "XAUUSD": "GC=F",  # Gold futures
        "XAGUSD": "SI=F",  # Silver futures
        "XPTUSD": "PL=F",  # Platinum futures
        # Forex
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X",
        # Crypto
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "BTCEUR": "BTC-EUR",
        "SOLUSD": "SOL-USD",
        "XRPUSD": "XRP-USD",
    }

    @property
    def provider_name(self) -> str:
        return "yahoo_finance"

    @property
    def supported_assets(self) -> List[AssetClass]:
        return [AssetClass.FOREX, AssetClass.CRYPTO, AssetClass.COMMODITY]

    def _get_yahoo_symbol(self, symbol: str) -> str:
        return self.SYMBOL_MAP.get(symbol, symbol)

    async def get_tick(self, symbol: str) -> MarketTick:
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        url = f"{self.BASE_URL}/chart/{yahoo_symbol}"
        params = {"interval": "1m", "range": "1d"}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [{}])[0]
            meta = result.get("meta", {})
            quote = result.get("indicators", {}).get("quote", [{}])[0]

            last_price = Decimal(str(meta.get("regularMarketPrice", 0)))
            # Simulate bid/ask from last price
            spread = last_price * Decimal("0.0002")  # 2 pip spread

            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=last_price - spread / 2,
                ask=last_price + spread / 2,
                last=last_price,
                volume=float(meta.get("regularMarketVolume", 0)),
                provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"Yahoo Finance tick error for {symbol}: {e}")
            raise

    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        yahoo_symbol = self._get_yahoo_symbol(symbol)

        # Map interval to Yahoo format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk"
        }
        yahoo_interval = interval_map.get(interval, "1d")

        url = f"{self.BASE_URL}/chart/{yahoo_symbol}"
        params = {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": yahoo_interval,
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [{}])[0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]

            ohlcv_data = []
            for i, ts in enumerate(timestamps):
                if quote.get("open", [None])[i] is None:
                    continue
                ohlcv_data.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ts),
                    open=Decimal(str(quote["open"][i])),
                    high=Decimal(str(quote["high"][i])),
                    low=Decimal(str(quote["low"][i])),
                    close=Decimal(str(quote["close"][i])),
                    volume=float(quote.get("volume", [0])[i] or 0),
                    interval=interval,
                    provider=self.provider_name,
                ))

            return ohlcv_data
        except Exception as e:
            logger.error(f"Yahoo Finance OHLCV error for {symbol}: {e}")
            raise


class AlphaVantageProvider(BaseMarketDataProvider):
    """Alpha Vantage data provider (API key required)"""

    BASE_URL = "https://www.alphavantage.co/query"

    @property
    def provider_name(self) -> str:
        return "alpha_vantage"

    @property
    def supported_assets(self) -> List[AssetClass]:
        return [AssetClass.FOREX, AssetClass.CRYPTO, AssetClass.COMMODITY]

    async def get_tick(self, symbol: str) -> MarketTick:
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        # Determine function based on asset type
        asset_info = STANDARD_ASSETS.get(symbol)
        if asset_info and asset_info.asset_class == AssetClass.CRYPTO:
            function = "CURRENCY_EXCHANGE_RATE"
            params = {
                "function": function,
                "from_currency": asset_info.base_currency,
                "to_currency": asset_info.quote_currency,
                "apikey": self.api_key,
            }
        else:
            function = "CURRENCY_EXCHANGE_RATE"
            base = symbol[:3]
            quote = symbol[3:]
            params = {
                "function": function,
                "from_currency": base,
                "to_currency": quote,
                "apikey": self.api_key,
            }

        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            rate_data = data.get("Realtime Currency Exchange Rate", {})
            price = Decimal(rate_data.get("5. Exchange Rate", "0"))
            bid = Decimal(rate_data.get("8. Bid Price", str(price)))
            ask = Decimal(rate_data.get("9. Ask Price", str(price)))

            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bid if bid > 0 else price - price * Decimal("0.0001"),
                ask=ask if ask > 0 else price + price * Decimal("0.0001"),
                last=price,
                provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"Alpha Vantage tick error for {symbol}: {e}")
            raise

    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        asset_info = STANDARD_ASSETS.get(symbol)

        if asset_info and asset_info.asset_class == AssetClass.CRYPTO:
            if interval in ["1d", "1w"]:
                function = "DIGITAL_CURRENCY_DAILY"
            else:
                function = "CRYPTO_INTRADAY"
        else:
            if interval in ["1d", "1w"]:
                function = "FX_DAILY"
            else:
                function = "FX_INTRADAY"

        base = asset_info.base_currency if asset_info else symbol[:3]
        quote = asset_info.quote_currency if asset_info else symbol[3:]

        params = {
            "function": function,
            "from_symbol": base,
            "to_symbol": quote,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        if "INTRADAY" in function:
            interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}
            params["interval"] = interval_map.get(interval, "60min")

        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Find the time series key
            ts_key = next((k for k in data.keys() if "Time Series" in k), None)
            if not ts_key:
                return []

            time_series = data[ts_key]
            ohlcv_data = []

            for timestamp_str, values in time_series.items():
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S" if " " in timestamp_str else "%Y-%m-%d")
                if start <= timestamp <= end:
                    ohlcv_data.append(OHLCV(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=Decimal(values.get("1. open", values.get("1a. open (USD)", "0"))),
                        high=Decimal(values.get("2. high", values.get("2a. high (USD)", "0"))),
                        low=Decimal(values.get("3. low", values.get("3a. low (USD)", "0"))),
                        close=Decimal(values.get("4. close", values.get("4a. close (USD)", "0"))),
                        volume=float(values.get("5. volume", values.get("5. volume", 0))),
                        interval=interval,
                        provider=self.provider_name,
                    ))

            return sorted(ohlcv_data, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"Alpha Vantage OHLCV error for {symbol}: {e}")
            raise


class BinanceProvider(BaseMarketDataProvider):
    """Binance data provider for crypto (no API key for public data)"""

    BASE_URL = "https://api.binance.com/api/v3"

    SYMBOL_MAP = {
        "BTCUSD": "BTCUSDT",
        "ETHUSD": "ETHUSDT",
        "SOLUSD": "SOLUSDT",
        "XRPUSD": "XRPUSDT",
        "ETHBTC": "ETHBTC",
    }

    @property
    def provider_name(self) -> str:
        return "binance"

    @property
    def supported_assets(self) -> List[AssetClass]:
        return [AssetClass.CRYPTO]

    def _get_binance_symbol(self, symbol: str) -> str:
        return self.SYMBOL_MAP.get(symbol, symbol)

    async def get_tick(self, symbol: str) -> MarketTick:
        binance_symbol = self._get_binance_symbol(symbol)
        url = f"{self.BASE_URL}/ticker/bookTicker"
        params = {"symbol": binance_symbol}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            bid = Decimal(data["bidPrice"])
            ask = Decimal(data["askPrice"])

            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bid,
                ask=ask,
                last=(bid + ask) / 2,
                provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"Binance tick error for {symbol}: {e}")
            raise

    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        binance_symbol = self._get_binance_symbol(symbol)

        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
        }
        binance_interval = interval_map.get(interval, "1h")

        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000,
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            ohlcv_data = []
            for kline in data:
                ohlcv_data.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=Decimal(kline[1]),
                    high=Decimal(kline[2]),
                    low=Decimal(kline[3]),
                    close=Decimal(kline[4]),
                    volume=float(kline[5]),
                    interval=interval,
                    provider=self.provider_name,
                ))

            return ohlcv_data
        except Exception as e:
            logger.error(f"Binance OHLCV error for {symbol}: {e}")
            raise


class PolygonProvider(BaseMarketDataProvider):
    """Polygon.io data provider (API key required)"""

    BASE_URL = "https://api.polygon.io"

    @property
    def provider_name(self) -> str:
        return "polygon"

    @property
    def supported_assets(self) -> List[AssetClass]:
        return [AssetClass.FOREX, AssetClass.CRYPTO, AssetClass.EQUITY]

    async def get_tick(self, symbol: str) -> MarketTick:
        if not self.api_key:
            raise ValueError("Polygon API key required")

        asset_info = STANDARD_ASSETS.get(symbol)

        if asset_info and asset_info.asset_class == AssetClass.CRYPTO:
            url = f"{self.BASE_URL}/v1/last/crypto/{asset_info.base_currency}/{asset_info.quote_currency}"
        else:
            base = symbol[:3]
            quote = symbol[3:]
            url = f"{self.BASE_URL}/v1/last_quote/currencies/{base}/{quote}"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if "last" in data:
                # Crypto response
                last = Decimal(str(data["last"]["price"]))
                bid = last - last * Decimal("0.0001")
                ask = last + last * Decimal("0.0001")
            else:
                # Forex response
                bid = Decimal(str(data.get("bid", 0)))
                ask = Decimal(str(data.get("ask", 0)))
                last = (bid + ask) / 2

            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bid,
                ask=ask,
                last=last,
                provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"Polygon tick error for {symbol}: {e}")
            raise

    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        if not self.api_key:
            raise ValueError("Polygon API key required")

        asset_info = STANDARD_ASSETS.get(symbol)

        # Map interval to Polygon format
        interval_map = {
            "1m": ("minute", 1), "5m": ("minute", 5), "15m": ("minute", 15),
            "30m": ("minute", 30), "1h": ("hour", 1), "4h": ("hour", 4),
            "1d": ("day", 1), "1w": ("week", 1)
        }
        timespan, multiplier = interval_map.get(interval, ("day", 1))

        if asset_info and asset_info.asset_class == AssetClass.CRYPTO:
            ticker = f"X:{asset_info.base_currency}{asset_info.quote_currency}"
        else:
            base = symbol[:3]
            quote = symbol[3:]
            ticker = f"C:{base}{quote}"

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"limit": 50000}

        try:
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            ohlcv_data = []
            for bar in data.get("results", []):
                ohlcv_data.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                    open=Decimal(str(bar["o"])),
                    high=Decimal(str(bar["h"])),
                    low=Decimal(str(bar["l"])),
                    close=Decimal(str(bar["c"])),
                    volume=float(bar.get("v", 0)),
                    interval=interval,
                    provider=self.provider_name,
                ))

            return ohlcv_data
        except Exception as e:
            logger.error(f"Polygon OHLCV error for {symbol}: {e}")
            raise


class MarketDataAggregator:
    """
    Aggregates data from multiple providers with fallback support.
    Provides unified interface for market data access.
    """

    def __init__(self):
        self.providers: Dict[str, BaseMarketDataProvider] = {}
        self.provider_priority: Dict[AssetClass, List[str]] = {
            AssetClass.FOREX: ["polygon", "alpha_vantage", "yahoo_finance"],
            AssetClass.CRYPTO: ["binance", "polygon", "yahoo_finance"],
            AssetClass.COMMODITY: ["yahoo_finance", "alpha_vantage", "polygon"],
        }
        self._cache: Dict[str, MarketTick] = {}
        self._cache_ttl = 5  # seconds

    def register_provider(self, provider: BaseMarketDataProvider):
        """Register a data provider"""
        self.providers[provider.provider_name] = provider
        logger.info(f"Registered market data provider: {provider.provider_name}")

    def _get_providers_for_asset(self, symbol: str) -> List[BaseMarketDataProvider]:
        """Get ordered list of providers for an asset"""
        asset_info = STANDARD_ASSETS.get(symbol)
        if not asset_info:
            # Default to all available providers
            return list(self.providers.values())

        priority = self.provider_priority.get(asset_info.asset_class, [])
        providers = []
        for name in priority:
            if name in self.providers:
                providers.append(self.providers[name])

        # Add remaining providers
        for name, provider in self.providers.items():
            if provider not in providers:
                providers.append(provider)

        return providers

    async def get_tick(self, symbol: str, use_cache: bool = True) -> MarketTick:
        """Get current tick with provider fallback"""
        # Check cache
        if use_cache and symbol in self._cache:
            cached = self._cache[symbol]
            if (datetime.now() - cached.timestamp).seconds < self._cache_ttl:
                return cached

        providers = self._get_providers_for_asset(symbol)
        last_error = None

        for provider in providers:
            try:
                tick = await provider.get_tick(symbol)
                self._cache[symbol] = tick
                return tick
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.provider_name} failed for {symbol}: {e}")
                continue

        raise last_error or ValueError(f"No provider available for {symbol}")

    async def get_ohlcv(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> List[OHLCV]:
        """Get OHLCV data with provider fallback"""
        providers = self._get_providers_for_asset(symbol)
        last_error = None

        for provider in providers:
            try:
                data = await provider.get_ohlcv(symbol, interval, start, end)
                if data:
                    return data
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.provider_name} failed for {symbol} OHLCV: {e}")
                continue

        raise last_error or ValueError(f"No provider available for {symbol}")

    async def get_multiple_ticks(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get ticks for multiple symbols concurrently"""
        tasks = [self.get_tick(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ticks = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get tick for {symbol}: {result}")
            else:
                ticks[symbol] = result

        return ticks

    async def close(self):
        """Close all provider connections"""
        for provider in self.providers.values():
            await provider.close()


# Factory function to create configured aggregator
def create_market_data_aggregator() -> MarketDataAggregator:
    """Create a configured market data aggregator"""
    aggregator = MarketDataAggregator()

    # Always register Yahoo Finance (free, no API key)
    aggregator.register_provider(YahooFinanceProvider())

    # Register Binance for crypto (free public API)
    aggregator.register_provider(BinanceProvider())

    # Register paid providers if API keys are available
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_key:
        aggregator.register_provider(AlphaVantageProvider(api_key=alpha_vantage_key))

    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        aggregator.register_provider(PolygonProvider(api_key=polygon_key))

    return aggregator


# Singleton instance
_aggregator: Optional[MarketDataAggregator] = None


def get_market_data_aggregator() -> MarketDataAggregator:
    """Get the singleton market data aggregator"""
    global _aggregator
    if _aggregator is None:
        _aggregator = create_market_data_aggregator()
    return _aggregator

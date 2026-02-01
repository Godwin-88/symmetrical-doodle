"""
Derivatives API Endpoints
FastAPI router for derivatives pricing, structuring, and backtesting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import uuid

from .market_data_providers import (
    STANDARD_ASSETS,
    AssetClass,
    MarketTick,
    OHLCV,
    get_market_data_aggregator,
)
from .derivatives_pricing import (
    BlackScholes,
    BinomialTree,
    DerivativesPricingEngine,
    FuturesPricer,
    Greeks,
    OptionContract,
    OptionType,
    OptionStyle,
    FuturesContract,
    StructuredProduct,
    ProductType,
    VolatilitySurface,
    PricingResult,
    get_pricing_engine,
    create_straddle,
    create_strangle,
    create_butterfly,
    create_iron_condor,
    create_calendar_spread,
)
from .derivatives_backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Strategy,
    CoveredCallStrategy,
    IronCondorStrategy,
    run_backtest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/derivatives", tags=["derivatives"])

# Active backtests storage
active_backtests: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class TickRequest(BaseModel):
    symbol: str = Field(..., description="Asset symbol (e.g., XAUUSD, BTCUSD, EURUSD)")


class TickResponse(BaseModel):
    symbol: str
    timestamp: str
    bid: float
    ask: float
    last: float
    mid: float
    spread_bps: float
    volume: Optional[float] = None
    provider: Optional[str] = None


class OHLCVRequest(BaseModel):
    symbol: str
    interval: str = Field(default="1d", description="Bar interval: 1m, 5m, 15m, 1h, 4h, 1d")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class OHLCVResponse(BaseModel):
    symbol: str
    interval: str
    data: List[Dict[str, Any]]
    count: int


class OptionPriceRequest(BaseModel):
    underlying: str
    spot_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    expiry_date: str = Field(..., description="Expiry date (YYYY-MM-DD)")
    option_type: str = Field(..., description="call or put")
    volatility: Optional[float] = Field(None, ge=0.01, le=5.0, description="Annual volatility (0.01-5.0)")
    option_style: str = Field(default="european", description="european or american")
    risk_free_rate: Optional[float] = Field(None, ge=0, le=0.5)
    dividend_yield: Optional[float] = Field(default=0.0, ge=0, le=0.5)


class OptionPriceResponse(BaseModel):
    price: float
    greeks: Dict[str, float]
    intrinsic_value: float
    time_value: float
    breakeven: float
    probability_itm: float
    pricing_model: str
    valuation_date: str


class ImpliedVolRequest(BaseModel):
    market_price: float = Field(..., gt=0)
    spot_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    expiry_date: str
    option_type: str


class FuturesPriceRequest(BaseModel):
    underlying: str
    spot_price: float = Field(..., gt=0)
    expiry_date: str
    risk_free_rate: Optional[float] = Field(None, ge=0, le=0.5)
    convenience_yield: float = Field(default=0.0, ge=0, le=0.5)
    storage_cost: float = Field(default=0.0, ge=0, le=0.5)


class StructuredProductRequest(BaseModel):
    product_type: str = Field(..., description="straddle, strangle, butterfly, iron_condor, calendar_spread")
    underlying: str
    spot_price: float
    volatility: float
    expiry_date: str

    # For straddle/strangle
    strike: Optional[float] = None
    call_strike: Optional[float] = None
    put_strike: Optional[float] = None

    # For butterfly
    lower_strike: Optional[float] = None
    middle_strike: Optional[float] = None
    upper_strike: Optional[float] = None

    # For iron condor
    put_lower: Optional[float] = None
    put_upper: Optional[float] = None
    call_lower: Optional[float] = None
    call_upper: Optional[float] = None

    # For calendar spread
    near_expiry: Optional[str] = None
    far_expiry: Optional[str] = None


class BacktestRequest(BaseModel):
    name: str = Field(..., description="Backtest name")
    strategy_type: str = Field(..., description="covered_call, iron_condor, custom")
    underlying: str
    start_date: str
    end_date: str
    initial_capital: float = Field(default=100000.0, gt=0)
    slippage_bps: float = Field(default=2.0, ge=0, le=100)
    commission_per_contract: float = Field(default=1.0, ge=0)
    risk_free_rate: float = Field(default=0.05, ge=0, le=0.5)

    # Strategy parameters
    strategy_params: Optional[Dict[str, Any]] = None


class BacktestStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    start_time: str
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# Market Data Endpoints
# ============================================================================

@router.get("/assets", response_model=List[Dict[str, Any]])
async def list_available_assets():
    """List all available assets for trading"""
    assets = []
    for symbol, info in STANDARD_ASSETS.items():
        assets.append({
            "symbol": symbol,
            "name": info.name,
            "asset_class": info.asset_class.value,
            "base_currency": info.base_currency,
            "quote_currency": info.quote_currency,
            "margin_requirement": info.margin_requirement,
        })
    return assets


@router.get("/assets/{asset_class}", response_model=List[Dict[str, Any]])
async def list_assets_by_class(asset_class: str):
    """List assets by asset class (forex, crypto, commodity)"""
    try:
        ac = AssetClass(asset_class.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid asset class: {asset_class}")

    assets = []
    for symbol, info in STANDARD_ASSETS.items():
        if info.asset_class == ac:
            assets.append({
                "symbol": symbol,
                "name": info.name,
                "base_currency": info.base_currency,
                "quote_currency": info.quote_currency,
                "margin_requirement": info.margin_requirement,
            })
    return assets


@router.post("/tick", response_model=TickResponse)
async def get_market_tick(request: TickRequest):
    """Get current market tick for an asset"""
    aggregator = get_market_data_aggregator()
    try:
        tick = await aggregator.get_tick(request.symbol)
        return TickResponse(
            symbol=tick.symbol,
            timestamp=tick.timestamp.isoformat(),
            bid=float(tick.bid),
            ask=float(tick.ask),
            last=float(tick.last),
            mid=float(tick.mid),
            spread_bps=tick.spread_bps,
            volume=tick.volume,
            provider=tick.provider,
        )
    except Exception as e:
        logger.error(f"Error getting tick for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ticks", response_model=Dict[str, TickResponse])
async def get_multiple_ticks(symbols: List[str]):
    """Get current market ticks for multiple assets"""
    aggregator = get_market_data_aggregator()
    try:
        ticks = await aggregator.get_multiple_ticks(symbols)
        return {
            symbol: TickResponse(
                symbol=tick.symbol,
                timestamp=tick.timestamp.isoformat(),
                bid=float(tick.bid),
                ask=float(tick.ask),
                last=float(tick.last),
                mid=float(tick.mid),
                spread_bps=tick.spread_bps,
                volume=tick.volume,
                provider=tick.provider,
            )
            for symbol, tick in ticks.items()
        }
    except Exception as e:
        logger.error(f"Error getting ticks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv_data(request: OHLCVRequest):
    """Get OHLCV historical data for an asset"""
    aggregator = get_market_data_aggregator()
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")

        data = await aggregator.get_ohlcv(
            request.symbol,
            request.interval,
            start,
            end,
        )

        return OHLCVResponse(
            symbol=request.symbol,
            interval=request.interval,
            data=[bar.to_dict() for bar in data],
            count=len(data),
        )
    except Exception as e:
        logger.error(f"Error getting OHLCV for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Options Pricing Endpoints
# ============================================================================

@router.post("/options/price", response_model=OptionPriceResponse)
async def price_option(request: OptionPriceRequest):
    """Price an option using Black-Scholes or Binomial Tree model"""
    try:
        expiry = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        option_type = OptionType(request.option_type.lower())
        option_style = OptionStyle(request.option_style.lower())

        contract = OptionContract(
            underlying=request.underlying,
            strike=request.strike,
            expiry=expiry,
            option_type=option_type,
            option_style=option_style,
        )

        # Get volatility from market if not provided
        volatility = request.volatility
        if volatility is None:
            # Calculate from recent data
            aggregator = get_market_data_aggregator()
            end = datetime.now()
            start = end - timedelta(days=60)
            try:
                ohlcv = await aggregator.get_ohlcv(request.underlying, "1d", start, end)
                if len(ohlcv) > 20:
                    import numpy as np
                    prices = [float(bar.close) for bar in ohlcv]
                    returns = np.diff(np.log(prices))
                    volatility = float(np.std(returns) * np.sqrt(252))
                else:
                    volatility = 0.2  # Default
            except:
                volatility = 0.2

        engine = get_pricing_engine()
        if request.risk_free_rate is not None:
            engine.set_risk_free_rate(request.risk_free_rate)
        if request.dividend_yield:
            engine.set_dividend_yield(request.underlying, request.dividend_yield)

        result = engine.price_option(contract, request.spot_price, volatility)

        return OptionPriceResponse(
            price=result.price,
            greeks=result.greeks.to_dict() if result.greeks else {},
            intrinsic_value=result.intrinsic_value,
            time_value=result.time_value,
            breakeven=result.breakeven or 0,
            probability_itm=result.probability_itm or 0,
            pricing_model=result.pricing_model,
            valuation_date=result.valuation_date.isoformat(),
        )
    except Exception as e:
        logger.error(f"Error pricing option: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/options/implied-vol")
async def calculate_implied_volatility(request: ImpliedVolRequest):
    """Calculate implied volatility from market price"""
    try:
        expiry = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        option_type = OptionType(request.option_type.lower())
        T = (expiry - datetime.now()).days / 365.25

        engine = get_pricing_engine()
        iv = engine.calculate_implied_vol(
            request.market_price,
            request.spot_price,
            request.strike,
            T,
            option_type,
        )

        if iv is None:
            raise HTTPException(status_code=400, detail="Could not calculate implied volatility")

        return {
            "implied_volatility": round(iv, 6),
            "implied_volatility_pct": round(iv * 100, 2),
            "spot_price": request.spot_price,
            "strike": request.strike,
            "market_price": request.market_price,
            "time_to_expiry_years": round(T, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating IV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/options/chain")
async def get_options_chain(
    underlying: str,
    spot_price: float,
    expiry_date: str,
    volatility: float = Query(default=0.2),
    strikes_count: int = Query(default=11, ge=5, le=51),
):
    """Generate options chain for an underlying"""
    try:
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        engine = get_pricing_engine()

        # Generate strikes around ATM
        strike_step = spot_price * 0.025  # 2.5% steps
        strikes = []
        center = round(spot_price / strike_step) * strike_step

        for i in range(-(strikes_count // 2), strikes_count // 2 + 1):
            strikes.append(center + i * strike_step)

        chain = []
        for strike in strikes:
            call_contract = OptionContract(
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                option_type=OptionType.CALL,
            )
            put_contract = OptionContract(
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                option_type=OptionType.PUT,
            )

            call_result = engine.price_option(call_contract, spot_price, volatility)
            put_result = engine.price_option(put_contract, spot_price, volatility)

            chain.append({
                "strike": strike,
                "call": {
                    "price": round(call_result.price, 4),
                    "delta": round(call_result.greeks.delta, 4) if call_result.greeks else 0,
                    "gamma": round(call_result.greeks.gamma, 6) if call_result.greeks else 0,
                    "theta": round(call_result.greeks.theta, 4) if call_result.greeks else 0,
                    "vega": round(call_result.greeks.vega, 4) if call_result.greeks else 0,
                    "iv": round(volatility * 100, 2),
                },
                "put": {
                    "price": round(put_result.price, 4),
                    "delta": round(put_result.greeks.delta, 4) if put_result.greeks else 0,
                    "gamma": round(put_result.greeks.gamma, 6) if put_result.greeks else 0,
                    "theta": round(put_result.greeks.theta, 4) if put_result.greeks else 0,
                    "vega": round(put_result.greeks.vega, 4) if put_result.greeks else 0,
                    "iv": round(volatility * 100, 2),
                },
            })

        return {
            "underlying": underlying,
            "spot_price": spot_price,
            "expiry_date": expiry_date,
            "volatility": volatility,
            "chain": chain,
        }
    except Exception as e:
        logger.error(f"Error generating options chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Futures Pricing Endpoints
# ============================================================================

@router.post("/futures/price")
async def price_futures(request: FuturesPriceRequest):
    """Calculate futures fair value"""
    try:
        expiry = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        T = (expiry - datetime.now()).days / 365.25

        r = request.risk_free_rate or 0.05

        fair_value = FuturesPricer.fair_value(
            spot=request.spot_price,
            r=r,
            T=T,
            q=request.convenience_yield,
            storage_cost=request.storage_cost,
        )

        basis = FuturesPricer.basis(request.spot_price, fair_value)
        implied_rate = FuturesPricer.implied_repo_rate(
            request.spot_price,
            fair_value,
            T,
            request.convenience_yield,
        )

        return {
            "underlying": request.underlying,
            "spot_price": request.spot_price,
            "fair_value": round(fair_value, 6),
            "basis": round(basis, 6),
            "basis_pct": round(basis / request.spot_price * 100, 4),
            "implied_repo_rate": round(implied_rate, 6),
            "time_to_expiry_years": round(T, 4),
            "risk_free_rate": r,
            "convenience_yield": request.convenience_yield,
            "storage_cost": request.storage_cost,
        }
    except Exception as e:
        logger.error(f"Error pricing futures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Structured Products Endpoints
# ============================================================================

@router.post("/structured/price")
async def price_structured_product(request: StructuredProductRequest):
    """Price a structured product (straddle, strangle, butterfly, etc.)"""
    try:
        expiry = datetime.strptime(request.expiry_date, "%Y-%m-%d")
        engine = get_pricing_engine()

        if request.product_type == "straddle":
            if request.strike is None:
                raise HTTPException(status_code=400, detail="Strike required for straddle")
            product = create_straddle(request.underlying, request.strike, expiry)

        elif request.product_type == "strangle":
            if request.call_strike is None or request.put_strike is None:
                raise HTTPException(status_code=400, detail="call_strike and put_strike required for strangle")
            product = create_strangle(request.underlying, request.call_strike, request.put_strike, expiry)

        elif request.product_type == "butterfly":
            if request.lower_strike is None or request.middle_strike is None or request.upper_strike is None:
                raise HTTPException(status_code=400, detail="lower_strike, middle_strike, upper_strike required")
            product = create_butterfly(request.underlying, request.lower_strike, request.middle_strike, request.upper_strike, expiry)

        elif request.product_type == "iron_condor":
            if not all([request.put_lower, request.put_upper, request.call_lower, request.call_upper]):
                raise HTTPException(status_code=400, detail="All four strikes required for iron condor")
            product = create_iron_condor(
                request.underlying,
                request.put_lower,
                request.put_upper,
                request.call_lower,
                request.call_upper,
                expiry,
            )

        elif request.product_type == "calendar_spread":
            if request.strike is None or request.near_expiry is None or request.far_expiry is None:
                raise HTTPException(status_code=400, detail="strike, near_expiry, far_expiry required")
            near = datetime.strptime(request.near_expiry, "%Y-%m-%d")
            far = datetime.strptime(request.far_expiry, "%Y-%m-%d")
            product = create_calendar_spread(request.underlying, request.strike, near, far)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown product type: {request.product_type}")

        result = engine.price_structured_product(product, request.spot_price, request.volatility)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pricing structured product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/structured/templates")
async def list_product_templates():
    """List available structured product templates"""
    return {
        "templates": [
            {
                "type": "straddle",
                "description": "Long call + long put at same strike. Profit from volatility.",
                "required_fields": ["underlying", "strike", "expiry_date"],
                "risk_profile": "Limited loss (premium paid), unlimited profit potential",
            },
            {
                "type": "strangle",
                "description": "Long OTM call + long OTM put. Cheaper than straddle.",
                "required_fields": ["underlying", "call_strike", "put_strike", "expiry_date"],
                "risk_profile": "Limited loss, unlimited profit potential",
            },
            {
                "type": "butterfly",
                "description": "Long low strike, short 2x middle strike, long high strike.",
                "required_fields": ["underlying", "lower_strike", "middle_strike", "upper_strike", "expiry_date"],
                "risk_profile": "Limited loss, limited profit. Neutral strategy.",
            },
            {
                "type": "iron_condor",
                "description": "Short put spread + short call spread. Income strategy.",
                "required_fields": ["underlying", "put_lower", "put_upper", "call_lower", "call_upper", "expiry_date"],
                "risk_profile": "Limited loss, limited profit. Range-bound market.",
            },
            {
                "type": "calendar_spread",
                "description": "Short near-term, long far-term at same strike.",
                "required_fields": ["underlying", "strike", "near_expiry", "far_expiry"],
                "risk_profile": "Profit from time decay differential.",
            },
        ]
    }


# ============================================================================
# Backtesting Endpoints
# ============================================================================

@router.post("/backtest/run", response_model=BacktestStatusResponse)
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a new backtest"""
    try:
        backtest_id = str(uuid.uuid4())
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")

        config = BacktestConfig(
            name=request.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=request.initial_capital,
            assets=[request.underlying],
            slippage_bps=request.slippage_bps,
            commission_per_contract=request.commission_per_contract,
            risk_free_rate=request.risk_free_rate,
        )

        # Create strategy
        if request.strategy_type == "covered_call":
            delta_target = request.strategy_params.get("delta_target", 0.3) if request.strategy_params else 0.3
            strategies = [CoveredCallStrategy(request.underlying, delta_target)]
        elif request.strategy_type == "iron_condor":
            wing_width = request.strategy_params.get("wing_width", 0.05) if request.strategy_params else 0.05
            strategies = [IronCondorStrategy(request.underlying, wing_width)]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy_type}")

        # Initialize backtest status
        active_backtests[backtest_id] = {
            "id": backtest_id,
            "name": request.name,
            "status": "running",
            "progress": 0.0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result": None,
        }

        # Run in background
        async def run_backtest_task():
            try:
                result = await run_backtest(config, strategies)
                active_backtests[backtest_id]["status"] = "completed"
                active_backtests[backtest_id]["progress"] = 1.0
                active_backtests[backtest_id]["end_time"] = datetime.now().isoformat()
                active_backtests[backtest_id]["result"] = result.to_dict()
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                active_backtests[backtest_id]["status"] = "failed"
                active_backtests[backtest_id]["error"] = str(e)

        background_tasks.add_task(asyncio.create_task, run_backtest_task())

        return BacktestStatusResponse(
            id=backtest_id,
            name=request.name,
            status="running",
            progress=0.0,
            start_time=active_backtests[backtest_id]["start_time"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(backtest_id: str):
    """Get backtest status and results"""
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    bt = active_backtests[backtest_id]
    return BacktestStatusResponse(
        id=bt["id"],
        name=bt["name"],
        status=bt["status"],
        progress=bt["progress"],
        start_time=bt["start_time"],
        end_time=bt.get("end_time"),
        result=bt.get("result"),
    )


@router.get("/backtest", response_model=List[BacktestStatusResponse])
async def list_backtests():
    """List all backtests"""
    return [
        BacktestStatusResponse(
            id=bt["id"],
            name=bt["name"],
            status=bt["status"],
            progress=bt["progress"],
            start_time=bt["start_time"],
            end_time=bt.get("end_time"),
            result=bt.get("result"),
        )
        for bt in active_backtests.values()
    ]


@router.delete("/backtest/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest"""
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    del active_backtests[backtest_id]
    return {"message": "Backtest deleted", "id": backtest_id}


# ============================================================================
# Strategy Templates
# ============================================================================

@router.get("/strategies/templates")
async def list_strategy_templates():
    """List available backtest strategy templates"""
    return {
        "strategies": [
            {
                "type": "covered_call",
                "name": "Covered Call",
                "description": "Long stock + short OTM call. Income generation strategy.",
                "parameters": {
                    "delta_target": {"type": "float", "default": 0.3, "description": "Target delta for short call"},
                },
                "risk_level": "low",
                "market_outlook": "neutral to slightly bullish",
            },
            {
                "type": "iron_condor",
                "name": "Iron Condor",
                "description": "Short put spread + short call spread. Range-bound strategy.",
                "parameters": {
                    "wing_width": {"type": "float", "default": 0.05, "description": "Width of wings as % of spot"},
                },
                "risk_level": "medium",
                "market_outlook": "neutral, low volatility",
            },
        ]
    }


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for derivatives API"""
    aggregator = get_market_data_aggregator()
    providers = list(aggregator.providers.keys())

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "market_data_providers": providers,
        "active_backtests": len(active_backtests),
        "pricing_engine": "active",
    }

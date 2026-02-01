"""
Derivatives Backtesting Engine
Supports: Options, Futures, Structured Products backtesting
Includes: Greeks-aware risk, margin simulation, slippage modeling
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd

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
    PricingResult,
    get_pricing_engine,
)
from .market_data_providers import (
    OHLCV,
    MarketTick,
    MarketDataAggregator,
    get_market_data_aggregator,
)

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionType(str, Enum):
    SPOT = "spot"
    OPTION = "option"
    FUTURE = "future"


@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    assets: List[str]

    # Execution parameters
    slippage_bps: float = 2.0  # Basis points
    commission_per_contract: float = 1.0
    commission_pct: float = 0.001  # 0.1%

    # Risk parameters
    max_position_pct: float = 0.1  # Max 10% per position
    max_leverage: float = 3.0
    margin_requirement: float = 0.2  # 20% margin

    # Options parameters
    vol_lookback_days: int = 30
    risk_free_rate: float = 0.05

    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalance

    # Data
    bar_interval: str = "1d"  # 1m, 5m, 15m, 1h, 4h, 1d

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "assets": self.assets,
            "slippage_bps": self.slippage_bps,
            "commission_per_contract": self.commission_per_contract,
            "commission_pct": self.commission_pct,
            "max_position_pct": self.max_position_pct,
            "max_leverage": self.max_leverage,
            "margin_requirement": self.margin_requirement,
            "risk_free_rate": self.risk_free_rate,
            "bar_interval": self.bar_interval,
        }


@dataclass
class Position:
    """Position in the portfolio"""
    id: str
    symbol: str
    position_type: PositionType
    side: OrderSide
    quantity: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0

    # For options
    option_contract: Optional[OptionContract] = None
    option_greeks: Optional[Greeks] = None

    # For futures
    futures_contract: Optional[FuturesContract] = None

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0

    @property
    def market_value(self) -> float:
        multiplier = 1 if self.side == OrderSide.BUY else -1
        if self.position_type == PositionType.OPTION:
            return self.quantity * self.current_price * multiplier * 100
        return self.quantity * self.current_price * multiplier

    @property
    def cost_basis(self) -> float:
        if self.position_type == PositionType.OPTION:
            return self.quantity * self.entry_price * 100
        return self.quantity * self.entry_price

    def update_pnl(self, current_price: float, current_greeks: Optional[Greeks] = None):
        self.current_price = current_price
        if current_greeks:
            self.option_greeks = current_greeks

        multiplier = 1 if self.side == OrderSide.BUY else -1

        if self.position_type == PositionType.OPTION:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity * 100 * multiplier
        else:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity * multiplier

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat(),
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "greeks": self.option_greeks.to_dict() if self.option_greeks else None,
        }


@dataclass
class Order:
    """Trade order"""
    id: str
    timestamp: datetime
    symbol: str
    position_type: PositionType
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # For options
    option_contract: Optional[OptionContract] = None

    # Execution
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    status: str = "pending"  # pending, filled, cancelled, rejected

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "status": self.status,
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    cash: float
    positions_value: float
    total_equity: float
    margin_used: float
    leverage: float
    positions: Dict[str, Position]

    # Greeks aggregation
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_equity": self.total_equity,
            "margin_used": self.margin_used,
            "leverage": self.leverage,
            "net_greeks": {
                "delta": self.net_delta,
                "gamma": self.net_gamma,
                "vega": self.net_vega,
                "theta": self.net_theta,
            },
            "var_95": self.var_95,
            "max_drawdown": self.max_drawdown,
            "position_count": len(self.positions),
        }


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    start_equity: float
    end_equity: float

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: float = 0.0  # hours
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Greeks statistics
    avg_net_delta: float = 0.0
    max_net_delta: float = 0.0
    avg_net_gamma: float = 0.0
    avg_net_vega: float = 0.0
    avg_net_theta: float = 0.0

    # Time series
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "performance": {
                "start_equity": self.start_equity,
                "end_equity": self.end_equity,
                "total_return": round(self.total_return, 4),
                "annualized_return": round(self.annualized_return, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "sortino_ratio": round(self.sortino_ratio, 4),
                "max_drawdown": round(self.max_drawdown, 4),
                "max_drawdown_duration": self.max_drawdown_duration,
                "calmar_ratio": round(self.calmar_ratio, 4),
                "win_rate": round(self.win_rate, 4),
                "profit_factor": round(self.profit_factor, 4),
            },
            "trade_statistics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "avg_trade_duration_hours": round(self.avg_trade_duration, 2),
                "total_commission": round(self.total_commission, 2),
                "total_slippage": round(self.total_slippage, 2),
            },
            "greeks_statistics": {
                "avg_net_delta": round(self.avg_net_delta, 4),
                "max_net_delta": round(self.max_net_delta, 4),
                "avg_net_gamma": round(self.avg_net_gamma, 4),
                "avg_net_vega": round(self.avg_net_vega, 4),
                "avg_net_theta": round(self.avg_net_theta, 4),
            },
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }


class Strategy:
    """Base strategy class for backtesting"""

    def __init__(self, name: str):
        self.name = name

    def on_bar(
        self,
        timestamp: datetime,
        bars: Dict[str, OHLCV],
        portfolio: "BacktestPortfolio",
        engine: "BacktestEngine",
    ) -> List[Order]:
        """Called on each bar. Override to implement strategy logic."""
        raise NotImplementedError

    def on_trade(self, order: Order, portfolio: "BacktestPortfolio"):
        """Called when a trade is executed"""
        pass


class BacktestPortfolio:
    """Portfolio manager for backtesting"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.orders: List[Order] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.pricing_engine = get_pricing_engine()
        self.pricing_engine.set_risk_free_rate(config.risk_free_rate)

        # Historical volatilities for options pricing
        self.volatilities: Dict[str, float] = {}

    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.positions_value

    @property
    def margin_used(self) -> float:
        margin = 0.0
        for pos in self.positions.values():
            if pos.position_type in [PositionType.OPTION, PositionType.FUTURE]:
                margin += abs(pos.market_value) * self.config.margin_requirement
        return margin

    @property
    def leverage(self) -> float:
        if self.total_equity <= 0:
            return 0.0
        return sum(abs(p.market_value) for p in self.positions.values()) / self.total_equity

    def get_net_greeks(self) -> Tuple[float, float, float, float]:
        """Get aggregated portfolio Greeks"""
        net_delta = net_gamma = net_vega = net_theta = 0.0
        for pos in self.positions.values():
            if pos.option_greeks:
                multiplier = 1 if pos.side == OrderSide.BUY else -1
                net_delta += pos.option_greeks.delta * pos.quantity * multiplier
                net_gamma += pos.option_greeks.gamma * pos.quantity * multiplier
                net_vega += pos.option_greeks.vega * pos.quantity * multiplier
                net_theta += pos.option_greeks.theta * pos.quantity * multiplier
        return net_delta, net_gamma, net_vega, net_theta

    def calculate_historical_vol(self, prices: List[float], window: int = 30) -> float:
        """Calculate historical volatility from price series"""
        if len(prices) < window + 1:
            return 0.2  # Default vol

        returns = np.diff(np.log(prices[-window-1:]))
        return float(np.std(returns) * np.sqrt(252))

    def update_positions(self, bars: Dict[str, OHLCV], timestamp: datetime):
        """Update all position prices and Greeks"""
        for symbol, pos in list(self.positions.items()):
            if pos.symbol in bars:
                bar = bars[pos.symbol]
                spot = float(bar.close)

                if pos.position_type == PositionType.OPTION and pos.option_contract:
                    vol = self.volatilities.get(pos.symbol, 0.2)
                    result = self.pricing_engine.price_option(
                        pos.option_contract,
                        spot,
                        vol,
                        timestamp,
                    )
                    pos.update_pnl(result.price / 100, result.greeks)  # Per-share price

                    # Check for expiry
                    if pos.option_contract.time_to_expiry(timestamp) <= 0:
                        self._close_expired_option(pos, spot, timestamp)
                else:
                    pos.update_pnl(spot)

    def _close_expired_option(self, pos: Position, spot: float, timestamp: datetime):
        """Handle option expiration"""
        if pos.option_contract is None:
            return

        # Calculate intrinsic value at expiry
        if pos.option_contract.option_type == OptionType.CALL:
            intrinsic = max(spot - pos.option_contract.strike, 0)
        else:
            intrinsic = max(pos.option_contract.strike - spot, 0)

        multiplier = 1 if pos.side == OrderSide.BUY else -1
        pos.realized_pnl = (intrinsic - pos.entry_price) * pos.quantity * 100 * multiplier
        pos.current_price = intrinsic

        # Move to closed
        self.closed_positions.append(pos)
        del self.positions[pos.id]

    def take_snapshot(self, timestamp: datetime):
        """Record portfolio state"""
        net_delta, net_gamma, net_vega, net_theta = self.get_net_greeks()

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=self.positions_value,
            total_equity=self.total_equity,
            margin_used=self.margin_used,
            leverage=self.leverage,
            positions={k: v for k, v in self.positions.items()},
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_vega=net_vega,
            net_theta=net_theta,
        )
        self.snapshots.append(snapshot)
        return snapshot


class BacktestEngine:
    """Main backtesting engine"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = BacktestPortfolio(config)
        self.market_data: Dict[str, List[OHLCV]] = {}
        self.current_bar_index = 0
        self.current_timestamp: Optional[datetime] = None
        self.strategies: List[Strategy] = []

    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the backtest"""
        self.strategies.append(strategy)

    def load_data(self, data: Dict[str, List[OHLCV]]):
        """Load historical market data"""
        self.market_data = data

        # Calculate historical volatilities
        for symbol, bars in data.items():
            if len(bars) > 30:
                prices = [float(bar.close) for bar in bars]
                self.portfolio.volatilities[symbol] = self.portfolio.calculate_historical_vol(prices)

    async def load_data_from_providers(self):
        """Load data from market data providers"""
        aggregator = get_market_data_aggregator()

        for symbol in self.config.assets:
            try:
                ohlcv = await aggregator.get_ohlcv(
                    symbol,
                    self.config.bar_interval,
                    self.config.start_date,
                    self.config.end_date,
                )
                self.market_data[symbol] = ohlcv

                # Calculate volatility
                if len(ohlcv) > 30:
                    prices = [float(bar.close) for bar in ohlcv]
                    self.portfolio.volatilities[symbol] = self.portfolio.calculate_historical_vol(prices)

            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")

    def execute_order(self, order: Order, current_bar: OHLCV) -> bool:
        """Execute an order with slippage and commission"""
        spot = float(current_bar.close)

        # Calculate fill price with slippage
        slippage_multiplier = 1 + (self.config.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            fill_price = spot * slippage_multiplier
        else:
            fill_price = spot / slippage_multiplier

        # For limit orders, check price
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return False  # Don't fill
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return False

        # Calculate commission
        if order.position_type == PositionType.OPTION:
            commission = order.quantity * self.config.commission_per_contract
        else:
            commission = fill_price * order.quantity * self.config.commission_pct

        # Check if we have enough capital
        if order.side == OrderSide.BUY:
            required = fill_price * order.quantity + commission
            if order.position_type == PositionType.OPTION:
                required = fill_price * order.quantity * 100 + commission
            if required > self.portfolio.cash:
                order.status = "rejected"
                return False

        # Execute the trade
        order.fill_price = fill_price
        order.fill_timestamp = current_bar.timestamp
        order.commission = commission
        order.slippage = abs(fill_price - spot) * order.quantity
        order.status = "filled"

        # Update portfolio
        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity
            if order.position_type == PositionType.OPTION:
                cost = fill_price * order.quantity * 100
            self.portfolio.cash -= cost + commission
        else:
            proceeds = fill_price * order.quantity
            if order.position_type == PositionType.OPTION:
                proceeds = fill_price * order.quantity * 100
            self.portfolio.cash += proceeds - commission

        # Create or update position
        position_key = f"{order.symbol}_{order.position_type.value}"
        if order.option_contract:
            position_key = f"{order.symbol}_{order.option_contract.strike}_{order.option_contract.option_type.value}_{order.option_contract.expiry.date()}"

        if position_key in self.portfolio.positions:
            # Update existing position
            pos = self.portfolio.positions[position_key]
            if pos.side == order.side:
                # Adding to position
                total_cost = pos.entry_price * pos.quantity + fill_price * order.quantity
                total_qty = pos.quantity + order.quantity
                pos.entry_price = total_cost / total_qty
                pos.quantity = total_qty
            else:
                # Reducing or closing position
                if order.quantity >= pos.quantity:
                    # Close position
                    pnl_per_unit = fill_price - pos.entry_price
                    if pos.side == OrderSide.SELL:
                        pnl_per_unit = -pnl_per_unit
                    pos.realized_pnl = pnl_per_unit * pos.quantity
                    if order.position_type == PositionType.OPTION:
                        pos.realized_pnl *= 100
                    self.portfolio.closed_positions.append(pos)
                    del self.portfolio.positions[position_key]

                    # Open new position with remaining quantity
                    remaining = order.quantity - pos.quantity
                    if remaining > 0:
                        new_pos = Position(
                            id=str(uuid.uuid4()),
                            symbol=order.symbol,
                            position_type=order.position_type,
                            side=order.side,
                            quantity=remaining,
                            entry_price=fill_price,
                            entry_date=current_bar.timestamp,
                            current_price=fill_price,
                            option_contract=order.option_contract,
                        )
                        self.portfolio.positions[position_key] = new_pos
                else:
                    # Partial close
                    pnl_per_unit = fill_price - pos.entry_price
                    if pos.side == OrderSide.SELL:
                        pnl_per_unit = -pnl_per_unit
                    pos.realized_pnl += pnl_per_unit * order.quantity
                    if order.position_type == PositionType.OPTION:
                        pos.realized_pnl *= 100
                    pos.quantity -= order.quantity
        else:
            # New position
            pos = Position(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                position_type=order.position_type,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_date=current_bar.timestamp,
                current_price=fill_price,
                option_contract=order.option_contract,
            )
            self.portfolio.positions[position_key] = pos

        self.portfolio.orders.append(order)
        return True

    def run(self) -> BacktestResult:
        """Run the backtest"""
        if not self.market_data:
            raise ValueError("No market data loaded")

        if not self.strategies:
            raise ValueError("No strategies added")

        # Get aligned timestamps
        all_timestamps = set()
        for bars in self.market_data.values():
            for bar in bars:
                if self.config.start_date <= bar.timestamp <= self.config.end_date:
                    all_timestamps.add(bar.timestamp)

        timestamps = sorted(all_timestamps)

        # Run simulation
        for i, timestamp in enumerate(timestamps):
            self.current_bar_index = i
            self.current_timestamp = timestamp

            # Get current bars
            current_bars = {}
            for symbol, bars in self.market_data.items():
                matching = [b for b in bars if b.timestamp == timestamp]
                if matching:
                    current_bars[symbol] = matching[0]

            if not current_bars:
                continue

            # Update positions
            self.portfolio.update_positions(current_bars, timestamp)

            # Run strategies
            for strategy in self.strategies:
                orders = strategy.on_bar(timestamp, current_bars, self.portfolio, self)
                for order in orders:
                    if order.symbol in current_bars:
                        self.execute_order(order, current_bars[order.symbol])
                        strategy.on_trade(order, self.portfolio)

            # Take snapshot
            self.portfolio.take_snapshot(timestamp)

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics"""
        snapshots = self.portfolio.snapshots
        if not snapshots:
            return BacktestResult(
                config=self.config,
                start_equity=self.config.initial_capital,
                end_equity=self.config.initial_capital,
            )

        # Equity curve
        equity_values = [s.total_equity for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]

        start_equity = equity_values[0]
        end_equity = equity_values[-1]

        # Returns
        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        total_return = (end_equity - start_equity) / start_equity

        # Trading days
        days = (timestamps[-1] - timestamps[0]).days
        years = max(days / 365.25, 1/365.25)
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = 0

        # Drawdown
        peak = equity_values[0]
        max_drawdown = 0
        drawdown_start = 0
        max_drawdown_duration = 0
        current_drawdown_start = 0

        for i, equity in enumerate(equity_values):
            if equity > peak:
                if current_drawdown_start > 0:
                    duration = i - current_drawdown_start
                    max_drawdown_duration = max(max_drawdown_duration, duration)
                peak = equity
                current_drawdown_start = 0
            else:
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                if current_drawdown_start == 0:
                    current_drawdown_start = i

        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        all_trades = self.portfolio.orders + [
            Order(
                id=p.id,
                timestamp=p.entry_date,
                symbol=p.symbol,
                position_type=p.position_type,
                side=p.side,
                order_type=OrderType.MARKET,
                quantity=p.quantity,
                fill_price=p.entry_price,
                status="closed",
            )
            for p in self.portfolio.closed_positions
        ]

        winning_trades = [p for p in self.portfolio.closed_positions if p.realized_pnl > 0]
        losing_trades = [p for p in self.portfolio.closed_positions if p.realized_pnl < 0]

        total_trades = len(self.portfolio.closed_positions)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(p.realized_pnl for p in winning_trades)
        total_loss = abs(sum(p.realized_pnl for p in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0

        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        total_commission = sum(o.commission for o in self.portfolio.orders)
        total_slippage = sum(o.slippage for o in self.portfolio.orders)

        # Greeks statistics
        deltas = [s.net_delta for s in snapshots]
        gammas = [s.net_gamma for s in snapshots]
        vegas = [s.net_vega for s in snapshots]
        thetas = [s.net_theta for s in snapshots]

        # Build equity curve
        equity_curve = [
            {"timestamp": s.timestamp.isoformat(), "equity": s.total_equity}
            for s in snapshots
        ]

        trades_data = [o.to_dict() for o in self.portfolio.orders if o.status == "filled"]

        return BacktestResult(
            config=self.config,
            start_equity=start_equity,
            end_equity=end_equity,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_commission=total_commission,
            total_slippage=total_slippage,
            avg_net_delta=np.mean(deltas) if deltas else 0,
            max_net_delta=max(abs(d) for d in deltas) if deltas else 0,
            avg_net_gamma=np.mean(gammas) if gammas else 0,
            avg_net_vega=np.mean(vegas) if vegas else 0,
            avg_net_theta=np.mean(thetas) if thetas else 0,
            equity_curve=equity_curve,
            trades=trades_data,
            snapshots=[s.to_dict() for s in snapshots],
        )


# Example strategies
class CoveredCallStrategy(Strategy):
    """Covered call strategy: long stock + short call"""

    def __init__(self, underlying: str, delta_target: float = 0.3):
        super().__init__("Covered Call")
        self.underlying = underlying
        self.delta_target = delta_target
        self.last_roll_date: Optional[datetime] = None
        self.roll_frequency_days = 30

    def on_bar(
        self,
        timestamp: datetime,
        bars: Dict[str, OHLCV],
        portfolio: BacktestPortfolio,
        engine: BacktestEngine,
    ) -> List[Order]:
        orders = []

        if self.underlying not in bars:
            return orders

        bar = bars[self.underlying]
        spot = float(bar.close)

        # Check if we need to roll or initiate
        should_trade = False
        if self.last_roll_date is None:
            should_trade = True
        elif (timestamp - self.last_roll_date).days >= self.roll_frequency_days:
            should_trade = True

        if should_trade:
            # Buy underlying if not already long
            has_stock = any(
                p.symbol == self.underlying and p.position_type == PositionType.SPOT
                for p in portfolio.positions.values()
            )

            if not has_stock:
                # Buy 100 shares
                orders.append(Order(
                    id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    symbol=self.underlying,
                    position_type=PositionType.SPOT,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100,
                ))

            # Sell OTM call
            strike = round(spot * 1.05, 2)  # 5% OTM
            expiry = timestamp + timedelta(days=30)

            contract = OptionContract(
                underlying=self.underlying,
                strike=strike,
                expiry=expiry,
                option_type=OptionType.CALL,
            )

            orders.append(Order(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                symbol=self.underlying,
                position_type=PositionType.OPTION,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1,  # 1 contract = 100 shares
                option_contract=contract,
            ))

            self.last_roll_date = timestamp

        return orders


class IronCondorStrategy(Strategy):
    """Iron condor strategy: sell put spread + sell call spread"""

    def __init__(self, underlying: str, wing_width: float = 0.05):
        super().__init__("Iron Condor")
        self.underlying = underlying
        self.wing_width = wing_width
        self.last_entry_date: Optional[datetime] = None
        self.expiry_days = 45

    def on_bar(
        self,
        timestamp: datetime,
        bars: Dict[str, OHLCV],
        portfolio: BacktestPortfolio,
        engine: BacktestEngine,
    ) -> List[Order]:
        orders = []

        if self.underlying not in bars:
            return orders

        bar = bars[self.underlying]
        spot = float(bar.close)

        # Check if we need to enter new position
        has_position = any(
            p.symbol == self.underlying and p.position_type == PositionType.OPTION
            for p in portfolio.positions.values()
        )

        if not has_position:
            expiry = timestamp + timedelta(days=self.expiry_days)

            # Put spread strikes
            put_short = round(spot * (1 - self.wing_width), 2)
            put_long = round(spot * (1 - 2 * self.wing_width), 2)

            # Call spread strikes
            call_short = round(spot * (1 + self.wing_width), 2)
            call_long = round(spot * (1 + 2 * self.wing_width), 2)

            # Sell put spread
            orders.append(Order(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                symbol=self.underlying,
                position_type=PositionType.OPTION,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1,
                option_contract=OptionContract(
                    underlying=self.underlying,
                    strike=put_short,
                    expiry=expiry,
                    option_type=OptionType.PUT,
                ),
            ))
            orders.append(Order(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                symbol=self.underlying,
                position_type=PositionType.OPTION,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1,
                option_contract=OptionContract(
                    underlying=self.underlying,
                    strike=put_long,
                    expiry=expiry,
                    option_type=OptionType.PUT,
                ),
            ))

            # Sell call spread
            orders.append(Order(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                symbol=self.underlying,
                position_type=PositionType.OPTION,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1,
                option_contract=OptionContract(
                    underlying=self.underlying,
                    strike=call_short,
                    expiry=expiry,
                    option_type=OptionType.CALL,
                ),
            ))
            orders.append(Order(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                symbol=self.underlying,
                position_type=PositionType.OPTION,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1,
                option_contract=OptionContract(
                    underlying=self.underlying,
                    strike=call_long,
                    expiry=expiry,
                    option_type=OptionType.CALL,
                ),
            ))

            self.last_entry_date = timestamp

        return orders


async def run_backtest(
    config: BacktestConfig,
    strategies: List[Strategy],
    data: Optional[Dict[str, List[OHLCV]]] = None,
) -> BacktestResult:
    """Run a backtest with the given configuration and strategies"""
    engine = BacktestEngine(config)

    for strategy in strategies:
        engine.add_strategy(strategy)

    if data:
        engine.load_data(data)
    else:
        await engine.load_data_from_providers()

    return engine.run()

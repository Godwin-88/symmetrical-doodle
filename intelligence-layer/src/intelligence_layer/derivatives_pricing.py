"""
Derivatives Pricing Engine
Supports: Options (European/American), Futures, Structured Products
Includes: Black-Scholes, Binomial Trees, Greeks, Volatility Surfaces
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class BarrierType(str, Enum):
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"


class ProductType(str, Enum):
    VANILLA_OPTION = "vanilla_option"
    BARRIER_OPTION = "barrier_option"
    BINARY_OPTION = "binary_option"
    FORWARD = "forward"
    FUTURE = "future"
    SWAP = "swap"
    SPREAD = "spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    IRON_CONDOR = "iron_condor"
    CALENDAR_SPREAD = "calendar_spread"


@dataclass
class Greeks:
    """Option Greeks"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0  # d(delta)/d(vol)
    volga: float = 0.0  # d(vega)/d(vol)
    charm: float = 0.0  # d(delta)/d(time)

    def to_dict(self) -> dict:
        return {
            "delta": round(self.delta, 6),
            "gamma": round(self.gamma, 6),
            "theta": round(self.theta, 6),
            "vega": round(self.vega, 6),
            "rho": round(self.rho, 6),
            "vanna": round(self.vanna, 6),
            "volga": round(self.volga, 6),
            "charm": round(self.charm, 6),
        }


@dataclass
class OptionContract:
    """Option contract specification"""
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    option_style: OptionStyle = OptionStyle.EUROPEAN
    contract_size: float = 100.0
    barrier: Optional[float] = None
    barrier_type: Optional[BarrierType] = None

    @property
    def is_barrier(self) -> bool:
        return self.barrier is not None

    def time_to_expiry(self, valuation_date: Optional[datetime] = None) -> float:
        """Time to expiry in years"""
        if valuation_date is None:
            valuation_date = datetime.now()
        delta = self.expiry - valuation_date
        return max(delta.total_seconds() / (365.25 * 24 * 60 * 60), 0)


@dataclass
class FuturesContract:
    """Futures contract specification"""
    underlying: str
    expiry: datetime
    contract_size: float
    tick_size: float
    tick_value: float
    margin_initial: float
    margin_maintenance: float
    settlement_type: str = "cash"  # cash or physical

    def time_to_expiry(self, valuation_date: Optional[datetime] = None) -> float:
        if valuation_date is None:
            valuation_date = datetime.now()
        delta = self.expiry - valuation_date
        return max(delta.total_seconds() / (365.25 * 24 * 60 * 60), 0)


@dataclass
class StructuredProduct:
    """Structured product composed of multiple legs"""
    name: str
    product_type: ProductType
    legs: List[Dict[str, Any]] = field(default_factory=list)
    underlying: Optional[str] = None
    notional: float = 100000.0

    def add_option_leg(
        self,
        option_type: OptionType,
        strike: float,
        expiry: datetime,
        quantity: int = 1,
        option_style: OptionStyle = OptionStyle.EUROPEAN,
    ):
        self.legs.append({
            "type": "option",
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry,
            "quantity": quantity,
            "option_style": option_style,
        })

    def add_future_leg(self, expiry: datetime, quantity: int = 1):
        self.legs.append({
            "type": "future",
            "expiry": expiry,
            "quantity": quantity,
        })


@dataclass
class PricingResult:
    """Result of derivative pricing"""
    price: float
    greeks: Optional[Greeks] = None
    implied_volatility: Optional[float] = None
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    breakeven: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    probability_itm: Optional[float] = None
    pricing_model: str = "black_scholes"
    valuation_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 6),
            "greeks": self.greeks.to_dict() if self.greeks else None,
            "implied_volatility": round(self.implied_volatility, 6) if self.implied_volatility else None,
            "intrinsic_value": round(self.intrinsic_value, 6),
            "time_value": round(self.time_value, 6),
            "breakeven": round(self.breakeven, 6) if self.breakeven else None,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "probability_itm": round(self.probability_itm, 4) if self.probability_itm else None,
            "pricing_model": self.pricing_model,
            "valuation_date": self.valuation_date.isoformat(),
        }


class BlackScholes:
    """
    Black-Scholes Option Pricing Model
    Supports European options on non-dividend paying assets
    """

    @staticmethod
    def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes._d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

    @classmethod
    def price(
        cls,
        S: float,  # Spot price
        K: float,  # Strike price
        r: float,  # Risk-free rate (annual)
        sigma: float,  # Volatility (annual)
        T: float,  # Time to expiry (years)
        option_type: OptionType,
        q: float = 0.0,  # Dividend yield
    ) -> float:
        """Calculate option price using Black-Scholes formula"""
        if T <= 0:
            # At expiry, return intrinsic value
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        # Adjust for dividend yield
        S_adj = S * math.exp(-q * T)

        d1 = cls._d1(S_adj, K, r, sigma, T)
        d2 = cls._d2(S_adj, K, r, sigma, T)

        if option_type == OptionType.CALL:
            price = S_adj * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S_adj * stats.norm.cdf(-d1)

        return max(price, 0)

    @classmethod
    def greeks(
        cls,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> Greeks:
        """Calculate option Greeks"""
        if T <= 0:
            # At expiry
            delta = 1.0 if option_type == OptionType.CALL and S > K else -1.0 if option_type == OptionType.PUT and S < K else 0.0
            return Greeks(delta=delta)

        S_adj = S * math.exp(-q * T)
        d1 = cls._d1(S_adj, K, r, sigma, T)
        d2 = cls._d2(S_adj, K, r, sigma, T)

        sqrt_T = math.sqrt(T)
        exp_q_T = math.exp(-q * T)
        exp_r_T = math.exp(-r * T)
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_d2 = stats.norm.cdf(d2)

        # Delta
        if option_type == OptionType.CALL:
            delta = exp_q_T * cdf_d1
        else:
            delta = exp_q_T * (cdf_d1 - 1)

        # Gamma (same for call and put)
        gamma = (exp_q_T * pdf_d1) / (S * sigma * sqrt_T)

        # Theta (per day)
        term1 = -(S * sigma * exp_q_T * pdf_d1) / (2 * sqrt_T)
        if option_type == OptionType.CALL:
            theta = term1 - r * K * exp_r_T * cdf_d2 + q * S * exp_q_T * cdf_d1
        else:
            theta = term1 + r * K * exp_r_T * (1 - cdf_d2) - q * S * exp_q_T * (1 - cdf_d1)
        theta = theta / 365  # Per day

        # Vega (for 1% vol change)
        vega = S * exp_q_T * sqrt_T * pdf_d1 / 100

        # Rho (for 1% rate change)
        if option_type == OptionType.CALL:
            rho = K * T * exp_r_T * cdf_d2 / 100
        else:
            rho = -K * T * exp_r_T * (1 - cdf_d2) / 100

        # Vanna: d(delta)/d(sigma)
        vanna = -exp_q_T * pdf_d1 * d2 / sigma

        # Volga: d(vega)/d(sigma)
        volga = vega * d1 * d2 / sigma

        # Charm: d(delta)/d(time)
        charm = exp_q_T * pdf_d1 * (q - (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T))

        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            vanna=vanna,
            volga=volga,
            charm=charm,
        )

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        option_type: OptionType,
        q: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> Optional[float]:
        """Calculate implied volatility from market price"""
        if T <= 0 or market_price <= 0:
            return None

        def objective(sigma):
            return cls.price(S, K, r, sigma, T, option_type, q) - market_price

        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations, xtol=tolerance)
            return iv
        except (ValueError, RuntimeError):
            # Newton-Raphson fallback
            sigma = 0.2  # Initial guess
            for _ in range(max_iterations):
                price = cls.price(S, K, r, sigma, T, option_type, q)
                vega = cls.greeks(S, K, r, sigma, T, option_type, q).vega * 100  # Unscale
                if abs(vega) < 1e-12:
                    break
                diff = price - market_price
                if abs(diff) < tolerance:
                    return sigma
                sigma = sigma - diff / vega
                sigma = max(0.001, min(sigma, 5.0))
            return sigma if abs(cls.price(S, K, r, sigma, T, option_type, q) - market_price) < 0.01 else None


class BinomialTree:
    """
    Binomial Tree Model for American and European Options
    More accurate for American options with early exercise
    """

    @classmethod
    def price(
        cls,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: OptionType,
        option_style: OptionStyle = OptionStyle.EUROPEAN,
        steps: int = 100,
        q: float = 0.0,
    ) -> float:
        """Price option using binomial tree"""
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)

        # Build price tree at expiration
        stock_prices = np.zeros(steps + 1)
        option_values = np.zeros(steps + 1)

        for i in range(steps + 1):
            stock_prices[i] = S * (u ** (steps - i)) * (d ** i)
            if option_type == OptionType.CALL:
                option_values[i] = max(stock_prices[i] - K, 0)
            else:
                option_values[i] = max(K - stock_prices[i], 0)

        # Backward induction
        discount = math.exp(-r * dt)
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                stock_price = S * (u ** (step - i)) * (d ** i)
                option_values[i] = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])

                # Early exercise for American options
                if option_style == OptionStyle.AMERICAN:
                    if option_type == OptionType.CALL:
                        intrinsic = max(stock_price - K, 0)
                    else:
                        intrinsic = max(K - stock_price, 0)
                    option_values[i] = max(option_values[i], intrinsic)

        return option_values[0]


class VolatilitySurface:
    """
    Volatility Surface for strike/expiry interpolation
    Supports SABR and SVI parameterizations
    """

    def __init__(self):
        self.strikes: np.ndarray = np.array([])
        self.expiries: np.ndarray = np.array([])
        self.vols: np.ndarray = np.array([])  # 2D: strikes x expiries
        self._interpolator: Optional[RectBivariateSpline] = None

    def build_from_quotes(
        self,
        strikes: List[float],
        expiries: List[float],  # In years
        implied_vols: List[List[float]],  # 2D: [strike_idx][expiry_idx]
    ):
        """Build surface from market quotes"""
        self.strikes = np.array(strikes)
        self.expiries = np.array(expiries)
        self.vols = np.array(implied_vols)
        self._interpolator = RectBivariateSpline(self.strikes, self.expiries, self.vols)

    def get_vol(self, strike: float, expiry: float) -> float:
        """Interpolate volatility for given strike and expiry"""
        if self._interpolator is None:
            raise ValueError("Surface not built. Call build_from_quotes first.")
        return float(self._interpolator(strike, expiry)[0, 0])

    def get_smile(self, expiry: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get volatility smile for a given expiry"""
        if self._interpolator is None:
            raise ValueError("Surface not built.")
        vols = np.array([self.get_vol(k, expiry) for k in self.strikes])
        return self.strikes, vols


class FuturesPricer:
    """Futures pricing and margin calculations"""

    @staticmethod
    def fair_value(
        spot: float,
        r: float,  # Risk-free rate
        T: float,  # Time to expiry in years
        q: float = 0.0,  # Dividend yield or convenience yield
        storage_cost: float = 0.0,  # For commodities
    ) -> float:
        """Calculate theoretical futures price"""
        return spot * math.exp((r - q + storage_cost) * T)

    @staticmethod
    def basis(spot: float, futures_price: float) -> float:
        """Calculate basis (spot - futures)"""
        return spot - futures_price

    @staticmethod
    def implied_repo_rate(
        spot: float,
        futures_price: float,
        T: float,
        q: float = 0.0,
    ) -> float:
        """Calculate implied repo/financing rate"""
        if T <= 0 or spot <= 0:
            return 0.0
        return (math.log(futures_price / spot) / T) + q

    @staticmethod
    def margin_call(
        position_value: float,
        maintenance_margin: float,
        current_margin: float,
    ) -> float:
        """Calculate margin call amount"""
        required = position_value * maintenance_margin
        if current_margin < required:
            # Need to restore to initial margin level
            return required - current_margin
        return 0.0


class DerivativesPricingEngine:
    """
    Main pricing engine that orchestrates all derivative pricing
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.volatility_surface: Optional[VolatilitySurface] = None
        self.dividend_yields: Dict[str, float] = {}

    def set_risk_free_rate(self, rate: float):
        """Update risk-free rate"""
        self.risk_free_rate = rate

    def set_dividend_yield(self, underlying: str, yield_rate: float):
        """Set dividend yield for an underlying"""
        self.dividend_yields[underlying] = yield_rate

    def set_volatility_surface(self, surface: VolatilitySurface):
        """Set volatility surface for interpolation"""
        self.volatility_surface = surface

    def price_option(
        self,
        contract: OptionContract,
        spot: float,
        volatility: float,
        valuation_date: Optional[datetime] = None,
    ) -> PricingResult:
        """Price an option contract"""
        if valuation_date is None:
            valuation_date = datetime.now()

        T = contract.time_to_expiry(valuation_date)
        q = self.dividend_yields.get(contract.underlying, 0.0)

        # Choose pricing model
        if contract.option_style == OptionStyle.AMERICAN:
            price = BinomialTree.price(
                S=spot,
                K=contract.strike,
                r=self.risk_free_rate,
                sigma=volatility,
                T=T,
                option_type=contract.option_type,
                option_style=contract.option_style,
                q=q,
            )
            pricing_model = "binomial_tree"
        else:
            price = BlackScholes.price(
                S=spot,
                K=contract.strike,
                r=self.risk_free_rate,
                sigma=volatility,
                T=T,
                option_type=contract.option_type,
                q=q,
            )
            pricing_model = "black_scholes"

        # Calculate Greeks using Black-Scholes (approximation for American)
        greeks = BlackScholes.greeks(
            S=spot,
            K=contract.strike,
            r=self.risk_free_rate,
            sigma=volatility,
            T=T,
            option_type=contract.option_type,
            q=q,
        )

        # Calculate intrinsic value
        if contract.option_type == OptionType.CALL:
            intrinsic = max(spot - contract.strike, 0)
            breakeven = contract.strike + price
        else:
            intrinsic = max(contract.strike - spot, 0)
            breakeven = contract.strike - price

        # Probability ITM (using Black-Scholes d2)
        if T > 0:
            d2 = BlackScholes._d2(spot, contract.strike, self.risk_free_rate, volatility, T)
            if contract.option_type == OptionType.CALL:
                prob_itm = stats.norm.cdf(d2)
            else:
                prob_itm = stats.norm.cdf(-d2)
        else:
            prob_itm = 1.0 if intrinsic > 0 else 0.0

        return PricingResult(
            price=price * contract.contract_size,
            greeks=greeks,
            intrinsic_value=intrinsic,
            time_value=price - intrinsic,
            breakeven=breakeven,
            probability_itm=prob_itm,
            pricing_model=pricing_model,
            valuation_date=valuation_date,
        )

    def price_futures(
        self,
        contract: FuturesContract,
        spot: float,
        convenience_yield: float = 0.0,
        storage_cost: float = 0.0,
        valuation_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Price a futures contract"""
        if valuation_date is None:
            valuation_date = datetime.now()

        T = contract.time_to_expiry(valuation_date)
        q = convenience_yield

        fair_value = FuturesPricer.fair_value(
            spot=spot,
            r=self.risk_free_rate,
            T=T,
            q=q,
            storage_cost=storage_cost,
        )

        return {
            "fair_value": fair_value,
            "spot": spot,
            "basis": FuturesPricer.basis(spot, fair_value),
            "time_to_expiry": T,
            "risk_free_rate": self.risk_free_rate,
            "convenience_yield": convenience_yield,
            "storage_cost": storage_cost,
            "margin_initial": contract.margin_initial,
            "margin_maintenance": contract.margin_maintenance,
            "valuation_date": valuation_date.isoformat(),
        }

    def price_structured_product(
        self,
        product: StructuredProduct,
        spot: float,
        volatility: float,
        valuation_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Price a structured product by pricing each leg"""
        if valuation_date is None:
            valuation_date = datetime.now()

        total_price = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        leg_results = []

        for leg in product.legs:
            quantity = leg.get("quantity", 1)

            if leg["type"] == "option":
                contract = OptionContract(
                    underlying=product.underlying or "",
                    strike=leg["strike"],
                    expiry=leg["expiry"],
                    option_type=leg["option_type"],
                    option_style=leg.get("option_style", OptionStyle.EUROPEAN),
                )
                result = self.price_option(contract, spot, volatility, valuation_date)
                leg_price = result.price * quantity
                total_price += leg_price
                if result.greeks:
                    total_delta += result.greeks.delta * quantity
                    total_gamma += result.greeks.gamma * quantity
                    total_vega += result.greeks.vega * quantity
                    total_theta += result.greeks.theta * quantity

                leg_results.append({
                    "type": "option",
                    "option_type": leg["option_type"].value,
                    "strike": leg["strike"],
                    "expiry": leg["expiry"].isoformat(),
                    "quantity": quantity,
                    "price": leg_price,
                    "delta": result.greeks.delta * quantity if result.greeks else 0,
                })

            elif leg["type"] == "future":
                # Simplified futures leg pricing
                T = (leg["expiry"] - valuation_date).days / 365.25
                fair_value = FuturesPricer.fair_value(spot, self.risk_free_rate, T)
                leg_price = (fair_value - spot) * quantity
                total_price += leg_price
                total_delta += quantity  # Futures have delta of 1

                leg_results.append({
                    "type": "future",
                    "expiry": leg["expiry"].isoformat(),
                    "quantity": quantity,
                    "fair_value": fair_value,
                    "price": leg_price,
                })

        # Calculate P&L at various spots for payoff diagram
        spot_range = np.linspace(spot * 0.7, spot * 1.3, 50)
        payoffs = []
        for s in spot_range:
            payoff = -total_price  # Initial cost
            for leg in product.legs:
                quantity = leg.get("quantity", 1)
                if leg["type"] == "option":
                    if leg["option_type"] == OptionType.CALL:
                        payoff += max(s - leg["strike"], 0) * quantity * 100
                    else:
                        payoff += max(leg["strike"] - s, 0) * quantity * 100
                elif leg["type"] == "future":
                    payoff += (s - spot) * quantity
            payoffs.append(payoff)

        return {
            "product_name": product.name,
            "product_type": product.product_type.value,
            "total_price": total_price,
            "net_greeks": {
                "delta": total_delta,
                "gamma": total_gamma,
                "vega": total_vega,
                "theta": total_theta,
            },
            "legs": leg_results,
            "payoff_diagram": {
                "spots": spot_range.tolist(),
                "payoffs": payoffs,
            },
            "max_profit": max(payoffs),
            "max_loss": min(payoffs),
            "breakevens": [spot_range[i] for i in range(1, len(payoffs)) if payoffs[i-1] * payoffs[i] < 0],
            "valuation_date": valuation_date.isoformat(),
        }

    def calculate_implied_vol(
        self,
        market_price: float,
        spot: float,
        strike: float,
        T: float,
        option_type: OptionType,
    ) -> Optional[float]:
        """Calculate implied volatility from market price"""
        return BlackScholes.implied_volatility(
            market_price=market_price,
            S=spot,
            K=strike,
            r=self.risk_free_rate,
            T=T,
            option_type=option_type,
        )


# Pre-built structured product templates
def create_straddle(
    underlying: str,
    strike: float,
    expiry: datetime,
    notional: float = 100000.0,
) -> StructuredProduct:
    """Create a straddle (long call + long put at same strike)"""
    product = StructuredProduct(
        name=f"Straddle {underlying} @ {strike}",
        product_type=ProductType.STRADDLE,
        underlying=underlying,
        notional=notional,
    )
    product.add_option_leg(OptionType.CALL, strike, expiry, quantity=1)
    product.add_option_leg(OptionType.PUT, strike, expiry, quantity=1)
    return product


def create_strangle(
    underlying: str,
    call_strike: float,
    put_strike: float,
    expiry: datetime,
    notional: float = 100000.0,
) -> StructuredProduct:
    """Create a strangle (long OTM call + long OTM put)"""
    product = StructuredProduct(
        name=f"Strangle {underlying} {put_strike}/{call_strike}",
        product_type=ProductType.STRANGLE,
        underlying=underlying,
        notional=notional,
    )
    product.add_option_leg(OptionType.CALL, call_strike, expiry, quantity=1)
    product.add_option_leg(OptionType.PUT, put_strike, expiry, quantity=1)
    return product


def create_butterfly(
    underlying: str,
    lower_strike: float,
    middle_strike: float,
    upper_strike: float,
    expiry: datetime,
    option_type: OptionType = OptionType.CALL,
    notional: float = 100000.0,
) -> StructuredProduct:
    """Create a butterfly spread"""
    product = StructuredProduct(
        name=f"Butterfly {underlying} {lower_strike}/{middle_strike}/{upper_strike}",
        product_type=ProductType.BUTTERFLY,
        underlying=underlying,
        notional=notional,
    )
    product.add_option_leg(option_type, lower_strike, expiry, quantity=1)
    product.add_option_leg(option_type, middle_strike, expiry, quantity=-2)
    product.add_option_leg(option_type, upper_strike, expiry, quantity=1)
    return product


def create_iron_condor(
    underlying: str,
    put_lower: float,
    put_upper: float,
    call_lower: float,
    call_upper: float,
    expiry: datetime,
    notional: float = 100000.0,
) -> StructuredProduct:
    """Create an iron condor (sell put spread + sell call spread)"""
    product = StructuredProduct(
        name=f"Iron Condor {underlying}",
        product_type=ProductType.IRON_CONDOR,
        underlying=underlying,
        notional=notional,
    )
    # Put spread (bull put spread - sell higher, buy lower)
    product.add_option_leg(OptionType.PUT, put_lower, expiry, quantity=1)  # Long
    product.add_option_leg(OptionType.PUT, put_upper, expiry, quantity=-1)  # Short
    # Call spread (bear call spread - sell lower, buy higher)
    product.add_option_leg(OptionType.CALL, call_lower, expiry, quantity=-1)  # Short
    product.add_option_leg(OptionType.CALL, call_upper, expiry, quantity=1)  # Long
    return product


def create_calendar_spread(
    underlying: str,
    strike: float,
    near_expiry: datetime,
    far_expiry: datetime,
    option_type: OptionType = OptionType.CALL,
    notional: float = 100000.0,
) -> StructuredProduct:
    """Create a calendar spread (short near, long far at same strike)"""
    product = StructuredProduct(
        name=f"Calendar Spread {underlying} @ {strike}",
        product_type=ProductType.CALENDAR_SPREAD,
        underlying=underlying,
        notional=notional,
    )
    product.add_option_leg(option_type, strike, near_expiry, quantity=-1)  # Short near
    product.add_option_leg(option_type, strike, far_expiry, quantity=1)  # Long far
    return product


# Singleton pricing engine instance
_pricing_engine: Optional[DerivativesPricingEngine] = None


def get_pricing_engine() -> DerivativesPricingEngine:
    """Get the singleton pricing engine"""
    global _pricing_engine
    if _pricing_engine is None:
        _pricing_engine = DerivativesPricingEngine()
    return _pricing_engine

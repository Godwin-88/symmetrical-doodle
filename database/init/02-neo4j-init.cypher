// Initialize Neo4j schema for algorithmic trading system

// Create constraints for node uniqueness
CREATE CONSTRAINT asset_id_unique IF NOT EXISTS FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE;
CREATE CONSTRAINT regime_id_unique IF NOT EXISTS FOR (r:MarketRegime) REQUIRE r.regime_id IS UNIQUE;
CREATE CONSTRAINT strategy_id_unique IF NOT EXISTS FOR (s:Strategy) REQUIRE s.strategy_id IS UNIQUE;
CREATE CONSTRAINT signal_id_unique IF NOT EXISTS FOR (i:IntelligenceSignal) REQUIRE i.signal_id IS UNIQUE;
CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:MacroEvent) REQUIRE e.event_id IS UNIQUE;

// Create indexes for performance
CREATE INDEX asset_class_index IF NOT EXISTS FOR (a:Asset) ON (a.asset_class);
CREATE INDEX regime_volatility_index IF NOT EXISTS FOR (r:MarketRegime) ON (r.volatility_level);
CREATE INDEX strategy_family_index IF NOT EXISTS FOR (s:Strategy) ON (s.family);
CREATE INDEX signal_timestamp_index IF NOT EXISTS FOR (i:IntelligenceSignal) ON (i.timestamp);

// Create sample assets
CREATE (eurusd:Asset {
    asset_id: 'EURUSD',
    asset_class: 'FX',
    venue: 'Deriv',
    base_currency: 'EUR',
    quote_currency: 'USD'
});

CREATE (gbpusd:Asset {
    asset_id: 'GBPUSD',
    asset_class: 'FX',
    venue: 'Deriv',
    base_currency: 'GBP',
    quote_currency: 'USD'
});

CREATE (usdjpy:Asset {
    asset_id: 'USDJPY',
    asset_class: 'FX',
    venue: 'Deriv',
    base_currency: 'USD',
    quote_currency: 'JPY'
});

CREATE (btcusd:Asset {
    asset_id: 'BTCUSD',
    asset_class: 'Crypto',
    venue: 'Deriv',
    base_currency: 'BTC',
    quote_currency: 'USD'
});

// Create market regimes
CREATE (low_vol_trending:MarketRegime {
    regime_id: 'low_vol_trending',
    volatility_level: 'low',
    trend_state: 'trending',
    liquidity_state: 'normal',
    description: 'Low volatility trending market with normal liquidity'
});

CREATE (high_vol_ranging:MarketRegime {
    regime_id: 'high_vol_ranging',
    volatility_level: 'high',
    trend_state: 'ranging',
    liquidity_state: 'stressed',
    description: 'High volatility ranging market with stressed liquidity'
});

CREATE (medium_vol_trending:MarketRegime {
    regime_id: 'medium_vol_trending',
    volatility_level: 'medium',
    trend_state: 'trending',
    liquidity_state: 'normal',
    description: 'Medium volatility trending market with normal liquidity'
});

// Create strategies
CREATE (trend_following:Strategy {
    strategy_id: 'trend_following_1',
    family: 'trend',
    horizon: 'daily',
    description: 'Momentum-based trend following strategy'
});

CREATE (mean_reversion:Strategy {
    strategy_id: 'mean_reversion_1',
    family: 'mean_reversion',
    horizon: 'intraday',
    description: 'Statistical arbitrage mean reversion strategy'
});

CREATE (volatility_strategy:Strategy {
    strategy_id: 'volatility_1',
    family: 'volatility',
    horizon: 'daily',
    description: 'Volatility breakout strategy'
});

// Create macro events
CREATE (fed_meeting:MacroEvent {
    event_id: 'fed_meeting_2024_01',
    category: 'monetary_policy',
    timestamp: datetime('2024-01-31T19:00:00Z'),
    surprise_score: 0.2
});

CREATE (nfp_release:MacroEvent {
    event_id: 'nfp_2024_01',
    category: 'employment',
    timestamp: datetime('2024-01-05T13:30:00Z'),
    surprise_score: -0.5
});

// Create relationships between assets (correlations)
MATCH (eur:Asset {asset_id: 'EURUSD'}), (gbp:Asset {asset_id: 'GBPUSD'})
CREATE (eur)-[:CORRELATED {window: '1d', strength: 0.75, sign: 1}]->(gbp);

MATCH (eur:Asset {asset_id: 'EURUSD'}), (jpy:Asset {asset_id: 'USDJPY'})
CREATE (eur)-[:CORRELATED {window: '1d', strength: -0.45, sign: -1}]->(jpy);

MATCH (btc:Asset {asset_id: 'BTCUSD'}), (eur:Asset {asset_id: 'EURUSD'})
CREATE (btc)-[:CORRELATED {window: '1d', strength: 0.15, sign: 1}]->(eur);

// Create regime transitions
MATCH (low:MarketRegime {regime_id: 'low_vol_trending'}), (med:MarketRegime {regime_id: 'medium_vol_trending'})
CREATE (low)-[:TRANSITIONS_TO {probability: 0.3, avg_duration: 5.2}]->(med);

MATCH (med:MarketRegime {regime_id: 'medium_vol_trending'}), (high:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (med)-[:TRANSITIONS_TO {probability: 0.25, avg_duration: 3.1}]->(high);

MATCH (high:MarketRegime {regime_id: 'high_vol_ranging'}), (low:MarketRegime {regime_id: 'low_vol_trending'})
CREATE (high)-[:TRANSITIONS_TO {probability: 0.4, avg_duration: 2.8}]->(low);

// Create strategy performance in regimes
MATCH (trend:Strategy {strategy_id: 'trend_following_1'}), (low:MarketRegime {regime_id: 'low_vol_trending'})
CREATE (trend)-[:PERFORMS_IN {sharpe: 1.8, max_dd: 0.05, sample_size: 120}]->(low);

MATCH (trend:Strategy {strategy_id: 'trend_following_1'}), (high:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (trend)-[:PERFORMS_IN {sharpe: -0.2, max_dd: 0.15, sample_size: 45}]->(high);

MATCH (mean:Strategy {strategy_id: 'mean_reversion_1'}), (high:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (mean)-[:PERFORMS_IN {sharpe: 2.1, max_dd: 0.08, sample_size: 67}]->(high);

MATCH (mean:Strategy {strategy_id: 'mean_reversion_1'}), (low:MarketRegime {regime_id: 'low_vol_trending'})
CREATE (mean)-[:PERFORMS_IN {sharpe: 0.3, max_dd: 0.12, sample_size: 89}]->(low);

// Create asset sensitivity to regimes
MATCH (eur:Asset {asset_id: 'EURUSD'}), (high:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (eur)-[:SENSITIVE_TO {beta: 1.2, lag: 0}]->(high);

MATCH (btc:Asset {asset_id: 'BTCUSD'}), (high:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (btc)-[:SENSITIVE_TO {beta: 2.5, lag: 1}]->(high);

// Create macro event impacts
MATCH (fed:MacroEvent {event_id: 'fed_meeting_2024_01'}), (eur:Asset {asset_id: 'EURUSD'})
CREATE (fed)-[:AFFECTS {impact_score: 0.8}]->(eur);

MATCH (nfp:MacroEvent {event_id: 'nfp_2024_01'}), (eur:Asset {asset_id: 'EURUSD'})
CREATE (nfp)-[:AFFECTS {impact_score: 0.6}]->(eur);

MATCH (nfp:MacroEvent {event_id: 'nfp_2024_01'}), (gbp:Asset {asset_id: 'GBPUSD'})
CREATE (nfp)-[:AFFECTS {impact_score: 0.4}]->(gbp);
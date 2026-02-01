// ============================================================================
// Neo4j GraphRAG Schema Extension for Derivatives Trading
// ============================================================================
// This script extends the base Neo4j schema with derivatives-specific nodes
// and relationships for advanced graph-based analytics and RAG.

// ============================================================================
// CONSTRAINTS AND INDEXES
// ============================================================================

// Derivatives contract constraints
CREATE CONSTRAINT option_contract_id_unique IF NOT EXISTS FOR (o:OptionContract) REQUIRE o.contract_id IS UNIQUE;
CREATE CONSTRAINT futures_contract_id_unique IF NOT EXISTS FOR (f:FuturesContract) REQUIRE f.contract_id IS UNIQUE;
CREATE CONSTRAINT structured_product_id_unique IF NOT EXISTS FOR (sp:StructuredProduct) REQUIRE sp.product_id IS UNIQUE;
CREATE CONSTRAINT backtest_id_unique IF NOT EXISTS FOR (b:Backtest) REQUIRE b.backtest_id IS UNIQUE;
CREATE CONSTRAINT strategy_template_id_unique IF NOT EXISTS FOR (st:StrategyTemplate) REQUIRE st.template_id IS UNIQUE;
CREATE CONSTRAINT volatility_surface_id_unique IF NOT EXISTS FOR (vs:VolatilitySurface) REQUIRE vs.surface_id IS UNIQUE;
CREATE CONSTRAINT greek_snapshot_id_unique IF NOT EXISTS FOR (gs:GreekSnapshot) REQUIRE gs.snapshot_id IS UNIQUE;

// Performance indexes
CREATE INDEX option_type_index IF NOT EXISTS FOR (o:OptionContract) ON (o.option_type);
CREATE INDEX option_expiry_index IF NOT EXISTS FOR (o:OptionContract) ON (o.expiry_date);
CREATE INDEX futures_expiry_index IF NOT EXISTS FOR (f:FuturesContract) ON (f.expiry_date);
CREATE INDEX product_type_index IF NOT EXISTS FOR (sp:StructuredProduct) ON (sp.product_type);
CREATE INDEX backtest_status_index IF NOT EXISTS FOR (b:Backtest) ON (b.status);
CREATE INDEX strategy_type_index IF NOT EXISTS FOR (st:StrategyTemplate) ON (st.strategy_type);

// ============================================================================
// EXTENDED ASSET NODES
// ============================================================================

// Add commodity assets
CREATE (gold:Asset {
    asset_id: 'XAUUSD',
    asset_class: 'Commodity',
    venue: 'Deriv',
    name: 'Gold',
    base_currency: 'XAU',
    quote_currency: 'USD',
    margin_requirement: 5.0
});

CREATE (silver:Asset {
    asset_id: 'XAGUSD',
    asset_class: 'Commodity',
    venue: 'Deriv',
    name: 'Silver',
    base_currency: 'XAG',
    quote_currency: 'USD',
    margin_requirement: 5.0
});

// Add more forex pairs
CREATE (audusd:Asset {
    asset_id: 'AUDUSD',
    asset_class: 'FX',
    venue: 'Deriv',
    base_currency: 'AUD',
    quote_currency: 'USD'
});

CREATE (usdchf:Asset {
    asset_id: 'USDCHF',
    asset_class: 'FX',
    venue: 'Deriv',
    base_currency: 'USD',
    quote_currency: 'CHF'
});

// Add crypto assets
CREATE (ethusd:Asset {
    asset_id: 'ETHUSD',
    asset_class: 'Crypto',
    venue: 'Deriv',
    base_currency: 'ETH',
    quote_currency: 'USD',
    margin_requirement: 50.0
});

// ============================================================================
// OPTION CONTRACT NODES
// ============================================================================

// Sample option contracts on Gold
CREATE (gold_call_2400:OptionContract {
    contract_id: 'XAUUSD_C_2400_2024Q1',
    underlying: 'XAUUSD',
    option_type: 'call',
    option_style: 'european',
    strike: 2400.0,
    expiry_date: date('2024-03-29'),
    multiplier: 1.0,
    description: 'Gold $2400 Call Mar-24'
});

CREATE (gold_put_2000:OptionContract {
    contract_id: 'XAUUSD_P_2000_2024Q1',
    underlying: 'XAUUSD',
    option_type: 'put',
    option_style: 'european',
    strike: 2000.0,
    expiry_date: date('2024-03-29'),
    multiplier: 1.0,
    description: 'Gold $2000 Put Mar-24'
});

// Sample option contracts on Bitcoin
CREATE (btc_call_50k:OptionContract {
    contract_id: 'BTCUSD_C_50000_2024Q1',
    underlying: 'BTCUSD',
    option_type: 'call',
    option_style: 'european',
    strike: 50000.0,
    expiry_date: date('2024-03-29'),
    multiplier: 1.0,
    description: 'Bitcoin $50K Call Mar-24'
});

CREATE (btc_put_40k:OptionContract {
    contract_id: 'BTCUSD_P_40000_2024Q1',
    underlying: 'BTCUSD',
    option_type: 'put',
    option_style: 'european',
    strike: 40000.0,
    expiry_date: date('2024-03-29'),
    multiplier: 1.0,
    description: 'Bitcoin $40K Put Mar-24'
});

// EURUSD options
CREATE (eur_call_1.10:OptionContract {
    contract_id: 'EURUSD_C_1.10_2024Q1',
    underlying: 'EURUSD',
    option_type: 'call',
    option_style: 'european',
    strike: 1.10,
    expiry_date: date('2024-03-29'),
    multiplier: 100000.0,
    description: 'EUR/USD 1.10 Call Mar-24'
});

// ============================================================================
// FUTURES CONTRACT NODES
// ============================================================================

CREATE (gold_futures_q1:FuturesContract {
    contract_id: 'XAUUSD_FUT_2024Q1',
    underlying: 'XAUUSD',
    expiry_date: date('2024-03-29'),
    contract_size: 100.0,
    tick_value: 10.0,
    description: 'Gold Futures Mar-24'
});

CREATE (btc_futures_q1:FuturesContract {
    contract_id: 'BTCUSD_FUT_2024Q1',
    underlying: 'BTCUSD',
    expiry_date: date('2024-03-29'),
    contract_size: 1.0,
    tick_value: 1.0,
    description: 'Bitcoin Futures Mar-24'
});

// ============================================================================
// STRUCTURED PRODUCT NODES
// ============================================================================

CREATE (gold_straddle:StructuredProduct {
    product_id: 'XAUUSD_STRADDLE_2200_2024Q1',
    product_name: 'Gold ATM Straddle',
    product_type: 'straddle',
    underlying: 'XAUUSD',
    strike: 2200.0,
    expiry_date: date('2024-03-29'),
    description: 'Long ATM straddle on Gold for volatility exposure'
});

CREATE (btc_iron_condor:StructuredProduct {
    product_id: 'BTCUSD_IC_2024Q1',
    product_name: 'Bitcoin Iron Condor',
    product_type: 'iron_condor',
    underlying: 'BTCUSD',
    put_lower: 35000.0,
    put_upper: 40000.0,
    call_lower: 50000.0,
    call_upper: 55000.0,
    expiry_date: date('2024-03-29'),
    description: 'Iron Condor on Bitcoin for premium collection'
});

CREATE (eur_butterfly:StructuredProduct {
    product_id: 'EURUSD_BUTTERFLY_2024Q1',
    product_name: 'EUR/USD Butterfly',
    product_type: 'butterfly',
    underlying: 'EURUSD',
    lower_strike: 1.08,
    middle_strike: 1.10,
    upper_strike: 1.12,
    expiry_date: date('2024-03-29'),
    description: 'Butterfly spread on EUR/USD'
});

// ============================================================================
// STRATEGY TEMPLATE NODES
// ============================================================================

CREATE (covered_call_template:StrategyTemplate {
    template_id: 'covered_call',
    strategy_type: 'covered_call',
    name: 'Covered Call Writing',
    description: 'Sell OTM calls against long underlying for income generation',
    risk_level: 'low',
    market_outlook: 'neutral',
    suitable_for: ['income', 'hedging'],
    parameters: '{"delta_target": 0.30, "roll_days": 7, "max_allocation": 0.20}'
});

CREATE (iron_condor_template:StrategyTemplate {
    template_id: 'iron_condor',
    strategy_type: 'iron_condor',
    name: 'Iron Condor',
    description: 'Sell OTM put and call spreads for premium in range-bound markets',
    risk_level: 'medium',
    market_outlook: 'neutral',
    suitable_for: ['premium_collection', 'range_trading'],
    parameters: '{"wing_width": 0.05, "delta_target": 0.15, "profit_target": 0.50}'
});

CREATE (straddle_template:StrategyTemplate {
    template_id: 'long_straddle',
    strategy_type: 'straddle',
    name: 'Long Straddle',
    description: 'Buy ATM call and put for volatility exposure ahead of events',
    risk_level: 'high',
    market_outlook: 'volatile',
    suitable_for: ['event_trading', 'volatility_speculation'],
    parameters: '{"entry_iv_percentile": 20, "exit_iv_percentile": 80}'
});

CREATE (calendar_template:StrategyTemplate {
    template_id: 'calendar_spread',
    strategy_type: 'calendar_spread',
    name: 'Calendar Spread',
    description: 'Sell near-term, buy far-term options at same strike',
    risk_level: 'medium',
    market_outlook: 'neutral',
    suitable_for: ['time_decay', 'iv_arbitrage'],
    parameters: '{"front_expiry_days": 30, "back_expiry_days": 60}'
});

// ============================================================================
// BACKTEST NODES
// ============================================================================

CREATE (gold_cc_backtest:Backtest {
    backtest_id: 'GOLD_CC_2023',
    name: 'Gold Covered Call 2023',
    strategy_type: 'covered_call',
    underlying: 'XAUUSD',
    start_date: date('2023-01-01'),
    end_date: date('2023-12-31'),
    initial_capital: 100000.0,
    status: 'completed',
    total_return: 0.124,
    sharpe_ratio: 1.45,
    max_drawdown: 0.087,
    win_rate: 0.72
});

CREATE (btc_ic_backtest:Backtest {
    backtest_id: 'BTC_IC_2023',
    name: 'Bitcoin Iron Condor 2023',
    strategy_type: 'iron_condor',
    underlying: 'BTCUSD',
    start_date: date('2023-01-01'),
    end_date: date('2023-12-31'),
    initial_capital: 100000.0,
    status: 'completed',
    total_return: 0.185,
    sharpe_ratio: 1.12,
    max_drawdown: 0.156,
    win_rate: 0.65
});

// ============================================================================
// VOLATILITY SURFACE NODES
// ============================================================================

CREATE (gold_vol_surface:VolatilitySurface {
    surface_id: 'XAUUSD_VOL_20240115',
    underlying: 'XAUUSD',
    timestamp: datetime('2024-01-15T16:00:00Z'),
    atm_vol: 0.145,
    skew: -0.02,
    kurtosis: 0.015,
    model: 'svi',
    term_structure: '{"1m": 0.14, "2m": 0.145, "3m": 0.15, "6m": 0.155}'
});

CREATE (btc_vol_surface:VolatilitySurface {
    surface_id: 'BTCUSD_VOL_20240115',
    underlying: 'BTCUSD',
    timestamp: datetime('2024-01-15T16:00:00Z'),
    atm_vol: 0.65,
    skew: -0.08,
    kurtosis: 0.05,
    model: 'svi',
    term_structure: '{"1m": 0.62, "2m": 0.64, "3m": 0.65, "6m": 0.68}'
});

// ============================================================================
// GREEK SNAPSHOT NODES (for time-series Greek tracking)
// ============================================================================

CREATE (gold_greeks_snapshot:GreekSnapshot {
    snapshot_id: 'XAUUSD_GREEKS_20240115',
    underlying: 'XAUUSD',
    timestamp: datetime('2024-01-15T16:00:00Z'),
    portfolio_delta: 0.35,
    portfolio_gamma: 0.002,
    portfolio_vega: 125.5,
    portfolio_theta: -45.2,
    delta_dollars: 35000.0,
    gamma_dollars: 200.0,
    vega_dollars: 1255.0
});

// ============================================================================
// RELATIONSHIPS: OPTIONS TO UNDERLYING
// ============================================================================

// Connect options to underlying assets
MATCH (o:OptionContract {contract_id: 'XAUUSD_C_2400_2024Q1'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (o)-[:HAS_UNDERLYING]->(a);

MATCH (o:OptionContract {contract_id: 'XAUUSD_P_2000_2024Q1'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (o)-[:HAS_UNDERLYING]->(a);

MATCH (o:OptionContract {contract_id: 'BTCUSD_C_50000_2024Q1'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (o)-[:HAS_UNDERLYING]->(a);

MATCH (o:OptionContract {contract_id: 'BTCUSD_P_40000_2024Q1'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (o)-[:HAS_UNDERLYING]->(a);

MATCH (o:OptionContract {contract_id: 'EURUSD_C_1.10_2024Q1'}), (a:Asset {asset_id: 'EURUSD'})
CREATE (o)-[:HAS_UNDERLYING]->(a);

// ============================================================================
// RELATIONSHIPS: FUTURES TO UNDERLYING
// ============================================================================

MATCH (f:FuturesContract {contract_id: 'XAUUSD_FUT_2024Q1'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (f)-[:HAS_UNDERLYING]->(a);

MATCH (f:FuturesContract {contract_id: 'BTCUSD_FUT_2024Q1'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (f)-[:HAS_UNDERLYING]->(a);

// ============================================================================
// RELATIONSHIPS: STRUCTURED PRODUCTS TO UNDERLYING
// ============================================================================

MATCH (sp:StructuredProduct {product_id: 'XAUUSD_STRADDLE_2200_2024Q1'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (sp)-[:HAS_UNDERLYING]->(a);

MATCH (sp:StructuredProduct {product_id: 'BTCUSD_IC_2024Q1'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (sp)-[:HAS_UNDERLYING]->(a);

MATCH (sp:StructuredProduct {product_id: 'EURUSD_BUTTERFLY_2024Q1'}), (a:Asset {asset_id: 'EURUSD'})
CREATE (sp)-[:HAS_UNDERLYING]->(a);

// ============================================================================
// RELATIONSHIPS: STRUCTURED PRODUCTS CONTAIN OPTIONS
// ============================================================================

MATCH (sp:StructuredProduct {product_id: 'XAUUSD_STRADDLE_2200_2024Q1'}),
      (c:OptionContract {contract_id: 'XAUUSD_C_2400_2024Q1'}),
      (p:OptionContract {contract_id: 'XAUUSD_P_2000_2024Q1'})
CREATE (sp)-[:CONTAINS_LEG {quantity: 1, leg_type: 'call'}]->(c),
       (sp)-[:CONTAINS_LEG {quantity: 1, leg_type: 'put'}]->(p);

// ============================================================================
// RELATIONSHIPS: BACKTESTS USE STRATEGIES
// ============================================================================

MATCH (b:Backtest {backtest_id: 'GOLD_CC_2023'}), (st:StrategyTemplate {template_id: 'covered_call'})
CREATE (b)-[:USES_STRATEGY]->(st);

MATCH (b:Backtest {backtest_id: 'BTC_IC_2023'}), (st:StrategyTemplate {template_id: 'iron_condor'})
CREATE (b)-[:USES_STRATEGY]->(st);

// ============================================================================
// RELATIONSHIPS: BACKTESTS ON UNDERLYING
// ============================================================================

MATCH (b:Backtest {backtest_id: 'GOLD_CC_2023'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (b)-[:BACKTESTED_ON]->(a);

MATCH (b:Backtest {backtest_id: 'BTC_IC_2023'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (b)-[:BACKTESTED_ON]->(a);

// ============================================================================
// RELATIONSHIPS: VOLATILITY SURFACES FOR ASSETS
// ============================================================================

MATCH (vs:VolatilitySurface {surface_id: 'XAUUSD_VOL_20240115'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (vs)-[:VOL_SURFACE_FOR]->(a);

MATCH (vs:VolatilitySurface {surface_id: 'BTCUSD_VOL_20240115'}), (a:Asset {asset_id: 'BTCUSD'})
CREATE (vs)-[:VOL_SURFACE_FOR]->(a);

// ============================================================================
// RELATIONSHIPS: GREEK SNAPSHOTS FOR ASSETS
// ============================================================================

MATCH (gs:GreekSnapshot {snapshot_id: 'XAUUSD_GREEKS_20240115'}), (a:Asset {asset_id: 'XAUUSD'})
CREATE (gs)-[:GREEKS_FOR]->(a);

// ============================================================================
// RELATIONSHIPS: STRATEGY SUITABILITY FOR REGIMES
// ============================================================================

// Covered calls work well in low volatility trending markets
MATCH (st:StrategyTemplate {template_id: 'covered_call'}), (r:MarketRegime {regime_id: 'low_vol_trending'})
CREATE (st)-[:SUITABLE_FOR_REGIME {confidence: 0.85, expected_sharpe: 1.4}]->(r);

// Iron condors work well in ranging markets
MATCH (st:StrategyTemplate {template_id: 'iron_condor'}), (r:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (st)-[:SUITABLE_FOR_REGIME {confidence: 0.75, expected_sharpe: 1.1}]->(r);

// Straddles work well in high volatility regimes
MATCH (st:StrategyTemplate {template_id: 'long_straddle'}), (r:MarketRegime {regime_id: 'high_vol_ranging'})
CREATE (st)-[:SUITABLE_FOR_REGIME {confidence: 0.80, expected_sharpe: 0.9}]->(r);

// ============================================================================
// RELATIONSHIPS: ASSET CORRELATIONS (Extended)
// ============================================================================

// Gold correlations
MATCH (gold:Asset {asset_id: 'XAUUSD'}), (silver:Asset {asset_id: 'XAGUSD'})
CREATE (gold)-[:CORRELATED {window: '1d', strength: 0.85, sign: 1}]->(silver);

MATCH (gold:Asset {asset_id: 'XAUUSD'}), (btc:Asset {asset_id: 'BTCUSD'})
CREATE (gold)-[:CORRELATED {window: '1d', strength: 0.25, sign: 1}]->(btc);

// Crypto correlations
MATCH (btc:Asset {asset_id: 'BTCUSD'}), (eth:Asset {asset_id: 'ETHUSD'})
CREATE (btc)-[:CORRELATED {window: '1d', strength: 0.90, sign: 1}]->(eth);

// ============================================================================
// GRAPHRAG: KNOWLEDGE GRAPH NODES
// ============================================================================

// Trading concepts for RAG
CREATE (delta_concept:TradingConcept {
    concept_id: 'delta',
    name: 'Delta',
    category: 'greeks',
    description: 'Rate of change of option price with respect to underlying price',
    formula: 'dV/dS for options',
    interpretation: 'Delta of 0.5 means option gains $0.50 for every $1 move in underlying'
});

CREATE (gamma_concept:TradingConcept {
    concept_id: 'gamma',
    name: 'Gamma',
    category: 'greeks',
    description: 'Rate of change of delta with respect to underlying price',
    formula: 'd²V/dS²',
    interpretation: 'Higher gamma means delta changes more rapidly as price moves'
});

CREATE (theta_concept:TradingConcept {
    concept_id: 'theta',
    name: 'Theta',
    category: 'greeks',
    description: 'Rate of time decay of option value',
    formula: 'dV/dt',
    interpretation: 'Negative theta means option loses value as time passes'
});

CREATE (vega_concept:TradingConcept {
    concept_id: 'vega',
    name: 'Vega',
    category: 'greeks',
    description: 'Sensitivity of option price to volatility changes',
    formula: 'dV/dσ',
    interpretation: 'Higher vega means option is more sensitive to volatility changes'
});

CREATE (iv_concept:TradingConcept {
    concept_id: 'implied_volatility',
    name: 'Implied Volatility',
    category: 'volatility',
    description: 'Market-implied expected volatility derived from option prices',
    calculation: 'Solved from Black-Scholes model given market price',
    interpretation: 'Higher IV suggests higher expected price movement'
});

CREATE (straddle_concept:TradingConcept {
    concept_id: 'straddle',
    name: 'Straddle',
    category: 'structured_product',
    description: 'Options strategy buying both call and put at same strike',
    profit_condition: 'Profits when price moves significantly in either direction',
    max_loss: 'Total premium paid'
});

// ============================================================================
// RELATIONSHIPS: CONCEPT RELATIONSHIPS
// ============================================================================

// Greeks hierarchy
MATCH (d:TradingConcept {concept_id: 'delta'}), (g:TradingConcept {concept_id: 'gamma'})
CREATE (g)-[:DERIVATIVE_OF]->(d);

// Vega relates to IV
MATCH (v:TradingConcept {concept_id: 'vega'}), (iv:TradingConcept {concept_id: 'implied_volatility'})
CREATE (v)-[:MEASURES_SENSITIVITY_TO]->(iv);

// Straddle uses theta (time decay)
MATCH (s:TradingConcept {concept_id: 'straddle'}), (t:TradingConcept {concept_id: 'theta'})
CREATE (s)-[:AFFECTED_BY]->(t);

// Straddle benefits from high vega
MATCH (s:TradingConcept {concept_id: 'straddle'}), (v:TradingConcept {concept_id: 'vega'})
CREATE (s)-[:BENEFITS_FROM_HIGH]->(v);

// ============================================================================
// GRAPHRAG: DOCUMENT EMBEDDINGS (for RAG system)
// ============================================================================

CREATE (doc_options_101:Document {
    doc_id: 'options_basics_101',
    title: 'Introduction to Options Trading',
    category: 'education',
    content_type: 'markdown',
    embedding_model: 'text-embedding-3-small',
    chunk_count: 15,
    created_at: datetime()
});

CREATE (doc_greeks_guide:Document {
    doc_id: 'greeks_comprehensive_guide',
    title: 'Comprehensive Guide to Options Greeks',
    category: 'education',
    content_type: 'markdown',
    embedding_model: 'text-embedding-3-small',
    chunk_count: 25,
    created_at: datetime()
});

CREATE (doc_strategies:Document {
    doc_id: 'derivatives_strategies_handbook',
    title: 'Derivatives Trading Strategies Handbook',
    category: 'strategy',
    content_type: 'markdown',
    embedding_model: 'text-embedding-3-small',
    chunk_count: 40,
    created_at: datetime()
});

// Connect documents to concepts
MATCH (doc:Document {doc_id: 'greeks_comprehensive_guide'}), (c:TradingConcept {concept_id: 'delta'})
CREATE (doc)-[:EXPLAINS]->(c);

MATCH (doc:Document {doc_id: 'greeks_comprehensive_guide'}), (c:TradingConcept {concept_id: 'gamma'})
CREATE (doc)-[:EXPLAINS]->(c);

MATCH (doc:Document {doc_id: 'greeks_comprehensive_guide'}), (c:TradingConcept {concept_id: 'theta'})
CREATE (doc)-[:EXPLAINS]->(c);

MATCH (doc:Document {doc_id: 'greeks_comprehensive_guide'}), (c:TradingConcept {concept_id: 'vega'})
CREATE (doc)-[:EXPLAINS]->(c);

MATCH (doc:Document {doc_id: 'derivatives_strategies_handbook'}), (c:TradingConcept {concept_id: 'straddle'})
CREATE (doc)-[:EXPLAINS]->(c);

// ============================================================================
// UTILITY QUERIES FOR GRAPHRAG
// ============================================================================

// Query: Find all options for an underlying
// MATCH (o:OptionContract)-[:HAS_UNDERLYING]->(a:Asset {asset_id: 'XAUUSD'}) RETURN o

// Query: Find strategies suitable for current regime
// MATCH (st:StrategyTemplate)-[:SUITABLE_FOR_REGIME]->(r:MarketRegime {regime_id: 'low_vol_trending'}) RETURN st

// Query: Find correlated assets for risk analysis
// MATCH (a:Asset {asset_id: 'BTCUSD'})-[c:CORRELATED]-(related:Asset) RETURN related, c.strength

// Query: Find backtest results for a strategy
// MATCH (b:Backtest)-[:USES_STRATEGY]->(st:StrategyTemplate {template_id: 'iron_condor'}) RETURN b

// Query: RAG - Find documents explaining a concept
// MATCH (d:Document)-[:EXPLAINS]->(c:TradingConcept {concept_id: 'delta'}) RETURN d

# Requirements Document

## Introduction

This document specifies the requirements for a full-stack, research-grade algorithmic trading system designed for master's thesis research. The system prioritizes intelligence completeness and academic rigor while maintaining strict separation between research intelligence and capital deployment through enforced architectural boundaries.

## Glossary

- **Intelligence_Layer**: Python-based components handling market analysis, prediction, and learning
- **Execution_Core**: Rust-based components managing orders, risk, and portfolio accounting
- **Simulation_Engine**: Deterministic backtesting and replay system for academic evaluation
- **Strategy_Orchestrator**: Component that converts intelligence outputs to trade intents
- **Execution_Adapter**: Interface layer for connecting to trading platforms (Deriv, MT5)
- **Research_Framework**: Tools for reproducible experimentation and thesis evaluation
- **Sandboxing**: Architectural isolation preventing intelligence from directly placing orders

## Requirements

### Requirement 1: Core System Architecture

**User Story:** As a research architect, I want a modular system with clear separation of concerns, so that I can safely experiment with intelligence components without risking capital.

#### Acceptance Criteria

1. THE Execution_Core SHALL be implemented in Rust and retain final execution authority
2. THE Intelligence_Layer SHALL be implemented in Python and provide advisory outputs only
3. WHEN intelligence components generate outputs, THE System SHALL express them as forecast distributions, regime labels, confidence scores, or suggested actions
4. THE Intelligence_Layer SHALL NOT place orders directly
5. THE System SHALL implement an event bus for inter-component communication
6. THE System SHALL include a global kill switch for emergency shutdown

### Requirement 2: Intelligence and Learning Components

**User Story:** As a researcher, I want advanced intelligence capabilities including graph analytics and machine learning, so that I can explore sophisticated market representation and learning techniques.

#### Acceptance Criteria

1. THE Intelligence_Layer SHALL implement feature extraction pipelines for market data
2. THE Intelligence_Layer SHALL include time-series predictive models
3. THE Intelligence_Layer SHALL implement regime detection capabilities
4. THE System SHALL integrate Neo4j with Graph Data Science for market relationship analysis
5. THE System SHALL use pgvector for state embeddings and vector memory
6. THE Intelligence_Layer SHALL include reinforcement learning environments
7. WHEN processing market data, THE Intelligence_Layer SHALL maintain explainability and interpretability of learned behaviors
8. THE Neo4j_Schema SHALL maintain clear separation between structural market knowledge and continuous latent representations
9. THE Neo4j_Schema SHALL implement Asset, MarketRegime, MacroEvent, Strategy, and IntelligenceSignal node types
10. THE Neo4j_Schema SHALL define relationships including CORRELATED, TRANSITIONS_TO, PERFORMS_IN, SENSITIVE_TO, and AFFECTS
11. THE pgvector_Schema SHALL store market_state_embeddings, strategy_state_embeddings, and regime_trajectory_embeddings
12. THE Intelligence_Layer SHALL construct composite RL states combining embeddings, regime labels, transition probabilities, and performance context

### Requirement 3: Simulation and Backtesting

**User Story:** As a researcher, I want deterministic simulation capabilities, so that I can evaluate strategies reproducibly and conduct academic analysis.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL implement event-driven backtesting
2. THE Simulation_Engine SHALL include realistic slippage and latency models
3. THE Simulation_Engine SHALL support deterministic replay of historical scenarios
4. THE Simulation_Engine SHALL allow scenario injection including market crashes, gaps, and trading halts
5. THE Simulation_Engine SHALL serve as the primary evaluation environment for intelligence components
6. WHEN running simulations, THE System SHALL prevent data leakage and lookahead bias
7. THE System SHALL implement a deterministic clock abstraction for time management

### Requirement 4: Strategy Management and Orchestration

**User Story:** As a quantitative researcher, I want a strategy orchestration layer, so that I can manage multiple strategies and convert intelligence outputs to actionable trade intents.

#### Acceptance Criteria

1. THE Strategy_Orchestrator SHALL implement a strategy registry for managing multiple strategies
2. THE Strategy_Orchestrator SHALL support strategy selection and weighting mechanisms
3. THE Strategy_Orchestrator SHALL include a sandboxed reinforcement learning meta-controller
4. THE Strategy_Orchestrator SHALL convert intelligence outputs to trade intents only, not direct orders
5. THE Strategy_Orchestrator SHALL implement policy evaluation hooks for performance monitoring

### Requirement 5: Execution and Market Connectivity

**User Story:** As a system operator, I want safe execution adapters with multiple broker connections, so that I can execute trades while maintaining system isolation.

#### Acceptance Criteria

1. THE System SHALL implement a Deriv API adapter as the primary execution interface
2. THE System SHALL optionally support MT5 adapter as secondary execution interface
3. THE Execution_Adapter SHALL provide normalized execution semantics across different brokers
4. THE System SHALL support shadow execution mode for testing without real trades
5. WHEN executing live trades, THE System SHALL mirror all operations in simulation for validation
6. THE System SHALL support future extensibility to FIX gateway protocols

### Requirement 6: Risk Management and Portfolio Accounting

**User Story:** As a risk manager, I want comprehensive risk controls and portfolio tracking, so that I can maintain capital safety and regulatory compliance.

#### Acceptance Criteria

1. THE Execution_Core SHALL implement portfolio accounting with real-time position tracking
2. THE System SHALL enforce configurable risk guardrails including position limits and drawdown controls
3. THE System SHALL maintain order and fill schemas with complete audit trails
4. WHEN risk limits are breached, THE System SHALL automatically halt trading and alert operators
5. THE System SHALL support manual override capabilities for emergency situations

### Requirement 7: Research and Evaluation Framework

**User Story:** As a master's student, I want comprehensive evaluation tools and reproducible experiments, so that I can produce academically rigorous thesis results.

#### Acceptance Criteria

1. THE Research_Framework SHALL implement offline evaluation metrics for strategy performance
2. THE Research_Framework SHALL support regime-conditioned performance analysis
3. THE Research_Framework SHALL provide attribution analysis capabilities
4. THE Research_Framework SHALL enable ablation studies for component evaluation
5. THE System SHALL maintain reproducible experiment configurations with version control
6. THE System SHALL log all model versions and experimental parameters
7. WHEN conducting experiments, THE System SHALL include negative findings in results
8. THE Research_Framework SHALL support formal ablation testing including RL with vs without graph features, RL with vs without embeddings, static vs dynamic regimes, and graph-only vs embedding-only intelligence
9. THE System SHALL enable evaluation of Neo4j GDS algorithms including Louvain clustering and PageRank for market structure analysis
10. THE Research_Framework SHALL provide clear ontology documentation and explicit vs implicit knowledge separation for academic evaluation

### Requirement 8: User Interface and Visualization

**User Story:** As a researcher, I want comprehensive visualization and control interfaces, so that I can inspect intelligence outputs and monitor system behavior.

#### Acceptance Criteria

1. THE System SHALL implement a React-based frontend with shadcn/ui components
2. THE Frontend SHALL provide intelligence visualizations including model outputs and confidence levels
3. THE Frontend SHALL enable inspection of graph relationships and embedding quality
4. THE Frontend SHALL display risk and exposure dashboards with real-time updates
5. THE Frontend SHALL support manual override controls for system intervention
6. THE Frontend SHALL serve as a research instrument for data exploration and analysis

### Requirement 9: Data Management and Processing

**User Story:** As a data engineer, I want robust data pipelines and storage, so that I can ensure data quality and support research requirements.

#### Acceptance Criteria

1. THE System SHALL implement FastAPI for API orchestration and data services
2. THE System SHALL support multiple data sources with normalized schemas
3. THE System SHALL implement data validation and quality checks
4. THE System SHALL maintain historical data storage for backtesting and research
5. WHEN processing market data, THE System SHALL handle missing data and market holidays gracefully

### Requirement 10: System Integration and Deployment

**User Story:** As a system administrator, I want reliable deployment and monitoring capabilities, so that I can maintain system availability and performance.

#### Acceptance Criteria

1. THE System SHALL support containerized deployment with Docker
2. THE System SHALL implement health checks and monitoring endpoints
3. THE System SHALL provide logging and alerting capabilities
4. THE System SHALL support configuration management for different environments
5. THE System SHALL implement graceful shutdown procedures for all components

### Requirement 11: Intelligence Layer Schema Design

**User Story:** As a quantitative researcher, I want a rigorous separation between structural market knowledge and continuous latent representations, so that I can conduct meaningful ablation studies and maintain academic rigor.

#### Acceptance Criteria

1. THE Neo4j_Schema SHALL implement Asset nodes with asset_id, asset_class, venue, base_currency, and quote_currency properties
2. THE Neo4j_Schema SHALL implement MarketRegime nodes with regime_id, volatility_level, trend_state, liquidity_state, and description properties
3. THE Neo4j_Schema SHALL implement MacroEvent nodes with event_id, category, timestamp, and surprise_score properties
4. THE Neo4j_Schema SHALL implement Strategy nodes with strategy_id, family, horizon, and description properties
5. THE Neo4j_Schema SHALL implement IntelligenceSignal nodes with signal_id, type, confidence, and timestamp properties
6. THE Neo4j_Schema SHALL define CORRELATED relationships between Assets with window, strength, and sign properties
7. THE Neo4j_Schema SHALL define TRANSITIONS_TO relationships between MarketRegimes with probability and avg_duration properties
8. THE Neo4j_Schema SHALL define PERFORMS_IN relationships between Strategies and MarketRegimes with sharpe, max_dd, and sample_size properties
9. THE Neo4j_Schema SHALL define SENSITIVE_TO relationships between Assets and MarketRegimes with beta and lag properties
10. THE Neo4j_Schema SHALL define AFFECTS relationships between MacroEvents and Assets with impact_score properties
11. THE pgvector_Schema SHALL implement market_state_embeddings table with timestamp, asset_id, regime_id, embedding vector, volatility, liquidity, horizon, source_model, and metadata columns
12. THE pgvector_Schema SHALL implement strategy_state_embeddings table with timestamp, strategy_id, embedding vector, pnl_state, drawdown, exposure, and metadata columns
13. THE pgvector_Schema SHALL implement regime_trajectory_embeddings table with start_time, end_time, embedding vector, realized_vol, and transition_path columns
14. THE pgvector_Schema SHALL implement appropriate vector indexing using ivfflat with cosine similarity operations
15. THE Intelligence_Layer SHALL construct composite RL states combining current embeddings from pgvector, regime labels from Neo4j, regime transition probabilities from Neo4j, strategy performance context from Neo4j, and risk metrics from Rust components

### Requirement 12: Reinforcement Learning Environment

**User Story:** As a quantitative researcher, I want a formally defined RL environment for meta-policy control, so that I can conduct rigorous academic evaluation of strategy allocation and risk management.

#### Acceptance Criteria

1. THE RL_Environment SHALL implement an episodic, partially observable MDP with strategy decision intervals
2. THE RL_Environment SHALL construct composite state vectors combining latent market embeddings, discrete regime representations, graph structural features, portfolio state, and confidence metrics
3. THE RL_Environment SHALL define continuous bounded action spaces for strategy weights, exposure multipliers, and execution aggressiveness
4. THE RL_Environment SHALL implement risk-adjusted reward functions incorporating Sharpe ratio, drawdown penalties, turnover costs, and regime violation penalties
5. THE RL_Environment SHALL terminate episodes based on drawdown breaches, regime shifts, or data exhaustion
6. THE RL_Environment SHALL serve as a meta-policy controller for capital allocation across strategies, NOT as a direct trading agent
7. WHEN constructing RL states, THE System SHALL include 128-dimensional market embeddings from pgvector
8. WHEN constructing RL states, THE System SHALL include regime IDs, transition probabilities, and regime entropy from Neo4j
9. WHEN constructing RL states, THE System SHALL include asset cluster IDs, centrality scores, and systemic risk proxies from Neo4j GDS
10. WHEN constructing RL states, THE System SHALL include net exposure, gross exposure, drawdown, and volatility utilization from Rust core

### Requirement 13: Embedding Model Training Protocol

**User Story:** As a machine learning researcher, I want a rigorous embedding model training protocol, so that I can generate stable, interpretable market representations for academic evaluation.

#### Acceptance Criteria

1. THE Embedding_Model SHALL implement Temporal Convolutional Networks as the primary architecture
2. THE Embedding_Model SHALL optionally support Variational Autoencoders for thesis comparison studies
3. THE Embedding_Model SHALL process input tensors with rolling windows of 64 time steps and engineered features
4. THE Embedding_Model SHALL use self-supervised representation learning without PnL leakage
5. THE Embedding_Model SHALL implement reconstruction loss, contrastive similarity loss, and temporal smoothness regularization
6. THE Training_Protocol SHALL use offline training only on pre-cut historical segments
7. THE Training_Protocol SHALL freeze model weights before simulation to prevent lookahead bias
8. THE Training_Protocol SHALL version and hash all models for reproducibility
9. THE Embedding_Model SHALL validate outputs for temporal continuity, regime separability, and similarity interpretability
10. WHEN embeddings fail validation criteria, THE System SHALL prevent progression to simulation or live trading

### Requirement 14: Neo4j Graph Data Science Integration

**User Story:** As a network analyst, I want formal Neo4j GDS job definitions, so that I can extract structural market intelligence for academic research.

#### Acceptance Criteria

1. THE Neo4j_GDS SHALL implement asset correlation graph projections with CORRELATED relationships and strength properties
2. THE Neo4j_GDS SHALL implement regime transition graph projections with TRANSITIONS_TO relationships and probability properties
3. THE Neo4j_GDS SHALL implement strategy-regime performance graph projections with PERFORMS_IN relationships
4. THE Neo4j_GDS SHALL execute Louvain clustering algorithms for market clustering analysis
5. THE Neo4j_GDS SHALL calculate degree centrality for systemic exposure measurement
6. THE Neo4j_GDS SHALL calculate betweenness centrality for contagion risk assessment
7. THE Neo4j_GDS SHALL execute PageRank algorithms on regime graphs for dominant regime identification
8. THE Neo4j_GDS SHALL perform path enumeration for crisis trajectory analysis
9. THE Neo4j_GDS SHALL calculate entropy measures for regime stability assessment
10. THE Neo4j_GDS SHALL materialize all algorithm outputs as node properties and snapshot exports to Parquet
11. WHEN executing GDS algorithms, THE System SHALL never query results synchronously during trading operations

### Requirement 15: FastAPI Intelligence Service Layer

**User Story:** As a system integrator, I want well-defined intelligence API endpoints, so that I can maintain clean separation between intelligence and execution layers.

#### Acceptance Criteria

1. THE FastAPI_Service SHALL implement stateless intelligence endpoints with deterministic responses
2. THE FastAPI_Service SHALL provide embedding inference endpoints accepting market window features and returning embedding IDs, similarity context, and confidence scores
3. THE FastAPI_Service SHALL provide regime inference endpoints returning regime probabilities, transition likelihoods, and regime entropy
4. THE FastAPI_Service SHALL provide graph features endpoints returning cluster membership, centrality metrics, and systemic risk proxies
5. THE FastAPI_Service SHALL provide RL state assembly endpoints returning complete composite states for strategy orchestration
6. THE FastAPI_Service SHALL implement strict version headers for all API responses
7. THE FastAPI_Service SHALL maintain full request logging for audit and reproducibility
8. THE FastAPI_Service SHALL serve as the intelligence boundary, not the control plane
9. WHEN processing intelligence requests, THE FastAPI_Service SHALL maintain complete request-response audit trails
10. WHEN serving intelligence data, THE FastAPI_Service SHALL ensure no direct trading capabilities are exposed
### Requirement 16: Admin Application Frontend Architecture

**User Story:** As a system administrator and researcher, I want a comprehensive admin interface following nLVE (List-View-Edit) patterns, so that I can efficiently manage, monitor, and configure the intelligence-centric trading platform.

#### Acceptance Criteria

1. THE Admin_Application SHALL be implemented using React with TypeScript and shadcn/ui component library
2. THE Admin_Application SHALL implement a sidebar-driven navigation structure with domain-based organization
3. THE Admin_Application SHALL follow nLVE (List-View-Edit) framework with clear separation between observation and modification
4. THE Admin_Application SHALL implement nine primary navigation domains: Dashboard, Markets, Intelligence, Strategies, Portfolio & Risk, Execution, Simulation & Experiments, Data & Models, and System
4. THE Admin_Application SHALL enforce "read-first, edit-second" principles with gated edit access
5. THE Admin_Application SHALL maintain "one mental model per screen" without mixing strategy logic, intelligence state, and execution outcomes
6. THE Admin_Application SHALL implement context flow from left to right (Sidebar → List → View → Edit)
7. THE Admin_Application SHALL ensure every edit has complete provenance with attribution and reversibility
8. THE Admin_Application SHALL implement staged edits with diff views, validation, and impact explanation before persistence
9. THE Admin_Application SHALL generate audit events for every modification operation
10. THE Admin_Application SHALL use shadcn/ui components for consistent design system and accessibility compliance
11. WHEN users navigate between domains, THE System SHALL maintain clear context boundaries and prevent cognitive overload

### Requirement 17: Intelligence Domain Interface

**User Story:** As a quantitative researcher, I want specialized intelligence management interfaces, so that I can monitor and configure market states, regimes, and intelligence signals effectively.

#### Acceptance Criteria

1. THE Intelligence_Interface SHALL provide list views for Market States, Regimes, Intelligence Signals, and Graph Snapshots
2. THE Intelligence_Interface SHALL display regime definitions, transition probabilities, duration statistics, affected assets, and strategy performance in view panels
3. THE Intelligence_Interface SHALL provide edit tabs for regime definitions, transition rules, graph context, validation settings, and audit history
4. THE Intelligence_Interface SHALL display embedding metadata, similarity neighbors, regime probabilities, and confidence scores for market states
5. THE Intelligence_Interface SHALL implement read-only market state interfaces with embedding info, similarity analysis, usage tracking, and audit metadata
6. WHEN viewing intelligence artifacts, THE System SHALL provide complete lineage and provenance information
7. WHEN editing intelligence configurations, THE System SHALL validate changes against academic and safety constraints

### Requirement 18: Strategy and Risk Management Interfaces

**User Story:** As a portfolio manager, I want comprehensive strategy and risk management interfaces, so that I can configure, monitor, and control trading strategies and risk policies.

#### Acceptance Criteria

1. THE Strategy_Interface SHALL provide list views for Strategy Catalog and Strategy Groups with filtering and search capabilities
2. THE Strategy_Interface SHALL display strategy descriptions, family classifications, enabled markets, historical performance, and regime affinity in view panels
3. THE Strategy_Interface SHALL provide edit tabs for strategy definitions, parameters, regime constraints, risk budgets, backtests, and audit trails
4. THE Risk_Interface SHALL provide list views for Portfolios and Risk Policies with real-time status indicators
5. THE Risk_Interface SHALL display exposure limits, drawdown rules, and kill switch conditions in view panels
6. THE Risk_Interface SHALL provide edit tabs for limits configuration, throttles, RL integration, scenarios, and audit history
7. WHEN configuring strategies or risk policies, THE System SHALL validate parameters against system constraints and regime compatibility

### Requirement 19: Execution and System Management Interfaces

**User Story:** As a system operator, I want execution monitoring and system management interfaces, so that I can oversee market connectivity, execution performance, and system health.

#### Acceptance Criteria

1. THE Execution_Interface SHALL provide list views for Execution Adapters, Venues, and Order Streams with real-time status
2. THE Execution_Interface SHALL display connectivity status, latency metrics, and rejection rates in view panels
3. THE Execution_Interface SHALL provide edit tabs for configuration, order mapping, safeguards, shadow mode, and audit trails
4. THE System_Interface SHALL provide list views for Users, Roles, Feature Flags, and Logs with search and filtering
5. THE System_Interface SHALL display user roles, permissions, and activity logs in view panels
6. THE System_Interface SHALL provide edit tabs for access control, restrictions, sessions, and security audit
7. WHEN managing execution or system configurations, THE System SHALL enforce security policies and change approval workflows

### Requirement 20: Simulation and Data Management Interfaces

**User Story:** As a research scientist, I want simulation and data management interfaces, so that I can conduct experiments, manage datasets, and track model performance.

#### Acceptance Criteria

1. THE Simulation_Interface SHALL provide list views for Experiments, Backtests, and Scenarios with status and progress indicators
2. THE Simulation_Interface SHALL display experiment objectives, data ranges, models used, and summary metrics in view panels
3. THE Simulation_Interface SHALL provide edit tabs for setup configuration, intelligence models, strategies, metrics, results, and audit logs
4. THE Data_Interface SHALL provide list views for Datasets, Feature Sets, and Models with version tracking
5. THE Data_Interface SHALL display model architecture, training data, validation metrics, and deployment status in view panels
6. THE Data_Interface SHALL provide edit tabs for configuration, training, versioning, deployment, and audit history
7. WHEN managing experiments or data artifacts, THE System SHALL maintain complete reproducibility and lineage tracking
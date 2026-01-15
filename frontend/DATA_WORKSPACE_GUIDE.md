# Data Workspace Guide (F10)

## Overview

The Data Workspace is a comprehensive analytics and visualization environment for exploring market data, performing advanced analysis, and visualizing relationships using both relational and graph databases.

## Access

Press **F10** or click **WORKSPACE** in the function key bar.

## Features

### 1. Data Source Selection

Connect to multiple data sources:

- **PostgreSQL**: Time-series market data, embeddings, positions, orders
- **Neo4j**: Graph relationships, asset correlations, regime transitions
- **Redis**: Real-time cache, live data streams
- **Live**: Direct streaming from market data feeds

### 2. External Data Import

Import data from external sources directly into the workspace:

#### Supported Sources
- **Yahoo Finance**: Stocks, forex, crypto, indices, commodities
- **Alpha Vantage**: Real-time and historical market data (future)
- **Quandl**: Economic and financial data (future)
- **FRED**: Federal Reserve economic data (future)
- **CryptoCompare**: Cryptocurrency data (future)

#### Import Workflow

1. **Click "SHOW IMPORT"** in the left panel
2. **Select Source**: Choose data provider (e.g., Yahoo Finance)
3. **Search Symbol**: 
   - Enter search query (e.g., "AAPL", "EUR", "BTC")
   - Press Enter or click Search button
   - Browse results with symbol, name, and type
4. **Select Symbol**: Click on desired symbol from results
5. **Set Date Range**: 
   - Start Date (optional, defaults to 30 days ago)
   - End Date (optional, defaults to today)
6. **Choose Interval**:
   - 1 Minute (1m)
   - 5 Minutes (5m)
   - 15 Minutes (15m)
   - 1 Hour (1h)
   - 1 Day (1d) - default
   - 1 Week (1wk)
   - 1 Month (1mo)
7. **Click "IMPORT DATA"**: Fetch and load data
8. **View Status**: Success/error message displayed
9. **Analyze**: Data ready for visualization and analysis

#### Symbol Search Examples

**Stocks:**
- Search: "Apple" → Results: AAPL (Apple Inc.)
- Search: "MSFT" → Results: MSFT (Microsoft Corporation)
- Search: "TSLA" → Results: TSLA (Tesla, Inc.)

**Forex:**
- Search: "EUR" → Results: EURUSD=X (EUR/USD)
- Search: "GBP" → Results: GBPUSD=X (GBP/USD)
- Search: "JPY" → Results: USDJPY=X (USD/JPY)

**Crypto:**
- Search: "BTC" → Results: BTC-USD (Bitcoin USD)
- Search: "ETH" → Results: ETH-USD (Ethereum USD)
- Search: "XRP" → Results: XRP-USD (Ripple USD)

**Indices:**
- Search: "^GSPC" → Results: S&P 500
- Search: "^DJI" → Results: Dow Jones Industrial Average
- Search: "^IXIC" → Results: NASDAQ Composite

#### Data Import Features

- **Real-time Search**: Instant symbol lookup
- **Symbol Metadata**: View name, type, exchange before importing
- **Flexible Date Ranges**: Custom or preset ranges
- **Multiple Intervals**: From 1-minute to monthly data
- **Progress Indication**: Loading states during import
- **Error Handling**: Clear error messages
- **Data Validation**: Automatic validation of imported data
- **Immediate Analysis**: Data ready for visualization instantly

### 3. Query Builder

#### Relational Queries (PostgreSQL)
```sql
SELECT * FROM market_data
WHERE asset_id = 'EURUSD'
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC
```

#### Graph Queries (Neo4j Cypher)
```cypher
MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset)
WHERE r.correlation > 0.7
RETURN a, r, b
```

#### Time Range Selection
- Last 1 Hour
- Last 24 Hours
- Last 7 Days
- Last 30 Days
- Custom Range

#### Asset Filtering
Multi-select assets for analysis:
- EURUSD
- GBPUSD
- USDJPY
- AUDUSD
- USDCHF
- And more...

### 3. Visualization Types

#### Time Series
- **Use Case**: Price movements, volume trends, indicator values
- **Features**: 
  - Real-time streaming
  - Multiple series overlay
  - Zoom and pan
  - Annotations
- **Analysis**: Trend detection, seasonality, autocorrelation

#### Scatter Plot
- **Use Case**: Correlation analysis, regime clustering
- **Features**:
  - Color by category
  - Size by value
  - Interactive tooltips
- **Analysis**: Relationship strength, outlier detection

#### Histogram
- **Use Case**: Distribution analysis, return distributions
- **Features**:
  - Adjustable bins
  - Overlay normal distribution
  - Cumulative distribution
- **Analysis**: Normality tests, skewness, kurtosis

#### Heatmap
- **Use Case**: Correlation matrices, time-of-day patterns
- **Features**:
  - Color gradients
  - Interactive cells
  - Hierarchical clustering
- **Analysis**: Correlation strength, pattern recognition

#### Graph Visualization
- **Use Case**: Asset relationships, regime transitions, strategy dependencies
- **Features**:
  - Force-directed layout
  - Community detection coloring
  - Node sizing by centrality
  - Edge thickness by weight
- **Analysis**: Network structure, clustering, centrality

#### Correlation Matrix
- **Use Case**: Asset correlations, factor analysis
- **Features**:
  - Rolling window correlations
  - Hierarchical clustering
  - Dendrogram view
- **Analysis**: Diversification, risk factors

#### Distribution Plot
- **Use Case**: Return distributions, risk analysis
- **Features**:
  - Kernel density estimation
  - Q-Q plots
  - Box plots
- **Analysis**: Fat tails, normality, outliers

#### Candlestick Chart
- **Use Case**: OHLC price data, technical analysis
- **Features**:
  - Volume bars
  - Technical indicators overlay
  - Pattern recognition
- **Analysis**: Support/resistance, trends, patterns

### 4. Analysis Types

#### Descriptive Statistics
Compute summary statistics:
- **Mean**: Average value
- **Median**: Middle value
- **Std Dev**: Volatility measure
- **Min/Max**: Range
- **Skewness**: Asymmetry
- **Kurtosis**: Tail heaviness
- **Observations**: Sample size

#### Correlation Analysis
Measure relationships:
- **Pearson**: Linear correlation
- **Spearman**: Rank correlation
- **Kendall**: Ordinal correlation
- **Rolling**: Time-varying correlation
- **Partial**: Controlling for other variables

#### Regression Analysis
Model relationships:
- **Linear**: Y = aX + b
- **Polynomial**: Higher-order relationships
- **Ridge/Lasso**: Regularized regression
- **Logistic**: Binary outcomes
- **Quantile**: Conditional quantiles

#### Time Series Analysis
Analyze temporal patterns:
- **Stationarity**: ADF test, KPSS test
- **Autocorrelation**: ACF, PACF
- **Seasonality**: Decomposition
- **Trend**: Linear, polynomial
- **ARIMA**: Forecasting models
- **GARCH**: Volatility modeling

#### Fourier Transform
Frequency domain analysis:
- **FFT**: Fast Fourier Transform
- **Power Spectrum**: Frequency components
- **Dominant Frequencies**: Peak detection
- **Harmonics**: Multiple frequencies
- **Spectral Density**: Energy distribution

#### Wavelet Analysis
Time-frequency analysis:
- **CWT**: Continuous Wavelet Transform
- **DWT**: Discrete Wavelet Transform
- **Scalogram**: Time-frequency representation
- **Denoising**: Signal cleaning
- **Feature Extraction**: Multi-scale features

#### Principal Component Analysis (PCA)
Dimensionality reduction:
- **Variance Explained**: Component importance
- **Loadings**: Variable contributions
- **Scores**: Transformed data
- **Scree Plot**: Component selection
- **Biplot**: Variables and observations

#### Clustering
Group similar observations:
- **K-Means**: Centroid-based
- **Hierarchical**: Dendrogram
- **DBSCAN**: Density-based
- **Gaussian Mixture**: Probabilistic
- **Spectral**: Graph-based

### 5. Graph Analytics (Neo4j GDS)

#### Centrality Algorithms
Identify important nodes:
- **PageRank**: Link-based importance
- **Betweenness**: Bridge nodes
- **Closeness**: Average distance
- **Degree**: Connection count
- **Eigenvector**: Influential connections

#### Community Detection
Find clusters:
- **Louvain**: Modularity optimization
- **Label Propagation**: Fast clustering
- **Weakly Connected**: Component detection
- **Strongly Connected**: Directed components

#### Similarity Algorithms
Find similar nodes:
- **Node Similarity**: Jaccard, Overlap
- **K-Nearest Neighbors**: Top-K similar
- **Cosine Similarity**: Vector similarity

#### Path Finding
Analyze connections:
- **Shortest Path**: Minimum distance
- **All Shortest Paths**: Multiple routes
- **Single Source**: From one node
- **A* Search**: Heuristic search

#### Link Prediction
Predict future connections:
- **Adamic Adar**: Common neighbors
- **Resource Allocation**: Network flow
- **Preferential Attachment**: Rich get richer

### 6. Real-Time Streaming

Enable live data updates:
- **Streaming Mode**: Continuous data flow
- **Update Frequency**: Configurable (1s, 5s, 10s, 30s)
- **Buffer Size**: Historical window
- **Auto-scroll**: Follow latest data
- **Pause/Resume**: Control streaming

### 7. Export Options

Export data and visualizations:

#### Data Formats
- **CSV**: Comma-separated values
- **JSON**: JavaScript Object Notation
- **Parquet**: Columnar format (efficient)
- **Excel**: XLSX format
- **HDF5**: Hierarchical data format

#### Graph Formats
- **GraphML**: XML-based graph format
- **GEXF**: Gephi exchange format
- **GML**: Graph Modeling Language
- **JSON**: Node-link format
- **Cypher**: Neo4j query export

#### Visualization Formats
- **PNG**: Raster image
- **SVG**: Vector graphics
- **PDF**: Portable document
- **HTML**: Interactive web page

## Use Cases

### 1. Market Microstructure Analysis
```
Data Source: PostgreSQL (market_data)
Visualization: Time Series
Analysis: Fourier Transform
Goal: Identify intraday patterns
```

### 2. Asset Correlation Network
```
Data Source: Neo4j (Asset relationships)
Visualization: Graph
Analysis: Community Detection (Louvain)
Goal: Find asset clusters for diversification
```

### 3. Return Distribution Analysis
```
Data Source: PostgreSQL (positions)
Visualization: Histogram + Distribution
Analysis: Descriptive Statistics
Goal: Assess risk and tail events
```

### 4. Regime Transition Patterns
```
Data Source: Neo4j (Regime transitions)
Visualization: Graph
Analysis: PageRank + Path Finding
Goal: Understand regime dynamics
```

### 5. Volatility Clustering
```
Data Source: PostgreSQL (market_data)
Visualization: Time Series + Heatmap
Analysis: GARCH + Clustering
Goal: Identify volatility regimes
```

### 6. Strategy Performance Attribution
```
Data Source: PostgreSQL (strategy_performance)
Visualization: Scatter + Correlation Matrix
Analysis: PCA + Regression
Goal: Understand performance drivers
```

## Workflow Examples

### Example 0: Import External Data (NEW)

1. **Open Data Workspace**: Press F10
2. **Show Import**: Click "SHOW IMPORT" button
3. **Select Source**: Yahoo Finance (default)
4. **Search Symbol**: 
   - Type "AAPL" in search box
   - Press Enter
5. **Select Result**: Click on "AAPL" from results
6. **Set Date Range**:
   - Start: 2024-01-01
   - End: 2024-12-31
7. **Choose Interval**: 1 Day
8. **Import**: Click "IMPORT DATA"
9. **Wait**: Progress indicator shows import status
10. **Success**: "Successfully imported X data points for AAPL"
11. **Visualize**: Data automatically available for analysis

### Example 1: Correlation Analysis

1. **Select Data Source**: PostgreSQL
2. **Query Builder**: 
   - Table: market_data
   - Time Range: Last 30 Days
   - Assets: EURUSD, GBPUSD, USDJPY, AUDUSD
3. **Visualization**: Correlation Matrix
4. **Analysis**: Correlation Analysis (Pearson)
5. **Run Analysis**: Click "RUN ANALYSIS"
6. **Results**: View correlation heatmap and statistics
7. **Export**: Download as CSV or PNG

### Example 2: Graph Community Detection

1. **Select Data Source**: Neo4j
2. **Graph Query**:
   ```cypher
   MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset)
   WHERE r.correlation > 0.5
   RETURN a, r, b
   ```
3. **Visualization**: Graph
4. **GDS Algorithm**: Louvain Community
5. **Run Analysis**: Execute query and algorithm
6. **Results**: View colored clusters
7. **Export**: Download as GraphML for Gephi

### Example 3: Time Series Forecasting

1. **Select Data Source**: PostgreSQL
2. **Query Builder**:
   - Table: market_data
   - Time Range: Last 7 Days
   - Assets: EURUSD
3. **Visualization**: Time Series
4. **Analysis**: Time Series Analysis (ARIMA)
5. **Run Analysis**: Fit model
6. **Results**: View forecast with confidence intervals
7. **Export**: Download predictions as CSV

## Keyboard Shortcuts

- **F10**: Open Data Workspace
- **Ctrl+R**: Run analysis
- **Ctrl+E**: Export data
- **Ctrl+S**: Save workspace configuration
- **Ctrl+L**: Toggle live streaming
- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo action

## Tips & Best Practices

### Performance
- **Limit Time Range**: Use shorter ranges for faster queries
- **Index Columns**: Ensure timestamp and asset_id are indexed
- **Batch Processing**: Process large datasets in chunks
- **Cache Results**: Save intermediate results

### Visualization
- **Choose Appropriate Type**: Match viz to data structure
- **Color Coding**: Use consistent color schemes
- **Labels**: Add clear axis labels and titles
- **Legends**: Include legends for multi-series plots

### Analysis
- **Check Assumptions**: Verify statistical assumptions
- **Multiple Methods**: Use multiple analysis types
- **Validate Results**: Cross-check with domain knowledge
- **Document Findings**: Export and save results

### Graph Analytics
- **Project Graphs**: Use GDS projections for performance
- **Filter Edges**: Remove weak connections
- **Normalize Metrics**: Scale centrality measures
- **Visualize Subgraphs**: Focus on relevant portions

## Integration with Other Domains

### Intelligence (F3)
- Import regime labels for clustering
- Visualize embedding spaces
- Analyze regime transitions

### Strategies (F4)
- Analyze strategy performance
- Correlate strategy returns
- Identify strategy clusters

### Portfolio (F5)
- Visualize position distributions
- Analyze risk metrics
- Correlation with market factors

### Simulation (F7)
- Compare backtest results
- Analyze scenario outcomes
- Visualize parameter sensitivity

## Future Enhancements

### Short-term
- [ ] Interactive chart controls (zoom, pan, select)
- [ ] Custom color schemes
- [ ] Save/load workspace configurations
- [ ] Annotation tools
- [ ] Chart templates

### Medium-term
- [ ] Machine learning model training
- [ ] Automated pattern recognition
- [ ] Alert creation from analysis
- [ ] Collaborative workspaces
- [ ] Version control for queries

### Long-term
- [ ] Natural language queries
- [ ] AI-powered insights
- [ ] Automated report generation
- [ ] Integration with Jupyter notebooks
- [ ] Custom plugin system

## Technical Details

### Data Flow
```
User Query → Query Builder → Data Source
    ↓
Raw Data → Transformation → Analysis Engine
    ↓
Results → Visualization Engine → Canvas
    ↓
Export → File System / Clipboard
```

### Performance Characteristics
- **Query Time**: 50-500ms (depending on data size)
- **Visualization Render**: 100-300ms
- **Analysis Computation**: 200ms-5s (depending on complexity)
- **Real-time Update**: 1-30s intervals

### Browser Requirements
- **Modern Browser**: Chrome 90+, Firefox 88+, Safari 14+
- **WebGL**: Required for advanced visualizations
- **Memory**: 4GB+ recommended for large datasets
- **Screen**: 1920x1080+ recommended

## Support

For issues or questions:
1. Check this guide
2. Review example workflows
3. Check browser console for errors
4. Contact system administrator

## Conclusion

The Data Workspace provides a powerful environment for exploring and analyzing trading data. Use it to gain insights, validate hypotheses, and make data-driven decisions.

Happy analyzing! 📊

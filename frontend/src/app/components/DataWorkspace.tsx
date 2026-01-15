import { useState } from 'react';
import { 
  LineChart, 
  BarChart3, 
  ScatterChart, 
  Network, 
  TrendingUp,
  Database,
  Filter,
  Download,
  Play,
  Settings,
  Maximize2,
  Search,
  Plus,
  Calendar,
} from 'lucide-react';
import { searchSymbols, getSymbolInfo, importExternalData, type SymbolSearchResult } from '../../services/dataImportService';

type VisualizationType = 
  | 'timeseries' 
  | 'scatter' 
  | 'histogram' 
  | 'heatmap' 
  | 'graph' 
  | 'correlation'
  | 'distribution'
  | 'candlestick';

type AnalysisType = 
  | 'descriptive' 
  | 'correlation' 
  | 'regression' 
  | 'timeseries' 
  | 'fourier'
  | 'wavelet'
  | 'pca'
  | 'clustering';

type DataSource = 'postgresql' | 'neo4j' | 'redis' | 'live';

export function DataWorkspace() {
  const [selectedViz, setSelectedViz] = useState<VisualizationType>('timeseries');
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisType>('descriptive');
  const [selectedSource, setSelectedSource] = useState<DataSource>('postgresql');
  const [isRunning, setIsRunning] = useState(false);
  
  // Data import state
  const [showImport, setShowImport] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SymbolSearchResult[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [importSource, setImportSource] = useState('yahoo_finance');
  const [importInterval, setImportInterval] = useState('1d');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [importStatus, setImportStatus] = useState<string>('');

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      const results = await searchSymbols(searchQuery, importSource, 10);
      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      setImportStatus('Search failed: ' + (error as Error).message);
    }
  };

  const handleImport = async () => {
    if (!selectedSymbol) {
      setImportStatus('Please select a symbol');
      return;
    }

    setIsImporting(true);
    setImportStatus('Importing data...');

    try {
      const result = await importExternalData(
        selectedSymbol,
        importSource,
        startDate || undefined,
        endDate || undefined,
        importInterval
      );

      setImportStatus(
        `Successfully imported ${result.data_points} data points for ${result.symbol}`
      );
      
      // TODO: Update visualization with imported data
      console.log('Imported data:', result);
      
    } catch (error) {
      console.error('Import failed:', error);
      setImportStatus('Import failed: ' + (error as Error).message);
    } finally {
      setIsImporting(false);
    }
  };

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  // Mock data for visualizations
  const timeSeriesData = Array.from({ length: 100 }, (_, i) => ({
    time: i,
    value: 1.1 + Math.sin(i / 10) * 0.01 + Math.random() * 0.005,
  }));

  const correlationMatrix = [
    { asset1: 'EURUSD', asset2: 'GBPUSD', corr: 0.87 },
    { asset1: 'EURUSD', asset2: 'USDJPY', corr: -0.45 },
    { asset1: 'EURUSD', asset2: 'AUDUSD', corr: 0.72 },
    { asset1: 'GBPUSD', asset2: 'USDJPY', corr: -0.38 },
    { asset1: 'GBPUSD', asset2: 'AUDUSD', corr: 0.65 },
    { asset1: 'USDJPY', asset2: 'AUDUSD', corr: -0.52 },
  ];

  const graphNodes = [
    { id: 'EURUSD', cluster: 'A', centrality: 0.85 },
    { id: 'GBPUSD', cluster: 'A', centrality: 0.72 },
    { id: 'USDJPY', cluster: 'B', centrality: 0.45 },
    { id: 'AUDUSD', cluster: 'A', centrality: 0.68 },
    { id: 'USDCHF', cluster: 'B', centrality: 0.52 },
  ];

  const statisticalSummary = {
    mean: 1.1045,
    median: 1.1042,
    std: 0.0087,
    min: 1.0892,
    max: 1.1198,
    skewness: 0.12,
    kurtosis: 2.98,
    observations: 10000,
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs bg-[#0a0a0a]">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 px-4">
        <div className="text-[#ff8c00] text-sm tracking-wide">
          DATA WORKSPACE - ADVANCED ANALYTICS & VISUALIZATION
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Data Source & Configuration */}
        <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
          <div className="p-4 space-y-6">
            {/* Data Source Selection */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <Database className="w-3 h-3" />
                DATA SOURCE
              </div>
              <div className="space-y-2">
                {(['postgresql', 'neo4j', 'redis', 'live'] as DataSource[]).map((source) => (
                  <button
                    key={source}
                    onClick={() => setSelectedSource(source)}
                    className={`
                      w-full py-2 px-3 border text-left text-[10px] transition-colors
                      ${selectedSource === source
                        ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                        : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                      }
                    `}
                  >
                    {source.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* External Data Import */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <Plus className="w-3 h-3" />
                IMPORT EXTERNAL DATA
              </div>
              <button
                onClick={() => setShowImport(!showImport)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] transition-colors"
              >
                {showImport ? 'HIDE IMPORT' : 'SHOW IMPORT'}
              </button>
              
              {showImport && (
                <div className="mt-3 space-y-3 border border-[#444] p-3 bg-[#000]">
                  {/* Source Selection */}
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">SOURCE</div>
                    <select
                      value={importSource}
                      onChange={(e) => setImportSource(e.target.value)}
                      className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]"
                    >
                      <option value="yahoo_finance">Yahoo Finance</option>
                      <option value="alpha_vantage">Alpha Vantage</option>
                      <option value="quandl">Quandl</option>
                    </select>
                  </div>

                  {/* Symbol Search */}
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">SEARCH SYMBOL</div>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="AAPL, EUR, BTC..."
                        className="flex-1 bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]"
                      />
                      <button
                        onClick={handleSearch}
                        className="px-3 py-1 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
                      >
                        <Search className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  {/* Search Results */}
                  {searchResults.length > 0 && (
                    <div>
                      <div className="text-[#666] text-[10px] mb-1">RESULTS</div>
                      <div className="max-h-32 overflow-y-auto space-y-1">
                        {searchResults.map((result, idx) => (
                          <button
                            key={idx}
                            onClick={() => setSelectedSymbol(result.symbol)}
                            className={`
                              w-full text-left px-2 py-1 text-[10px] border transition-colors
                              ${selectedSymbol === result.symbol
                                ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                                : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                              }
                            `}
                          >
                            <div>{result.symbol}</div>
                            <div className="text-[#666] text-[9px]">
                              {result.name} | {result.type}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Date Range */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <div className="text-[#666] text-[10px] mb-1">START DATE</div>
                      <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]"
                      />
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px] mb-1">END DATE</div>
                      <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]"
                      />
                    </div>
                  </div>

                  {/* Interval */}
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">INTERVAL</div>
                    <select
                      value={importInterval}
                      onChange={(e) => setImportInterval(e.target.value)}
                      className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]"
                    >
                      <option value="1m">1 Minute</option>
                      <option value="5m">5 Minutes</option>
                      <option value="15m">15 Minutes</option>
                      <option value="1h">1 Hour</option>
                      <option value="1d">1 Day</option>
                      <option value="1wk">1 Week</option>
                      <option value="1mo">1 Month</option>
                    </select>
                  </div>

                  {/* Import Button */}
                  <button
                    onClick={handleImport}
                    disabled={isImporting || !selectedSymbol}
                    className="w-full py-2 px-3 border border-[#00ff00] bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isImporting ? 'IMPORTING...' : 'IMPORT DATA'}
                  </button>

                  {/* Status */}
                  {importStatus && (
                    <div className={`text-[10px] p-2 border ${
                      importStatus.includes('Success') 
                        ? 'border-[#00ff00] text-[#00ff00]' 
                        : importStatus.includes('failed')
                        ? 'border-[#ff0000] text-[#ff0000]'
                        : 'border-[#ffff00] text-[#ffff00]'
                    }`}>
                      {importStatus}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Query Builder */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <Filter className="w-3 h-3" />
                QUERY BUILDER
              </div>
              <div className="border border-[#444] bg-[#000] p-3 space-y-2">
                <div>
                  <div className="text-[#666] text-[10px] mb-1">TABLE/COLLECTION</div>
                  <select className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]">
                    <option>market_data</option>
                    <option>embeddings</option>
                    <option>regimes</option>
                    <option>positions</option>
                    <option>orders</option>
                  </select>
                </div>
                <div>
                  <div className="text-[#666] text-[10px] mb-1">TIME RANGE</div>
                  <select className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]">
                    <option>Last 1 Hour</option>
                    <option>Last 24 Hours</option>
                    <option>Last 7 Days</option>
                    <option>Last 30 Days</option>
                    <option>Custom Range</option>
                  </select>
                </div>
                <div>
                  <div className="text-[#666] text-[10px] mb-1">ASSETS</div>
                  <select className="w-full bg-[#0a0a0a] border border-[#444] text-[#00ff00] px-2 py-1 text-[10px]" multiple size={3}>
                    <option>EURUSD</option>
                    <option>GBPUSD</option>
                    <option>USDJPY</option>
                    <option>AUDUSD</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Visualization Type */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <LineChart className="w-3 h-3" />
                VISUALIZATION
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { type: 'timeseries' as VisualizationType, label: 'TIME SERIES', icon: TrendingUp },
                  { type: 'scatter' as VisualizationType, label: 'SCATTER', icon: ScatterChart },
                  { type: 'histogram' as VisualizationType, label: 'HISTOGRAM', icon: BarChart3 },
                  { type: 'heatmap' as VisualizationType, label: 'HEATMAP', icon: Maximize2 },
                  { type: 'graph' as VisualizationType, label: 'GRAPH', icon: Network },
                  { type: 'correlation' as VisualizationType, label: 'CORRELATION', icon: ScatterChart },
                ].map(({ type, label, icon: Icon }) => (
                  <button
                    key={type}
                    onClick={() => setSelectedViz(type)}
                    className={`
                      py-2 px-2 border text-[10px] transition-colors flex items-center gap-1
                      ${selectedViz === type
                        ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                        : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                      }
                    `}
                  >
                    <Icon className="w-3 h-3" />
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Analysis Type */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <Settings className="w-3 h-3" />
                ANALYSIS
              </div>
              <div className="space-y-2">
                {[
                  { type: 'descriptive' as AnalysisType, label: 'DESCRIPTIVE STATS' },
                  { type: 'correlation' as AnalysisType, label: 'CORRELATION' },
                  { type: 'regression' as AnalysisType, label: 'REGRESSION' },
                  { type: 'timeseries' as AnalysisType, label: 'TIME SERIES' },
                  { type: 'fourier' as AnalysisType, label: 'FOURIER TRANSFORM' },
                  { type: 'wavelet' as AnalysisType, label: 'WAVELET' },
                  { type: 'pca' as AnalysisType, label: 'PCA' },
                  { type: 'clustering' as AnalysisType, label: 'CLUSTERING' },
                ].map(({ type, label }) => (
                  <button
                    key={type}
                    onClick={() => setSelectedAnalysis(type)}
                    className={`
                      w-full py-2 px-3 border text-left text-[10px] transition-colors
                      ${selectedAnalysis === type
                        ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                        : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                      }
                    `}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="space-y-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="w-full py-2 px-3 border border-[#00ff00] bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 transition-colors flex items-center justify-center gap-2"
              >
                <Play className="w-3 h-3" />
                {isRunning ? 'STOP ANALYSIS' : 'RUN ANALYSIS'}
              </button>
              <button className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors flex items-center justify-center gap-2">
                <Download className="w-3 h-3" />
                EXPORT DATA
              </button>
            </div>
          </div>
        </div>

        {/* Center Panel - Visualization Canvas */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Visualization Area */}
          <div className="flex-1 p-4 overflow-auto">
            <div className="border border-[#444] bg-[#000] h-full p-4">
              {/* Visualization Title */}
              <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">
                {selectedViz.toUpperCase()} - {selectedAnalysis.toUpperCase()}
              </div>

              {/* Time Series Visualization */}
              {selectedViz === 'timeseries' && (
                <div className="space-y-4">
                  <div className="h-64 border border-[#444] bg-[#0a0a0a] relative">
                    {/* Y-axis labels */}
                    <div className="absolute left-2 top-2 text-[#666] text-[10px]">1.12</div>
                    <div className="absolute left-2 top-1/2 text-[#666] text-[10px]">1.10</div>
                    <div className="absolute left-2 bottom-2 text-[#666] text-[10px]">1.08</div>
                    
                    {/* Chart area */}
                    <svg className="w-full h-full" viewBox="0 0 800 200">
                      <polyline
                        points={timeSeriesData.map((d, i) => 
                          `${i * 8},${200 - (d.value - 1.08) * 5000}`
                        ).join(' ')}
                        fill="none"
                        stroke="#00ff00"
                        strokeWidth="1"
                      />
                    </svg>
                    
                    {/* X-axis label */}
                    <div className="absolute bottom-2 right-2 text-[#666] text-[10px]">TIME</div>
                  </div>
                  
                  <div className="text-[#00ff00] text-[10px]">
                    REAL-TIME STREAMING: {isRunning ? 'ACTIVE' : 'PAUSED'} | POINTS: {timeSeriesData.length}
                  </div>
                </div>
              )}

              {/* Correlation Matrix */}
              {selectedViz === 'correlation' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-4 gap-2">
                    {correlationMatrix.map((item, idx) => (
                      <div key={idx} className="border border-[#444] p-3 bg-[#0a0a0a]">
                        <div className="text-[#00ff00] text-[10px] mb-2">
                          {item.asset1} × {item.asset2}
                        </div>
                        <div className={`text-sm ${item.corr > 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                          {formatNumber(item.corr)}
                        </div>
                        <div className="mt-2 h-2 bg-[#222]">
                          <div
                            className={`h-full ${item.corr > 0 ? 'bg-[#00ff00]' : 'bg-[#ff0000]'}`}
                            style={{ width: `${Math.abs(item.corr) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Graph Visualization */}
              {selectedViz === 'graph' && (
                <div className="space-y-4">
                  <div className="h-96 border border-[#444] bg-[#0a0a0a] relative flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 400 300">
                      {/* Edges */}
                      <line x1="200" y1="150" x2="100" y2="80" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="300" y2="80" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="100" y2="220" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="300" y2="220" stroke="#444" strokeWidth="1" />
                      <line x1="100" y1="80" x2="300" y2="80" stroke="#444" strokeWidth="1" />
                      
                      {/* Nodes */}
                      {[
                        { x: 200, y: 150, label: 'EURUSD', cluster: 'A' },
                        { x: 100, y: 80, label: 'GBPUSD', cluster: 'A' },
                        { x: 300, y: 80, label: 'USDJPY', cluster: 'B' },
                        { x: 100, y: 220, label: 'AUDUSD', cluster: 'A' },
                        { x: 300, y: 220, label: 'USDCHF', cluster: 'B' },
                      ].map((node, idx) => (
                        <g key={idx}>
                          <circle
                            cx={node.x}
                            cy={node.y}
                            r="20"
                            fill={node.cluster === 'A' ? '#ff8c00' : '#00ff00'}
                            opacity="0.3"
                          />
                          <circle
                            cx={node.x}
                            cy={node.y}
                            r="15"
                            fill="none"
                            stroke={node.cluster === 'A' ? '#ff8c00' : '#00ff00'}
                            strokeWidth="2"
                          />
                          <text
                            x={node.x}
                            y={node.y + 35}
                            textAnchor="middle"
                            fill="#fff"
                            fontSize="10"
                          >
                            {node.label}
                          </text>
                        </g>
                      ))}
                    </svg>
                  </div>
                  <div className="text-[#00ff00] text-[10px]">
                    GRAPH LAYOUT: FORCE-DIRECTED | NODES: 5 | EDGES: 6 | CLUSTERS: 2
                  </div>
                </div>
              )}

              {/* Histogram */}
              {selectedViz === 'histogram' && (
                <div className="space-y-4">
                  <div className="h-64 border border-[#444] bg-[#0a0a0a] flex items-end justify-around p-4">
                    {Array.from({ length: 20 }, (_, i) => {
                      const height = Math.random() * 100;
                      return (
                        <div
                          key={i}
                          className="bg-[#00ff00] w-6"
                          style={{ height: `${height}%` }}
                        />
                      );
                    })}
                  </div>
                  <div className="text-[#00ff00] text-[10px]">
                    DISTRIBUTION: NORMAL | BINS: 20 | SAMPLES: 10000
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Results Panel */}
          <div className="h-48 border-t border-[#444] bg-[#0a0a0a] overflow-y-auto">
            <div className="p-4">
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
                ANALYSIS RESULTS - {selectedAnalysis.toUpperCase()}
              </div>
              
              {selectedAnalysis === 'descriptive' && (
                <div className="grid grid-cols-4 gap-4">
                  {Object.entries(statisticalSummary).map(([key, value]) => (
                    <div key={key} className="border border-[#444] p-2">
                      <div className="text-[#666] text-[10px]">{key.toUpperCase()}</div>
                      <div className="text-[#00ff00] text-sm mt-1">
                        {typeof value === 'number' ? formatNumber(value, 4) : value}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {selectedAnalysis === 'timeseries' && (
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">STATIONARITY (ADF TEST)</span>
                    <span className="text-[#00ff00]">STATIONARY (p=0.001)</span>
                  </div>
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">AUTOCORRELATION (LAG 1)</span>
                    <span className="text-[#00ff00]">0.87</span>
                  </div>
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">SEASONALITY</span>
                    <span className="text-[#ffff00]">DETECTED (24H PERIOD)</span>
                  </div>
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">TREND</span>
                    <span className="text-[#00ff00]">UPWARD (0.0001/HOUR)</span>
                  </div>
                </div>
              )}

              {selectedAnalysis === 'fourier' && (
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">DOMINANT FREQUENCY</span>
                    <span className="text-[#00ff00]">0.042 HZ (24H CYCLE)</span>
                  </div>
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">POWER SPECTRUM PEAK</span>
                    <span className="text-[#00ff00]">0.0087</span>
                  </div>
                  <div className="flex justify-between border-b border-[#444] pb-2">
                    <span className="text-[#666]">HARMONICS DETECTED</span>
                    <span className="text-[#00ff00]">3 (12H, 8H, 6H)</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Graph Query & Neo4j */}
        <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
          <div className="p-4 space-y-6">
            {/* Graph Query Builder */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider flex items-center gap-2">
                <Network className="w-3 h-3" />
                GRAPH QUERY (CYPHER)
              </div>
              <textarea
                className="w-full h-32 bg-[#000] border border-[#444] text-[#00ff00] p-2 text-[10px] font-mono"
                placeholder="MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset)&#10;WHERE r.correlation > 0.7&#10;RETURN a, r, b"
                defaultValue="MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset)
WHERE r.correlation > 0.7
RETURN a, r, b"
              />
              <button className="w-full mt-2 py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
                EXECUTE QUERY
              </button>
            </div>

            {/* Graph Nodes */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
                GRAPH NODES
              </div>
              <div className="space-y-2">
                {graphNodes.map((node, idx) => (
                  <div key={idx} className="border border-[#444] p-2 bg-[#000]">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[#00ff00] text-[10px]">{node.id}</span>
                      <span className={`text-[10px] ${node.cluster === 'A' ? 'text-[#ff8c00]' : 'text-[#00ff00]'}`}>
                        CLUSTER {node.cluster}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-[#666]">CENTRALITY</span>
                      <span className="text-[#fff]">{formatNumber(node.centrality)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* GDS Algorithms */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
                NEO4J GDS ALGORITHMS
              </div>
              <div className="space-y-2">
                {[
                  'PageRank',
                  'Betweenness Centrality',
                  'Louvain Community',
                  'Label Propagation',
                  'Node Similarity',
                  'Triangle Count',
                ].map((algo, idx) => (
                  <button
                    key={idx}
                    className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                  >
                    {algo}
                  </button>
                ))}
              </div>
            </div>

            {/* Export Options */}
            <div>
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
                EXPORT FORMAT
              </div>
              <div className="space-y-2">
                {['CSV', 'JSON', 'PARQUET', 'GRAPHML', 'GEPHI'].map((format, idx) => (
                  <button
                    key={idx}
                    className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                  >
                    {format}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

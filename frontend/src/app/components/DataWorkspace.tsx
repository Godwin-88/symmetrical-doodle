import { useState, useEffect } from 'react';
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
  FileText,
  Table,
  Layers,
} from 'lucide-react';
import { searchSymbols, getSymbolInfo, importExternalData, type SymbolSearchResult } from '../../services/dataImportService';

// Types
type WorkspaceCategory = 'SOURCES' | 'IMPORTS' | 'QUERIES' | 'VISUALIZATIONS' | 'ANALYSIS' | 'EXPORTS';

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

type DataSource = 'postgresql' | 'neo4j' | 'redis' | 'live' | 'nautilus';

interface DataImport {
  id: string;
  symbol: string;
  source: string;
  status: 'PENDING' | 'IMPORTING' | 'COMPLETED' | 'FAILED';
  progress: number;
  dataPoints: number;
  startDate: string;
  endDate: string;
  timestamp: string;
}

interface SavedQuery {
  id: string;
  name: string;
  source: DataSource;
  query: string;
  lastRun?: string;
  resultCount?: number;
}

export function DataWorkspace() {
  // Category selection state
  const [selectedCategory, setSelectedCategory] = useState<WorkspaceCategory>('SOURCES');

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

  // Data state
  const [dataImports, setDataImports] = useState<DataImport[]>([
    { id: '1', symbol: 'EURUSD', source: 'yahoo_finance', status: 'COMPLETED', progress: 100, dataPoints: 15000, startDate: '2023-01-01', endDate: '2024-01-01', timestamp: new Date(Date.now() - 86400000).toISOString() },
    { id: '2', symbol: 'GBPUSD', source: 'yahoo_finance', status: 'COMPLETED', progress: 100, dataPoints: 12000, startDate: '2023-06-01', endDate: '2024-01-01', timestamp: new Date(Date.now() - 172800000).toISOString() },
    { id: '3', symbol: 'BTC-USD', source: 'yahoo_finance', status: 'IMPORTING', progress: 65, dataPoints: 8500, startDate: '2023-01-01', endDate: '2024-01-01', timestamp: new Date().toISOString() },
  ]);

  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([
    { id: '1', name: 'Recent Market Data', source: 'postgresql', query: 'SELECT * FROM market_data WHERE timestamp > NOW() - INTERVAL \'24 hours\'', lastRun: new Date(Date.now() - 3600000).toISOString(), resultCount: 1440 },
    { id: '2', name: 'Asset Correlations', source: 'neo4j', query: 'MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset) WHERE r.correlation > 0.7 RETURN a, r, b', lastRun: new Date(Date.now() - 7200000).toISOString(), resultCount: 24 },
    { id: '3', name: 'Active Regimes', source: 'postgresql', query: 'SELECT * FROM regimes WHERE end_time IS NULL', lastRun: new Date(Date.now() - 1800000).toISOString(), resultCount: 3 },
  ]);

  const [selectedQuery, setSelectedQuery] = useState<SavedQuery | null>(null);
  const [selectedImport, setSelectedImport] = useState<DataImport | null>(null);

  // Action status
  const [actionStatus, setActionStatus] = useState<string | null>(null);

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

      // Add to imports list
      const newImport: DataImport = {
        id: Date.now().toString(),
        symbol: result.symbol,
        source: importSource,
        status: 'COMPLETED',
        progress: 100,
        dataPoints: result.data_points,
        startDate: startDate || 'N/A',
        endDate: endDate || 'N/A',
        timestamp: new Date().toISOString(),
      };
      setDataImports([newImport, ...dataImports]);

      setImportStatus(
        `Successfully imported ${result.data_points} data points for ${result.symbol}`
      );
      setActionStatus('Import completed successfully');
      setTimeout(() => setActionStatus(null), 3000);

    } catch (error) {
      console.error('Import failed:', error);
      setImportStatus('Import failed: ' + (error as Error).message);
    } finally {
      setIsImporting(false);
    }
  };

  const handleRunQuery = (query: SavedQuery) => {
    setActionStatus(`Running query: ${query.name}...`);
    setTimeout(() => {
      setActionStatus(`Query completed: ${query.resultCount} results`);
      setSavedQueries(savedQueries.map(q =>
        q.id === query.id ? { ...q, lastRun: new Date().toISOString() } : q
      ));
      setTimeout(() => setActionStatus(null), 3000);
    }, 1000);
  };

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
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

  // Categories configuration
  const categories: { key: WorkspaceCategory; label: string; icon: any; count?: number }[] = [
    { key: 'SOURCES', label: 'DATA SOURCES', icon: Database, count: 5 },
    { key: 'IMPORTS', label: 'DATA IMPORTS', icon: Plus, count: dataImports.length },
    { key: 'QUERIES', label: 'SAVED QUERIES', icon: FileText, count: savedQueries.length },
    { key: 'VISUALIZATIONS', label: 'VISUALIZATIONS', icon: LineChart },
    { key: 'ANALYSIS', label: 'ANALYSIS', icon: TrendingUp },
    { key: 'EXPORTS', label: 'EXPORTS', icon: Download },
  ];

  // Render left panel content based on category
  const renderLeftPanelContent = () => {
    switch (selectedCategory) {
      case 'SOURCES':
        return (
          <div className="space-y-2">
            {(['postgresql', 'neo4j', 'redis', 'live', 'nautilus'] as DataSource[]).map((source) => (
              <div
                key={source}
                onClick={() => setSelectedSource(source)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedSource === source
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="text-[#00ff00] text-[11px]">{source.toUpperCase()}</span>
                  <span className="text-[10px] px-2 py-0.5 border border-[#00ff00] text-[#00ff00]">
                    CONNECTED
                  </span>
                </div>
                <div className="text-[#666] text-[9px]">
                  {source === 'postgresql' && 'TimescaleDB + pgvector'}
                  {source === 'neo4j' && 'Graph Database + GDS'}
                  {source === 'redis' && 'Cache Layer'}
                  {source === 'live' && 'Real-time Market Data'}
                  {source === 'nautilus' && 'Parquet Data Catalog'}
                </div>
              </div>
            ))}
          </div>
        );

      case 'IMPORTS':
        return (
          <div className="space-y-2">
            {dataImports.map((imp) => (
              <div
                key={imp.id}
                onClick={() => setSelectedImport(imp)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedImport?.id === imp.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-[#00ff00]">{imp.symbol}</span>
                  <span className={`text-[10px] px-1 ${
                    imp.status === 'COMPLETED' ? 'text-[#00ff00]' :
                    imp.status === 'IMPORTING' ? 'text-[#ffff00]' :
                    'text-[#ff0000]'
                  }`}>
                    {imp.status}
                  </span>
                </div>
                <div className="text-[#666] text-[9px]">
                  {imp.dataPoints.toLocaleString()} points | {imp.source}
                </div>
                {imp.status === 'IMPORTING' && (
                  <div className="mt-2 h-1 bg-[#222]">
                    <div className="h-full bg-[#ffff00]" style={{ width: `${imp.progress}%` }} />
                  </div>
                )}
              </div>
            ))}
            <button
              onClick={() => {
                setSelectedCategory('IMPORTS');
                setShowImport(true);
              }}
              className="w-full py-2 border border-dashed border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
            >
              + NEW IMPORT
            </button>
          </div>
        );

      case 'QUERIES':
        return (
          <div className="space-y-2">
            {savedQueries.map((query) => (
              <div
                key={query.id}
                onClick={() => setSelectedQuery(query)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedQuery?.id === query.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-[#00ff00]">{query.name}</span>
                  <span className="text-[#666] text-[9px]">{query.source.toUpperCase()}</span>
                </div>
                <div className="text-[#666] text-[9px] truncate">
                  {query.query.substring(0, 50)}...
                </div>
                {query.lastRun && (
                  <div className="text-[#666] text-[9px] mt-1">
                    Last: {formatTime(query.lastRun)} | {query.resultCount} results
                  </div>
                )}
              </div>
            ))}
            <button
              className="w-full py-2 border border-dashed border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
            >
              + NEW QUERY
            </button>
          </div>
        );

      case 'VISUALIZATIONS':
        return (
          <div className="space-y-2">
            {[
              { type: 'timeseries' as VisualizationType, label: 'TIME SERIES', icon: TrendingUp },
              { type: 'scatter' as VisualizationType, label: 'SCATTER PLOT', icon: ScatterChart },
              { type: 'histogram' as VisualizationType, label: 'HISTOGRAM', icon: BarChart3 },
              { type: 'heatmap' as VisualizationType, label: 'HEATMAP', icon: Maximize2 },
              { type: 'graph' as VisualizationType, label: 'NETWORK GRAPH', icon: Network },
              { type: 'correlation' as VisualizationType, label: 'CORRELATION', icon: Table },
              { type: 'candlestick' as VisualizationType, label: 'CANDLESTICK', icon: Layers },
            ].map(({ type, label, icon: Icon }) => (
              <div
                key={type}
                onClick={() => setSelectedViz(type)}
                className={`
                  border p-3 cursor-pointer transition-colors flex items-center gap-2
                  ${selectedViz === type
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <Icon className="w-4 h-4 text-[#ff8c00]" />
                <span className={selectedViz === type ? 'text-[#ff8c00]' : 'text-[#00ff00]'}>{label}</span>
              </div>
            ))}
          </div>
        );

      case 'ANALYSIS':
        return (
          <div className="space-y-2">
            {[
              { type: 'descriptive' as AnalysisType, label: 'DESCRIPTIVE STATS' },
              { type: 'correlation' as AnalysisType, label: 'CORRELATION ANALYSIS' },
              { type: 'regression' as AnalysisType, label: 'REGRESSION' },
              { type: 'timeseries' as AnalysisType, label: 'TIME SERIES ANALYSIS' },
              { type: 'fourier' as AnalysisType, label: 'FOURIER TRANSFORM' },
              { type: 'wavelet' as AnalysisType, label: 'WAVELET ANALYSIS' },
              { type: 'pca' as AnalysisType, label: 'PCA / DIMENSIONALITY' },
              { type: 'clustering' as AnalysisType, label: 'CLUSTERING' },
            ].map(({ type, label }) => (
              <div
                key={type}
                onClick={() => setSelectedAnalysis(type)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedAnalysis === type
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <span className={selectedAnalysis === type ? 'text-[#ff8c00]' : 'text-[#00ff00]'}>{label}</span>
              </div>
            ))}
          </div>
        );

      case 'EXPORTS':
        return (
          <div className="space-y-2">
            {['CSV', 'JSON', 'PARQUET', 'GRAPHML', 'GEPHI', 'EXCEL'].map((format) => (
              <div
                key={format}
                className="border border-[#333] p-3 cursor-pointer hover:border-[#ff8c00] transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Download className="w-4 h-4 text-[#ff8c00]" />
                  <span className="text-[#00ff00]">{format}</span>
                </div>
              </div>
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  // Render center panel content
  const renderCenterPanelContent = () => {
    switch (selectedCategory) {
      case 'SOURCES':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                DATA SOURCE: {selectedSource.toUpperCase()}
              </div>
            </div>

            <div className="space-y-6">
              {/* Connection Details */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">CONNECTION DETAILS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Status:</span>
                    <span className="text-[#00ff00]">CONNECTED</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Host:</span>
                    <span className="text-[#fff]">localhost:5432</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Database:</span>
                    <span className="text-[#fff]">trading_platform</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Latency:</span>
                    <span className="text-[#00ff00]">2ms</span>
                  </div>
                </div>
              </div>

              {/* Available Tables */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">AVAILABLE TABLES</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  {['market_data', 'embeddings', 'regimes', 'positions', 'orders', 'strategies'].map((table, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 border-b border-[#222] last:border-0 text-[10px]">
                      <span className="text-[#00ff00]">{table}</span>
                      <span className="text-[#666]">{Math.floor(Math.random() * 100000)} rows</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Query Interface */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">QUERY INTERFACE</div>
                <div className="border border-[#444] p-3 bg-[#000]">
                  <textarea
                    className="w-full h-24 bg-transparent border-none text-[#00ff00] text-[10px] font-mono outline-none resize-none"
                    placeholder="SELECT * FROM market_data LIMIT 100"
                    defaultValue="SELECT * FROM market_data WHERE timestamp > NOW() - INTERVAL '1 hour' LIMIT 100"
                  />
                </div>
                <button className="w-full mt-2 py-2 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black">
                  EXECUTE QUERY
                </button>
              </div>
            </div>
          </>
        );

      case 'IMPORTS':
        if (showImport) {
          return (
            <>
              <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
                <div className="text-[#ff8c00] text-sm tracking-wider">NEW DATA IMPORT</div>
              </div>

              <div className="space-y-4 max-w-xl">
                {/* Source Selection */}
                <div>
                  <div className="text-[#666] text-[10px] mb-1">DATA SOURCE</div>
                  <select
                    value={importSource}
                    onChange={(e) => setImportSource(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]"
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
                      className="flex-1 bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]"
                    />
                    <button
                      onClick={handleSearch}
                      className="px-4 py-2 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
                    >
                      SEARCH
                    </button>
                  </div>
                </div>

                {/* Search Results */}
                {searchResults.length > 0 && (
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">RESULTS</div>
                    <div className="border border-[#444] max-h-40 overflow-y-auto">
                      {searchResults.map((result, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedSymbol(result.symbol)}
                          className={`
                            w-full text-left p-2 text-[10px] border-b border-[#222] last:border-0
                            ${selectedSymbol === result.symbol
                              ? 'bg-[#1a1a1a] text-[#ff8c00]'
                              : 'text-[#00ff00] hover:bg-[#1a1a1a]'
                            }
                          `}
                        >
                          <div>{result.symbol}</div>
                          <div className="text-[#666] text-[9px]">{result.name} | {result.type}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Date Range */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">START DATE</div>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]"
                    />
                  </div>
                  <div>
                    <div className="text-[#666] text-[10px] mb-1">END DATE</div>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]"
                    />
                  </div>
                </div>

                {/* Interval */}
                <div>
                  <div className="text-[#666] text-[10px] mb-1">INTERVAL</div>
                  <select
                    value={importInterval}
                    onChange={(e) => setImportInterval(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]"
                  >
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="1h">1 Hour</option>
                    <option value="1d">1 Day</option>
                    <option value="1wk">1 Week</option>
                  </select>
                </div>

                {/* Import Button */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowImport(false)}
                    className="flex-1 py-2 border border-[#666] text-[#666] text-[10px] hover:text-[#fff] hover:border-[#fff]"
                  >
                    CANCEL
                  </button>
                  <button
                    onClick={handleImport}
                    disabled={isImporting || !selectedSymbol}
                    className="flex-1 py-2 bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 disabled:opacity-50"
                  >
                    {isImporting ? 'IMPORTING...' : 'START IMPORT'}
                  </button>
                </div>

                {/* Status */}
                {importStatus && (
                  <div className={`text-[10px] p-3 border ${
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
            </>
          );
        }

        if (!selectedImport) return <div className="text-[#666] p-4">Select an import to view details or create a new one</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                IMPORT: {selectedImport.symbol}
              </div>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">STATUS</div>
                  <div className={`text-[14px] ${selectedImport.status === 'COMPLETED' ? 'text-[#00ff00]' : 'text-[#ffff00]'}`}>
                    {selectedImport.status}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">DATA POINTS</div>
                  <div className="text-[#00ff00] text-[14px]">{selectedImport.dataPoints.toLocaleString()}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SOURCE</div>
                  <div className="text-[#fff] text-[14px]">{selectedImport.source}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">PROGRESS</div>
                  <div className="text-[#00ff00] text-[14px]">{selectedImport.progress}%</div>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">IMPORT DETAILS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Date Range:</span>
                    <span className="text-[#fff]">{selectedImport.startDate} to {selectedImport.endDate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Imported:</span>
                    <span className="text-[#fff]">{formatTime(selectedImport.timestamp)}</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'QUERIES':
        if (!selectedQuery) return <div className="text-[#666] p-4">Select a query to view details</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                QUERY: {selectedQuery.name}
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">QUERY CODE</div>
                <div className="border border-[#444] p-3 bg-[#000]">
                  <pre className="text-[#00ff00] text-[10px] whitespace-pre-wrap">{selectedQuery.query}</pre>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SOURCE</div>
                  <div className="text-[#00ff00] text-[12px]">{selectedQuery.source.toUpperCase()}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">LAST RUN</div>
                  <div className="text-[#fff] text-[12px]">{selectedQuery.lastRun ? formatTime(selectedQuery.lastRun) : 'Never'}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">RESULTS</div>
                  <div className="text-[#00ff00] text-[12px]">{selectedQuery.resultCount || 0}</div>
                </div>
              </div>

              <button
                onClick={() => handleRunQuery(selectedQuery)}
                className="w-full py-2 bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80"
              >
                RUN QUERY
              </button>
            </div>
          </>
        );

      case 'VISUALIZATIONS':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                {selectedViz.toUpperCase()} VISUALIZATION
              </div>
            </div>

            <div className="space-y-4">
              {/* Time Series Visualization */}
              {selectedViz === 'timeseries' && (
                <div className="border border-[#444] bg-[#000] p-4">
                  <div className="h-64 relative">
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
                  </div>
                  <div className="text-[#00ff00] text-[10px] mt-2">
                    POINTS: {timeSeriesData.length} | STREAMING: {isRunning ? 'ACTIVE' : 'PAUSED'}
                  </div>
                </div>
              )}

              {/* Correlation Matrix */}
              {selectedViz === 'correlation' && (
                <div className="grid grid-cols-3 gap-4">
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
              )}

              {/* Graph Visualization */}
              {selectedViz === 'graph' && (
                <div className="border border-[#444] bg-[#000] p-4">
                  <div className="h-80 relative flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 400 300">
                      <line x1="200" y1="150" x2="100" y2="80" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="300" y2="80" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="100" y2="220" stroke="#444" strokeWidth="1" />
                      <line x1="200" y1="150" x2="300" y2="220" stroke="#444" strokeWidth="1" />

                      {[
                        { x: 200, y: 150, label: 'EURUSD', cluster: 'A' },
                        { x: 100, y: 80, label: 'GBPUSD', cluster: 'A' },
                        { x: 300, y: 80, label: 'USDJPY', cluster: 'B' },
                        { x: 100, y: 220, label: 'AUDUSD', cluster: 'A' },
                        { x: 300, y: 220, label: 'USDCHF', cluster: 'B' },
                      ].map((node, idx) => (
                        <g key={idx}>
                          <circle cx={node.x} cy={node.y} r="20" fill={node.cluster === 'A' ? '#ff8c00' : '#00ff00'} opacity="0.3" />
                          <circle cx={node.x} cy={node.y} r="15" fill="none" stroke={node.cluster === 'A' ? '#ff8c00' : '#00ff00'} strokeWidth="2" />
                          <text x={node.x} y={node.y + 35} textAnchor="middle" fill="#fff" fontSize="10">{node.label}</text>
                        </g>
                      ))}
                    </svg>
                  </div>
                </div>
              )}

              {/* Histogram */}
              {selectedViz === 'histogram' && (
                <div className="border border-[#444] bg-[#000] p-4">
                  <div className="h-64 flex items-end justify-around">
                    {Array.from({ length: 20 }, (_, i) => (
                      <div key={i} className="bg-[#00ff00] w-6" style={{ height: `${Math.random() * 100}%` }} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        );

      case 'ANALYSIS':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                {selectedAnalysis.toUpperCase()} ANALYSIS
              </div>
            </div>

            <div className="space-y-4">
              {selectedAnalysis === 'descriptive' && (
                <div className="grid grid-cols-4 gap-4">
                  {Object.entries(statisticalSummary).map(([key, value]) => (
                    <div key={key} className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">{key.toUpperCase()}</div>
                      <div className="text-[#00ff00] text-[12px] mt-1">
                        {typeof value === 'number' ? formatNumber(value, 4) : value}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {selectedAnalysis === 'timeseries' && (
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between border-b border-[#222] pb-2">
                    <span className="text-[#666]">STATIONARITY (ADF TEST)</span>
                    <span className="text-[#00ff00]">STATIONARY (p=0.001)</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-2">
                    <span className="text-[#666]">AUTOCORRELATION (LAG 1)</span>
                    <span className="text-[#00ff00]">0.87</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-2">
                    <span className="text-[#666]">SEASONALITY</span>
                    <span className="text-[#ffff00]">DETECTED (24H PERIOD)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">TREND</span>
                    <span className="text-[#00ff00]">UPWARD (0.0001/HOUR)</span>
                  </div>
                </div>
              )}
            </div>
          </>
        );

      case 'EXPORTS':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">EXPORT DATA</div>
            </div>

            <div className="space-y-4 max-w-xl">
              <div>
                <div className="text-[#666] text-[10px] mb-1">SELECT DATA</div>
                <select className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]">
                  <option>market_data (Last 24h)</option>
                  <option>All Embeddings</option>
                  <option>Query Results</option>
                  <option>Current Visualization</option>
                </select>
              </div>

              <div>
                <div className="text-[#666] text-[10px] mb-1">FORMAT</div>
                <select className="w-full bg-[#1a1a1a] border border-[#444] text-[#00ff00] px-2 py-2 text-[10px]">
                  <option>CSV</option>
                  <option>JSON</option>
                  <option>Parquet</option>
                  <option>Excel</option>
                </select>
              </div>

              <button className="w-full py-2 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500]">
                EXPORT
              </button>
            </div>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Header */}
          <div className="mb-4">
            <div className="text-[#ff8c00] text-[10px] tracking-wider">DATA WORKSPACE</div>
          </div>

          {/* Category Selection */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">CATEGORY</div>
            <div className="space-y-1">
              {categories.map((cat) => {
                const Icon = cat.icon;
                return (
                  <button
                    key={cat.key}
                    onClick={() => {
                      setSelectedCategory(cat.key);
                      setShowImport(false);
                      setSelectedImport(null);
                      setSelectedQuery(null);
                    }}
                    className={`
                      w-full py-2 px-3 text-left text-[10px] border transition-colors flex items-center gap-2
                      ${selectedCategory === cat.key
                        ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                        : 'border-[#333] text-[#666] hover:border-[#ff8c00] hover:text-[#ff8c00]'
                      }
                    `}
                  >
                    <Icon className="w-3 h-3" />
                    <span className="flex-1">{cat.label}</span>
                    {cat.count !== undefined && <span>{cat.count}</span>}
                  </button>
                );
              })}
            </div>
          </div>

          {/* List Content */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
              {categories.find(c => c.key === selectedCategory)?.label}
            </div>
            {renderLeftPanelContent()}
          </div>
        </div>
      </div>

      {/* Center Panel */}
      <div className="flex-1 overflow-y-auto p-4">
        {renderCenterPanelContent()}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">ACTIONS</div>

          {/* Action Status */}
          {actionStatus && (
            <div className="mb-4 p-2 border border-[#ffff00] bg-[#1a1a1a] text-[#ffff00] text-[10px]">
              {actionStatus}
            </div>
          )}

          {/* Analysis Controls */}
          <div className="space-y-2 mb-6">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className="w-full py-2 px-3 border border-[#00ff00] bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 flex items-center justify-center gap-2"
            >
              <Play className="w-3 h-3" />
              {isRunning ? 'STOP ANALYSIS' : 'RUN ANALYSIS'}
            </button>
            <button className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black flex items-center justify-center gap-2">
              <Download className="w-3 h-3" />
              EXPORT DATA
            </button>
          </div>

          {/* Graph Query (if Neo4j) */}
          {selectedSource === 'neo4j' && (
            <div className="mb-6">
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">CYPHER QUERY</div>
              <textarea
                className="w-full h-24 bg-[#000] border border-[#444] text-[#00ff00] p-2 text-[9px] font-mono"
                placeholder="MATCH (a:Asset)-[r:CORRELATES_WITH]->(b:Asset) RETURN a, r, b"
              />
              <button className="w-full mt-2 py-2 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black">
                EXECUTE
              </button>
            </div>
          )}

          {/* GDS Algorithms */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">GRAPH ALGORITHMS</div>
            <div className="space-y-2">
              {['PageRank', 'Betweenness', 'Louvain', 'Node2Vec'].map((algo) => (
                <button
                  key={algo}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a]"
                >
                  {algo}
                </button>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK ACTIONS</div>
            <div className="space-y-2">
              <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff]">
                REFRESH DATA
              </button>
              <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff]">
                CLEAR CACHE
              </button>
              <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff]">
                VIEW LOGS
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

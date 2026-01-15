import { useTradingStore } from '@/app/store/tradingStore';
import { useState, useEffect } from 'react';
import { trainRegimeModel, runGraphAnalysis } from '../../services/intelligenceService';
import { 
  listModels, 
  getModelCategories, 
  getUseCases, 
  getModelDetails,
  recommendModels,
  formatCategory,
  formatUseCase,
  getLatencyColor,
  getDataReqColor,
  getExplainabilityColor,
  type ModelSpec,
  type ModelCategory as ModelCategoryType,
  type UseCase as UseCaseType,
} from '../../services/modelsService';

interface RegimeConfig {
  id: string;
  name: string;
  volatility: 'LOW' | 'MEDIUM' | 'HIGH';
  trend: 'TRENDING' | 'RANGING' | 'REVERTING';
  liquidity: 'NORMAL' | 'HIGH' | 'LOW';
  minDuration: number; // minutes
  maxDuration: number; // minutes
}

interface EmbeddingConfig {
  assetId: string;
  windowSize: number;
  features: string[];
  modelType: 'TCN' | 'VAE' | 'LSTM';
}

export function Intelligence() {
  const {
    regimes,
    embeddings,
    intelligenceSignals,
    selectedRegime,
    setSelectedRegime,
    fetchRegimeData,
    fetchGraphFeatures,
    isLoading,
    error,
  } = useTradingStore();
  
  // State for CRUD operations
  const [showEmbeddingModal, setShowEmbeddingModal] = useState(false);
  const [showRegimeModal, setShowRegimeModal] = useState(false);
  const [showGraphModal, setShowGraphModal] = useState(false);
  const [showTrainModal, setShowTrainModal] = useState(false);
  const [showModelBrowser, setShowModelBrowser] = useState(false);
  const [editingRegime, setEditingRegime] = useState<RegimeConfig | null>(null);
  const [editingEmbedding, setEditingEmbedding] = useState<EmbeddingConfig | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  
  // Model browser state
  const [availableModels, setAvailableModels] = useState<ModelSpec[]>([]);
  const [modelCategories, setModelCategories] = useState<ModelCategoryType[]>([]);
  const [useCases, setUseCases] = useState<UseCaseType[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedUseCase, setSelectedUseCase] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelSpec | null>(null);
  const [productionOnlyFilter, setProductionOnlyFilter] = useState(true);
  
  // Local state for configurations
  const [regimeConfigs, setRegimeConfigs] = useState<RegimeConfig[]>([
    { id: '1', name: 'LOW_VOL_TRENDING', volatility: 'LOW', trend: 'TRENDING', liquidity: 'NORMAL', minDuration: 60, maxDuration: 720 },
    { id: '2', name: 'HIGH_VOL_RANGING', volatility: 'HIGH', trend: 'RANGING', liquidity: 'NORMAL', minDuration: 30, maxDuration: 360 },
    { id: '3', name: 'CRISIS', volatility: 'HIGH', trend: 'REVERTING', liquidity: 'LOW', minDuration: 15, maxDuration: 180 },
  ]);
  
  const [embeddingConfigs, setEmbeddingConfigs] = useState<EmbeddingConfig[]>([
    { assetId: 'EURUSD', windowSize: 100, features: ['price', 'volume', 'volatility'], modelType: 'TCN' },
    { assetId: 'GBPUSD', windowSize: 100, features: ['price', 'volume'], modelType: 'VAE' },
  ]);
  
  // Fetch regime data on mount and periodically
  useEffect(() => {
    const fetchData = async () => {
      try {
        await fetchRegimeData('EURUSD');
        await fetchGraphFeatures('EURUSD');
        setIsBackendConnected(true);
      } catch (err) {
        console.warn('Backend unavailable:', err);
        setIsBackendConnected(false);
      }
    };
    
    fetchData();
    
    // Poll every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchRegimeData, fetchGraphFeatures]);

  // Fetch models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const [models, categories, cases] = await Promise.all([
          listModels({ production_ready: productionOnlyFilter }),
          getModelCategories(),
          getUseCases(),
        ]);
        setAvailableModels(models);
        setModelCategories(categories);
        setUseCases(cases);
      } catch (err) {
        console.warn('Failed to fetch models:', err);
      }
    };
    
    fetchModels();
  }, [productionOnlyFilter]);

  // CRUD operations for regime configs
  const createRegimeConfig = (config: Omit<RegimeConfig, 'id'>) => {
    const newConfig: RegimeConfig = {
      ...config,
      id: Date.now().toString(),
    };
    setRegimeConfigs([...regimeConfigs, newConfig]);
    setActionStatus('Regime configuration created');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const updateRegimeConfig = (id: string, updates: Partial<RegimeConfig>) => {
    setRegimeConfigs(regimeConfigs.map(c => c.id === id ? { ...c, ...updates } : c));
    setActionStatus('Regime configuration updated');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const deleteRegimeConfig = (id: string) => {
    setRegimeConfigs(regimeConfigs.filter(c => c.id !== id));
    setActionStatus('Regime configuration deleted');
    setTimeout(() => setActionStatus(null), 3000);
  };

  // CRUD operations for embedding configs
  const createEmbeddingConfig = (config: EmbeddingConfig) => {
    setEmbeddingConfigs([...embeddingConfigs, config]);
    setActionStatus('Embedding configuration created');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const updateEmbeddingConfig = (assetId: string, updates: Partial<EmbeddingConfig>) => {
    setEmbeddingConfigs(embeddingConfigs.map(c => c.assetId === assetId ? { ...c, ...updates } : c));
    setActionStatus('Embedding configuration updated');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const deleteEmbeddingConfig = (assetId: string) => {
    setEmbeddingConfigs(embeddingConfigs.filter(c => c.assetId !== assetId));
    setActionStatus('Embedding configuration deleted');
    setTimeout(() => setActionStatus(null), 3000);
  };

  // Action handlers
  const handleTrainRegimeModel = async () => {
    setActionStatus('Training regime model...');
    try {
      // In production, this would use real historical data
      const mockHistoricalData = Array.from({ length: 200 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 3600000).toISOString(),
        asset_id: 'EURUSD',
        open: 1.08 + Math.random() * 0.01,
        high: 1.08 + Math.random() * 0.01,
        low: 1.08 + Math.random() * 0.01,
        close: 1.08 + Math.random() * 0.01,
        volume: 1000000 + Math.random() * 500000,
      }));
      
      await trainRegimeModel(mockHistoricalData);
      setActionStatus('Regime model trained successfully');
      await fetchRegimeData('EURUSD');
    } catch (err: any) {
      setActionStatus(`Training failed: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 5000);
  };

  const handleRunGraphAnalysis = async (analysisType: 'asset_correlations' | 'regime_transitions') => {
    setActionStatus(`Running ${analysisType} analysis...`);
    try {
      await runGraphAnalysis(analysisType);
      setActionStatus(`${analysisType} analysis completed`);
      await fetchGraphFeatures('EURUSD');
    } catch (err: any) {
      setActionStatus(`Analysis failed: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 5000);
  };

  const handleGenerateEmbedding = async (assetId: string) => {
    setActionStatus(`Generating embedding for ${assetId}...`);
    // In production, this would call the embedding API
    setTimeout(() => {
      setActionStatus(`Embedding generated for ${assetId}`);
      setTimeout(() => setActionStatus(null), 3000);
    }, 2000);
  };

  const currentRegime = regimes.find(r => r.id === selectedRegime) || regimes[0];

  // Mock data for regime details
  const regimeTransitions = [
    { to: 'MEDIUM_VOL_TRENDING', probability: 30.0 },
    { to: 'HIGH_VOL_RANGING', probability: 25.0 },
    { to: 'CRISIS', probability: 5.0 },
  ];

  const durationStats = {
    avg: '5.2 HOURS',
    min: '1.5H',
    max: '12.8H',
  };

  const affectedAssets = [
    { asset: 'EURUSD', sensitivity: 0.85 },
    { asset: 'GBPUSD', sensitivity: 0.72 },
    { asset: 'USDJPY', sensitivity: 0.45 },
  ];

  const strategyPerformance = [
    { strategy: 'TREND_ALPHA', sharpe: 1.80, maxDD: 5.0, sample: 120 },
    { strategy: 'MEAN_REVERSION', sharpe: 0.30, maxDD: 12.0, sample: 89 },
    { strategy: 'VOLATILITY_ARB', sharpe: 1.20, maxDD: 8.0, sample: 67 },
  ];

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Loading/Error Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="text-[#ff8c00] text-sm">LOADING INTELLIGENCE DATA...</div>
        </div>
      )}
      {error && (
        <div className="absolute top-4 right-4 bg-[#ff0000] text-black px-4 py-2 z-50">
          ERROR: {error}
        </div>
      )}
      
      {/* Left Panel - List View */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Market State Embeddings */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">MARKET STATE EMBEDDINGS</div>
            <div className="space-y-2">
              {embeddings.map((emb, idx) => (
                <div key={idx} className="border border-[#333] p-2 hover:border-[#ff8c00] cursor-pointer transition-colors">
                  <div className="text-[#00ff00]">{emb.asset}_{emb.timestamp}</div>
                  <div className="text-[#666] text-[10px] mt-1">
                    [CONF: {formatNumber(emb.confidence)}]
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detected Regimes */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">DETECTED REGIMES</div>
            <div className="space-y-2">
              {regimes.map((regime) => (
                <div
                  key={regime.id}
                  onClick={() => setSelectedRegime(regime.id)}
                  className={`
                    border p-2 cursor-pointer transition-colors
                    ${selectedRegime === regime.id || (!selectedRegime && regime === regimes[0])
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                    }
                  `}
                >
                  <div className="text-[#00ff00]">{regime.name}</div>
                  <div className="text-[#666] text-[10px] mt-1">
                    [PROB: {regime.probability}%]
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Intelligence Signals */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">INTELLIGENCE SIGNALS</div>
            <div className="space-y-2">
              {intelligenceSignals.map((signal, idx) => (
                <div key={idx} className="border border-[#333] p-2">
                  <div className="text-[#ffff00]">{signal.type}</div>
                  <div className="text-[#666] text-[10px] mt-1">[{signal.timestamp}]</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Center Panel - View */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            REGIME: {currentRegime.name}
          </div>
        </div>

        <div className="space-y-6">
          {/* Definition */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DEFINITION</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="space-y-1 text-[10px]">
                <div>
                  <span className="text-[#666]">VOLATILITY:</span>
                  <span className="ml-2 text-[#00ff00]">{currentRegime.volatility}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#666]">TREND:</span>
                  <span className="ml-2 text-[#00ff00]">{currentRegime.trend}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#666]">LIQUIDITY:</span>
                  <span className="ml-2 text-[#00ff00]">{currentRegime.liquidity}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Transition Probabilities */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">TRANSITION PROBABILITIES</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2">
              {regimeTransitions.map((trans, idx) => (
                <div key={idx} className="flex items-center gap-4">
                  <div className="text-[#666]">→</div>
                  <div className="w-48 text-[#00ff00]">{trans.to}</div>
                  <div className="w-16 text-right text-[#fff]">{formatNumber(trans.probability, 1)}%</div>
                  <div className="flex-1">
                    <div className="h-2 bg-[#222]">
                      <div
                        className="h-full bg-[#ff8c00]"
                        style={{ width: `${trans.probability}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Duration Statistics */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DURATION STATISTICS</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="space-y-1 text-[10px]">
                <div>
                  <span className="text-[#666]">AVG:</span>
                  <span className="ml-2 text-[#00ff00]">{durationStats.avg}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#666]">MIN:</span>
                  <span className="ml-2 text-[#00ff00]">{durationStats.min}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#666]">MAX:</span>
                  <span className="ml-2 text-[#00ff00]">{durationStats.max}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Affected Assets */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">AFFECTED ASSETS</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2">
              {affectedAssets.map((asset, idx) => (
                <div key={idx} className="flex items-center gap-4">
                  <div className="w-32 text-[#00ff00]">{asset.asset}</div>
                  <div className="text-[#666]">SENSITIVITY:</div>
                  <div className="w-16 text-right text-[#fff]">{formatNumber(asset.sensitivity)}</div>
                  <div className="flex-1">
                    <div className="h-2 bg-[#222]">
                      <div
                        className="h-full bg-[#ff8c00]"
                        style={{ width: `${asset.sensitivity * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Strategy Performance in this Regime */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">
              STRATEGY PERFORMANCE IN THIS REGIME
            </div>
            <div className="border border-[#444] bg-[#0a0a0a]">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">STRATEGY</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">SHARPE</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">MAX DD</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">SAMPLE</th>
                  </tr>
                </thead>
                <tbody>
                  {strategyPerformance.map((perf, idx) => (
                    <tr key={idx} className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#00ff00]">{perf.strategy}</td>
                      <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(perf.sharpe)}</td>
                      <td className="px-3 py-2 text-right text-[#ff0000]">{formatNumber(perf.maxDD, 1)}%</td>
                      <td className="px-3 py-2 text-right text-[#fff]">{perf.sample} DAYS</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Graph Context */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">GRAPH CONTEXT</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="space-y-2 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">CLUSTER:</span>
                  <span className="text-[#00ff00]">CLUSTER_2</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">CENTRALITY:</span>
                  <span className="text-[#00ff00]">0.67</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">SYSTEMIC RISK:</span>
                  <span className="text-[#ffff00]">0.34</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Actions & Configuration */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Status Indicator */}
          <div className="mb-4 flex items-center justify-between">
            <div className="text-[#ff8c00] text-[10px] tracking-wider">INTELLIGENCE CONTROL</div>
            <div className={`px-2 py-1 border text-[10px] ${isBackendConnected ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#ff0000] text-[#ff0000]'}`}>
              {isBackendConnected ? '● LIVE' : '● OFFLINE'}
            </div>
          </div>

          {/* Action Status */}
          {actionStatus && (
            <div className="mb-4 p-2 border border-[#ffff00] bg-[#1a1a1a] text-[#ffff00] text-[10px]">
              {actionStatus}
            </div>
          )}
          
          {/* Embedding Actions */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">EMBEDDINGS</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowModelBrowser(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                BROWSE MODELS ({availableModels.length})
              </button>
              <button
                onClick={() => setShowEmbeddingModal(true)}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                CONFIGURE EMBEDDINGS
              </button>
              <button
                onClick={() => handleGenerateEmbedding('EURUSD')}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                GENERATE EMBEDDING
              </button>
            </div>
          </div>

          {/* Regime Actions */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">REGIME DETECTION</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowRegimeModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                CONFIGURE REGIMES
              </button>
              <button
                onClick={() => setShowTrainModal(true)}
                className="w-full py-2 px-3 border border-[#00ff00] text-[10px] text-[#00ff00] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                TRAIN MODEL
              </button>
              <button
                onClick={() => fetchRegimeData('EURUSD')}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                REFRESH REGIMES
              </button>
            </div>
          </div>

          {/* Graph Analytics Actions */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">GRAPH ANALYTICS</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowGraphModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                CONFIGURE GRAPH
              </button>
              <button
                onClick={() => handleRunGraphAnalysis('asset_correlations')}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                ANALYZE CORRELATIONS
              </button>
              <button
                onClick={() => handleRunGraphAnalysis('regime_transitions')}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                ANALYZE TRANSITIONS
              </button>
              <button
                onClick={() => fetchGraphFeatures('EURUSD')}
                className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#00ff00] hover:border-[#00ff00] transition-colors"
              >
                REFRESH FEATURES
              </button>
            </div>
          </div>

          {/* Validation & Audit */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">VALIDATION</div>
            <div className="space-y-2">
              <button className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors">
                VALIDATE CONFIG
              </button>
              <button className="w-full py-2 px-3 border border-[#444] text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors">
                VIEW AUDIT LOG
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Embedding Configuration Modal */}
      {showEmbeddingModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">EMBEDDING CONFIGURATIONS</h2>
              <button
                onClick={() => {
                  setShowEmbeddingModal(false);
                  setEditingEmbedding(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Embedding List */}
            <div className="space-y-2 mb-4">
              {embeddingConfigs.map(config => (
                <div key={config.assetId} className="border border-[#444] p-3 flex justify-between items-center">
                  <div className="flex-1">
                    <div className="text-[#00ff00]">{config.assetId}</div>
                    <div className="text-[#666] text-[10px]">
                      Model: {config.modelType} | Window: {config.windowSize} | Features: {config.features.join(', ')}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleGenerateEmbedding(config.assetId)}
                      className="px-2 py-1 border border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black text-[10px]"
                    >
                      GENERATE
                    </button>
                    <button
                      onClick={() => setEditingEmbedding(config)}
                      className="px-2 py-1 border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black text-[10px]"
                    >
                      EDIT
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Delete embedding config for ${config.assetId}?`)) {
                          deleteEmbeddingConfig(config.assetId);
                        }
                      }}
                      className="px-2 py-1 border border-[#ff0000] text-[#ff0000] hover:bg-[#ff0000] hover:text-black text-[10px]"
                    >
                      DELETE
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Create/Edit Form */}
            <div className="border border-[#ff8c00] p-4">
              <h3 className="text-[#ff8c00] mb-3 text-[10px]">{editingEmbedding ? 'EDIT EMBEDDING CONFIG' : 'CREATE NEW EMBEDDING CONFIG'}</h3>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.currentTarget);
                  const config: EmbeddingConfig = {
                    assetId: formData.get('assetId') as string,
                    windowSize: parseInt(formData.get('windowSize') as string),
                    features: (formData.get('features') as string).split(',').map(f => f.trim()),
                    modelType: formData.get('modelType') as 'TCN' | 'VAE' | 'LSTM',
                  };
                  
                  if (editingEmbedding) {
                    updateEmbeddingConfig(editingEmbedding.assetId, config);
                    setEditingEmbedding(null);
                  } else {
                    createEmbeddingConfig(config);
                  }
                  e.currentTarget.reset();
                }}
              >
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">ASSET ID</label>
                    <input
                      type="text"
                      name="assetId"
                      defaultValue={editingEmbedding?.assetId}
                      required
                      disabled={!!editingEmbedding}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                      placeholder="e.g., EURUSD"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">WINDOW SIZE</label>
                    <input
                      type="number"
                      name="windowSize"
                      defaultValue={editingEmbedding?.windowSize || 100}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">MODEL TYPE</label>
                    <select
                      name="modelType"
                      defaultValue={editingEmbedding?.modelType || 'TCN'}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    >
                      {availableModels
                        .filter(m => m.category === 'time_series' || m.category === 'representation')
                        .map(model => (
                          <option key={model.id} value={model.id}>
                            {model.name} ({model.category})
                          </option>
                        ))
                      }
                      {/* Fallback options if no models loaded */}
                      {availableModels.length === 0 && (
                        <>
                          <option value="TCN">TCN (Temporal Convolutional Network)</option>
                          <option value="VAE">VAE (Variational Autoencoder)</option>
                          <option value="LSTM">LSTM (Long Short-Term Memory)</option>
                        </>
                      )}
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">FEATURES (comma-separated)</label>
                    <input
                      type="text"
                      name="features"
                      defaultValue={editingEmbedding?.features.join(', ')}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                      placeholder="e.g., price, volume, volatility"
                    />
                  </div>
                </div>
                <div className="flex gap-2 mt-4">
                  <button
                    type="submit"
                    className="px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                  >
                    {editingEmbedding ? 'UPDATE' : 'CREATE'}
                  </button>
                  {editingEmbedding && (
                    <button
                      type="button"
                      onClick={() => setEditingEmbedding(null)}
                      className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                    >
                      CANCEL
                    </button>
                  )}
                </div>
              </form>
            </div>
          </div>
        </div>
      )}


      {/* Regime Configuration Modal */}
      {showRegimeModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">REGIME CONFIGURATIONS</h2>
              <button
                onClick={() => {
                  setShowRegimeModal(false);
                  setEditingRegime(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Regime List */}
            <div className="space-y-2 mb-4">
              {regimeConfigs.map(config => (
                <div key={config.id} className="border border-[#444] p-3 flex justify-between items-center">
                  <div className="flex-1">
                    <div className="text-[#00ff00]">{config.name}</div>
                    <div className="text-[#666] text-[10px]">
                      Vol: {config.volatility} | Trend: {config.trend} | Liq: {config.liquidity} | Duration: {config.minDuration}-{config.maxDuration}min
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setEditingRegime(config)}
                      className="px-2 py-1 border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black text-[10px]"
                    >
                      EDIT
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Delete regime "${config.name}"?`)) {
                          deleteRegimeConfig(config.id);
                        }
                      }}
                      className="px-2 py-1 border border-[#ff0000] text-[#ff0000] hover:bg-[#ff0000] hover:text-black text-[10px]"
                    >
                      DELETE
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Create/Edit Form */}
            <div className="border border-[#ff8c00] p-4">
              <h3 className="text-[#ff8c00] mb-3 text-[10px]">{editingRegime ? 'EDIT REGIME CONFIG' : 'CREATE NEW REGIME CONFIG'}</h3>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.currentTarget);
                  const config = {
                    name: formData.get('name') as string,
                    volatility: formData.get('volatility') as 'LOW' | 'MEDIUM' | 'HIGH',
                    trend: formData.get('trend') as 'TRENDING' | 'RANGING' | 'REVERTING',
                    liquidity: formData.get('liquidity') as 'NORMAL' | 'HIGH' | 'LOW',
                    minDuration: parseInt(formData.get('minDuration') as string),
                    maxDuration: parseInt(formData.get('maxDuration') as string),
                  };
                  
                  if (editingRegime) {
                    updateRegimeConfig(editingRegime.id, config);
                    setEditingRegime(null);
                  } else {
                    createRegimeConfig(config);
                  }
                  e.currentTarget.reset();
                }}
              >
                <div className="grid grid-cols-2 gap-3">
                  <div className="col-span-2">
                    <label className="text-[#666] block mb-1 text-[10px]">REGIME NAME</label>
                    <input
                      type="text"
                      name="name"
                      defaultValue={editingRegime?.name}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                      placeholder="e.g., MEDIUM_VOL_TRENDING"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">VOLATILITY</label>
                    <select
                      name="volatility"
                      defaultValue={editingRegime?.volatility || 'MEDIUM'}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    >
                      <option value="LOW">LOW</option>
                      <option value="MEDIUM">MEDIUM</option>
                      <option value="HIGH">HIGH</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">TREND</label>
                    <select
                      name="trend"
                      defaultValue={editingRegime?.trend || 'TRENDING'}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    >
                      <option value="TRENDING">TRENDING</option>
                      <option value="RANGING">RANGING</option>
                      <option value="REVERTING">REVERTING</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">LIQUIDITY</label>
                    <select
                      name="liquidity"
                      defaultValue={editingRegime?.liquidity || 'NORMAL'}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    >
                      <option value="LOW">LOW</option>
                      <option value="NORMAL">NORMAL</option>
                      <option value="HIGH">HIGH</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">MIN DURATION (minutes)</label>
                    <input
                      type="number"
                      name="minDuration"
                      defaultValue={editingRegime?.minDuration || 30}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">MAX DURATION (minutes)</label>
                    <input
                      type="number"
                      name="maxDuration"
                      defaultValue={editingRegime?.maxDuration || 360}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none text-[10px]"
                    />
                  </div>
                </div>
                <div className="flex gap-2 mt-4">
                  <button
                    type="submit"
                    className="px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                  >
                    {editingRegime ? 'UPDATE' : 'CREATE'}
                  </button>
                  {editingRegime && (
                    <button
                      type="button"
                      onClick={() => setEditingRegime(null)}
                      className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                    >
                      CANCEL
                    </button>
                  )}
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Train Model Modal */}
      {showTrainModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">TRAIN REGIME MODEL</h2>
              <button
                onClick={() => setShowTrainModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div className="border border-[#444] p-4">
                <div className="text-[#666] text-[10px] mb-3">TRAINING CONFIGURATION</div>
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Data Points:</span>
                    <span className="text-[#00ff00]">200 (Mock)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Asset:</span>
                    <span className="text-[#00ff00]">EURUSD</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Model Type:</span>
                    <span className="text-[#00ff00]">Hidden Markov Model</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Regimes:</span>
                    <span className="text-[#00ff00]">{regimeConfigs.length} configured</span>
                  </div>
                </div>
              </div>

              <div className="border border-[#ffff00] bg-[#1a1a1a] p-3 text-[#ffff00] text-[10px]">
                ⚠ Training will use mock historical data. In production, connect to real data sources.
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => {
                    handleTrainRegimeModel();
                    setShowTrainModal(false);
                  }}
                  className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px]"
                >
                  START TRAINING
                </button>
                <button
                  onClick={() => setShowTrainModal(false)}
                  className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CANCEL
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Graph Analytics Modal */}
      {showGraphModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">GRAPH ANALYTICS CONFIGURATION</h2>
              <button
                onClick={() => setShowGraphModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div className="border border-[#444] p-4">
                <div className="text-[#666] text-[10px] mb-3">AVAILABLE ALGORITHMS</div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 border border-[#333] hover:border-[#ff8c00]">
                    <div>
                      <div className="text-[#00ff00] text-[10px]">PageRank</div>
                      <div className="text-[#666] text-[9px]">Centrality measure for asset importance</div>
                    </div>
                    <button
                      onClick={() => {
                        handleRunGraphAnalysis('asset_correlations');
                        setShowGraphModal(false);
                      }}
                      className="px-3 py-1 border border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black text-[10px]"
                    >
                      RUN
                    </button>
                  </div>
                  <div className="flex items-center justify-between p-2 border border-[#333] hover:border-[#ff8c00]">
                    <div>
                      <div className="text-[#00ff00] text-[10px]">Louvain Community Detection</div>
                      <div className="text-[#666] text-[9px]">Find asset clusters and communities</div>
                    </div>
                    <button
                      onClick={() => {
                        handleRunGraphAnalysis('asset_correlations');
                        setShowGraphModal(false);
                      }}
                      className="px-3 py-1 border border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black text-[10px]"
                    >
                      RUN
                    </button>
                  </div>
                  <div className="flex items-center justify-between p-2 border border-[#333] hover:border-[#ff8c00]">
                    <div>
                      <div className="text-[#00ff00] text-[10px]">Betweenness Centrality</div>
                      <div className="text-[#666] text-[9px]">Identify systemic risk bridges</div>
                    </div>
                    <button
                      onClick={() => {
                        handleRunGraphAnalysis('regime_transitions');
                        setShowGraphModal(false);
                      }}
                      className="px-3 py-1 border border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black text-[10px]"
                    >
                      RUN
                    </button>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setShowGraphModal(false)}
                className="w-full px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Browser Modal */}
      {showModelBrowser && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">ML MODEL REGISTRY</h2>
              <button
                onClick={() => {
                  setShowModelBrowser(false);
                  setSelectedModel(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Filters */}
            <div className="flex gap-4 mb-4 flex-wrap">
              <div>
                <label className="text-[#666] text-[10px] block mb-1">CATEGORY</label>
                <select
                  value={selectedCategory || ''}
                  onChange={(e) => setSelectedCategory(e.target.value || null)}
                  className="bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">All Categories</option>
                  {modelCategories.map(cat => (
                    <option key={cat.id} value={cat.id}>{cat.name} ({cat.count})</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[#666] text-[10px] block mb-1">USE CASE</label>
                <select
                  value={selectedUseCase || ''}
                  onChange={(e) => setSelectedUseCase(e.target.value || null)}
                  className="bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">All Use Cases</option>
                  {useCases.map(uc => (
                    <option key={uc.id} value={uc.id}>{uc.name} ({uc.count})</option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-[10px] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={productionOnlyFilter}
                    onChange={(e) => setProductionOnlyFilter(e.target.checked)}
                    className="form-checkbox"
                  />
                  <span className="text-[#666]">Production Ready Only</span>
                </label>
              </div>
            </div>

            {/* Model List and Details */}
            <div className="flex-1 flex gap-4 overflow-hidden">
              {/* Model List */}
              <div className="w-1/2 border border-[#444] overflow-y-auto">
                <div className="space-y-1">
                  {availableModels
                    .filter(m => !selectedCategory || m.category === selectedCategory)
                    .filter(m => !selectedUseCase || m.use_cases.includes(selectedUseCase))
                    .map(model => (
                      <div
                        key={model.id}
                        onClick={() => setSelectedModel(model)}
                        className={`p-3 border-b border-[#222] cursor-pointer transition-colors ${
                          selectedModel?.id === model.id
                            ? 'bg-[#1a1a1a] border-l-4 border-l-[#ff8c00]'
                            : 'hover:bg-[#0f0f0f]'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="text-[#00ff00] text-[11px] font-bold">{model.name}</div>
                            <div className="text-[#666] text-[9px] mt-1">{formatCategory(model.category)}</div>
                            <div className="text-[#fff] text-[10px] mt-1 line-clamp-2">{model.description}</div>
                          </div>
                          {model.production_ready && (
                            <div className="ml-2 px-2 py-1 bg-[#00ff00] text-black text-[8px]">PROD</div>
                          )}
                        </div>
                        <div className="flex gap-2 mt-2 text-[9px]">
                          <span style={{ color: getLatencyColor(model.latency_class) }}>
                            {model.latency_class.toUpperCase()} LATENCY
                          </span>
                          <span className="text-[#666]">|</span>
                          <span style={{ color: getDataReqColor(model.data_requirements) }}>
                            {model.data_requirements.toUpperCase()} DATA
                          </span>
                          {model.gpu_required && (
                            <>
                              <span className="text-[#666]">|</span>
                              <span className="text-[#ffff00]">GPU</span>
                            </>
                          )}
                        </div>
                      </div>
                    ))
                  }
                </div>
              </div>

              {/* Model Details */}
              <div className="w-1/2 border border-[#444] overflow-y-auto p-4">
                {selectedModel ? (
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-[#ff8c00] text-[12px] mb-2">{selectedModel.name}</h3>
                      <p className="text-[#fff] text-[10px]">{selectedModel.description}</p>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">STRENGTHS</div>
                      <ul className="space-y-1">
                        {selectedModel.strengths.map((s, i) => (
                          <li key={i} className="text-[#00ff00] text-[9px]">✓ {s}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">WEAKNESSES</div>
                      <ul className="space-y-1">
                        {selectedModel.weaknesses.map((w, i) => (
                          <li key={i} className="text-[#ff0000] text-[9px]">✗ {w}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">BEST FOR</div>
                      <div className="flex flex-wrap gap-1">
                        {selectedModel.best_for.map((b, i) => (
                          <span key={i} className="px-2 py-1 bg-[#1a1a1a] border border-[#444] text-[#00ff00] text-[9px]">
                            {b}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[#666] text-[9px]">Latency</div>
                        <div style={{ color: getLatencyColor(selectedModel.latency_class) }} className="text-[10px]">
                          {selectedModel.latency_class.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Data Requirements</div>
                        <div style={{ color: getDataReqColor(selectedModel.data_requirements) }} className="text-[10px]">
                          {selectedModel.data_requirements.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Explainability</div>
                        <div style={{ color: getExplainabilityColor(selectedModel.explainability) }} className="text-[10px]">
                          {selectedModel.explainability.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">GPU Required</div>
                        <div className={selectedModel.gpu_required ? 'text-[#ffff00]' : 'text-[#00ff00]'} style={{ fontSize: '10px' }}>
                          {selectedModel.gpu_required ? 'YES' : 'NO'}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Memory</div>
                        <div className="text-[#fff] text-[10px]">{selectedModel.memory_mb} MB</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Min Samples</div>
                        <div className="text-[#fff] text-[10px]">{selectedModel.min_samples.toLocaleString()}</div>
                      </div>
                    </div>

                    {selectedModel.paper_url && (
                      <div>
                        <a
                          href={selectedModel.paper_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[#ff8c00] text-[9px] hover:text-[#ffa500]"
                        >
                          📄 Read Paper →
                        </a>
                      </div>
                    )}

                    <button
                      onClick={() => {
                        // Use this model in embedding config
                        setEditingEmbedding({
                          assetId: 'EURUSD',
                          windowSize: 100,
                          features: ['price', 'volume', 'volatility'],
                          modelType: selectedModel.id as any,
                        });
                        setShowModelBrowser(false);
                        setShowEmbeddingModal(true);
                      }}
                      className="w-full px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                    >
                      USE THIS MODEL
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-[#666] text-[10px]">
                    Select a model to view details
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { create } from 'zustand';
import { 
  getRegimeInference, 
  getGraphFeatures, 
  assembleRLState,
  type RegimeResponse,
  type GraphFeaturesResponse,
  type RLStateResponse 
} from '../../services/intelligenceService';
import { checkIntelligenceHealth, checkExecutionHealth } from '../../services/api';

export type Domain = 'DASH' | 'MKTS' | 'INTL' | 'STRT' | 'PORT' | 'EXEC' | 'SIMU' | 'DATA' | 'SYST' | 'WORK';

export interface Position {
  symbol: string;
  size: number;
  pnl: number;
  exposure: number;
}

export interface Strategy {
  name: string;
  status: 'ACTIVE' | 'PAUSED' | 'STOPPED';
  allocation: number;
  pnl: number;
  sharpe: number;
}

export interface MarketRegime {
  id: string;
  name: string;
  probability: number;
  duration: string;
  volatility: 'LOW' | 'MEDIUM' | 'HIGH';
  trend: 'TRENDING' | 'RANGING' | 'REVERTING';
  liquidity: 'NORMAL' | 'HIGH' | 'LOW';
}

export interface Embedding {
  asset: string;
  timestamp: string;
  confidence: number;
}

export interface IntelligenceSignal {
  type: string;
  timestamp: string;
  description: string;
}

interface TradingState {
  currentDomain: Domain;
  setCurrentDomain: (domain: Domain) => void;
  
  // System status
  systemStatus: 'OPERATIONAL' | 'DEGRADED' | 'DOWN';
  netPnl: number;
  dailyPnlPercent: number;
  riskUtilization: number;
  maxRisk: number;
  activeStrategies: number;
  totalStrategies: number;
  
  // Positions
  positions: Position[];
  
  // Strategies
  strategies: Strategy[];
  
  // Intelligence
  currentRegime: MarketRegime | null;
  regimes: MarketRegime[];
  embeddings: Embedding[];
  intelligenceSignals: IntelligenceSignal[];
  selectedRegime: string | null;
  setSelectedRegime: (regimeId: string | null) => void;
  
  // Connection status
  connectionStatus: 'LIVE' | 'DELAYED' | 'DISCONNECTED';
  latency: number;
  executionMode: 'LIVE' | 'SHADOW' | 'SIMULATION';
  
  // Backend integration
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
  
  // Actions
  fetchRegimeData: (assetId: string) => Promise<void>;
  fetchGraphFeatures: (assetId: string) => Promise<void>;
  fetchRLState: (assetIds: string[], strategyIds?: string[]) => Promise<void>;
  checkHealth: () => Promise<void>;
  clearError: () => void;
}

export const useTradingStore = create<TradingState>((set, get) => ({
  currentDomain: 'DASH',
  setCurrentDomain: (domain) => set({ currentDomain: domain }),
  
  systemStatus: 'OPERATIONAL',
  netPnl: 4127.89,
  dailyPnlPercent: 2.34,
  riskUtilization: 47.3,
  maxRisk: 100,
  activeStrategies: 3,
  totalStrategies: 4,
  
  positions: [
    { symbol: 'EUR/USD', size: 250000, pnl: 1234.56, exposure: 12.5 },
    { symbol: 'GBP/USD', size: 180000, pnl: -456.78, exposure: 9.0 },
    { symbol: 'USD/JPY', size: 320000, pnl: 789.12, exposure: 16.0 },
    { symbol: 'AUD/USD', size: 150000, pnl: 123.45, exposure: 7.5 },
  ],
  
  strategies: [
    { name: 'MOMENTUM ALPHA', status: 'ACTIVE', allocation: 35, pnl: 2456.78, sharpe: 1.42 },
    { name: 'MEAN REVERSION', status: 'ACTIVE', allocation: 25, pnl: 1234.56, sharpe: 1.18 },
    { name: 'REGIME SWITCH', status: 'PAUSED', allocation: 20, pnl: -234.56, sharpe: 0.89 },
    { name: 'VOLATILITY ARB', status: 'ACTIVE', allocation: 20, pnl: 567.89, sharpe: 1.67 },
  ],
  
  currentRegime: {
    id: 'LOW_VOL_TRENDING',
    name: 'LOW VOLATILITY TRENDING',
    probability: 65,
    duration: '2H 34M',
    volatility: 'LOW',
    trend: 'TRENDING',
    liquidity: 'NORMAL',
  },
  
  regimes: [
    {
      id: 'LOW_VOL_TRENDING',
      name: 'LOW VOLATILITY TRENDING',
      probability: 65,
      duration: '2H 34M',
      volatility: 'LOW',
      trend: 'TRENDING',
      liquidity: 'NORMAL',
    },
    {
      id: 'HIGH_VOL_RANGING',
      name: 'HIGH VOLATILITY RANGING',
      probability: 25,
      duration: '0H 00M',
      volatility: 'HIGH',
      trend: 'RANGING',
      liquidity: 'NORMAL',
    },
    {
      id: 'CRISIS',
      name: 'CRISIS MODE',
      probability: 10,
      duration: '0H 00M',
      volatility: 'HIGH',
      trend: 'REVERTING',
      liquidity: 'LOW',
    },
  ],
  
  embeddings: [
    { asset: 'EURUSD', timestamp: '2024-01-15 14:30', confidence: 0.87 },
    { asset: 'GBPUSD', timestamp: '2024-01-15 14:30', confidence: 0.92 },
    { asset: 'BTCUSD', timestamp: '2024-01-15 14:30', confidence: 0.78 },
  ],
  
  intelligenceSignals: [
    { type: 'REGIME_TRANSITION_ALERT', timestamp: '14:28:45', description: 'Potential regime shift detected' },
    { type: 'CORRELATION_SHIFT', timestamp: '14:15:22', description: 'Asset correlation matrix changed' },
    { type: 'VOLATILITY_SPIKE', timestamp: '13:45:10', description: 'Abnormal volatility detected' },
  ],
  
  selectedRegime: null,
  setSelectedRegime: (regimeId) => set({ selectedRegime: regimeId }),
  
  connectionStatus: 'LIVE',
  latency: 12,
  executionMode: 'SHADOW',
  
  // Backend integration state
  isLoading: false,
  error: null,
  lastUpdate: null,
  
  // Actions
  fetchRegimeData: async (assetId: string) => {
    set({ isLoading: true, error: null });
    try {
      const regimeData = await getRegimeInference(assetId);
      
      // Convert backend response to frontend format
      const regimes: MarketRegime[] = Object.entries(regimeData.regime_probabilities).map(([key, prob]) => ({
        id: key.toUpperCase(),
        name: key.toUpperCase().replace(/_/g, ' '),
        probability: Math.round(prob * 100),
        duration: '0H 00M', // TODO: Get from backend
        volatility: key.includes('high_vol') ? 'HIGH' : key.includes('low_vol') ? 'LOW' : 'MEDIUM',
        trend: key.includes('trending') ? 'TRENDING' : key.includes('ranging') ? 'RANGING' : 'REVERTING',
        liquidity: 'NORMAL',
      }));
      
      // Sort by probability
      regimes.sort((a, b) => b.probability - a.probability);
      
      set({ 
        regimes,
        currentRegime: regimes[0] || null,
        isLoading: false,
        lastUpdate: new Date(),
      });
    } catch (error: any) {
      console.error('Failed to fetch regime data:', error);
      set({ 
        error: error.message || 'Failed to fetch regime data',
        isLoading: false,
      });
    }
  },
  
  fetchGraphFeatures: async (assetId: string) => {
    set({ isLoading: true, error: null });
    try {
      const graphData = await getGraphFeatures(assetId);
      
      // Update intelligence signals with graph features
      const signals = get().intelligenceSignals;
      const newSignal: IntelligenceSignal = {
        type: 'GRAPH_FEATURES_UPDATE',
        timestamp: new Date().toLocaleTimeString(),
        description: `Cluster: ${graphData.cluster_membership || 'N/A'}, Centrality: ${Object.values(graphData.centrality_metrics)[0]?.toFixed(2) || 'N/A'}`,
      };
      
      set({ 
        intelligenceSignals: [newSignal, ...signals.slice(0, 9)],
        isLoading: false,
        lastUpdate: new Date(),
      });
    } catch (error: any) {
      console.error('Failed to fetch graph features:', error);
      set({ 
        error: error.message || 'Failed to fetch graph features',
        isLoading: false,
      });
    }
  },
  
  fetchRLState: async (assetIds: string[], strategyIds: string[] = []) => {
    set({ isLoading: true, error: null });
    try {
      const rlState = await assembleRLState(assetIds, strategyIds);
      
      // Update regime information from RL state
      if (rlState.composite_state.current_regime_label) {
        const currentRegime: MarketRegime = {
          id: rlState.composite_state.current_regime_label.toUpperCase(),
          name: rlState.composite_state.current_regime_label.toUpperCase().replace(/_/g, ' '),
          probability: Math.round((rlState.composite_state.regime_confidence || 0) * 100),
          duration: '0H 00M',
          volatility: 'MEDIUM',
          trend: 'TRENDING',
          liquidity: 'NORMAL',
        };
        
        set({ 
          currentRegime,
          isLoading: false,
          lastUpdate: new Date(),
        });
      }
    } catch (error: any) {
      console.error('Failed to fetch RL state:', error);
      set({ 
        error: error.message || 'Failed to fetch RL state',
        isLoading: false,
      });
    }
  },
  
  checkHealth: async () => {
    try {
      const [intelligenceHealth, executionHealth] = await Promise.all([
        checkIntelligenceHealth().catch(() => null),
        checkExecutionHealth().catch(() => null),
      ]);
      
      if (intelligenceHealth && executionHealth) {
        set({ 
          systemStatus: 'OPERATIONAL',
          connectionStatus: 'LIVE',
        });
      } else if (intelligenceHealth || executionHealth) {
        set({ 
          systemStatus: 'DEGRADED',
          connectionStatus: 'DELAYED',
        });
      } else {
        set({ 
          systemStatus: 'DOWN',
          connectionStatus: 'DISCONNECTED',
        });
      }
    } catch (error) {
      console.error('Health check failed:', error);
      set({ 
        systemStatus: 'DOWN',
        connectionStatus: 'DISCONNECTED',
      });
    }
  },
  
  clearError: () => set({ error: null }),
}));

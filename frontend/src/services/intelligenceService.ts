/**
 * Intelligence Layer API Service
 */

import { intelligenceApi } from './api';

export interface RegimeResponse {
  regime_probabilities: Record<string, number>;
  transition_likelihoods: Record<string, number>;
  regime_entropy: number;
  confidence: number;
  timestamp: string;
}

export interface GraphFeaturesResponse {
  cluster_membership: string | null;
  centrality_metrics: Record<string, number>;
  systemic_risk_proxies: Record<string, number>;
  timestamp: string;
}

export interface IntelligenceState {
  embedding_similarity_context: any[];
  current_regime_label: string | null;
  regime_transition_probabilities: Record<string, number>;
  regime_confidence: number | null;
  graph_structural_features: any | null;
  confidence_scores: Record<string, number>;
  timestamp: string;
  version: string;
}

export interface RLStateResponse {
  composite_state: IntelligenceState;
  state_components: Record<string, any>;
  assembly_metadata: Record<string, any>;
  timestamp: string;
}

export interface MarketData {
  timestamp: string;
  asset_id: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  metadata?: Record<string, any>;
}

/**
 * Get regime inference for an asset
 */
export async function getRegimeInference(assetId: string): Promise<RegimeResponse> {
  return intelligenceApi.get(`/intelligence/regime?asset_id=${assetId}`);
}

/**
 * Infer regime from market data
 */
export async function inferRegime(marketData: MarketData[]): Promise<RegimeResponse> {
  return intelligenceApi.post('/intelligence/regime', marketData);
}

/**
 * Train regime model with historical data
 */
export async function trainRegimeModel(historicalData: MarketData[]): Promise<{
  status: string;
  data_points: number;
  regime_definitions: any;
  timestamp: string;
}> {
  return intelligenceApi.post('/intelligence/regime/train', historicalData);
}

/**
 * Get graph features for an asset
 */
export async function getGraphFeatures(assetId: string): Promise<GraphFeaturesResponse> {
  return intelligenceApi.get(`/intelligence/graph-features?asset_id=${assetId}`);
}

/**
 * Run graph analysis
 */
export async function runGraphAnalysis(analysisType: 'asset_correlations' | 'regime_transitions'): Promise<{
  analysis_type: string;
  status: string;
  timestamp: string;
  algorithms_executed: string[];
}> {
  return intelligenceApi.post(`/intelligence/graph/analyze?analysis_type=${analysisType}`);
}

/**
 * Assemble RL state for strategy orchestration
 */
export async function assembleRLState(
  assetIds: string[],
  strategyIds: string[] = []
): Promise<RLStateResponse> {
  const assetIdsParam = assetIds.join(',');
  const strategyIdsParam = strategyIds.join(',');
  return intelligenceApi.get(
    `/intelligence/state?asset_ids=${assetIdsParam}&strategy_ids=${strategyIdsParam}`
  );
}

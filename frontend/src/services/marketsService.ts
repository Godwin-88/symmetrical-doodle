/**
 * Markets API Service
 */

import { intelligenceApi } from './api';

export interface LiveMarketData {
  asset_id: string;
  timestamp: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  spread_bps: number;
  depth_bid: number;
  depth_ask: number;
  tick_frequency: number;
  order_book?: {
    bids: [number, number][];
    asks: [number, number][];
  };
}

export interface CorrelationData {
  timestamp: string;
  assets: string[];
  correlation_matrix: number[][];
  significance: number[][];
  window: string;
  method: string;
  clusters: string[][];
}

export interface MicrostructureData {
  asset_id: string;
  timestamp: string;
  spread_bps: number;
  effective_spread_bps: number;
  quoted_spread_bps: number;
  depth_bid: number;
  depth_ask: number;
  imbalance_ratio: number;
  tick_frequency: number;
  price_impact_bps: number;
}

export interface LiquidityData {
  asset_id: string;
  timestamp: string;
  bid_liquidity_usd: number;
  ask_liquidity_usd: number;
  total_liquidity_usd: number;
  liquidity_score: number;
  resilience_score: number;
  toxicity_score: number;
}

export interface MarketEvent {
  timestamp: string;
  asset_id: string;
  event_type: string;
  severity: number;
  description: string;
  recommended_action: string;
}

/**
 * Get live market data for assets
 */
export async function getLiveMarketData(
  assets: string[],
  includeDepth: boolean = false
): Promise<{ data: LiveMarketData[]; timestamp: string }> {
  return intelligenceApi.get(
    `/markets/live-data?assets=${assets.join(',')}&include_depth=${includeDepth}`
  );
}

/**
 * Get correlation matrix for assets
 */
export async function getCorrelationMatrix(
  assets: string[],
  window: string = '24H',
  method: string = 'pearson'
): Promise<CorrelationData> {
  return intelligenceApi.get(
    `/markets/correlations?assets=${assets.join(',')}&window=${window}&method=${method}`
  );
}

/**
 * Get microstructure metrics for an asset
 */
export async function getMicrostructure(
  assetId: string
): Promise<MicrostructureData> {
  return intelligenceApi.get(`/markets/microstructure?asset_id=${assetId}`);
}

/**
 * Get liquidity analysis for assets
 */
export async function getLiquidityAnalysis(
  assets: string[]
): Promise<{ data: LiquidityData[]; timestamp: string }> {
  return intelligenceApi.get(`/markets/liquidity?assets=${assets.join(',')}`);
}

/**
 * Get recent market events
 */
export async function getMarketEvents(
  since?: string,
  eventTypes?: string[],
  severityMin: number = 0.5
): Promise<MarketEvent[]> {
  const params = new URLSearchParams();
  if (since) params.append('since', since);
  if (eventTypes) params.append('event_types', eventTypes.join(','));
  params.append('severity_min', severityMin.toString());
  
  return intelligenceApi.get(`/markets/events?${params.toString()}`);
}

/**
 * Connect to market data WebSocket stream
 */
export function connectMarketStream(
  assets: string[],
  onData: (data: LiveMarketData) => void,
  onEvent: (event: MarketEvent) => void,
  onError?: (error: Error) => void
): WebSocket {
  const wsUrl = `ws://localhost:8000/markets/stream?assets=${assets.join(',')}`;
  const ws = new WebSocket(wsUrl);
  
  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === 'tick') {
        onData(message.data);
      } else if (message.type === 'event') {
        onEvent(message.data);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      if (onError) onError(error as Error);
    }
  };
  
  ws.onerror = (event) => {
    console.error('WebSocket error:', event);
    if (onError) onError(new Error('WebSocket connection error'));
  };
  
  ws.onclose = () => {
    console.log('WebSocket connection closed');
  };
  
  return ws;
}

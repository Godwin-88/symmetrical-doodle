/**
 * Data Import Service for external data sources
 */

import { intelligenceApi } from './api';

export interface SymbolSearchResult {
  symbol: string;
  name: string;
  type: string;
  exchange: string;
}

export interface SymbolInfo {
  symbol: string;
  name: string;
  type: string;
  exchange: string;
  currency?: string;
  market_cap?: number;
  sector?: string;
  industry?: string;
  description?: string;
}

export interface ImportedDataPoint {
  timestamp: string;
  asset_id: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ImportDataResponse {
  symbol: string;
  source: string;
  interval: string;
  data_points: number;
  start_date: string;
  end_date: string;
  data: ImportedDataPoint[];
}

/**
 * Search for symbols across data sources
 */
export async function searchSymbols(
  query: string,
  source: string = 'yahoo_finance',
  limit: number = 10
): Promise<SymbolSearchResult[]> {
  return intelligenceApi.get(
    `/data/search?query=${encodeURIComponent(query)}&source=${source}&limit=${limit}`
  );
}

/**
 * Get detailed information about a symbol
 */
export async function getSymbolInfo(
  symbol: string,
  source: string = 'yahoo_finance'
): Promise<SymbolInfo> {
  return intelligenceApi.get(
    `/data/symbol-info?symbol=${encodeURIComponent(symbol)}&source=${source}`
  );
}

/**
 * Import market data from external source
 */
export async function importExternalData(
  symbol: string,
  source: string = 'yahoo_finance',
  startDate?: string,
  endDate?: string,
  interval: string = '1d'
): Promise<ImportDataResponse> {
  const params = new URLSearchParams({
    symbol,
    source,
    interval,
  });
  
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  
  return intelligenceApi.post(`/data/import?${params.toString()}`);
}

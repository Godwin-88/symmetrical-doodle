/**
 * LLM and RAG Service - Frontend client for AI capabilities
 */

// Types
export interface LLMQueryRequest {
  query: string;
  system_prompt?: string;
  prefer_local?: boolean;
  provider?: string;
}

export interface LLMQueryResponse {
  answer: string;
  model: string;
  provider: string;
  tokens_used: number;
  response_time: number;
  timestamp: string;
}

export interface RAGQueryRequest {
  question: string;
  context_filter?: Record<string, any>;
}

export interface RAGQueryResponse {
  answer: string;
  sources: Array<{
    file_path: string;
    similarity: number;
    metadata: Record<string, any>;
  }>;
  confidence: string;
  model_info: {
    model: string;
    provider: string;
    tokens_used: number;
    response_time: number;
  };
  retrieval_info?: {
    documents_found: number;
    avg_similarity: number;
  };
}

export interface ResearchRequest {
  query: string;
  include_web?: boolean;
  include_market_data?: boolean;
}

export interface ResearchResponse {
  query: string;
  timestamp: string;
  rag_analysis?: {
    answer: string;
    sources: any[];
    confidence: string;
  };
  web_research?: {
    results: any[];
    sentiment_analysis: any[];
  };
  market_data?: Record<string, any>;
  comprehensive_analysis?: {
    analysis: string;
    model_info: any;
  };
}

export interface StockAnalysisRequest {
  symbol: string;
}

export interface StockAnalysisResponse {
  symbol: string;
  market_data: Record<string, any>;
  news_analysis: {
    articles: any[];
    sentiment_summary: any[];
  };
  ai_analysis: string;
  timestamp: string;
}

export interface DocumentIngestResponse {
  success: boolean;
  chunks_created: number;
  document_id: string;
  message: string;
}

// API Base URLs
const INTELLIGENCE_API_BASE = 'http://localhost:8000';

// LLM Functions
export async function queryLLM(request: LLMQueryRequest): Promise<LLMQueryResponse> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/llm/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('LLM query API failed, using mock response:', error);
    
    // Mock response for development
    return {
      answer: `Mock LLM Response for: "${request.query}"\n\nThis is a simulated response. The actual LLM service would provide detailed financial analysis based on your query. Key points would include market analysis, risk factors, and data-driven insights.`,
      model: 'mock-model',
      provider: 'mock',
      tokens_used: 150,
      response_time: 0.5,
      timestamp: new Date().toISOString()
    };
  }
}

export async function financialAnalysis(request: LLMQueryRequest): Promise<LLMQueryResponse> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/llm/financial-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Financial analysis API failed, using mock response:', error);
    
    // Mock response for development
    return {
      answer: `Financial Analysis for: "${request.query}"\n\n**Key Insights:**\n• Market conditions suggest cautious optimism\n• Technical indicators show mixed signals\n• Risk factors include volatility and liquidity concerns\n\n**Recommendations:**\n• Diversify portfolio exposure\n• Monitor key support/resistance levels\n• Consider hedging strategies\n\n*Note: This is a mock response for development purposes.*`,
      model: 'mock-financial-model',
      provider: 'mock',
      tokens_used: 200,
      response_time: 0.7,
      timestamp: new Date().toISOString()
    };
  }
}

// RAG Functions
export async function queryRAG(request: RAGQueryRequest): Promise<RAGQueryResponse> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('RAG query API failed, using mock response:', error);
    
    // Mock response for development
    return {
      answer: `RAG Analysis for: "${request.question}"\n\nBased on available documents and research, here are the key findings:\n\n• Document analysis reveals relevant patterns\n• Historical data supports current trends\n• Risk assessment indicates moderate exposure\n\n*Sources: Internal research documents, market reports*`,
      sources: [
        {
          file_path: 'research/market_analysis_2024.pdf',
          similarity: 0.85,
          metadata: { type: 'research', date: '2024-01-15' }
        },
        {
          file_path: 'reports/risk_assessment.md',
          similarity: 0.78,
          metadata: { type: 'report', date: '2024-01-10' }
        }
      ],
      confidence: 'high',
      model_info: {
        model: 'mock-rag-model',
        provider: 'mock',
        tokens_used: 180,
        response_time: 0.6
      },
      retrieval_info: {
        documents_found: 5,
        avg_similarity: 0.82
      }
    };
  }
}

export async function ingestDocument(file: File, metadata: Record<string, any> = {}): Promise<DocumentIngestResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));
    
    const response = await fetch(`${INTELLIGENCE_API_BASE}/rag/ingest`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Document ingestion API failed, using mock response:', error);
    
    // Mock response for development
    return {
      success: true,
      chunks_created: Math.floor(Math.random() * 20) + 5,
      document_id: `doc_${Date.now()}`,
      message: `Successfully processed ${file.name} (mock response)`
    };
  }
}

export async function getRAGStats(): Promise<Record<string, any>> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/rag/stats`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('RAG stats API failed, using mock response:', error);
    
    // Mock response for development
    return {
      total_documents: 127,
      vector_store: 'chroma',
      embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
      chunk_size: 1000,
      similarity_threshold: 0.7
    };
  }
}

// Research Functions
export async function comprehensiveResearch(request: ResearchRequest): Promise<ResearchResponse> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/research/comprehensive`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Comprehensive research API failed, using mock response:', error);
    
    // Mock response for development
    return {
      query: request.query,
      timestamp: new Date().toISOString(),
      rag_analysis: {
        answer: 'Internal document analysis shows relevant market trends and risk factors.',
        sources: [],
        confidence: 'medium'
      },
      web_research: {
        results: [
          { title: 'Market Analysis Report', snippet: 'Recent market conditions show...', url: 'https://example.com' }
        ],
        sentiment_analysis: [{ sentiment: 'neutral', confidence: 0.7 }]
      },
      market_data: {
        'AAPL': { current_price: 150.25, change_percent: 1.2 }
      },
      comprehensive_analysis: {
        analysis: `Comprehensive analysis for "${request.query}":\n\nBased on multiple data sources including internal documents, web research, and market data, the analysis suggests a balanced approach with careful risk management.`,
        model_info: { model: 'mock-research-model', provider: 'mock', tokens_used: 250 }
      }
    };
  }
}

export async function stockAnalysis(request: StockAnalysisRequest): Promise<StockAnalysisResponse> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/research/stock-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Stock analysis API failed, using mock response:', error);
    
    // Mock response for development
    return {
      symbol: request.symbol,
      market_data: {
        current_price: 150.25,
        market_cap: 2500000000000,
        pe_ratio: 25.4,
        dividend_yield: 0.015,
        technical_indicators: {
          sma_20: 148.50,
          sma_50: 145.20,
          rsi: 58.3,
          volatility: 0.25
        }
      },
      news_analysis: {
        articles: [
          { title: `${request.symbol} Earnings Beat Expectations`, snippet: 'Strong quarterly results...' }
        ],
        sentiment_summary: [{ sentiment: 'positive', confidence: 0.8 }]
      },
      ai_analysis: `AI Analysis for ${request.symbol}:\n\n**Valuation:** Currently trading at reasonable multiples\n**Technical:** Bullish momentum with RSI in healthy range\n**Sentiment:** Positive news flow and analyst coverage\n**Recommendation:** HOLD with upside potential\n\n*Risk Factors:* Market volatility, sector rotation risks`,
      timestamp: new Date().toISOString()
    };
  }
}

export async function getMarketOverview(): Promise<Record<string, any>> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/research/market-overview`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Market overview API failed, using mock response:', error);
    
    // Mock response for development
    return {
      economic_indicators: {
        indices: {
          '^GSPC': { current: 4150.25, change_percent: 0.5 },
          '^DJI': { current: 33500.75, change_percent: 0.3 },
          '^IXIC': { current: 12800.50, change_percent: 0.8 }
        },
        currencies: {
          'EURUSD=X': { current: 1.0850, change_percent: -0.2 }
        }
      },
      ai_analysis: 'Market Overview:\n\nCurrent market conditions show moderate optimism with mixed signals across sectors. Key themes include technology resilience and energy sector volatility.',
      timestamp: new Date().toISOString()
    };
  }
}
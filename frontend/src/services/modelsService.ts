/**
 * ML Models API Service
 */

import { intelligenceApi } from './api';

export interface ModelSpec {
  id: string;
  name: string;
  category: string;
  use_cases: string[];
  description: string;
  strengths: string[];
  weaknesses: string[];
  best_for: string[];
  production_ready: boolean;
  latency_class: 'low' | 'medium' | 'high';
  data_requirements: 'small' | 'medium' | 'large';
  explainability: 'low' | 'medium' | 'high';
  hyperparameters: Record<string, any>;
  dependencies?: string[];
  min_samples: number;
  recommended_samples: number;
  supports_online_learning: boolean;
  supports_transfer_learning: boolean;
  gpu_required: boolean;
  memory_mb: number;
  paper_url?: string;
  implementation_url?: string;
}

export interface ModelCategory {
  id: string;
  name: string;
  count: number;
}

export interface UseCase {
  id: string;
  name: string;
  count: number;
}

export interface ModelRecommendation {
  id: string;
  name: string;
  category: string;
  description: string;
  strengths: string[];
  best_for: string[];
  production_ready: boolean;
  latency_class: string;
  data_requirements: string;
  explainability: string;
  gpu_required: boolean;
}

// ============================================================================
// HARDCODED MODEL DATA (Fallback when backend is unavailable)
// ============================================================================

const HARDCODED_MODELS: ModelSpec[] = [
  // TIME-SERIES MODELS
  {
    id: 'tft',
    name: 'Temporal Fusion Transformer',
    category: 'time_series',
    use_cases: ['price_forecasting', 'volatility_forecasting'],
    description: 'State-of-the-art multi-horizon forecasting with attention mechanisms',
    strengths: [
      'Handles long-range dependencies',
      'Attention provides interpretability',
      'Scales across thousands of instruments',
      'Multi-horizon forecasting',
      'Regime-aware prediction'
    ],
    weaknesses: [
      'Data-hungry (needs large datasets)',
      'Overfits noisy price series',
      'Expensive inference for low-latency trading'
    ],
    best_for: [
      'Portfolio forecasting',
      'Risk modeling',
      'Volatility surfaces',
      'Factor return forecasting'
    ],
    production_ready: true,
    latency_class: 'high',
    data_requirements: 'large',
    explainability: 'medium',
    hyperparameters: {
      hidden_size: 128,
      num_attention_heads: 4,
      dropout: 0.1,
      learning_rate: 0.001,
      batch_size: 64
    },
    min_samples: 10000,
    recommended_samples: 100000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: true,
    memory_mb: 2048,
    paper_url: 'https://arxiv.org/abs/1912.09363'
  },
  {
    id: 'informer',
    name: 'Informer',
    category: 'time_series',
    use_cases: ['price_forecasting', 'volatility_forecasting'],
    description: 'Efficient transformer for long sequence time-series forecasting',
    strengths: [
      'Efficient for long sequences',
      'Lower memory footprint than TFT',
      'Good for macro forecasting'
    ],
    weaknesses: [
      'Less interpretable than TFT',
      'Requires careful tuning'
    ],
    best_for: [
      'Long-horizon forecasting',
      'Macro indicators',
      'Multi-asset prediction'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'large',
    explainability: 'low',
    hyperparameters: {
      d_model: 512,
      n_heads: 8,
      dropout: 0.05
    },
    min_samples: 5000,
    recommended_samples: 50000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: true,
    memory_mb: 1536
  },
  {
    id: 'patchtst',
    name: 'PatchTST',
    category: 'time_series',
    use_cases: ['price_forecasting', 'volatility_forecasting'],
    description: 'Patch-based transformer for efficient time series forecasting',
    strengths: [
      'Very efficient',
      'Good performance with less data',
      'Fast inference'
    ],
    weaknesses: [
      'Newer model, less battle-tested',
      'Limited interpretability'
    ],
    best_for: [
      'Medium-frequency trading',
      'Resource-constrained environments',
      'Multi-variate forecasting'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'medium',
    explainability: 'low',
    hyperparameters: {
      patch_len: 16,
      stride: 8,
      d_model: 128
    },
    min_samples: 2000,
    recommended_samples: 20000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 512
  },
  {
    id: 'lstm',
    name: 'LSTM',
    category: 'time_series',
    use_cases: ['price_forecasting', 'credit_risk', 'fraud_detection'],
    description: 'Classic recurrent network, still relevant for many tasks',
    strengths: [
      'Stable training',
      'Low inference cost',
      'Works well with engineered features',
      'Battle-tested in production'
    ],
    weaknesses: [
      'Weak regime adaptation',
      'Poor long-range memory',
      'Sequential processing limits parallelization'
    ],
    best_for: [
      'Medium-frequency signals',
      'Small to mid-sized datasets',
      'Latency-sensitive systems',
      'Credit risk'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'small',
    explainability: 'low',
    hyperparameters: {
      hidden_size: 128,
      num_layers: 2,
      dropout: 0.2
    },
    min_samples: 500,
    recommended_samples: 5000,
    supports_online_learning: true,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 256
  },
  {
    id: 'gru',
    name: 'GRU',
    category: 'time_series',
    use_cases: ['price_forecasting', 'fraud_detection'],
    description: 'Simplified LSTM variant with fewer parameters',
    strengths: [
      'Faster training than LSTM',
      'Lower memory footprint',
      'Good for real-time systems'
    ],
    weaknesses: [
      'Similar limitations to LSTM',
      'Less expressive than LSTM'
    ],
    best_for: [
      'Intraday signal generation',
      'Real-time fraud detection',
      'Resource-constrained deployment'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'small',
    explainability: 'low',
    hyperparameters: {
      hidden_size: 128,
      num_layers: 2,
      dropout: 0.2
    },
    min_samples: 500,
    recommended_samples: 5000,
    supports_online_learning: true,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 200
  },
  
  // REPRESENTATION MODELS
  {
    id: 'vae',
    name: 'Variational Autoencoder',
    category: 'representation',
    use_cases: ['regime_detection', 'anomaly_detection'],
    description: 'Probabilistic autoencoder for latent representation learning',
    strengths: [
      'Latent factor discovery',
      'Noise reduction',
      'Regime compression',
      'Generates new samples',
      'Uncertainty quantification'
    ],
    weaknesses: [
      'Training can be unstable',
      'Requires careful architecture design',
      'Latent space interpretation not always clear'
    ],
    best_for: [
      'Alpha factor construction',
      'Anomaly detection',
      'Market state embeddings',
      'Regime clustering'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'medium',
    explainability: 'medium',
    hyperparameters: {
      latent_dim: 32,
      beta: 1.0,
      learning_rate: 0.001
    },
    min_samples: 1000,
    recommended_samples: 10000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 512
  },
  {
    id: 'denoising_ae',
    name: 'Denoising Autoencoder',
    category: 'representation',
    use_cases: ['anomaly_detection'],
    description: 'Autoencoder trained to reconstruct clean data from noisy input',
    strengths: [
      'Robust to noise',
      'Good for feature extraction',
      'Simple and stable training'
    ],
    weaknesses: [
      'No probabilistic interpretation',
      'Requires noise model design'
    ],
    best_for: [
      'Noisy market data',
      'Feature denoising',
      'Outlier detection'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'medium',
    explainability: 'low',
    hyperparameters: {
      noise_factor: 0.2,
      learning_rate: 0.001
    },
    min_samples: 1000,
    recommended_samples: 10000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 384
  },
  {
    id: 'contrastive',
    name: 'Contrastive Learning',
    category: 'representation',
    use_cases: ['regime_detection'],
    description: 'Self-supervised learning via contrastive objectives',
    strengths: [
      'Works with unlabeled data',
      'Learns robust representations',
      'Good for regime separation'
    ],
    weaknesses: [
      'Requires careful augmentation design',
      'Computationally expensive',
      'Needs large batch sizes'
    ],
    best_for: [
      'Label-scarce scenarios',
      'Regime clustering',
      'Instrument similarity'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'large',
    explainability: 'low',
    hyperparameters: {
      projection_dim: 128,
      temperature: 0.5,
      batch_size: 256
    },
    min_samples: 5000,
    recommended_samples: 50000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: true,
    memory_mb: 1024
  },
  
  // GRAPH MODELS
  {
    id: 'gcn',
    name: 'Graph Convolutional Network',
    category: 'graph',
    use_cases: ['cross_asset_effects', 'regime_detection'],
    description: 'Convolutional operations on graph-structured data',
    strengths: [
      'Captures network effects',
      'Spillover risk modeling',
      'Cross-asset reasoning',
      'Interpretable via attention'
    ],
    weaknesses: [
      'Requires graph construction',
      'Sensitive to graph quality',
      'Over-smoothing in deep networks'
    ],
    best_for: [
      'Correlation networks',
      'Sector analysis',
      'Contagion modeling',
      'Supply chain risk'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'medium',
    explainability: 'medium',
    hyperparameters: {
      hidden_channels: 64,
      num_layers: 3,
      dropout: 0.5
    },
    min_samples: 500,
    recommended_samples: 5000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 512
  },
  {
    id: 'gat',
    name: 'Graph Attention Network',
    category: 'graph',
    use_cases: ['cross_asset_effects'],
    description: 'GNN with attention mechanisms for adaptive aggregation',
    strengths: [
      'Learns edge importance',
      'More expressive than GCN',
      'Better interpretability'
    ],
    weaknesses: [
      'More parameters than GCN',
      'Slower training'
    ],
    best_for: [
      'Dynamic correlation networks',
      'Heterogeneous graphs',
      'Systemic risk'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'medium',
    explainability: 'high',
    hyperparameters: {
      hidden_channels: 64,
      num_heads: 4,
      num_layers: 2
    },
    min_samples: 500,
    recommended_samples: 5000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 768
  },
  {
    id: 'temporal_gnn',
    name: 'Temporal Graph Neural Network',
    category: 'graph',
    use_cases: ['cross_asset_effects', 'regime_detection'],
    description: 'GNN for time-evolving graphs',
    strengths: [
      'Handles dynamic networks',
      'Temporal + spatial reasoning',
      'Regime transition modeling'
    ],
    weaknesses: [
      'Complex architecture',
      'High computational cost',
      'Requires temporal graph data'
    ],
    best_for: [
      'Time-varying correlations',
      'Regime transitions',
      'Flow-of-funds analysis'
    ],
    production_ready: false,
    latency_class: 'high',
    data_requirements: 'large',
    explainability: 'medium',
    hyperparameters: {
      hidden_channels: 64,
      num_layers: 3,
      temporal_window: 10
    },
    min_samples: 2000,
    recommended_samples: 20000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: true,
    memory_mb: 1536
  },
  
  // REINFORCEMENT LEARNING
  {
    id: 'ppo',
    name: 'Proximal Policy Optimization',
    category: 'reinforcement',
    use_cases: ['trading_execution', 'portfolio_allocation'],
    description: 'Stable policy gradient method for continuous control',
    strengths: [
      'Stable training',
      'Good sample efficiency',
      'Works well with continuous actions',
      'Industry standard'
    ],
    weaknesses: [
      'Reward hacking risk',
      'Non-stationarity challenges',
      'Simulation bias'
    ],
    best_for: [
      'Execution optimization',
      'Market making',
      'Dynamic allocation'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'large',
    explainability: 'low',
    hyperparameters: {
      learning_rate: 0.0003,
      gamma: 0.99,
      clip_range: 0.2
    },
    min_samples: 10000,
    recommended_samples: 100000,
    supports_online_learning: true,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 512
  },
  {
    id: 'sac',
    name: 'Soft Actor-Critic',
    category: 'reinforcement',
    use_cases: ['trading_execution'],
    description: 'Off-policy RL with maximum entropy objective',
    strengths: [
      'Sample efficient',
      'Stable training',
      'Exploration via entropy',
      'Off-policy learning'
    ],
    weaknesses: [
      'Hyperparameter sensitive',
      'Requires replay buffer'
    ],
    best_for: [
      'Market making',
      'Execution',
      'Continuous control'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'large',
    explainability: 'low',
    hyperparameters: {
      learning_rate: 0.0003,
      gamma: 0.99,
      tau: 0.005
    },
    min_samples: 10000,
    recommended_samples: 100000,
    supports_online_learning: true,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 1024
  },
  
  // NLP MODELS
  {
    id: 'finbert',
    name: 'FinBERT',
    category: 'nlp',
    use_cases: ['news_analysis'],
    description: 'BERT fine-tuned on financial text',
    strengths: [
      'Domain-specific vocabulary',
      'Good sentiment analysis',
      'Pre-trained on financial corpus'
    ],
    weaknesses: [
      'Limited context window',
      'Requires fine-tuning',
      'Computationally expensive'
    ],
    best_for: [
      'Earnings calls',
      'SEC filings',
      'News sentiment',
      'Event detection'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'medium',
    explainability: 'medium',
    hyperparameters: {
      max_length: 512,
      learning_rate: 2e-5,
      batch_size: 16
    },
    min_samples: 1000,
    recommended_samples: 10000,
    supports_online_learning: false,
    supports_transfer_learning: true,
    gpu_required: true,
    memory_mb: 2048
  },
  {
    id: 'longformer',
    name: 'Longformer',
    category: 'nlp',
    use_cases: ['news_analysis'],
    description: 'Transformer for long documents',
    strengths: [
      'Handles long documents',
      'Efficient attention',
      'Good for filings'
    ],
    weaknesses: [
      'Memory intensive',
      'Slower than BERT'
    ],
    best_for: [
      'SEC filings',
      'Research reports',
      'Long-form analysis'
    ],
    production_ready: true,
    latency_class: 'high',
    data_requirements: 'large',
    explainability: 'medium',
    hyperparameters: {
      max_length: 4096,
      attention_window: 512,
      learning_rate: 3e-5
    },
    min_samples: 500,
    recommended_samples: 5000,
    supports_online_learning: false,
    supports_transfer_learning: true,
    gpu_required: true,
    memory_mb: 4096
  },
  
  // TABULAR MODELS
  {
    id: 'tabnet',
    name: 'TabNet',
    category: 'tabular',
    use_cases: ['credit_risk', 'fraud_detection'],
    description: 'Attention-based tabular learning',
    strengths: [
      'Interpretable feature selection',
      'Works well on tabular data',
      'No feature engineering needed',
      'Competitive with XGBoost'
    ],
    weaknesses: [
      'Slower than tree models',
      'Requires tuning'
    ],
    best_for: [
      'Credit scoring',
      'AML',
      'Fraud detection',
      'Structured data'
    ],
    production_ready: true,
    latency_class: 'low',
    data_requirements: 'medium',
    explainability: 'high',
    hyperparameters: {
      n_d: 64,
      n_a: 64,
      n_steps: 5
    },
    min_samples: 1000,
    recommended_samples: 10000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 512
  },
  {
    id: 'ft_transformer',
    name: 'FT-Transformer',
    category: 'tabular',
    use_cases: ['credit_risk', 'fraud_detection'],
    description: 'Feature Tokenizer + Transformer for tabular data',
    strengths: [
      'State-of-the-art on tabular',
      'Handles mixed data types',
      'Good generalization'
    ],
    weaknesses: [
      'Slower than TabNet',
      'More parameters'
    ],
    best_for: [
      'Complex tabular tasks',
      'Mixed data types',
      'High-stakes decisions'
    ],
    production_ready: true,
    latency_class: 'medium',
    data_requirements: 'medium',
    explainability: 'medium',
    hyperparameters: {
      d_token: 192,
      n_blocks: 3,
      attention_dropout: 0.2
    },
    min_samples: 1000,
    recommended_samples: 10000,
    supports_online_learning: false,
    supports_transfer_learning: false,
    gpu_required: false,
    memory_mb: 768
  },
];

/**
 * List all available models with optional filtering
 * Falls back to hardcoded data if backend is unavailable
 */
export async function listModels(filters?: {
  category?: string;
  use_case?: string;
  production_ready?: boolean;
}): Promise<ModelSpec[]> {
  try {
    const params = new URLSearchParams();
    if (filters?.category) params.append('category', filters.category);
    if (filters?.use_case) params.append('use_case', filters.use_case);
    if (filters?.production_ready !== undefined) {
      params.append('production_ready', filters.production_ready.toString());
    }
    
    const response = await intelligenceApi.get(`/models/list?${params.toString()}`);
    return (response as any).models as ModelSpec[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded models:', error);
    
    // Filter hardcoded models
    let models = [...HARDCODED_MODELS];
    
    if (filters?.category) {
      models = models.filter(m => m.category === filters.category);
    }
    
    if (filters?.use_case) {
      models = models.filter(m => m.use_cases.includes(filters.use_case!));
    }
    
    if (filters?.production_ready !== undefined) {
      models = models.filter(m => m.production_ready === filters.production_ready);
    }
    
    return models;
  }
}

/**
 * Get detailed information about a specific model
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getModelDetails(modelId: string): Promise<ModelSpec> {
  try {
    const response = await intelligenceApi.get(`/models/${modelId}`);
    return response as ModelSpec;
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded model details:', error);
    
    const model = HARDCODED_MODELS.find(m => m.id === modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }
    
    return model;
  }
}

/**
 * Get recommended models for a specific use case
 * Falls back to hardcoded data if backend is unavailable
 */
export async function recommendModels(useCase: string): Promise<{
  use_case: string;
  recommendations: ModelRecommendation[];
  count: number;
}> {
  try {
    const response = await intelligenceApi.get(`/models/recommend?use_case=${useCase}`);
    return response as {
      use_case: string;
      recommendations: ModelRecommendation[];
      count: number;
    };
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded recommendations:', error);
    
    // Filter models by use case
    const filteredModels = HARDCODED_MODELS.filter(m => 
      m.use_cases.includes(useCase)
    );
    
    // Sort by production readiness and data requirements
    const dataReqOrder = { small: 0, medium: 1, large: 2 };
    const sortedModels = filteredModels
      .filter(m => m.production_ready)
      .sort((a, b) => {
        // Prefer production-ready models with smaller data requirements
        return dataReqOrder[a.data_requirements] - dataReqOrder[b.data_requirements];
      });
    
    // Convert to recommendation format
    const recommendations: ModelRecommendation[] = sortedModels.map(m => ({
      id: m.id,
      name: m.name,
      category: m.category,
      description: m.description,
      strengths: m.strengths,
      best_for: m.best_for,
      production_ready: m.production_ready,
      latency_class: m.latency_class,
      data_requirements: m.data_requirements,
      explainability: m.explainability,
      gpu_required: m.gpu_required,
    }));
    
    return {
      use_case: useCase,
      recommendations,
      count: recommendations.length,
    };
  }
}

/**
 * Get all model categories
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getModelCategories(): Promise<ModelCategory[]> {
  try {
    const response = await intelligenceApi.get('/models/categories');
    return (response as any).categories as ModelCategory[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded categories:', error);
    
    // Extract unique categories from hardcoded models
    const categoryMap = new Map<string, number>();
    
    HARDCODED_MODELS.forEach(model => {
      const count = categoryMap.get(model.category) || 0;
      categoryMap.set(model.category, count + 1);
    });
    
    return Array.from(categoryMap.entries()).map(([id, count]) => ({
      id,
      name: formatCategory(id),
      count,
    }));
  }
}

/**
 * Get all use cases
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getUseCases(): Promise<UseCase[]> {
  try {
    const response = await intelligenceApi.get('/models/use-cases');
    return (response as any).use_cases as UseCase[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded use cases:', error);
    
    // Extract unique use cases from hardcoded models
    const useCaseMap = new Map<string, number>();
    
    HARDCODED_MODELS.forEach(model => {
      model.use_cases.forEach(useCase => {
        const count = useCaseMap.get(useCase) || 0;
        useCaseMap.set(useCase, count + 1);
      });
    });
    
    return Array.from(useCaseMap.entries()).map(([id, count]) => ({
      id,
      name: formatUseCase(id),
      count,
    }));
  }
}

/**
 * Helper function to format model category for display
 */
export function formatCategory(category: string): string {
  return category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Helper function to format use case for display
 */
export function formatUseCase(useCase: string): string {
  return useCase.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Helper function to get latency class color
 */
export function getLatencyColor(latency: string): string {
  switch (latency) {
    case 'low': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'high': return '#ff8c00';
    default: return '#666';
  }
}

/**
 * Helper function to get data requirements color
 */
export function getDataReqColor(dataReq: string): string {
  switch (dataReq) {
    case 'small': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'large': return '#ff8c00';
    default: return '#666';
  }
}

/**
 * Helper function to get explainability color
 */
export function getExplainabilityColor(explainability: string): string {
  switch (explainability) {
    case 'high': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'low': return '#ff8c00';
    default: return '#666';
  }
}

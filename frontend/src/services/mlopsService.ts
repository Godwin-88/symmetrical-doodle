/**
 * MLOps Service - Machine Learning Operations API Client
 * Provides comprehensive MLOps functionality for financial engineering
 */

// Types
export interface TrainingDataset {
  id: string;
  name: string;
  records: number;
  features: number;
  quality: number;
  size: string;
  dateRange: string;
  assets: string[];
  status: 'READY' | 'PROCESSING' | 'ERROR';
  path: string;
  created: string;
  description: string;
}

export interface TrainingJob {
  id: string;
  model_id: string;
  model_name: string;
  dataset_id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED' | 'QUEUED' | 'PAUSED';
  current_epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  accuracy: number;
  started_at: string;
  eta: string;
  config: TrainingConfig;
}

export interface TrainingConfig {
  learning_rate: number;
  batch_size: number;
  epochs: number;
  optimizer: string;
  loss_function: string;
  validation_split: number;
  early_stopping: boolean;
  patience: number;
}

export interface DeployedModel {
  id: string;
  name: string;
  type: string;
  status: 'ACTIVE' | 'TESTING' | 'DEPRECATED' | 'STAGING';
  accuracy: number;
  trained: string;
  version: string;
  hash: string;
  dataset: string;
  endpoint: string;
  replicas: number;
  cpu_usage: number;
  memory_usage: number;
  requests_per_minute: number;
}

export interface DeploymentConfig {
  model_id: string;
  environment: string;
  replicas: number;
  cpu_limit: string;
  memory_limit: string;
  auto_scale: boolean;
}

export interface ModelAlert {
  id: string;
  model_id: string;
  type: 'DRIFT' | 'PERFORMANCE' | 'ERROR' | 'WARNING';
  message: string;
  timestamp: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  acknowledged: boolean;
}

export interface ValidationMetrics {
  temporal_continuity: number;
  regime_separability: number;
  feature_stability: number;
  prediction_consistency: number;
  drift_detection: number;
  overall_score: number;
}
// API Base URLs
const INTELLIGENCE_API_BASE = 'http://localhost:8000';
const EXECUTION_API_BASE = 'http://localhost:8001';

// Dataset Management
export async function createDataset(dataset: Partial<TrainingDataset>): Promise<TrainingDataset> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/datasets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dataset)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Dataset creation API failed, using mock response:', error);
    
    // Mock response for development
    return {
      id: `ds_${Date.now()}`,
      name: dataset.name || 'New Dataset',
      records: 1000000,
      features: 100,
      quality: 95.0,
      size: '500MB',
      dateRange: dataset.dateRange || '2023-01-01 to 2024-01-15',
      assets: dataset.assets || ['EURUSD'],
      status: 'PROCESSING',
      path: dataset.path || '/data/new_dataset.parquet',
      created: new Date().toISOString().split('T')[0],
      description: dataset.description || 'New training dataset'
    };
  }
}

export async function listDatasets(): Promise<TrainingDataset[]> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/datasets`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('List datasets API failed, using mock data:', error);
    
    // Mock data for development
    return [
      {
        id: 'ds_001',
        name: 'FX_REGIME_FEATURES_2024',
        records: 2847392,
        features: 127,
        quality: 94.2,
        size: '1.2GB',
        dateRange: '2020-01-01 to 2024-01-15',
        assets: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        status: 'READY',
        path: '/data/fx_regime_features_2024.parquet',
        created: '2024-01-10',
        description: 'High-frequency FX regime detection features with technical indicators'
      },
      {
        id: 'ds_002', 
        name: 'EQUITY_MOMENTUM_SIGNALS',
        records: 1923847,
        features: 89,
        quality: 91.7,
        size: '850MB',
        dateRange: '2019-06-01 to 2024-01-15',
        assets: ['SPY', 'QQQ', 'IWM', 'VTI'],
        status: 'READY',
        path: '/data/equity_momentum_signals.parquet',
        created: '2024-01-08',
        description: 'Equity momentum signals with sector rotation indicators'
      }
    ];
  }
}

export async function deleteDataset(datasetId: string): Promise<void> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/datasets/${datasetId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Delete dataset API failed:', error);
    // Mock success for development
  }
}

// Training Job Management
export async function startTraining(modelId: string, datasetId: string, config: TrainingConfig): Promise<TrainingJob> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, dataset_id: datasetId, config })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Start training API failed, using mock response:', error);
    
    // Mock response for development
    return {
      id: `job_${Date.now()}`,
      model_id: modelId,
      model_name: 'Training Job',
      dataset_id: datasetId,
      status: 'QUEUED',
      current_epoch: 0,
      total_epochs: config.epochs,
      train_loss: 0,
      val_loss: 0,
      accuracy: 0,
      started_at: new Date().toISOString(),
      eta: 'Calculating...',
      config
    };
  }
}

export async function pauseTraining(jobId: string): Promise<void> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/training/${jobId}/pause`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Pause training API failed:', error);
    // Mock success for development
  }
}

export async function stopTraining(jobId: string): Promise<void> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/training/${jobId}/stop`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Stop training API failed:', error);
    // Mock success for development
  }
}

export async function listTrainingJobs(): Promise<TrainingJob[]> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/training/jobs`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('List training jobs API failed, using mock data:', error);
    
    // Mock data for development
    return [
      {
        id: 'job_001',
        model_id: 'tcn_regime_v2',
        model_name: 'TCN Regime Detector v2.1',
        dataset_id: 'ds_001',
        status: 'RUNNING',
        current_epoch: 47,
        total_epochs: 100,
        train_loss: 0.0234,
        val_loss: 0.0287,
        accuracy: 87.3,
        started_at: '2024-01-15 09:30:00',
        eta: '2h 15m',
        config: {
          learning_rate: 0.001,
          batch_size: 64,
          epochs: 100,
          optimizer: 'adam',
          loss_function: 'categorical_crossentropy',
          validation_split: 0.2,
          early_stopping: true,
          patience: 15
        }
      }
    ];
  }
}
// Model Deployment Management
export async function deployModel(config: DeploymentConfig): Promise<DeployedModel> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Deploy model API failed, using mock response:', error);
    
    // Mock response for development
    return {
      id: `deploy_${Date.now()}`,
      name: 'Deployed Model',
      type: 'ML Model',
      status: 'STAGING',
      accuracy: 85.0,
      trained: new Date().toISOString().split('T')[0],
      version: 'v1.0',
      hash: Math.random().toString(36).substr(2, 8),
      dataset: 'training_dataset',
      endpoint: `https://staging.trading.com/models/${config.model_id}/predict`,
      replicas: config.replicas,
      cpu_usage: 25.0,
      memory_usage: 40.0,
      requests_per_minute: 0
    };
  }
}

export async function scaleModel(modelId: string, replicas: number): Promise<void> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/models/${modelId}/scale`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ replicas })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Scale model API failed:', error);
    // Mock success for development
  }
}

export async function rollbackModel(modelId: string, version: string): Promise<void> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/models/${modelId}/rollback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Rollback model API failed:', error);
    // Mock success for development
  }
}

export async function promoteModel(modelId: string): Promise<void> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/models/${modelId}/promote`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Promote model API failed:', error);
    // Mock success for development
  }
}

export async function stopModel(modelId: string): Promise<void> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/models/${modelId}/stop`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Stop model API failed:', error);
    // Mock success for development
  }
}

export async function listDeployedModels(): Promise<DeployedModel[]> {
  try {
    const response = await fetch(`${EXECUTION_API_BASE}/mlops/models`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('List deployed models API failed, using mock data:', error);
    
    // Mock data for development
    return [
      {
        id: 'prod_001',
        name: 'TCN Regime Detector',
        type: 'Regime Classification',
        status: 'ACTIVE',
        accuracy: 89.4,
        trained: '2024-01-10',
        version: 'v2.0',
        hash: 'a7f3c9e1',
        dataset: 'FX_REGIME_FEATURES_2024',
        endpoint: 'https://api.trading.com/models/tcn-regime/predict',
        replicas: 3,
        cpu_usage: 45.2,
        memory_usage: 67.8,
        requests_per_minute: 1247
      },
      {
        id: 'prod_002',
        name: 'VAE Market Embeddings',
        type: 'Feature Extraction',
        status: 'ACTIVE',
        accuracy: 91.2,
        trained: '2024-01-08',
        version: 'v1.3',
        hash: 'b2d8f4a6',
        dataset: 'EQUITY_MOMENTUM_SIGNALS',
        endpoint: 'https://api.trading.com/models/vae-embeddings/embed',
        replicas: 2,
        cpu_usage: 32.1,
        memory_usage: 54.3,
        requests_per_minute: 892
      }
    ];
  }
}

// Model Monitoring
export async function getValidationMetrics(): Promise<ValidationMetrics> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/metrics/validation`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Get validation metrics API failed, using mock data:', error);
    
    // Mock data for development
    return {
      temporal_continuity: 92.3,
      regime_separability: 87.9,
      feature_stability: 94.1,
      prediction_consistency: 89.6,
      drift_detection: 91.2,
      overall_score: 91.0
    };
  }
}

export async function getModelAlerts(): Promise<ModelAlert[]> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/alerts`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Get model alerts API failed, using mock data:', error);
    
    // Mock data for development
    return [
      {
        id: 'alert_001',
        model_id: 'prod_003',
        type: 'DRIFT',
        message: 'LSTM Volatility model showing high data drift (threshold: 0.15, current: 0.23)',
        timestamp: '14:25:30',
        severity: 'HIGH',
        acknowledged: false
      },
      {
        id: 'alert_002',
        model_id: 'prod_002',
        type: 'PERFORMANCE',
        message: 'VAE Embeddings performance degraded to 91.2% (threshold: 92%)',
        timestamp: '13:45:12',
        severity: 'MEDIUM',
        acknowledged: false
      }
    ];
  }
}

export async function acknowledgeAlert(alertId: string): Promise<void> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/alerts/${alertId}/acknowledge`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Acknowledge alert API failed:', error);
    // Mock success for development
  }
}

// Model Configuration
export async function updateTrainingConfig(config: TrainingConfig): Promise<void> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/config/training`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn('Update training config API failed:', error);
    // Mock success for development
  }
}

export async function retrainModel(modelId: string, datasetId: string): Promise<TrainingJob> {
  try {
    const response = await fetch(`${INTELLIGENCE_API_BASE}/mlops/models/${modelId}/retrain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Retrain model API failed, using mock response:', error);
    
    // Mock response for development
    return {
      id: `retrain_${Date.now()}`,
      model_id: modelId,
      model_name: 'Retrained Model',
      dataset_id: datasetId,
      status: 'QUEUED',
      current_epoch: 0,
      total_epochs: 100,
      train_loss: 0,
      val_loss: 0,
      accuracy: 0,
      started_at: new Date().toISOString(),
      eta: 'Calculating...',
      config: {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 100,
        optimizer: 'adam',
        loss_function: 'mse',
        validation_split: 0.2,
        early_stopping: true,
        patience: 10
      }
    };
  }
}

// Utility functions
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'ACTIVE':
    case 'READY':
    case 'COMPLETED':
      return 'text-green-400';
    case 'RUNNING':
    case 'PROCESSING':
    case 'TESTING':
    case 'STAGING':
      return 'text-yellow-400';
    case 'FAILED':
    case 'ERROR':
    case 'DEPRECATED':
      return 'text-red-400';
    case 'QUEUED':
    case 'PAUSED':
      return 'text-blue-400';
    default:
      return 'text-gray-400';
  }
}

export function getSeverityColor(severity: string): string {
  switch (severity) {
    case 'CRITICAL': return 'text-red-400';
    case 'HIGH': return 'text-orange-400';
    case 'MEDIUM': return 'text-yellow-400';
    case 'LOW': return 'text-green-400';
    default: return 'text-gray-400';
  }
}
import { useState, useEffect } from 'react';
import {
  listModels,
  getModelDetails,
  formatCategory,
  type ModelSpec,
} from '../../services/modelsService';

interface TrainingDataset {
  id: string;
  name: string;
  records: number;
  features: number;
  quality: number;
  size: string;
  dateRange: string;
  assets: string[];
  status: 'READY' | 'PROCESSING' | 'ERROR';
}

interface TrainingJob {
  id: string;
  model_id: string;
  model_name: string;
  dataset_id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED' | 'QUEUED';
  current_epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  accuracy: number;
  started_at: string;
  eta: string;
}

interface DeployedModel {
  id: string;
  name: string;
  type: string;
  status: 'ACTIVE' | 'TESTING' | 'DEPRECATED';
  accuracy: number;
  trained: string;
  version: string;
  hash: string;
  dataset: string;
}

interface ValidationMetrics {
  temporal_continuity: number;
  regime_separability: number;
  similarity_coherence: number;
  prediction_accuracy: number;
  confusion_matrix?: number[][];
  precision: number;
  recall: number;
  f1_score: number;
}

export function DataModels() {
  // State
  const [availableModels, setAvailableModels] = useState<ModelSpec[]>([]);
  const [deployedModels, setDeployedModels] = useState<DeployedModel[]>([]);
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [selectedModel, setSelectedModel] = useState<DeployedModel | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<TrainingDataset | null>(null);
  
  // Modal states
  const [showTrainModal, setShowTrainModal] = useState(false);
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [showModelBrowser, setShowModelBrowser] = useState(false);
  const [selectedBrowserModel, setSelectedBrowserModel] = useState<ModelSpec | null>(null);
  
  // Form states
  const [trainingConfig, setTrainingConfig] = useState({
    model_id: '',
    dataset_id: '',
    epochs: 100,
    batch_size: 64,
    learning_rate: 0.001,
    validation_split: 0.2,
  });

  const [validationMetrics, setValidationMetrics] = useState<ValidationMetrics>({
    temporal_continuity: 0.92,
    regime_separability: 0.87,
    similarity_coherence: 0.94,
    prediction_accuracy: 0.89,
    precision: 0.91,
    recall: 0.88,
    f1_score: 0.89,
  });

  // Mock data initialization
  useEffect(() => {
    const fetchData = async () => {
      try {
        const models = await listModels();
        setAvailableModels(models);
      } catch (err) {
        console.warn('Failed to fetch models:', err);
      }
    };
    
    fetchData();
    
    // Initialize mock deployed models
    setDeployedModels([
      {
        id: 'TCN_V1.2',
        name: 'TCN EMBEDDING MODEL',
        type: 'TCN',
        status: 'ACTIVE',
        accuracy: 92.3,
        trained: '2024-01-10',
        version: '1.2.0',
        hash: 'a3f5b2c8d9e1f4a6',
        dataset: 'EURUSD_2020-2023_CLEAN'
      },
      {
        id: 'HMM_V2.0',
        name: 'HMM REGIME DETECTOR',
        type: 'HMM',
        status: 'ACTIVE',
        accuracy: 87.5,
        trained: '2024-01-08',
        version: '2.0.0',
        hash: 'b4c6d7e8f9a0b1c2',
        dataset: 'MULTI_ASSET_2022-2024'
      },
      {
        id: 'VAE_V1.0',
        name: 'VAE EMBEDDING',
        type: 'VAE',
        status: 'TESTING',
        accuracy: 89.1,
        trained: '2024-01-12',
        version: '1.0.0',
        hash: 'c5d7e8f9a0b1c2d3',
        dataset: 'EURUSD_2020-2023_CLEAN'
      },
    ]);

    // Initialize mock datasets
    setDatasets([
      {
        id: 'DS001',
        name: 'EURUSD_2020-2023_CLEAN',
        records: 1250000,
        features: 32,
        quality: 98.5,
        size: '2.4 GB',
        dateRange: '2020-01-01 to 2023-12-31',
        assets: ['EURUSD'],
        status: 'READY'
      },
      {
        id: 'DS002',
        name: 'MULTI_ASSET_2022-2024',
        records: 3450000,
        features: 48,
        quality: 97.2,
        size: '6.8 GB',
        dateRange: '2022-01-01 to 2024-01-15',
        assets: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        status: 'READY'
      },
      {
        id: 'DS003',
        name: 'CRISIS_SCENARIOS_2008-2020',
        records: 450000,
        features: 32,
        quality: 99.1,
        size: '890 MB',
        dateRange: '2008-01-01 to 2020-12-31',
        assets: ['EURUSD', 'GBPUSD'],
        status: 'READY'
      },
    ]);

    // Initialize mock training job
    setTrainingJobs([
      {
        id: 'JOB001',
        model_id: 'vae',
        model_name: 'Variational Autoencoder',
        dataset_id: 'DS001',
        status: 'RUNNING',
        current_epoch: 45,
        total_epochs: 100,
        train_loss: 0.0234,
        val_loss: 0.0289,
        accuracy: 89.1,
        started_at: '2024-01-15 12:00:00',
        eta: '2H 15M'
      }
    ]);
  }, []);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  // CRUD Operations
  const createDataset = (dataset: Omit<TrainingDataset, 'id'>) => {
    const newDataset: TrainingDataset = {
      ...dataset,
      id: `DS${String(datasets.length + 1).padStart(3, '0')}`,
    };
    setDatasets([...datasets, newDataset]);
  };

  const deleteDataset = (id: string) => {
    setDatasets(datasets.filter(d => d.id !== id));
  };

  const startTraining = () => {
    const newJob: TrainingJob = {
      id: `JOB${String(trainingJobs.length + 1).padStart(3, '0')}`,
      model_id: trainingConfig.model_id,
      model_name: availableModels.find(m => m.id === trainingConfig.model_id)?.name || 'Unknown',
      dataset_id: trainingConfig.dataset_id,
      status: 'RUNNING',
      current_epoch: 0,
      total_epochs: trainingConfig.epochs,
      train_loss: 0.5,
      val_loss: 0.6,
      accuracy: 0,
      started_at: new Date().toISOString(),
      eta: `${Math.floor(trainingConfig.epochs / 20)}H ${(trainingConfig.epochs % 20) * 3}M`
    };
    setTrainingJobs([...trainingJobs, newJob]);
    setShowTrainModal(false);
  };

  const deployModel = (jobId: string) => {
    const job = trainingJobs.find(j => j.id === jobId);
    if (!job) return;

    const newModel: DeployedModel = {
      id: `${job.model_id.toUpperCase()}_V${deployedModels.length + 1}.0`,
      name: job.model_name.toUpperCase(),
      type: job.model_id.toUpperCase(),
      status: 'TESTING',
      accuracy: job.accuracy,
      trained: new Date().toISOString().split('T')[0],
      version: `${deployedModels.length + 1}.0.0`,
      hash: Math.random().toString(36).substring(2, 18),
      dataset: datasets.find(d => d.id === job.dataset_id)?.name || 'Unknown'
    };
    setDeployedModels([...deployedModels, newModel]);
  };

  const updateModelStatus = (id: string, status: 'ACTIVE' | 'TESTING' | 'DEPRECATED') => {
    setDeployedModels(deployedModels.map(m => m.id === id ? { ...m, status } : m));
  };

  const deleteModel = (id: string) => {
    setDeployedModels(deployedModels.filter(m => m.id !== id));
  };

  const runValidation = (modelId: string) => {
    // Generate mock validation metrics
    setValidationMetrics({
      temporal_continuity: 0.85 + Math.random() * 0.15,
      regime_separability: 0.80 + Math.random() * 0.15,
      similarity_coherence: 0.88 + Math.random() * 0.12,
      prediction_accuracy: 0.82 + Math.random() * 0.15,
      precision: 0.85 + Math.random() * 0.12,
      recall: 0.83 + Math.random() * 0.12,
      f1_score: 0.84 + Math.random() * 0.12,
      confusion_matrix: [
        [850, 50, 30, 20],
        [40, 880, 45, 35],
        [35, 55, 870, 40],
        [25, 45, 55, 875]
      ]
    });
    setShowValidationModal(true);
  };

  return (
    <div className="flex h-full font-mono text-xs">

      {/* Left Panel - Models & Datasets List */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Deployed Models */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">DEPLOYED MODELS</div>
            <div className="space-y-2">
              {deployedModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model)}
                  className={`
                    border p-3 cursor-pointer transition-colors
                    ${selectedModel?.id === model.id
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                    }
                  `}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="text-[#00ff00] text-[10px]">{model.name}</div>
                    <span className={`
                      text-[8px] px-1
                      ${model.status === 'ACTIVE' ? 'text-[#00ff00]' : ''}
                      ${model.status === 'TESTING' ? 'text-[#ffff00]' : ''}
                      ${model.status === 'DEPRECATED' ? 'text-[#666]' : ''}
                    `}>
                      {model.status}
                    </span>
                  </div>
                  <div className="space-y-1 text-[9px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">TYPE:</span>
                      <span className="text-[#fff]">{model.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">ACCURACY:</span>
                      <span className="text-[#00ff00]">{formatNumber(model.accuracy, 1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">VERSION:</span>
                      <span className="text-[#fff]">{model.version}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Training Datasets */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">TRAINING DATASETS</div>
            <div className="space-y-2">
              {datasets.map((dataset) => (
                <div
                  key={dataset.id}
                  onClick={() => setSelectedDataset(dataset)}
                  className={`
                    border p-3 cursor-pointer transition-colors
                    ${selectedDataset?.id === dataset.id
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                    }
                  `}
                >
                  <div className="text-[#00ff00] text-[10px] mb-2">{dataset.name}</div>
                  <div className="space-y-1 text-[9px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">RECORDS:</span>
                      <span className="text-[#fff]">{formatNumber(dataset.records, 0)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">QUALITY:</span>
                      <span className="text-[#00ff00]">{formatNumber(dataset.quality, 1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">SIZE:</span>
                      <span className="text-[#fff]">{dataset.size}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Training Jobs */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">TRAINING JOBS</div>
            <div className="space-y-2">
              {trainingJobs.map((job) => (
                <div
                  key={job.id}
                  className="border border-[#333] p-3"
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="text-[#ffff00] text-[10px]">{job.model_name}</div>
                    <span className={`
                      text-[8px]
                      ${job.status === 'RUNNING' ? 'text-[#ffff00]' : ''}
                      ${job.status === 'COMPLETED' ? 'text-[#00ff00]' : ''}
                      ${job.status === 'FAILED' ? 'text-[#ff0000]' : ''}
                    `}>
                      {job.status}
                    </span>
                  </div>
                  <div className="space-y-1 text-[9px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">EPOCH:</span>
                      <span className="text-[#fff]">{job.current_epoch}/{job.total_epochs}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">LOSS:</span>
                      <span className="text-[#00ff00]">{formatNumber(job.train_loss, 4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">ETA:</span>
                      <span className="text-[#fff]">{job.eta}</span>
                    </div>
                  </div>
                  {job.status === 'COMPLETED' && (
                    <button
                      onClick={() => deployModel(job.id)}
                      className="w-full mt-2 py-1 border border-[#00ff00] text-[#00ff00] text-[8px] hover:bg-[#00ff00] hover:text-black"
                    >
                      DEPLOY MODEL
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Center Panel - Details View */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            {selectedModel ? `MODEL: ${selectedModel.name}` : selectedDataset ? `DATASET: ${selectedDataset.name}` : 'DATA & MODELS - TRAINING & VALIDATION'}
          </div>
        </div>

        {selectedModel && (
          <div className="space-y-6">
            {/* Model Information */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">MODEL INFORMATION</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-[#666]">MODEL ID</div>
                    <div className="text-[#00ff00]">{selectedModel.id}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">TYPE</div>
                    <div className="text-[#fff]">{selectedModel.type}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">STATUS</div>
                    <div className={`
                      ${selectedModel.status === 'ACTIVE' ? 'text-[#00ff00]' : ''}
                      ${selectedModel.status === 'TESTING' ? 'text-[#ffff00]' : ''}
                      ${selectedModel.status === 'DEPRECATED' ? 'text-[#666]' : ''}
                    `}>
                      {selectedModel.status}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#666]">ACCURACY</div>
                    <div className="text-[#00ff00]">{formatNumber(selectedModel.accuracy, 1)}%</div>
                  </div>
                  <div>
                    <div className="text-[#666]">VERSION</div>
                    <div className="text-[#fff]">{selectedModel.version}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">TRAINED</div>
                    <div className="text-[#fff]">{selectedModel.trained}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Provenance */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">MODEL PROVENANCE</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="space-y-2 text-[10px]">
                  <div className="flex gap-4">
                    <span className="text-[#666] w-32">MODEL HASH:</span>
                    <span className="text-[#00ff00]">{selectedModel.hash}</span>
                  </div>
                  <div className="flex gap-4">
                    <span className="text-[#666] w-32">DATASET:</span>
                    <span className="text-[#fff]">{selectedModel.dataset}</span>
                  </div>
                  <div className="flex gap-4">
                    <span className="text-[#666] w-32">ARCHITECTURE:</span>
                    <span className="text-[#fff]">{selectedModel.type} (8 LAYERS, 256 CHANNELS)</span>
                  </div>
                  <div className="flex gap-4">
                    <span className="text-[#666] w-32">PARAMETERS:</span>
                    <span className="text-[#fff]">2.4M</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Validation Metrics */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">VALIDATION METRICS</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">TEMPORAL CONTINUITY</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.temporal_continuity)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">REGIME SEPARABILITY</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.regime_separability)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">SIMILARITY COHERENCE</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.similarity_coherence)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">PREDICTION ACCURACY</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.prediction_accuracy)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">PRECISION</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.precision)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">RECALL</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.recall)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">F1 SCORE</span>
                    <span className="text-[#00ff00]">{formatNumber(validationMetrics.f1_score)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedDataset && (
          <div className="space-y-6">
            {/* Dataset Information */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DATASET INFORMATION</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-[#666]">DATASET ID</div>
                    <div className="text-[#00ff00]">{selectedDataset.id}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">STATUS</div>
                    <div className="text-[#00ff00]">{selectedDataset.status}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">RECORDS</div>
                    <div className="text-[#fff]">{formatNumber(selectedDataset.records, 0)}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">FEATURES</div>
                    <div className="text-[#fff]">{selectedDataset.features}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">QUALITY SCORE</div>
                    <div className="text-[#00ff00]">{formatNumber(selectedDataset.quality, 1)}%</div>
                  </div>
                  <div>
                    <div className="text-[#666]">SIZE</div>
                    <div className="text-[#fff]">{selectedDataset.size}</div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-[#666]">DATE RANGE</div>
                    <div className="text-[#fff]">{selectedDataset.dateRange}</div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-[#666]">ASSETS</div>
                    <div className="text-[#fff]">{selectedDataset.assets.join(', ')}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Data Quality Metrics */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DATA QUALITY METRICS</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">COMPLETENESS</span>
                    <span className="text-[#00ff00]">99.2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">CONSISTENCY</span>
                    <span className="text-[#00ff00]">98.7%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">OUTLIERS DETECTED</span>
                    <span className="text-[#ffff00]">0.3%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">MISSING VALUES</span>
                    <span className="text-[#00ff00]">0.1%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!selectedModel && !selectedDataset && (
          <div className="text-center text-[#666] text-[10px] mt-20">
            SELECT A MODEL OR DATASET TO VIEW DETAILS
          </div>
        )}
      </div>


      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">MODEL ACTIONS</div>
          
          <div className="space-y-2 mb-6">
            <button
              onClick={() => setShowModelBrowser(true)}
              className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              BROWSE MODEL ARCHITECTURES ({availableModels.length})
            </button>
            <button
              onClick={() => setShowTrainModal(true)}
              className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
            >
              TRAIN NEW MODEL
            </button>
            <button
              onClick={() => selectedModel && runValidation(selectedModel.id)}
              disabled={!selectedModel}
              className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              RUN VALIDATION
            </button>
          </div>

          {selectedModel && (
            <div className="space-y-2 mb-6">
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">MODEL MANAGEMENT</div>
              <button
                onClick={() => updateModelStatus(selectedModel.id, 'ACTIVE')}
                disabled={selectedModel.status === 'ACTIVE'}
                className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors disabled:opacity-50"
              >
                ACTIVATE MODEL
              </button>
              <button
                onClick={() => updateModelStatus(selectedModel.id, 'TESTING')}
                disabled={selectedModel.status === 'TESTING'}
                className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors disabled:opacity-50"
              >
                MOVE TO TESTING
              </button>
              <button
                onClick={() => updateModelStatus(selectedModel.id, 'DEPRECATED')}
                disabled={selectedModel.status === 'DEPRECATED'}
                className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors disabled:opacity-50"
              >
                DEPRECATE MODEL
              </button>
              <button
                onClick={() => {
                  if (confirm(`Delete model ${selectedModel.name}?`)) {
                    deleteModel(selectedModel.id);
                    setSelectedModel(null);
                  }
                }}
                className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
              >
                DELETE MODEL
              </button>
            </div>
          )}

          <div className="space-y-2 mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">DATASET ACTIONS</div>
            <button
              onClick={() => setShowDatasetModal(true)}
              className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
            >
              IMPORT DATASET
            </button>
            {selectedDataset && (
              <>
                <button
                  className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  VALIDATE DATA QUALITY
                </button>
                <button
                  className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  EXPORT DATASET
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Delete dataset ${selectedDataset.name}?`)) {
                      deleteDataset(selectedDataset.id);
                      setSelectedDataset(null);
                    }
                  }}
                  className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                >
                  DELETE DATASET
                </button>
              </>
            )}
          </div>

          <div className="space-y-2">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">EXPERIMENT TRACKING</div>
            <button className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              VIEW EXPERIMENTS
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              COMPARE MODELS
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              EXPORT METRICS
            </button>
          </div>
        </div>
      </div>


      {/* Model Browser Modal */}
      {showModelBrowser && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">MODEL ARCHITECTURE REGISTRY</h2>
              <button
                onClick={() => {
                  setShowModelBrowser(false);
                  setSelectedBrowserModel(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 flex gap-4 overflow-hidden">
              {/* Model List */}
              <div className="w-1/2 border border-[#444] overflow-y-auto">
                <div className="space-y-1">
                  {availableModels.map(model => (
                    <div
                      key={model.id}
                      onClick={() => setSelectedBrowserModel(model)}
                      className={`p-3 border-b border-[#222] cursor-pointer transition-colors ${
                        selectedBrowserModel?.id === model.id
                          ? 'bg-[#1a1a1a] border-l-4 border-l-[#ff8c00]'
                          : 'hover:bg-[#0f0f0f]'
                      }`}
                    >
                      <div className="text-[#00ff00] text-[11px] font-bold">{model.name}</div>
                      <div className="text-[#666] text-[9px] mt-1">{formatCategory(model.category)}</div>
                      <div className="text-[#fff] text-[10px] mt-1 line-clamp-2">{model.description}</div>
                      <div className="flex gap-2 mt-2 text-[9px]">
                        <span className="text-[#00ff00]">MIN: {model.min_samples}</span>
                        <span className="text-[#666]">|</span>
                        <span className="text-[#ffff00]">REC: {model.recommended_samples}</span>
                        {model.gpu_required && (
                          <>
                            <span className="text-[#666]">|</span>
                            <span className="text-[#ff8c00]">GPU</span>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Model Details */}
              <div className="w-1/2 border border-[#444] overflow-y-auto p-4">
                {selectedBrowserModel ? (
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-[#ff8c00] text-[12px] mb-2">{selectedBrowserModel.name}</h3>
                      <p className="text-[#fff] text-[10px]">{selectedBrowserModel.description}</p>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">STRENGTHS</div>
                      <ul className="space-y-1">
                        {selectedBrowserModel.strengths.map((s, i) => (
                          <li key={i} className="text-[#00ff00] text-[9px]">✓ {s}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">WEAKNESSES</div>
                      <ul className="space-y-1">
                        {selectedBrowserModel.weaknesses.map((w, i) => (
                          <li key={i} className="text-[#ff0000] text-[9px]">✗ {w}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[#666] text-[9px]">Data Requirements</div>
                        <div className="text-[#fff] text-[10px]">{selectedBrowserModel.data_requirements.toUpperCase()}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">GPU Required</div>
                        <div className="text-[#fff] text-[10px]">{selectedBrowserModel.gpu_required ? 'YES' : 'NO'}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Memory</div>
                        <div className="text-[#fff] text-[10px]">{selectedBrowserModel.memory_mb} MB</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Min Samples</div>
                        <div className="text-[#fff] text-[10px]">{selectedBrowserModel.min_samples}</div>
                      </div>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">HYPERPARAMETERS</div>
                      <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                        <pre className="text-[#00ff00] text-[9px]">
                          {JSON.stringify(selectedBrowserModel.hyperparameters, null, 2)}
                        </pre>
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        setTrainingConfig({ ...trainingConfig, model_id: selectedBrowserModel.id });
                        setShowModelBrowser(false);
                        setShowTrainModal(true);
                      }}
                      className="w-full px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                    >
                      USE THIS MODEL FOR TRAINING
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-[#666] text-[10px]">
                    SELECT A MODEL TO VIEW DETAILS
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Train Model Modal */}
      {showTrainModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">TRAIN NEW MODEL</h2>
              <button
                onClick={() => setShowTrainModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">MODEL ARCHITECTURE</label>
                <select
                  value={trainingConfig.model_id}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, model_id: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">Select Model...</option>
                  {availableModels.map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-[#666] block mb-1 text-[10px]">TRAINING DATASET</label>
                <select
                  value={trainingConfig.dataset_id}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, dataset_id: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">Select Dataset...</option>
                  {datasets.map(dataset => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">EPOCHS</label>
                  <input
                    type="number"
                    value={trainingConfig.epochs}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">BATCH SIZE</label>
                  <input
                    type="number"
                    value={trainingConfig.batch_size}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">LEARNING RATE</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingConfig.learning_rate}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">VALIDATION SPLIT</label>
                  <input
                    type="number"
                    step="0.05"
                    value={trainingConfig.validation_split}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, validation_split: parseFloat(e.target.value) })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
              </div>

              <div className="border border-[#ffff00] bg-[#1a1a1a] p-3 text-[#ffff00] text-[10px]">
                ⚠ Training will use GPU if available. Estimated time: {Math.floor(trainingConfig.epochs / 20)}H {(trainingConfig.epochs % 20) * 3}M
              </div>

              <div className="flex gap-2">
                <button
                  onClick={startTraining}
                  disabled={!trainingConfig.model_id || !trainingConfig.dataset_id}
                  className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px] disabled:opacity-50 disabled:cursor-not-allowed"
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


      {/* Dataset Import Modal */}
      {showDatasetModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">IMPORT TRAINING DATASET</h2>
              <button
                onClick={() => setShowDatasetModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                createDataset({
                  name: formData.get('name') as string,
                  records: parseInt(formData.get('records') as string),
                  features: parseInt(formData.get('features') as string),
                  quality: parseFloat(formData.get('quality') as string),
                  size: formData.get('size') as string,
                  dateRange: formData.get('dateRange') as string,
                  assets: (formData.get('assets') as string).split(',').map(a => a.trim()),
                  status: 'READY'
                });
                setShowDatasetModal(false);
              }}
            >
              <div className="space-y-4">
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">DATASET NAME</label>
                  <input
                    type="text"
                    name="name"
                    required
                    placeholder="e.g., EURUSD_2024_CLEAN"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">RECORDS</label>
                    <input
                      type="number"
                      name="records"
                      required
                      defaultValue={1000000}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">FEATURES</label>
                    <input
                      type="number"
                      name="features"
                      required
                      defaultValue={32}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">QUALITY SCORE (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      name="quality"
                      required
                      defaultValue={98.5}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">SIZE</label>
                    <input
                      type="text"
                      name="size"
                      required
                      defaultValue="2.4 GB"
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">DATE RANGE</label>
                  <input
                    type="text"
                    name="dateRange"
                    required
                    placeholder="2020-01-01 to 2023-12-31"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>

                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">ASSETS (comma-separated)</label>
                  <input
                    type="text"
                    name="assets"
                    required
                    placeholder="EURUSD, GBPUSD, USDJPY"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>

                <div className="flex gap-2">
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px]"
                  >
                    IMPORT DATASET
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowDatasetModal(false)}
                    className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                  >
                    CANCEL
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Validation Results Modal */}
      {showValidationModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">VALIDATION RESULTS</h2>
              <button
                onClick={() => setShowValidationModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-6">
              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">TEMPORAL CONTINUITY</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.temporal_continuity)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">REGIME SEPARABILITY</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.regime_separability)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SIMILARITY COHERENCE</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.similarity_coherence)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">PREDICTION ACCURACY</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.prediction_accuracy)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">PRECISION</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.precision)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">RECALL</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.recall)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">F1 SCORE</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(validationMetrics.f1_score)}</div>
                </div>
              </div>

              {/* Confusion Matrix */}
              {validationMetrics.confusion_matrix && (
                <div>
                  <div className="text-[#ff8c00] text-[10px] mb-2">CONFUSION MATRIX</div>
                  <div className="border border-[#444] p-4 bg-[#0a0a0a]">
                    <table className="w-full text-center text-[10px]">
                      <thead>
                        <tr className="text-[#ff8c00]">
                          <th className="p-2"></th>
                          <th className="p-2">PRED: R1</th>
                          <th className="p-2">PRED: R2</th>
                          <th className="p-2">PRED: R3</th>
                          <th className="p-2">PRED: R4</th>
                        </tr>
                      </thead>
                      <tbody>
                        {validationMetrics.confusion_matrix.map((row, i) => (
                          <tr key={i} className="border-t border-[#222]">
                            <td className="p-2 text-[#ff8c00]">TRUE: R{i + 1}</td>
                            {row.map((val, j) => (
                              <td
                                key={j}
                                className={`p-2 ${i === j ? 'text-[#00ff00] font-bold' : 'text-[#fff]'}`}
                              >
                                {val}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={() => setShowValidationModal(false)}
                  className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CLOSE
                </button>
                <button
                  className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px]"
                >
                  EXPORT RESULTS
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

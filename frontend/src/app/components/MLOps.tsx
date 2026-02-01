import { useState, useEffect, useMemo } from 'react';
import {
  listModels,
  getModelDetails,
  formatCategory,
  type ModelSpec,
} from '../../services/modelsService';
import {
  createDataset,
  listDatasets,
  deleteDataset,
  startTraining,
  pauseTraining,
  stopTraining,
  listTrainingJobs,
  deployModel,
  scaleModel,
  rollbackModel,
  promoteModel,
  stopModel,
  listDeployedModels,
  getValidationMetrics,
  getModelAlerts,
  acknowledgeAlert,
  retrainModel,
  type TrainingDataset,
  type TrainingJob,
  type TrainingConfig,
  type DeployedModel,
  type DeploymentConfig,
  type ValidationMetrics,
  type ModelAlert,
  // MLflow types and functions
  type MLflowExperiment,
  type MLflowRun,
  type MLflowRegisteredModel,
  type MLflowModelVersion,
  type MLflowRunComparison,
  type MLflowServingEndpoint,
  getMLflowStatus,
  listMLflowExperiments,
  listMLflowRuns,
  listMLflowModels,
  compareMLflowRuns,
  transitionMLflowModelStage,
  listMLflowServingEndpoints,
  deployMLflowModel,
} from '../../services/mlopsService';
import {
  Play,
  Pause,
  Square,
  Upload,
  Download,
  Settings,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Plus,
  Trash2,
  Edit3,
  Save,
  X,
  RefreshCw,
  Database,
  Cpu,
  BarChart3,
  TrendingUp,
  Clock,
  Target,
  Zap,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  GitBranch,
  Layers,
  Server,
  Activity,
  FileCode,
  Box,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

type SubcategoryType = 'experiments' | 'runs' | 'models' | 'serving' | null;

export function MLOps() {
  const [activeTab, setActiveTab] = useState<'registry' | 'training' | 'deployment' | 'monitoring'>('registry');
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelSpec | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Real data state
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [deployedModels, setDeployedModels] = useState<DeployedModel[]>([]);
  const [validationMetrics, setValidationMetrics] = useState<ValidationMetrics | null>(null);
  const [modelAlerts, setModelAlerts] = useState<ModelAlert[]>([]);

  // Collapsible subcategory panel state
  const [isSubcategoryPanelOpen, setIsSubcategoryPanelOpen] = useState(true);
  const [activeSubcategory, setActiveSubcategory] = useState<SubcategoryType>(null);

  // MLflow state
  const [mlflowStatus, setMlflowStatus] = useState<{ status: string; tracking_uri?: string } | null>(null);
  const [mlflowExperiments, setMlflowExperiments] = useState<MLflowExperiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<MLflowExperiment | null>(null);
  const [mlflowRuns, setMlflowRuns] = useState<MLflowRun[]>([]);
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [runComparison, setRunComparison] = useState<MLflowRunComparison | null>(null);
  const [mlflowModels, setMlflowModels] = useState<MLflowRegisteredModel[]>([]);
  const [servingEndpoints, setServingEndpoints] = useState<MLflowServingEndpoint[]>([]);
  
  // Modal states
  const [showNewDatasetModal, setShowNewDatasetModal] = useState(false);
  const [showNewTrainingModal, setShowNewTrainingModal] = useState(false);
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  
  // Form states
  const [newDataset, setNewDataset] = useState({
    name: '',
    description: '',
    assets: '',
    dateRange: '',
    path: ''
  });
  
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    optimizer: 'adam',
    loss_function: 'mse',
    validation_split: 0.2,
    early_stopping: true,
    patience: 10
  });
  
  const [deployConfig, setDeployConfig] = useState<DeploymentConfig>({
    model_id: '',
    environment: 'staging',
    replicas: 1,
    cpu_limit: '500m',
    memory_limit: '1Gi',
    auto_scale: false
  });

  useEffect(() => {
    loadModels();
    loadMLOpsData();
    loadMLflowData();
  }, []);

  const loadMLOpsData = async () => {
    try {
      const [datasets, jobs, models, metrics, alerts] = await Promise.all([
        listDatasets(),
        listTrainingJobs(),
        listDeployedModels(),
        getValidationMetrics(),
        getModelAlerts()
      ]);
      
      setTrainingDatasets(datasets);
      setTrainingJobs(jobs);
      setDeployedModels(models);
      setValidationMetrics(metrics);
      setModelAlerts(alerts);
    } catch (error) {
      console.error('Failed to load MLOps data:', error);
    }
  };

  const loadModels = async () => {
    setIsLoading(true);
    try {
      const modelList = await listModels();
      setModels(modelList);
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMLflowData = async () => {
    try {
      const [status, experiments, mlModels, endpoints] = await Promise.all([
        getMLflowStatus(),
        listMLflowExperiments(),
        listMLflowModels(),
        listMLflowServingEndpoints(),
      ]);

      setMlflowStatus(status);
      setMlflowExperiments(experiments);
      setMlflowModels(mlModels);
      setServingEndpoints(endpoints.endpoints || []);
    } catch (error) {
      console.error('Failed to load MLflow data:', error);
    }
  };

  const loadExperimentRuns = async (experimentName: string) => {
    try {
      const runs = await listMLflowRuns(experimentName);
      setMlflowRuns(runs);
    } catch (error) {
      console.error('Failed to load experiment runs:', error);
    }
  };

  const handleCompareRuns = async () => {
    if (selectedRuns.length < 2) return;
    try {
      const comparison = await compareMLflowRuns(selectedRuns);
      setRunComparison(comparison);
    } catch (error) {
      console.error('Failed to compare runs:', error);
    }
  };

  const handleTransitionStage = async (modelName: string, version: string, stage: 'None' | 'Staging' | 'Production' | 'Archived') => {
    try {
      await transitionMLflowModelStage(modelName, version, stage);
      await loadMLflowData();
    } catch (error) {
      console.error('Failed to transition model stage:', error);
    }
  };

  const toggleRunSelection = (runId: string) => {
    setSelectedRuns(prev =>
      prev.includes(runId)
        ? prev.filter(id => id !== runId)
        : [...prev, runId]
    );
  };

  // Prepare chart data for run comparison
  const comparisonChartData = useMemo(() => {
    if (!runComparison) return [];
    const metricKeys = Object.keys(runComparison.metrics);
    return metricKeys.map(metric => {
      const dataPoint: Record<string, any> = { metric };
      runComparison.runs.forEach(run => {
        dataPoint[run.run_name || run.run_id] = runComparison.metrics[metric][run.run_id];
      });
      return dataPoint;
    });
  }, [runComparison]);

  const handleModelSelect = async (modelId: string) => {
    try {
      const modelDetails = await getModelDetails(modelId);
      setSelectedModel(modelDetails);
    } catch (error) {
      console.error('Failed to load model details:', error);
    }
  };
  // Action handlers with real API calls
  const handleCreateDataset = async () => {
    try {
      const dataset = await createDataset({
        name: newDataset.name,
        description: newDataset.description,
        assets: newDataset.assets.split(',').map(s => s.trim()),
        dateRange: newDataset.dateRange,
        path: newDataset.path
      });
      
      setTrainingDatasets(prev => [...prev, dataset]);
      setShowNewDatasetModal(false);
      setNewDataset({ name: '', description: '', assets: '', dateRange: '', path: '' });
    } catch (error) {
      console.error('Failed to create dataset:', error);
    }
  };

  const handleStartTraining = async (modelId: string, datasetId: string) => {
    try {
      const job = await startTraining(modelId, datasetId, trainingConfig);
      setTrainingJobs(prev => [...prev, job]);
      setShowNewTrainingModal(false);
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const handlePauseTraining = async (jobId: string) => {
    try {
      await pauseTraining(jobId);
      setTrainingJobs(prev => prev.map(job => 
        job.id === jobId ? { ...job, status: 'PAUSED' as const } : job
      ));
    } catch (error) {
      console.error('Failed to pause training:', error);
    }
  };

  const handleStopTraining = async (jobId: string) => {
    try {
      await stopTraining(jobId);
      setTrainingJobs(prev => prev.filter(job => job.id !== jobId));
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const handleDeployModel = async () => {
    try {
      const model = await deployModel(deployConfig);
      setDeployedModels(prev => [...prev, model]);
      setShowDeployModal(false);
    } catch (error) {
      console.error('Failed to deploy model:', error);
    }
  };

  const handleScaleModel = async (modelId: string, replicas: number) => {
    try {
      await scaleModel(modelId, replicas);
      setDeployedModels(prev => prev.map(model => 
        model.id === modelId ? { ...model, replicas } : model
      ));
    } catch (error) {
      console.error('Failed to scale model:', error);
    }
  };

  const handleRollbackModel = async (modelId: string, version: string) => {
    try {
      await rollbackModel(modelId, version);
      const models = await listDeployedModels();
      setDeployedModels(models);
    } catch (error) {
      console.error('Failed to rollback model:', error);
    }
  };

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId);
      setModelAlerts(prev => prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      await deleteDataset(datasetId);
      setTrainingDatasets(prev => prev.filter(ds => ds.id !== datasetId));
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    }
  };

  const handlePromoteModel = async (modelId: string) => {
    try {
      await promoteModel(modelId);
      setDeployedModels(prev => prev.map(model => 
        model.id === modelId ? { ...model, status: 'ACTIVE' as const } : model
      ));
    } catch (error) {
      console.error('Failed to promote model:', error);
    }
  };

  const handleStopModel = async (modelId: string) => {
    try {
      await stopModel(modelId);
      setDeployedModels(prev => prev.map(model => 
        model.id === modelId ? { ...model, status: 'DEPRECATED' as const } : model
      ));
    } catch (error) {
      console.error('Failed to stop model:', error);
    }
  };

  const handleRetrainModel = async (modelId: string) => {
    try {
      const datasetId = trainingDatasets[0]?.id;
      if (!datasetId) {
        console.error('No datasets available for retraining');
        return;
      }
      
      const job = await retrainModel(modelId, datasetId);
      setTrainingJobs(prev => [...prev, job]);
    } catch (error) {
      console.error('Failed to retrain model:', error);
    }
  };
  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const getStatusColor = (status: string) => {
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
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'text-red-400';
      case 'HIGH': return 'text-orange-400';
      case 'MEDIUM': return 'text-yellow-400';
      case 'LOW': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs overflow-hidden">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2">
        <div className="text-center text-[#ff8c00] text-sm tracking-wide">
          MLOPS - MACHINE LEARNING OPERATIONS
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-600">
        {[
          { key: 'registry', label: 'Model Registry' },
          { key: 'training', label: 'Training Jobs' },
          { key: 'deployment', label: 'Deployment' },
          { key: 'monitoring', label: 'Model Monitoring' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-orange-900 text-orange-400 border-b-2 border-orange-400'
                : 'text-gray-400 hover:text-orange-400'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Main Content Area with Collapsible Subcategory Panel */}
      <div className="flex-1 flex overflow-hidden">
        {/* Collapsible Subcategory Panel */}
        <div className={`${isSubcategoryPanelOpen ? 'w-56' : 'w-12'} border-r border-[#444] bg-[#0a0a0a] transition-all duration-300 flex-shrink-0 flex flex-col`}>
          <button
            onClick={() => setIsSubcategoryPanelOpen(!isSubcategoryPanelOpen)}
            className="w-full py-2 px-2 flex items-center justify-center text-[#ff8c00] hover:bg-[#1a1a1a] border-b border-[#444]"
          >
            {isSubcategoryPanelOpen ? (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span className="ml-2 text-[10px]">MLFLOW</span>
              </>
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>

          {isSubcategoryPanelOpen && (
            <div className="flex-1 overflow-y-auto">
              {/* MLflow Status */}
              <div className="px-2 py-2 border-b border-[#333]">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${mlflowStatus?.status === 'available' ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className="text-[9px] text-[#666]">
                    {mlflowStatus?.status === 'available' ? 'CONNECTED' : 'OFFLINE'}
                  </span>
                </div>
              </div>

              {/* Subcategory Navigation */}
              <div className="py-2">
                <div className="px-2 text-[9px] text-[#666] mb-2">MLFLOW</div>

                <button
                  onClick={() => setActiveSubcategory(activeSubcategory === 'experiments' ? null : 'experiments')}
                  className={`w-full px-2 py-2 flex items-center space-x-2 text-[10px] hover:bg-[#1a1a1a] ${activeSubcategory === 'experiments' ? 'bg-[#1a1a1a] text-[#ff8c00]' : 'text-[#999]'}`}
                >
                  <FlaskConical className="w-3 h-3" />
                  <span>Experiments</span>
                  <span className="ml-auto text-[#666]">{mlflowExperiments.length}</span>
                </button>

                <button
                  onClick={() => setActiveSubcategory(activeSubcategory === 'runs' ? null : 'runs')}
                  className={`w-full px-2 py-2 flex items-center space-x-2 text-[10px] hover:bg-[#1a1a1a] ${activeSubcategory === 'runs' ? 'bg-[#1a1a1a] text-[#ff8c00]' : 'text-[#999]'}`}
                >
                  <GitBranch className="w-3 h-3" />
                  <span>Runs</span>
                  <span className="ml-auto text-[#666]">{mlflowRuns.length}</span>
                </button>

                <button
                  onClick={() => setActiveSubcategory(activeSubcategory === 'models' ? null : 'models')}
                  className={`w-full px-2 py-2 flex items-center space-x-2 text-[10px] hover:bg-[#1a1a1a] ${activeSubcategory === 'models' ? 'bg-[#1a1a1a] text-[#ff8c00]' : 'text-[#999]'}`}
                >
                  <Layers className="w-3 h-3" />
                  <span>Models</span>
                  <span className="ml-auto text-[#666]">{mlflowModels.length}</span>
                </button>

                <button
                  onClick={() => setActiveSubcategory(activeSubcategory === 'serving' ? null : 'serving')}
                  className={`w-full px-2 py-2 flex items-center space-x-2 text-[10px] hover:bg-[#1a1a1a] ${activeSubcategory === 'serving' ? 'bg-[#1a1a1a] text-[#ff8c00]' : 'text-[#999]'}`}
                >
                  <Server className="w-3 h-3" />
                  <span>Serving</span>
                  <span className="ml-auto text-[#666]">{servingEndpoints.length}</span>
                </button>
              </div>

              {/* Subcategory Content */}
              {activeSubcategory === 'experiments' && (
                <div className="border-t border-[#333] p-2">
                  <div className="text-[9px] text-[#666] mb-2">EXPERIMENTS</div>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {mlflowExperiments.map(exp => (
                      <button
                        key={exp.experiment_id}
                        onClick={() => {
                          setSelectedExperiment(exp);
                          loadExperimentRuns(exp.name);
                        }}
                        className={`w-full px-2 py-1 text-left text-[9px] hover:bg-[#222] rounded ${selectedExperiment?.experiment_id === exp.experiment_id ? 'bg-[#222] text-[#ff8c00]' : 'text-[#999]'}`}
                      >
                        <div className="truncate">{exp.name}</div>
                        <div className="text-[8px] text-[#555]">{exp.lifecycle_stage}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {activeSubcategory === 'runs' && selectedExperiment && (
                <div className="border-t border-[#333] p-2">
                  <div className="text-[9px] text-[#666] mb-2">RUNS - {selectedExperiment.name}</div>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {mlflowRuns.map(run => (
                      <button
                        key={run.run_id}
                        onClick={() => toggleRunSelection(run.run_id)}
                        className={`w-full px-2 py-1 text-left text-[9px] hover:bg-[#222] rounded flex items-center ${selectedRuns.includes(run.run_id) ? 'bg-[#222] text-[#ff8c00]' : 'text-[#999]'}`}
                      >
                        <input
                          type="checkbox"
                          checked={selectedRuns.includes(run.run_id)}
                          onChange={() => {}}
                          className="mr-2 w-3 h-3"
                        />
                        <div className="flex-1 truncate">
                          <div>{run.run_name || run.run_id.slice(0, 8)}</div>
                          <div className={`text-[8px] ${run.status === 'FINISHED' ? 'text-green-400' : run.status === 'RUNNING' ? 'text-yellow-400' : 'text-red-400'}`}>
                            {run.status}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                  {selectedRuns.length >= 2 && (
                    <button
                      onClick={handleCompareRuns}
                      className="w-full mt-2 px-2 py-1 bg-[#ff8c00] text-black text-[9px] hover:bg-[#ffaa33]"
                    >
                      COMPARE ({selectedRuns.length})
                    </button>
                  )}
                </div>
              )}

              {activeSubcategory === 'models' && (
                <div className="border-t border-[#333] p-2">
                  <div className="text-[9px] text-[#666] mb-2">REGISTERED MODELS</div>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {mlflowModels.map(model => (
                      <div key={model.name} className="px-2 py-1 text-[9px] text-[#999] hover:bg-[#222] rounded">
                        <div className="text-[#fff]">{model.name}</div>
                        <div className="flex items-center space-x-2 mt-1">
                          {model.latest_versions.map(v => (
                            <span
                              key={v.version}
                              className={`px-1 py-0.5 text-[8px] rounded ${
                                v.current_stage === 'Production' ? 'bg-green-900 text-green-400' :
                                v.current_stage === 'Staging' ? 'bg-yellow-900 text-yellow-400' :
                                'bg-gray-800 text-gray-400'
                              }`}
                            >
                              v{v.version} {v.current_stage}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeSubcategory === 'serving' && (
                <div className="border-t border-[#333] p-2">
                  <div className="text-[9px] text-[#666] mb-2">SERVING ENDPOINTS</div>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {servingEndpoints.length > 0 ? (
                      servingEndpoints.map((ep, idx) => (
                        <div key={idx} className="px-2 py-1 text-[9px] text-[#999] hover:bg-[#222] rounded">
                          <div className="text-[#00ff00]">{ep.model_name}</div>
                          <div className="text-[8px]">v{ep.model_version} - {ep.status}</div>
                        </div>
                      ))
                    ) : (
                      <div className="text-[9px] text-[#555]">No active endpoints</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'registry' && (
            <div className="p-4 space-y-4">
              {/* MLflow Run Comparison Chart */}
              {runComparison && runComparison.runs.length > 0 && (
                <div className="border border-[#444] mb-4">
                  <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444] flex justify-between items-center">
                    <div className="text-[#ff8c00]">RUN COMPARISON</div>
                    <button
                      onClick={() => setRunComparison(null)}
                      className="text-[#666] hover:text-[#fff]"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                  <div className="p-3">
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={comparisonChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="metric" tick={{ fontSize: 9, fill: '#999' }} />
                          <YAxis tick={{ fontSize: 9, fill: '#999' }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelStyle={{ color: '#ff8c00' }}
                          />
                          <Legend wrapperStyle={{ fontSize: 9 }} />
                          {runComparison.runs.map((run, idx) => (
                            <Bar
                              key={run.run_id}
                              dataKey={run.run_name || run.run_id}
                              fill={['#ff8c00', '#00ff00', '#00ffff', '#ff00ff'][idx % 4]}
                            />
                          ))}
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Registry Header with Actions */}
              <div className="flex justify-between items-center">
                <div className="text-[#ff8c00] text-sm">MODEL REGISTRY & DATASETS</div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setShowNewDatasetModal(true)}
                    className="px-3 py-1 bg-[#ff8c00] text-black text-xs font-medium hover:bg-[#ffaa33] transition-colors flex items-center space-x-1"
                  >
                    <Plus className="w-3 h-3" />
                    <span>NEW DATASET</span>
                  </button>
                  <button
                    onClick={() => {
                      loadModels();
                      loadMLOpsData();
                      loadMLflowData();
                    }}
                    className="px-3 py-1 bg-gray-700 text-white text-xs font-medium hover:bg-gray-600 transition-colors flex items-center space-x-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>REFRESH</span>
                  </button>
                </div>
              </div>

            {/* Training Datasets */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">TRAINING DATASETS</div>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">DATASET</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">RECORDS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">FEATURES</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">QUALITY</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingDatasets.map((dataset) => (
                    <tr key={dataset.id} className="border-b border-[#222]">
                      <td className="px-3 py-2">
                        <div className="text-[#00ff00]">{dataset.name}</div>
                        <div className="text-[#666] text-[9px]">{dataset.description}</div>
                        <div className="text-[#666] text-[9px]">{dataset.path}</div>
                      </td>
                      <td className="px-3 py-2 text-right text-[#fff]">{dataset.records.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right text-[#fff]">{dataset.features}</td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">{formatNumber(dataset.quality, 1)}%</td>
                      <td className="px-3 py-2 text-center">
                        <span className={getStatusColor(dataset.status)}>{dataset.status}</span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <div className="flex justify-center space-x-2">
                          <button 
                            className="text-[#00ff00] hover:text-[#fff] text-[9px]"
                            title="View Dataset"
                          >
                            <Database className="w-3 h-3" />
                          </button>
                          <button 
                            className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
                            title="Edit Dataset"
                          >
                            <Edit3 className="w-3 h-3" />
                          </button>
                          <button 
                            onClick={() => handleDeleteDataset(dataset.id)}
                            className="text-[#ff0000] hover:text-[#fff] text-[9px]"
                            title="Delete Dataset"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Available Models */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">AVAILABLE MODELS</div>
              </div>
              <div className="p-3">
                {isLoading ? (
                  <div className="text-center text-gray-400 py-8">Loading models...</div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {models.map((model) => (
                      <div
                        key={model.id}
                        onClick={() => handleModelSelect(model.id)}
                        className="border border-[#444] p-3 hover:border-[#ff8c00] cursor-pointer transition-colors"
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div className="text-[#00ff00] font-medium">{model.name}</div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setShowConfigModal(true);
                              setSelectedModel(model);
                            }}
                            className="text-[#666] hover:text-[#ff8c00] transition-colors"
                          >
                            <Settings className="w-3 h-3" />
                          </button>
                        </div>
                        <div className="text-[#666] text-[10px] mb-2">{formatCategory(model.category)}</div>
                        <div className="space-y-1 text-[10px]">
                          <div className="flex justify-between">
                            <span className="text-[#666]">Production Ready:</span>
                            <span className={model.production_ready ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                              {model.production_ready ? 'YES' : 'NO'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-[#666]">Latency:</span>
                            <span className="text-[#fff]">{model.latency_class}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-[#666]">Memory:</span>
                            <span className="text-[#fff]">{model.memory_mb}MB</span>
                          </div>
                        </div>
                        <div className="mt-2 pt-2 border-t border-[#333] flex space-x-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setShowNewTrainingModal(true);
                              setSelectedModel(model);
                            }}
                            className="text-[#00ff00] hover:text-[#fff] text-[9px] flex items-center space-x-1"
                          >
                            <Play className="w-3 h-3" />
                            <span>TRAIN</span>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setShowDeployModal(true);
                              setSelectedModel(model);
                              setDeployConfig(prev => ({...prev, model_id: model.id}));
                            }}
                            className="text-[#ff8c00] hover:text-[#fff] text-[9px] flex items-center space-x-1"
                          >
                            <Upload className="w-3 h-3" />
                            <span>DEPLOY</span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        {activeTab === 'training' && (
          <div className="p-4 space-y-4">
            {/* Training Jobs Header */}
            <div className="flex justify-between items-center">
              <div className="text-[#ff8c00] text-sm">TRAINING JOBS</div>
              <button
                onClick={() => setShowNewTrainingModal(true)}
                className="px-3 py-1 bg-[#ff8c00] text-black text-xs font-medium hover:bg-[#ffaa33] transition-colors flex items-center space-x-1"
              >
                <Play className="w-3 h-3" />
                <span>START TRAINING</span>
              </button>
            </div>

            {/* Active Training Jobs */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">ACTIVE TRAINING JOBS</div>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">MODEL</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">PROGRESS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">TRAIN LOSS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">VAL LOSS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">ACCURACY</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingJobs.map((job) => (
                    <tr key={job.id} className="border-b border-[#222]">
                      <td className="px-3 py-2">
                        <div className="text-[#00ff00]">{job.model_name}</div>
                        <div className="text-[#666] text-[9px]">Dataset: {job.dataset_id}</div>
                        <div className="text-[#666] text-[9px]">Started: {job.started_at}</div>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={getStatusColor(job.status)}>{job.status}</span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="text-[#fff]">
                          {job.status === 'RUNNING' ? `${job.current_epoch}/${job.total_epochs}` : 
                           job.status === 'COMPLETED' ? '100%' : '-'}
                        </div>
                        {job.status === 'RUNNING' && (
                          <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                            <div 
                              className="bg-[#ff8c00] h-1 rounded-full" 
                              style={{ width: `${(job.current_epoch / job.total_epochs) * 100}%` }}
                            ></div>
                          </div>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">
                        {job.train_loss > 0 ? formatNumber(job.train_loss, 4) : '-'}
                      </td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">
                        {job.val_loss > 0 ? formatNumber(job.val_loss, 4) : '-'}
                      </td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">
                        {job.accuracy > 0 ? formatNumber(job.accuracy, 1) + '%' : '-'}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <div className="flex justify-center space-x-2">
                          {job.status === 'RUNNING' && (
                            <>
                              <button
                                onClick={() => handlePauseTraining(job.id)}
                                className="text-[#ffff00] hover:text-[#fff] text-[9px]"
                                title="Pause Training"
                              >
                                <Pause className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => handleStopTraining(job.id)}
                                className="text-[#ff0000] hover:text-[#fff] text-[9px]"
                                title="Stop Training"
                              >
                                <Square className="w-3 h-3" />
                              </button>
                            </>
                          )}
                          {job.status === 'PAUSED' && (
                            <button
                              onClick={() => handleStartTraining(job.model_id, job.dataset_id)}
                              className="text-[#00ff00] hover:text-[#fff] text-[9px]"
                              title="Resume Training"
                            >
                              <Play className="w-3 h-3" />
                            </button>
                          )}
                          <button
                            onClick={() => {
                              setTrainingConfig(job.config);
                              setShowConfigModal(true);
                            }}
                            className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
                            title="View Config"
                          >
                            <Settings className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Training Configuration Panel */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">TRAINING CONFIGURATION</div>
              </div>
              <div className="p-3 grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">LEARNING RATE</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={trainingConfig.learning_rate}
                      onChange={(e) => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">BATCH SIZE</label>
                    <input
                      type="number"
                      value={trainingConfig.batch_size}
                      onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">EPOCHS</label>
                    <input
                      type="number"
                      value={trainingConfig.epochs}
                      onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">OPTIMIZER</label>
                    <select
                      value={trainingConfig.optimizer}
                      onChange={(e) => setTrainingConfig({...trainingConfig, optimizer: e.target.value})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    >
                      <option value="adam">Adam</option>
                      <option value="adamw">AdamW</option>
                      <option value="sgd">SGD</option>
                      <option value="rmsprop">RMSprop</option>
                    </select>
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">LOSS FUNCTION</label>
                    <select
                      value={trainingConfig.loss_function}
                      onChange={(e) => setTrainingConfig({...trainingConfig, loss_function: e.target.value})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    >
                      <option value="mse">Mean Squared Error</option>
                      <option value="categorical_crossentropy">Categorical Crossentropy</option>
                      <option value="binary_crossentropy">Binary Crossentropy</option>
                      <option value="huber">Huber Loss</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">VALIDATION SPLIT</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.5"
                      value={trainingConfig.validation_split}
                      onChange={(e) => setTrainingConfig({...trainingConfig, validation_split: parseFloat(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">PATIENCE (Early Stopping)</label>
                    <input
                      type="number"
                      value={trainingConfig.patience}
                      onChange={(e) => setTrainingConfig({...trainingConfig, patience: parseInt(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={trainingConfig.early_stopping}
                      onChange={(e) => setTrainingConfig({...trainingConfig, early_stopping: e.target.checked})}
                      className="text-[#ff8c00]"
                    />
                    <label className="text-[#666] text-[10px]">ENABLE EARLY STOPPING</label>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'deployment' && (
          <div className="p-4 space-y-4">
            {/* Deployment Header */}
            <div className="flex justify-between items-center">
              <div className="text-[#ff8c00] text-sm">MODEL DEPLOYMENT</div>
              <button
                onClick={() => setShowDeployModal(true)}
                className="px-3 py-1 bg-[#ff8c00] text-black text-xs font-medium hover:bg-[#ffaa33] transition-colors flex items-center space-x-1"
              >
                <Upload className="w-3 h-3" />
                <span>DEPLOY MODEL</span>
              </button>
            </div>

            {/* Deployed Models */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">PRODUCTION MODELS</div>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">MODEL</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">REPLICAS</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">CPU</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">MEMORY</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">REQ/MIN</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {deployedModels.map((model) => (
                    <tr key={model.id} className="border-b border-[#222]">
                      <td className="px-3 py-2">
                        <div className="text-[#00ff00]">{model.name}</div>
                        <div className="text-[#666] text-[9px]">{model.version} ({model.hash})</div>
                        <div className="text-[#666] text-[9px]">{model.endpoint}</div>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={getStatusColor(model.status)}>{model.status}</span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex items-center justify-end space-x-1">
                          <span className="text-[#fff]">{model.replicas}</span>
                          <div className="flex space-x-1">
                            <button
                              onClick={() => handleScaleModel(model.id, Math.max(1, model.replicas - 1))}
                              className="text-[#ff0000] hover:text-[#fff] text-[8px] w-4 h-4 flex items-center justify-center border border-[#444]"
                              disabled={model.replicas <= 1}
                            >
                              -
                            </button>
                            <button
                              onClick={() => handleScaleModel(model.id, model.replicas + 1)}
                              className="text-[#00ff00] hover:text-[#fff] text-[8px] w-4 h-4 flex items-center justify-center border border-[#444]"
                            >
                              +
                            </button>
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="text-[#fff]">{formatNumber(model.cpu_usage, 1)}%</div>
                        <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                          <div 
                            className={`h-1 rounded-full ${model.cpu_usage > 80 ? 'bg-red-400' : model.cpu_usage > 60 ? 'bg-yellow-400' : 'bg-green-400'}`}
                            style={{ width: `${model.cpu_usage}%` }}
                          ></div>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="text-[#fff]">{formatNumber(model.memory_usage, 1)}%</div>
                        <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                          <div 
                            className={`h-1 rounded-full ${model.memory_usage > 80 ? 'bg-red-400' : model.memory_usage > 60 ? 'bg-yellow-400' : 'bg-green-400'}`}
                            style={{ width: `${model.memory_usage}%` }}
                          ></div>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">{model.requests_per_minute.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center">
                        <div className="flex justify-center space-x-2">
                          <button
                            onClick={() => handleRollbackModel(model.id, 'previous')}
                            className="text-[#ffff00] hover:text-[#fff] text-[9px]"
                            title="Rollback"
                          >
                            <RefreshCw className="w-3 h-3" />
                          </button>
                          {model.status === 'TESTING' && (
                            <button
                              onClick={() => handlePromoteModel(model.id)}
                              className="text-[#00ff00] hover:text-[#fff] text-[9px]"
                              title="Promote to Production"
                            >
                              <CheckCircle className="w-3 h-3" />
                            </button>
                          )}
                          <button
                            onClick={() => handleStopModel(model.id)}
                            className="text-[#ff0000] hover:text-[#fff] text-[9px]"
                            title="Stop Model"
                          >
                            <XCircle className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Deployment Configuration */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">DEPLOYMENT CONFIGURATION</div>
              </div>
              <div className="p-3 grid grid-cols-3 gap-4">
                <div className="space-y-3">
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">ENVIRONMENT</label>
                    <select
                      value={deployConfig.environment}
                      onChange={(e) => setDeployConfig({...deployConfig, environment: e.target.value})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    >
                      <option value="staging">Staging</option>
                      <option value="production">Production</option>
                      <option value="testing">Testing</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">REPLICAS</label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={deployConfig.replicas}
                      onChange={(e) => setDeployConfig({...deployConfig, replicas: parseInt(e.target.value)})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                    />
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">CPU LIMIT</label>
                    <input
                      type="text"
                      value={deployConfig.cpu_limit}
                      onChange={(e) => setDeployConfig({...deployConfig, cpu_limit: e.target.value})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                      placeholder="500m"
                    />
                  </div>
                  <div>
                    <label className="block text-[#666] text-[10px] mb-1">MEMORY LIMIT</label>
                    <input
                      type="text"
                      value={deployConfig.memory_limit}
                      onChange={(e) => setDeployConfig({...deployConfig, memory_limit: e.target.value})}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                      placeholder="1Gi"
                    />
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={deployConfig.auto_scale}
                      onChange={(e) => setDeployConfig({...deployConfig, auto_scale: e.target.checked})}
                      className="text-[#ff8c00]"
                    />
                    <label className="text-[#666] text-[10px]">AUTO SCALING</label>
                  </div>
                  <button
                    onClick={handleDeployModel}
                    className="w-full px-3 py-2 bg-[#ff8c00] text-black text-xs font-medium hover:bg-[#ffaa33] transition-colors"
                  >
                    DEPLOY MODEL
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'monitoring' && (
          <div className="p-4 space-y-4">
            {/* Model Alerts */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444] flex justify-between items-center">
                <div className="text-[#ff8c00]">MODEL ALERTS</div>
                <div className="text-[#666] text-[10px]">
                  {modelAlerts.filter(a => !a.acknowledged).length} unacknowledged
                </div>
              </div>
              <div className="p-3 space-y-2 text-[10px] max-h-48 overflow-y-auto">
                {modelAlerts.map((alert) => (
                  <div key={alert.id} className={`flex items-center justify-between p-2 border border-[#333] ${alert.acknowledged ? 'opacity-50' : ''}`}>
                    <div className="flex items-center space-x-3">
                      <div className={`w-2 h-2 rounded-full ${getSeverityColor(alert.severity).replace('text-', 'bg-')}`}></div>
                      <span className="text-[#666]">{alert.timestamp}</span>
                      <span className="text-[#fff]">{alert.message}</span>
                      <span className={getSeverityColor(alert.severity)}>[{alert.severity}]</span>
                    </div>
                    <div className="flex space-x-2">
                      {!alert.acknowledged && (
                        <button
                          onClick={() => handleAcknowledgeAlert(alert.id)}
                          className="text-[#00ff00] hover:text-[#fff] text-[9px]"
                          title="Acknowledge"
                        >
                          <CheckCircle className="w-3 h-3" />
                        </button>
                      )}
                      <button
                        onClick={() => console.log('View alert details:', alert.id)}
                        className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
                        title="Details"
                      >
                        <AlertTriangle className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Performance Metrics */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">MODEL VALIDATION METRICS</div>
              </div>
              <div className="p-3">
                {validationMetrics && (
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-[#666]">TEMPORAL CONTINUITY</span>
                        <span className="text-[#00ff00]">{formatNumber(validationMetrics.temporal_continuity, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-[#00ff00] h-2 rounded-full" style={{ width: `${validationMetrics.temporal_continuity}%` }}></div>
                      </div>
                      
                      <div className="flex justify-between text-[10px]">
                        <span className="text-[#666]">REGIME SEPARABILITY</span>
                        <span className="text-[#00ff00]">{formatNumber(validationMetrics.regime_separability, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-[#00ff00] h-2 rounded-full" style={{ width: `${validationMetrics.regime_separability}%` }}></div>
                      </div>
                      
                      <div className="flex justify-between text-[10px]">
                        <span className="text-[#666]">FEATURE STABILITY</span>
                        <span className="text-[#00ff00]">{formatNumber(validationMetrics.feature_stability, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-[#00ff00] h-2 rounded-full" style={{ width: `${validationMetrics.feature_stability}%` }}></div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-[#666]">PREDICTION CONSISTENCY</span>
                        <span className="text-[#00ff00]">{formatNumber(validationMetrics.prediction_consistency, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-[#00ff00] h-2 rounded-full" style={{ width: `${validationMetrics.prediction_consistency}%` }}></div>
                      </div>
                      
                      <div className="flex justify-between text-[10px]">
                        <span className="text-[#666]">DRIFT DETECTION</span>
                        <span className="text-[#00ff00]">{formatNumber(validationMetrics.drift_detection, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-[#00ff00] h-2 rounded-full" style={{ width: `${validationMetrics.drift_detection}%` }}></div>
                      </div>
                      
                      <div className="flex justify-between text-[10px] border-t border-[#444] pt-2">
                        <span className="text-[#ff8c00]">OVERALL SCORE</span>
                        <span className="text-[#ff8c00] font-bold">{formatNumber(validationMetrics.overall_score, 1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-3">
                        <div className="bg-[#ff8c00] h-3 rounded-full" style={{ width: `${validationMetrics.overall_score}%` }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Model Drift Detection */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">MODEL DRIFT MONITORING</div>
              </div>
              <div className="p-3">
                <div className="grid grid-cols-3 gap-4 text-[10px]">
                  {deployedModels.map((model) => (
                    <div key={model.id} className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 flex justify-between items-center">
                        <span>{model.name.toUpperCase()}</span>
                        <button
                          onClick={() => handleRetrainModel(model.id)}
                          className="text-[#00ff00] hover:text-[#fff] text-[9px]"
                          title="Retrain Model"
                        >
                          <RefreshCw className="w-3 h-3" />
                        </button>
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-[#666]">Data Drift:</span>
                          <span className={model.id === 'prod_003' ? 'text-[#ff0000]' : model.id === 'prod_002' ? 'text-[#ffff00]' : 'text-[#00ff00]'}>
                            {model.id === 'prod_003' ? 'HIGH' : model.id === 'prod_002' ? 'MEDIUM' : 'LOW'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">Concept Drift:</span>
                          <span className={model.id === 'prod_003' ? 'text-[#ffff00]' : 'text-[#00ff00]'}>
                            {model.id === 'prod_003' ? 'MODERATE' : 'STABLE'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">Performance:</span>
                          <span className="text-[#00ff00]">{formatNumber(model.accuracy, 1)}%</span>
                        </div>
                        <div className="mt-2 pt-2 border-t border-[#333]">
                          <div className="flex justify-between text-[9px]">
                            <button
                              onClick={() => console.log('View drift details:', model.id)}
                              className="text-[#ff8c00] hover:text-[#fff]"
                            >
                              VIEW DETAILS
                            </button>
                            <button
                              onClick={() => console.log('Configure alerts:', model.id)}
                              className="text-[#666] hover:text-[#fff]"
                            >
                              ALERTS
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>
      {/* Modals */}
      {/* New Dataset Modal */}
      {showNewDatasetModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#444] p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-sm">CREATE NEW DATASET</h3>
              <button
                onClick={() => setShowNewDatasetModal(false)}
                className="text-[#666] hover:text-[#fff]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-[#666] text-[10px] mb-1">DATASET NAME</label>
                <input
                  type="text"
                  value={newDataset.name}
                  onChange={(e) => setNewDataset({...newDataset, name: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  placeholder="FX_REGIME_FEATURES_2024"
                />
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">DESCRIPTION</label>
                <textarea
                  value={newDataset.description}
                  onChange={(e) => setNewDataset({...newDataset, description: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1 h-16"
                  placeholder="High-frequency FX regime detection features..."
                />
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">ASSETS (comma-separated)</label>
                <input
                  type="text"
                  value={newDataset.assets}
                  onChange={(e) => setNewDataset({...newDataset, assets: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  placeholder="EURUSD, GBPUSD, USDJPY"
                />
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">DATE RANGE</label>
                <input
                  type="text"
                  value={newDataset.dateRange}
                  onChange={(e) => setNewDataset({...newDataset, dateRange: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  placeholder="2020-01-01 to 2024-01-15"
                />
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">DATA PATH</label>
                <input
                  type="text"
                  value={newDataset.path}
                  onChange={(e) => setNewDataset({...newDataset, path: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  placeholder="/data/fx_regime_features_2024.parquet"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => setShowNewDatasetModal(false)}
                className="px-3 py-1 bg-gray-700 text-white text-xs hover:bg-gray-600"
              >
                CANCEL
              </button>
              <button
                onClick={handleCreateDataset}
                className="px-3 py-1 bg-[#ff8c00] text-black text-xs hover:bg-[#ffaa33]"
              >
                CREATE DATASET
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Training Modal */}
      {showNewTrainingModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#444] p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-sm">START TRAINING JOB</h3>
              <button
                onClick={() => setShowNewTrainingModal(false)}
                className="text-[#666] hover:text-[#fff]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-[#666] text-[10px] mb-1">MODEL</label>
                <select
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  onChange={(e) => setSelectedModel(models.find(m => m.id === e.target.value) || null)}
                >
                  <option value="">Select Model</option>
                  {models.map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">DATASET</label>
                <select
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                >
                  <option value="">Select Dataset</option>
                  {trainingDatasets.filter(ds => ds.status === 'READY').map(dataset => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">EXPERIMENT NAME</label>
                <input
                  type="text"
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                  placeholder="tcn_regime_v2_experiment_001"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => setShowNewTrainingModal(false)}
                className="px-3 py-1 bg-gray-700 text-white text-xs hover:bg-gray-600"
              >
                CANCEL
              </button>
              <button
                onClick={() => {
                  if (selectedModel && trainingDatasets.length > 0) {
                    handleStartTraining(selectedModel.id, trainingDatasets[0].id);
                  }
                }}
                className="px-3 py-1 bg-[#ff8c00] text-black text-xs hover:bg-[#ffaa33]"
              >
                START TRAINING
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Deploy Model Modal */}
      {showDeployModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#444] p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-sm">DEPLOY MODEL</h3>
              <button
                onClick={() => setShowDeployModal(false)}
                className="text-[#666] hover:text-[#fff]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-[#666] text-[10px] mb-1">MODEL</label>
                <select
                  value={deployConfig.model_id}
                  onChange={(e) => setDeployConfig({...deployConfig, model_id: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                >
                  <option value="">Select Model</option>
                  {models.filter(m => m.production_ready).map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">ENVIRONMENT</label>
                <select
                  value={deployConfig.environment}
                  onChange={(e) => setDeployConfig({...deployConfig, environment: e.target.value})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                >
                  <option value="staging">Staging</option>
                  <option value="production">Production</option>
                  <option value="testing">Testing</option>
                </select>
              </div>
              <div>
                <label className="block text-[#666] text-[10px] mb-1">INITIAL REPLICAS</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={deployConfig.replicas}
                  onChange={(e) => setDeployConfig({...deployConfig, replicas: parseInt(e.target.value)})}
                  className="w-full bg-[#0a0a0a] border border-[#444] text-[#fff] text-xs px-2 py-1"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => setShowDeployModal(false)}
                className="px-3 py-1 bg-gray-700 text-white text-xs hover:bg-gray-600"
              >
                CANCEL
              </button>
              <button
                onClick={handleDeployModel}
                className="px-3 py-1 bg-[#ff8c00] text-black text-xs hover:bg-[#ffaa33]"
              >
                DEPLOY
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
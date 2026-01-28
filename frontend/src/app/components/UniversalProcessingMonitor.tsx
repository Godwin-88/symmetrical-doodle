import React, { useState, useEffect } from 'react';
import {
  RefreshCw,
  X,
  Play,
  Pause,
  RotateCcw,
  AlertCircle,
  CheckCircle,
  Clock,
  Download,
  FileText,
  Layers,
  Zap,
  Database,
  Wifi,
  WifiOff,
  Signal,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Settings,
  Activity,
  TrendingUp,
  AlertTriangle,
  Info
} from 'lucide-react';
import {
  MultiSourceIngestionJob,
  FileProcessingStatus,
  DataSourceType,
  getSourceName,
  getStatusColor,
  formatFileSize
} from '../../services/multiSourceService';
import {
  useMultiSourceWebSocket,
  ProcessingUpdate,
  ProcessingPhase,
  DetailedFileProgress,
  SourceProgressUpdate
} from '../../services/multiSourceWebSocket';

interface UniversalProcessingMonitorProps {
  jobs: MultiSourceIngestionJob[];
  onJobCancel: (jobId: string) => void;
  onJobPause: (jobId: string) => void;
  onJobResume: (jobId: string) => void;
  onJobRetry: (jobId: string) => void;
  onFileRetry: (jobId: string, fileId: string) => void;
}

export function UniversalProcessingMonitor({
  jobs,
  onJobCancel,
  onJobPause,
  onJobResume,
  onJobRetry,
  onFileRetry
}: UniversalProcessingMonitorProps) {
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set());
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [showDetailedProgress, setShowDetailedProgress] = useState(true);
  const [detailedFileProgress, setDetailedFileProgress] = useState<Map<string, DetailedFileProgress>>(new Map());
  const [sourceProgress, setSourceProgress] = useState<Map<string, SourceProgressUpdate>>(new Map());
  const [connectionStatus, setConnectionStatus] = useState<{
    isConnected: boolean;
    quality: string;
    queuedMessages: number;
  }>({ isConnected: false, quality: 'disconnected', queuedMessages: 0 });

  const {
    subscribe,
    subscribeToJob,
    subscribeToSource,
    cancelJob,
    pauseJob,
    resumeJob,
    retryFile,
    isConnected,
    getConnectionState,
    getConnectionQuality,
    getQueuedMessageCount
  } = useMultiSourceWebSocket();

  useEffect(() => {
    // Update connection status periodically
    const updateConnectionStatus = () => {
      setConnectionStatus({
        isConnected: isConnected(),
        quality: getConnectionQuality(),
        queuedMessages: getQueuedMessageCount()
      });
    };

    updateConnectionStatus();
    const interval = setInterval(updateConnectionStatus, 1000);

    // Subscribe to all processing updates
    const unsubscribe = subscribe((update: ProcessingUpdate) => {
      handleProcessingUpdate(update);
    });

    return () => {
      clearInterval(interval);
      unsubscribe();
    };
  }, []);

  const handleProcessingUpdate = (update: ProcessingUpdate) => {
    switch (update.type) {
      case 'file_progress':
        if (update.data.detailedProgress) {
          setDetailedFileProgress(prev => {
            const newMap = new Map(prev);
            newMap.set(update.data.fileId, update.data.detailedProgress);
            return newMap;
          });
        }
        break;

      case 'source_progress':
        if (update.sourceType && update.data.sourceProgress) {
          setSourceProgress(prev => {
            const newMap = new Map(prev);
            newMap.set(update.sourceType!, update.data.sourceProgress);
            return newMap;
          });
        }
        break;

      case 'phase_progress':
        // Handle individual phase updates
        if (update.fileId && update.phase) {
          setDetailedFileProgress(prev => {
            const newMap = new Map(prev);
            const existing = newMap.get(update.fileId!) || createEmptyFileProgress(update.fileId!);
            
            // Update the specific phase
            const updatedPhases = existing.phases.map(phase => 
              phase.name === update.phase!.name ? { ...phase, ...update.phase } : phase
            );
            
            newMap.set(update.fileId!, {
              ...existing,
              phases: updatedPhases,
              currentPhase: update.phase,
              overallProgress: calculateOverallProgress(updatedPhases)
            });
            
            return newMap;
          });
        }
        break;

      case 'error':
        // Handle error updates with retry information
        console.error('Processing error:', update.data);
        break;
    }
  };

  const createEmptyFileProgress = (fileId: string): DetailedFileProgress => ({
    fileId,
    fileName: `File ${fileId}`,
    sourceType: 'unknown',
    overallProgress: 0,
    currentPhase: {
      name: 'downloading',
      displayName: 'Downloading',
      progress: 0,
      status: 'pending'
    },
    phases: [
      { name: 'downloading', displayName: 'Downloading', progress: 0, status: 'pending' },
      { name: 'parsing', displayName: 'Parsing', progress: 0, status: 'pending' },
      { name: 'chunking', displayName: 'Chunking', progress: 0, status: 'pending' },
      { name: 'embedding', displayName: 'Embedding', progress: 0, status: 'pending' },
      { name: 'storing', displayName: 'Storing', progress: 0, status: 'pending' }
    ],
    estimatedTimeRemaining: 0,
    processingStartTime: new Date().toISOString()
  });

  const calculateOverallProgress = (phases: ProcessingPhase[]): number => {
    const totalProgress = phases.reduce((sum, phase) => sum + phase.progress, 0);
    return totalProgress / phases.length;
  };

  const toggleJobExpansion = (jobId: string) => {
    setExpandedJobs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(jobId)) {
        newSet.delete(jobId);
      } else {
        newSet.add(jobId);
      }
      return newSet;
    });
  };

  const toggleSourceExpansion = (sourceKey: string) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sourceKey)) {
        newSet.delete(sourceKey);
      } else {
        newSet.add(sourceKey);
      }
      return newSet;
    });
  };

  const getPhaseIcon = (phaseName: string) => {
    switch (phaseName) {
      case 'downloading':
        return <Download className="w-4 h-4" />;
      case 'parsing':
        return <FileText className="w-4 h-4" />;
      case 'chunking':
        return <Layers className="w-4 h-4" />;
      case 'embedding':
        return <Zap className="w-4 h-4" />;
      case 'storing':
        return <Database className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  const getPhaseStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'active':
        return 'text-yellow-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getConnectionIcon = () => {
    if (!connectionStatus.isConnected) {
      return <WifiOff className="w-4 h-4 text-red-400" />;
    }
    
    switch (connectionStatus.quality) {
      case 'excellent':
        return <Wifi className="w-4 h-4 text-green-400" />;
      case 'good':
        return <Signal className="w-4 h-4 text-yellow-400" />;
      case 'poor':
        return <Signal className="w-4 h-4 text-red-400" />;
      default:
        return <WifiOff className="w-4 h-4 text-gray-400" />;
    }
  };

  const formatTimeRemaining = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  };

  if (jobs.length === 0) {
    return (
      <div className="bg-[#1a1a1a] p-6 rounded border border-[#444]">
        <div className="text-center text-gray-400">
          <Activity className="w-12 h-12 mx-auto mb-3" />
          <div className="text-lg font-medium mb-2">No Active Processing Jobs</div>
          <div className="text-sm">
            Select files and start batch processing to monitor progress here
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Connection Status Header */}
      <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              {getConnectionIcon()}
              <span className="text-sm font-medium text-white">
                Real-time Monitor
              </span>
              <span className={`text-xs ${
                connectionStatus.isConnected ? 'text-green-400' : 'text-red-400'
              }`}>
                {connectionStatus.isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {connectionStatus.queuedMessages > 0 && (
              <div className="flex items-center space-x-1 text-xs text-yellow-400">
                <Clock className="w-3 h-3" />
                <span>{connectionStatus.queuedMessages} queued</span>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowDetailedProgress(!showDetailedProgress)}
              className={`p-1 rounded ${
                showDetailedProgress ? 'text-[#ff8c00]' : 'text-gray-400 hover:text-white'
              }`}
              title="Toggle detailed progress"
            >
              {showDetailedProgress ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
            <Settings className="w-4 h-4 text-gray-400 hover:text-white cursor-pointer" />
          </div>
        </div>
      </div>

      {/* Processing Jobs */}
      {jobs.map(job => (
        <UniversalJobCard
          key={job.id}
          job={job}
          isExpanded={expandedJobs.has(job.id)}
          onToggleExpansion={() => toggleJobExpansion(job.id)}
          onCancel={() => {
            cancelJob(job.id);
            onJobCancel(job.id);
          }}
          onPause={() => {
            pauseJob(job.id);
            onJobPause(job.id);
          }}
          onResume={() => {
            resumeJob(job.id);
            onJobResume(job.id);
          }}
          onRetry={() => onJobRetry(job.id)}
          onFileRetry={(fileId) => {
            retryFile(job.id, fileId);
            onFileRetry(job.id, fileId);
          }}
          showDetailedProgress={showDetailedProgress}
          detailedFileProgress={detailedFileProgress}
          sourceProgress={sourceProgress}
          expandedSources={expandedSources}
          onToggleSourceExpansion={toggleSourceExpansion}
        />
      ))}
    </div>
  );
}

// Enhanced Job Card Component
interface UniversalJobCardProps {
  job: MultiSourceIngestionJob;
  isExpanded: boolean;
  onToggleExpansion: () => void;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  onRetry: () => void;
  onFileRetry: (fileId: string) => void;
  showDetailedProgress: boolean;
  detailedFileProgress: Map<string, DetailedFileProgress>;
  sourceProgress: Map<string, SourceProgressUpdate>;
  expandedSources: Set<string>;
  onToggleSourceExpansion: (sourceKey: string) => void;
}

function UniversalJobCard({
  job,
  isExpanded,
  onToggleExpansion,
  onCancel,
  onPause,
  onResume,
  onRetry,
  onFileRetry,
  showDetailedProgress,
  detailedFileProgress,
  sourceProgress,
  expandedSources,
  onToggleSourceExpansion
}: UniversalJobCardProps) {
  const progressPercentage = job.totalFiles > 0 ? (job.processedFiles / job.totalFiles) * 100 : 0;
  const hasErrors = job.failedFiles > 0;
  const isPaused = job.status === 'paused';

  const getJobStatusIcon = () => {
    switch (job.status) {
      case 'running':
        return <RefreshCw className="w-4 h-4 text-yellow-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-blue-400" />;
      case 'cancelled':
        return <X className="w-4 h-4 text-gray-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getJobActions = () => {
    switch (job.status) {
      case 'running':
        return (
          <>
            <button
              onClick={onPause}
              className="p-1 text-blue-400 hover:text-blue-300"
              title="Pause job"
            >
              <Pause className="w-4 h-4" />
            </button>
            <button
              onClick={onCancel}
              className="p-1 text-red-400 hover:text-red-300"
              title="Cancel job"
            >
              <X className="w-4 h-4" />
            </button>
          </>
        );
      case 'paused':
        return (
          <>
            <button
              onClick={onResume}
              className="p-1 text-green-400 hover:text-green-300"
              title="Resume job"
            >
              <Play className="w-4 h-4" />
            </button>
            <button
              onClick={onCancel}
              className="p-1 text-red-400 hover:text-red-300"
              title="Cancel job"
            >
              <X className="w-4 h-4" />
            </button>
          </>
        );
      case 'failed':
        return (
          <button
            onClick={onRetry}
            className="p-1 text-blue-400 hover:text-blue-300"
            title="Retry job"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-[#2a2a2a] rounded border border-[#444] overflow-hidden">
      {/* Job Header */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            {getJobStatusIcon()}
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-white font-medium">
                  Multi-Source Job {job.id.slice(-8)}
                </span>
                {hasErrors && (
                  <AlertTriangle className="w-4 h-4 text-yellow-400" title="Has errors" />
                )}
                {isPaused && (
                  <Info className="w-4 h-4 text-blue-400" title="Job is paused" />
                )}
              </div>
              <div className="text-xs text-gray-400">
                {job.sourceProgress.length} sources • {job.totalFiles} files
                {job.priority !== 'normal' && (
                  <span className="ml-2 px-1 bg-[#ff8c00] text-black rounded text-xs">
                    {job.priority.toUpperCase()}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className={`text-xs font-medium ${getStatusColor(job.status)}`}>
              {job.status.toUpperCase()}
            </span>
            <button
              onClick={onToggleExpansion}
              className="p-1 text-gray-400 hover:text-white"
              title="Toggle details"
            >
              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
            {getJobActions()}
          </div>
        </div>

        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">
              {job.processedFiles} / {job.totalFiles} files processed
              {job.failedFiles > 0 && (
                <span className="text-red-400 ml-2">({job.failedFiles} failed)</span>
              )}
            </span>
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">{Math.round(progressPercentage)}%</span>
              {job.estimatedTimeRemaining && job.estimatedTimeRemaining > 0 && (
                <span className="text-xs text-gray-500">
                  ~{formatTimeRemaining(job.estimatedTimeRemaining)} remaining
                </span>
              )}
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                job.status === 'failed' ? 'bg-red-400' :
                job.status === 'completed' ? 'bg-green-400' :
                job.status === 'paused' ? 'bg-blue-400' : 'bg-[#ff8c00]'
              }`}
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-[#333] p-4 space-y-4">
          {/* Source-wise Progress */}
          {job.sourceProgress.map(sourceProgressData => {
            const sourceKey = `${job.id}-${sourceProgressData.sourceType}`;
            const isSourceExpanded = expandedSources.has(sourceKey);
            const sourceProgressUpdate = sourceProgress.get(sourceProgressData.sourceType);
            
            return (
              <div key={sourceProgressData.sourceType} className="space-y-3">
                <div 
                  className="flex items-center justify-between cursor-pointer hover:bg-[#1a1a1a] p-2 rounded"
                  onClick={() => onToggleSourceExpansion(sourceKey)}
                >
                  <div className="flex items-center space-x-3">
                    {isSourceExpanded ? 
                      <ChevronDown className="w-4 h-4 text-gray-400" /> : 
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                    }
                    <span className="text-sm font-medium text-white">
                      {getSourceName(sourceProgressData.sourceType as DataSourceType)}
                    </span>
                    <div className="flex items-center space-x-2 text-xs text-gray-400">
                      <span>{sourceProgressData.completed} / {sourceProgressData.total}</span>
                      {sourceProgressData.failed > 0 && (
                        <span className="text-red-400">({sourceProgressData.failed} failed)</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-400">
                    {Math.round((sourceProgressData.completed / sourceProgressData.total) * 100)}%
                  </div>
                </div>
                
                <div className="bg-gray-700 rounded-full h-1 ml-6">
                  <div 
                    className="bg-[#ff8c00] h-1 rounded-full transition-all duration-300"
                    style={{ width: `${(sourceProgressData.completed / sourceProgressData.total) * 100}%` }}
                  />
                </div>

                {/* Enhanced Source Details */}
                {isSourceExpanded && (
                  <div className="ml-6 space-y-3">
                    {/* Source Statistics */}
                    {sourceProgressUpdate && (
                      <div className="bg-[#1a1a1a] p-3 rounded border border-[#333]">
                        <div className="grid grid-cols-2 gap-4 text-xs">
                          <div>
                            <span className="text-gray-400">Avg Processing Time:</span>
                            <span className="text-white ml-2">
                              {formatTimeRemaining(sourceProgressUpdate.averageProcessingTime)}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-400">Est. Remaining:</span>
                            <span className="text-white ml-2">
                              {formatTimeRemaining(sourceProgressUpdate.estimatedTimeRemaining)}
                            </span>
                          </div>
                        </div>
                        
                        {/* Phase Distribution */}
                        {sourceProgressUpdate.currentPhaseDistribution && (
                          <div className="mt-3">
                            <div className="text-xs text-gray-400 mb-2">Current Phase Distribution:</div>
                            <div className="flex space-x-2">
                              {Object.entries(sourceProgressUpdate.currentPhaseDistribution).map(([phase, count]) => (
                                <div key={phase} className="flex items-center space-x-1 text-xs">
                                  {getPhaseIcon(phase)}
                                  <span className="text-white">{count}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Individual File Progress */}
                    {sourceProgressData.files.map(file => (
                      <DetailedFileProgressCard
                        key={file.id}
                        file={file}
                        detailedProgress={detailedFileProgress.get(file.id)}
                        onRetry={() => onFileRetry(file.id)}
                        showDetailedProgress={showDetailedProgress}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Detailed File Progress Card Component
interface DetailedFileProgressCardProps {
  file: FileProcessingStatus;
  detailedProgress?: DetailedFileProgress;
  onRetry: () => void;
  showDetailedProgress: boolean;
}

function DetailedFileProgressCard({
  file,
  detailedProgress,
  onRetry,
  showDetailedProgress
}: DetailedFileProgressCardProps) {
  const hasError = file.status === 'failed' || file.error;
  const canRetry = hasError && (file.retryCount || 0) < 3;

  const getPhaseIcon = (phaseName: string) => {
    switch (phaseName) {
      case 'downloading':
        return <Download className="w-3 h-3" />;
      case 'parsing':
        return <FileText className="w-3 h-3" />;
      case 'chunking':
        return <Layers className="w-3 h-3" />;
      case 'embedding':
        return <Zap className="w-3 h-3" />;
      case 'storing':
        return <Database className="w-3 h-3" />;
      default:
        return <Clock className="w-3 h-3" />;
    }
  };

  const getPhaseStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'active':
        return 'text-yellow-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className={`bg-[#1a1a1a] p-3 rounded border ${
      hasError ? 'border-red-400/30' : 'border-[#333]'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-white truncate max-w-[200px]" title={file.name}>
            {file.name}
          </span>
          <span className={`text-xs ${getStatusColor(file.status)}`}>
            {file.status.toUpperCase()}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          {file.estimatedTimeRemaining && file.estimatedTimeRemaining > 0 && (
            <span className="text-xs text-gray-400">
              ~{formatTimeRemaining(file.estimatedTimeRemaining)}
            </span>
          )}
          {canRetry && (
            <button
              onClick={onRetry}
              className="p-1 text-blue-400 hover:text-blue-300"
              title="Retry file processing"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{file.currentStep}</span>
          <span>{file.progress}%</span>
        </div>
        <div className="bg-gray-700 rounded-full h-1">
          <div 
            className={`h-1 rounded-full transition-all duration-300 ${
              hasError ? 'bg-red-400' : 'bg-[#ff8c00]'
            }`}
            style={{ width: `${file.progress}%` }}
          />
        </div>
      </div>

      {/* Detailed Phase Progress */}
      {showDetailedProgress && detailedProgress && (
        <div className="space-y-2">
          <div className="text-xs text-gray-400 mb-2">Processing Phases:</div>
          <div className="grid grid-cols-5 gap-2">
            {detailedProgress.phases.map(phase => (
              <div key={phase.name} className="text-center">
                <div className={`flex items-center justify-center mb-1 ${getPhaseStatusColor(phase.status)}`}>
                  {getPhaseIcon(phase.name)}
                </div>
                <div className="text-xs text-gray-400 mb-1">{phase.displayName}</div>
                <div className="bg-gray-700 rounded-full h-1">
                  <div 
                    className={`h-1 rounded-full transition-all duration-300 ${
                      phase.status === 'failed' ? 'bg-red-400' :
                      phase.status === 'completed' ? 'bg-green-400' :
                      phase.status === 'active' ? 'bg-yellow-400' : 'bg-gray-600'
                    }`}
                    style={{ width: `${phase.progress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">{phase.progress}%</div>
              </div>
            ))}
          </div>

          {/* Phase Metrics */}
          {detailedProgress.currentPhase.metrics && (
            <div className="bg-[#2a2a2a] p-2 rounded border border-[#333] mt-2">
              <div className="text-xs text-gray-400 mb-1">Current Phase Metrics:</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {detailedProgress.currentPhase.metrics.bytesProcessed && (
                  <div>
                    <span className="text-gray-400">Processed:</span>
                    <span className="text-white ml-1">
                      {formatFileSize(detailedProgress.currentPhase.metrics.bytesProcessed)}
                    </span>
                  </div>
                )}
                {detailedProgress.currentPhase.metrics.chunksCreated && (
                  <div>
                    <span className="text-gray-400">Chunks:</span>
                    <span className="text-white ml-1">
                      {detailedProgress.currentPhase.metrics.chunksCreated}
                    </span>
                  </div>
                )}
                {detailedProgress.currentPhase.metrics.embeddingsGenerated && (
                  <div>
                    <span className="text-gray-400">Embeddings:</span>
                    <span className="text-white ml-1">
                      {detailedProgress.currentPhase.metrics.embeddingsGenerated}
                    </span>
                  </div>
                )}
                {detailedProgress.currentPhase.metrics.qualityScore && (
                  <div>
                    <span className="text-gray-400">Quality:</span>
                    <span className="text-white ml-1">
                      {Math.round(detailedProgress.currentPhase.metrics.qualityScore * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Information */}
      {hasError && file.error && (
        <div className="bg-red-400/10 border border-red-400/30 p-2 rounded mt-2">
          <div className="flex items-center space-x-2 mb-1">
            <AlertCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm text-red-400 font-medium">Processing Error</span>
          </div>
          <div className="text-xs text-gray-300">{file.error}</div>
          {file.retryCount && (
            <div className="text-xs text-gray-400 mt-1">
              Retry attempt {file.retryCount} of 3
            </div>
          )}
        </div>
      )}

      {/* Processing Statistics */}
      {file.status === 'completed' && (
        <div className="bg-green-400/10 border border-green-400/30 p-2 rounded mt-2">
          <div className="grid grid-cols-3 gap-2 text-xs">
            {file.processingTime && (
              <div>
                <span className="text-gray-400">Time:</span>
                <span className="text-white ml-1">{formatTimeRemaining(file.processingTime)}</span>
              </div>
            )}
            {file.chunks && (
              <div>
                <span className="text-gray-400">Chunks:</span>
                <span className="text-white ml-1">{file.chunks}</span>
              </div>
            )}
            {file.qualityScore && (
              <div>
                <span className="text-gray-400">Quality:</span>
                <span className="text-white ml-1">{Math.round(file.qualityScore * 100)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function formatTimeRemaining(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}
import React, { useState, useEffect } from 'react';
import {
  Cloud,
  Folder,
  Upload,
  Archive,
  CheckCircle,
  AlertCircle,
  Settings,
  RefreshCw,
  X,
  Search,
  Filter,
  Eye,
  Trash2,
  Edit,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  FileText,
  SortAsc,
  SortDesc,
  Grid,
  List,
  Calendar,
  HardDrive,
  Clock,
  Tag,
  Star,
  MoreHorizontal
} from 'lucide-react';
import {
  DataSourceType,
  DataSource,
  UniversalFileMetadata,
  CrossSourceSelection,
  MultiSourceIngestionJob,
  ProcessedDocument,
  FileTreeNode,
  ProcessingOptions,
  getAvailableSources,
  connectGoogleDrive,
  disconnectGoogleDrive,
  browseSource,
  searchAcrossSources,
  startMultiSourceIngestion,
  getIngestionJobStatus,
  cancelIngestionJob,
  pauseIngestionJob,
  resumeIngestionJob,
  retryIngestionJob,
  getProcessedDocuments,
  updateDocumentMetadata,
  deleteDocument,
  reprocessDocument,
  getSourceName,
  getSourceIcon,
  formatFileSize,
  getStatusColor
} from '../../services/multiSourceService';
import { ConnectionModal } from './ConnectionModal';
import { UniversalProcessingMonitor } from './UniversalProcessingMonitor';
import { useMultiSourceWebSocket } from '../../services/multiSourceWebSocket';

interface MultiSourcePanelProps {
  onDocumentProcessed?: (document: ProcessedDocument) => void;
  onError?: (error: string) => void;
}

export function MultiSourcePanel({ onDocumentProcessed, onError }: MultiSourcePanelProps) {
  // State management
  const [availableSources, setAvailableSources] = useState<DataSource[]>([]);
  const [connectedSources, setConnectedSources] = useState<DataSource[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Map<string, UniversalFileMetadata>>(new Map());
  const [activeIngestionJobs, setActiveIngestionJobs] = useState<MultiSourceIngestionJob[]>([]);
  const [processedDocuments, setProcessedDocuments] = useState<ProcessedDocument[]>([]);
  
  // UI state
  const [activeTab, setActiveTab] = useState<'sources' | 'browse' | 'processing' | 'documents'>('sources');
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showConnectionModal, setShowConnectionModal] = useState<DataSourceType | null>(null);
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);
  const [showBatchConfig, setShowBatchConfig] = useState(false);
  const [batchProcessingOptions, setBatchProcessingOptions] = useState<Map<DataSourceType, ProcessingOptions>>(new Map());
  const [estimatedProcessingTime, setEstimatedProcessingTime] = useState<number>(0);
  const [jobQueue, setJobQueue] = useState<MultiSourceIngestionJob[]>([]);

  // WebSocket for real-time updates
  const { subscribe } = useMultiSourceWebSocket();

  useEffect(() => {
    loadInitialData();
    
    // Subscribe to WebSocket updates
    const unsubscribe = subscribe((update: any) => {
      switch (update.type) {
        case 'job_status':
          setActiveIngestionJobs(prev => 
            prev.map(job => job.id === update.jobId ? { ...job, ...update.data } : job)
          );
          break;
        
        case 'completion':
          // Refresh processed documents when a job completes
          if (update.data.status === 'completed') {
            getProcessedDocuments().then(setProcessedDocuments);
            if (onDocumentProcessed) {
              update.data.documents?.forEach((doc: ProcessedDocument) => {
                onDocumentProcessed(doc);
              });
            }
          }
          break;
        
        case 'error':
          onError?.(update.data.message);
          break;
      }
    });

    return unsubscribe;
  }, []);

  const loadInitialData = async () => {
    setIsLoading(true);
    try {
      const [sources, documents] = await Promise.all([
        getAvailableSources(),
        getProcessedDocuments()
      ]);
      
      setAvailableSources(sources);
      setConnectedSources(sources.filter(s => s.isConnected));
      setProcessedDocuments(documents);
    } catch (error) {
      console.error('Failed to load initial data:', error);
      onError?.('Failed to load data sources');
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnect = async (sourceType: DataSourceType, credentials?: any) => {
    setIsLoading(true);
    try {
      switch (sourceType) {
        case DataSourceType.GOOGLE_DRIVE:
          const result = await connectGoogleDrive();
          if (result.success) {
            await loadInitialData(); // Refresh sources
          } else {
            onError?.(result.error || 'Failed to connect to Google Drive');
          }
          break;
        
        default:
          // For local sources, they're automatically "connected"
          await loadInitialData();
          break;
      }
    } catch (error) {
      console.error('Connection failed:', error);
      onError?.(`Failed to connect to ${getSourceName(sourceType)}`);
    } finally {
      setIsLoading(false);
      setShowConnectionModal(null);
    }
  };

  const handleDisconnect = async (sourceType: DataSourceType) => {
    setIsLoading(true);
    try {
      switch (sourceType) {
        case DataSourceType.GOOGLE_DRIVE:
          const connectedSource = connectedSources.find(s => s.type === sourceType);
          if (connectedSource) {
            await disconnectGoogleDrive('connection_id'); // Would use actual connection ID
            await loadInitialData();
          }
          break;
        
        default:
          await loadInitialData();
          break;
      }
    } catch (error) {
      console.error('Disconnection failed:', error);
      onError?.(`Failed to disconnect from ${getSourceName(sourceType)}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (file: UniversalFileMetadata, selected: boolean) => {
    const newSelection = new Map(selectedFiles);
    if (selected) {
      newSelection.set(file.file_id, file);
    } else {
      newSelection.delete(file.file_id);
    }
    setSelectedFiles(newSelection);
  };

  const handleBatchIngestion = async () => {
    if (selectedFiles.size === 0) return;

    // Show batch configuration modal if not configured
    if (batchProcessingOptions.size === 0) {
      setShowBatchConfig(true);
      return;
    }

    setIsLoading(true);
    try {
      // Group files by source type
      const selectionsBySource = new Map<DataSourceType, string[]>();
      selectedFiles.forEach(file => {
        const existing = selectionsBySource.get(file.source_type) || [];
        existing.push(file.file_id);
        selectionsBySource.set(file.source_type, existing);
      });

      // Create cross-source selections with per-source processing options
      const selections: CrossSourceSelection[] = Array.from(selectionsBySource.entries()).map(
        ([sourceType, fileIds]) => ({
          sourceType,
          fileIds,
          processingOptions: batchProcessingOptions.get(sourceType) || {
            chunkSize: 1000,
            embeddingModel: 'text-embedding-3-large',
            preserveMath: true,
            category: 'Multi-Source Batch',
            tags: ['multi-source', 'batch', sourceType]
          }
        })
      );

      const job = await startMultiSourceIngestion(selections);
      setActiveIngestionJobs(prev => [...prev, job]);
      setJobQueue(prev => [...prev, job]);
      setSelectedFiles(new Map()); // Clear selection
      setActiveTab('processing'); // Switch to processing tab
      
      // Start polling for job status
      pollJobStatus(job.id);
      
    } catch (error) {
      console.error('Batch ingestion failed:', error);
      onError?.('Failed to start batch ingestion');
    } finally {
      setIsLoading(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const poll = async () => {
      try {
        const job = await getIngestionJobStatus(jobId);
        setActiveIngestionJobs(prev => 
          prev.map(j => j.id === jobId ? job : j)
        );

        if (job.status === 'completed' || job.status === 'failed') {
          // Refresh processed documents
          const documents = await getProcessedDocuments();
          setProcessedDocuments(documents);
          return; // Stop polling
        }

        // Continue polling if job is still running
        if (job.status === 'running') {
          setTimeout(poll, 2000); // Poll every 2 seconds
        }
      } catch (error) {
        console.error('Failed to poll job status:', error);
      }
    };

    poll();
  };

  const handleAdvancedSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    try {
      const results = await searchAcrossSources(searchQuery);
      
      // Update the file browser with search results
      // This would typically update the sourceFiles state with search results
      console.log('Search results:', results);
      
    } catch (error) {
      console.error('Advanced search failed:', error);
      onError?.('Search failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const getSelectedSourceCount = (selection: Map<string, UniversalFileMetadata>): number => {
    const sources = new Set();
    selection.forEach(file => sources.add(file.source_type));
    return sources.size;
  };

  const calculateProcessingTimeEstimate = (selection: Map<string, UniversalFileMetadata>): number => {
    let totalEstimate = 0;
    
    selection.forEach(file => {
      // Base processing time estimates (in seconds) per MB
      const baseTimePerMB = {
        [DataSourceType.GOOGLE_DRIVE]: 15, // Includes download time
        [DataSourceType.LOCAL_ZIP]: 8,     // Includes extraction time
        [DataSourceType.LOCAL_DIRECTORY]: 5, // Direct file access
        [DataSourceType.INDIVIDUAL_UPLOAD]: 5, // Already uploaded
        [DataSourceType.AWS_S3]: 12,       // Cloud download time
        [DataSourceType.AZURE_BLOB]: 12,   // Cloud download time
        [DataSourceType.GOOGLE_CLOUD_STORAGE]: 12 // Cloud download time
      };
      
      const fileSizeMB = file.size / (1024 * 1024);
      const baseTime = baseTimePerMB[file.source_type] || 10;
      
      // Add processing complexity factors
      const processingOptions = batchProcessingOptions.get(file.source_type);
      let complexityMultiplier = 1.0;
      
      if (processingOptions?.preserveMath) complexityMultiplier += 0.3;
      if (processingOptions?.embeddingModel === 'text-embedding-3-large') complexityMultiplier += 0.2;
      if (processingOptions?.chunkSize && processingOptions.chunkSize < 500) complexityMultiplier += 0.4;
      
      totalEstimate += fileSizeMB * baseTime * complexityMultiplier;
    });
    
    // Add queue processing overhead
    const queueOverhead = Math.max(30, totalEstimate * 0.1);
    return Math.ceil(totalEstimate + queueOverhead);
  };

  const updateProcessingTimeEstimate = () => {
    const estimate = calculateProcessingTimeEstimate(selectedFiles);
    setEstimatedProcessingTime(estimate);
  };

  // Update processing time estimate when selection changes
  useEffect(() => {
    updateProcessingTimeEstimate();
  }, [selectedFiles, batchProcessingOptions]);

  const formatEstimatedTime = (seconds: number): string => {
    if (seconds < 60) return `~${seconds}s`;
    if (seconds < 3600) return `~${Math.ceil(seconds / 60)}m`;
    return `~${Math.ceil(seconds / 3600)}h ${Math.ceil((seconds % 3600) / 60)}m`;
  };

  const renderSourcesTab = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[#ff8c00] font-medium">Data Sources</h3>
        <span className="text-sm text-gray-400">
          {connectedSources.length} connected
        </span>
      </div>
      
      <div className="space-y-2">
        {availableSources.map(source => (
          <SourceConnectionItem
            key={source.type}
            source={source}
            onConnect={() => setShowConnectionModal(source.type)}
            onDisconnect={() => handleDisconnect(source.type)}
            isLoading={isLoading}
          />
        ))}
      </div>
    </div>
  );

  const renderBrowseTab = () => (
    <div className="space-y-4">
      {/* Enhanced Search Bar */}
      <div className="space-y-3">
        <div className="flex items-center space-x-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search across all sources..."
              className="w-full bg-[#1a1a1a] border border-[#444] text-white text-sm pl-10 pr-3 py-2 rounded focus:border-[#ff8c00] focus:ring-1 focus:ring-[#ff8c00]"
            />
          </div>
          <button 
            className="px-3 py-2 bg-gray-700 text-white text-sm rounded hover:bg-gray-600 flex items-center space-x-1"
            title="Advanced Filters"
          >
            <Filter className="w-4 h-4" />
            <span>Filters</span>
          </button>
          <button 
            className="px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 flex items-center space-x-1"
            onClick={() => handleAdvancedSearch()}
            title="Advanced Search"
          >
            <Search className="w-4 h-4" />
            <span>Search</span>
          </button>
        </div>

        {/* Quick Filters */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400">Quick filters:</span>
          <button className="px-2 py-1 bg-[#2a2a2a] text-gray-300 rounded hover:bg-[#3a3a3a]">
            <Tag className="w-3 h-3 inline mr-1" />
            Processed
          </button>
          <button className="px-2 py-1 bg-[#2a2a2a] text-gray-300 rounded hover:bg-[#3a3a3a]">
            <Clock className="w-3 h-3 inline mr-1" />
            Pending
          </button>
          <button className="px-2 py-1 bg-[#2a2a2a] text-gray-300 rounded hover:bg-[#3a3a3a]">
            <Calendar className="w-3 h-3 inline mr-1" />
            Recent
          </button>
          <button className="px-2 py-1 bg-[#2a2a2a] text-gray-300 rounded hover:bg-[#3a3a3a]">
            <Star className="w-3 h-3 inline mr-1" />
            Favorites
          </button>
        </div>
      </div>

      {/* Enhanced Batch Actions */}
      <div className="bg-[#1a1a1a] p-4 rounded border border-[#444] space-y-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-sm text-gray-400">
              {selectedFiles.size} files selected across {getSelectedSourceCount(selectedFiles)} sources
            </div>
            {selectedFiles.size > 0 && (
              <div className="text-xs text-gray-500">
                Estimated processing time: {formatEstimatedTime(estimatedProcessingTime)}
              </div>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setSelectedFiles(new Map())}
              disabled={selectedFiles.size === 0}
              className="px-2 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-50"
            >
              Clear All
            </button>
            <button
              onClick={() => setShowBatchConfig(true)}
              disabled={selectedFiles.size === 0}
              className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-500 disabled:opacity-50 flex items-center space-x-1"
            >
              <Settings className="w-3 h-3" />
              <span>Configure</span>
            </button>
            <button
              onClick={handleBatchIngestion}
              disabled={selectedFiles.size === 0 || isLoading}
              className="px-4 py-1 bg-[#ff8c00] text-black text-xs rounded hover:bg-orange-600 disabled:opacity-50 flex items-center space-x-1"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Process Selected ({selectedFiles.size})</span>
            </button>
          </div>
        </div>

        {/* Source-wise Selection Summary */}
        {selectedFiles.size > 0 && (
          <div className="border-t border-[#333] pt-3">
            <div className="text-xs text-gray-400 mb-2">Selection by source:</div>
            <div className="flex flex-wrap gap-2">
              {Array.from(
                selectedFiles.values().reduce((acc, file) => {
                  const count = acc.get(file.source_type) || 0;
                  acc.set(file.source_type, count + 1);
                  return acc;
                }, new Map<DataSourceType, number>())
              ).map(([sourceType, count]) => (
                <div
                  key={sourceType}
                  className="flex items-center space-x-1 px-2 py-1 bg-[#2a2a2a] rounded text-xs"
                >
                  {getSourceIconComponent(sourceType)}
                  <span className="text-white">{getSourceName(sourceType)}</span>
                  <span className="text-[#ff8c00] font-medium">{count}</span>
                  {batchProcessingOptions.has(sourceType) && (
                    <CheckCircle className="w-3 h-3 text-green-400" title="Configured" />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* File Browser */}
      <UnifiedFileBrowser
        connectedSources={connectedSources}
        selectedFiles={selectedFiles}
        onFileSelect={handleFileSelect}
        searchQuery={searchQuery}
      />
    </div>
  );

  const renderProcessingTab = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[#ff8c00] font-medium">Processing Jobs</h3>
        <div className="flex items-center space-x-3">
          <span className="text-sm text-gray-400">
            {activeIngestionJobs.filter(j => j.status === 'running').length} active
          </span>
          <span className="text-sm text-gray-400">•</span>
          <span className="text-sm text-gray-400">
            {jobQueue.length} in queue
          </span>
        </div>
      </div>

      {/* Enhanced Universal Processing Monitor */}
      <UniversalProcessingMonitor
        jobs={activeIngestionJobs}
        onJobCancel={async (jobId) => {
          try {
            await cancelIngestionJob(jobId);
            setActiveIngestionJobs(prev => prev.filter(job => job.id !== jobId));
          } catch (error) {
            console.error('Failed to cancel job:', error);
            onError?.('Failed to cancel processing job');
          }
        }}
        onJobPause={async (jobId) => {
          try {
            await pauseIngestionJob(jobId);
            setActiveIngestionJobs(prev => 
              prev.map(job => job.id === jobId ? { ...job, status: 'paused' as const } : job)
            );
          } catch (error) {
            console.error('Failed to pause job:', error);
            onError?.('Failed to pause processing job');
          }
        }}
        onJobResume={async (jobId) => {
          try {
            await resumeIngestionJob(jobId);
            setActiveIngestionJobs(prev => 
              prev.map(job => job.id === jobId ? { ...job, status: 'running' as const } : job)
            );
          } catch (error) {
            console.error('Failed to resume job:', error);
            onError?.('Failed to resume processing job');
          }
        }}
        onJobRetry={async (jobId) => {
          try {
            const result = await retryIngestionJob(jobId);
            if (result.success) {
              // Add the new retry job to the list
              const newJob = await getIngestionJobStatus(result.newJobId);
              setActiveIngestionJobs(prev => [...prev, newJob]);
              pollJobStatus(result.newJobId);
            }
          } catch (error) {
            console.error('Failed to retry job:', error);
            onError?.('Failed to retry processing job');
          }
        }}
        onFileRetry={async (jobId, fileId) => {
          try {
            // File retry is handled via WebSocket
            console.log(`Retrying file ${fileId} in job ${jobId}`);
          } catch (error) {
            console.error('Failed to retry file:', error);
            onError?.('Failed to retry file processing');
          }
        }}
      />

      {/* Job Queue Management */}
      {jobQueue.length > 0 && (
        <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-white text-sm font-medium">Job Queue</h4>
            <button
              onClick={() => setJobQueue([])}
              className="text-xs text-gray-400 hover:text-white"
            >
              Clear Completed
            </button>
          </div>
          <div className="space-y-2">
            {jobQueue.slice(0, 5).map((job, index) => (
              <div key={job.id} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="text-gray-400">#{index + 1}</span>
                  <span className="text-white">{job.totalFiles} files</span>
                  <span className={`${getStatusColor(job.status)}`}>
                    {job.status.toUpperCase()}
                  </span>
                </div>
                <div className="text-gray-400">
                  {job.status === 'running' ? 'Processing...' : 
                   job.status === 'pending' ? 'Queued' :
                   job.status === 'completed' ? 'Completed' : 'Failed'}
                </div>
              </div>
            ))}
            {jobQueue.length > 5 && (
              <div className="text-xs text-gray-400 text-center">
                +{jobQueue.length - 5} more jobs
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const renderDocumentsTab = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[#ff8c00] font-medium">Processed Documents</h3>
        <span className="text-sm text-gray-400">
          {processedDocuments.length} documents
        </span>
      </div>

      <div className="space-y-2">
        {processedDocuments.map(doc => (
          <ProcessedDocumentCard
            key={doc.id}
            document={doc}
            onUpdate={(updates) => updateDocumentMetadata(doc.id, updates)}
            onDelete={() => deleteDocument(doc.id)}
            onReprocess={(options) => reprocessDocument(doc.id, options)}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="bg-[#1a1a1a] p-4 rounded border border-[#444] mb-4">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-[#2a2a2a] p-1 rounded mb-4">
        {[
          { key: 'sources', label: 'Sources', count: connectedSources.length },
          { key: 'browse', label: 'Browse', count: selectedFiles.size },
          { key: 'processing', label: 'Processing', count: activeIngestionJobs.filter(j => j.status === 'running').length },
          { key: 'documents', label: 'Documents', count: processedDocuments.length }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-3 py-1 text-xs rounded flex items-center space-x-1 ${
              activeTab === tab.key 
                ? 'bg-[#ff8c00] text-black' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <span>{tab.label}</span>
            {tab.count > 0 && (
              <span className={`text-xs px-1 rounded ${
                activeTab === tab.key ? 'bg-black text-[#ff8c00]' : 'bg-gray-600 text-white'
              }`}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === 'sources' && renderSourcesTab()}
        {activeTab === 'browse' && renderBrowseTab()}
        {activeTab === 'processing' && renderProcessingTab()}
        {activeTab === 'documents' && renderDocumentsTab()}
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded">
          <div className="flex items-center space-x-2 text-white">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Loading...</span>
          </div>
        </div>
      )}

      {/* Connection Modal */}
      <ConnectionModal
        sourceType={showConnectionModal!}
        isOpen={showConnectionModal !== null}
        onClose={() => setShowConnectionModal(null)}
        onConnect={(credentials) => handleConnect(showConnectionModal!, credentials)}
        isLoading={isLoading}
      />

      {/* Batch Configuration Modal */}
      <BatchConfigurationModal
        isOpen={showBatchConfig}
        onClose={() => setShowBatchConfig(false)}
        selectedFiles={selectedFiles}
        processingOptions={batchProcessingOptions}
        onUpdateOptions={setBatchProcessingOptions}
        onStartProcessing={handleBatchIngestion}
        estimatedTime={estimatedProcessingTime}
      />
    </div>
  );
}

// Source Connection Item Component
interface SourceConnectionItemProps {
  source: DataSource;
  onConnect: () => void;
  onDisconnect: () => void;
  isLoading: boolean;
}

function SourceConnectionItem({ source, onConnect, onDisconnect, isLoading }: SourceConnectionItemProps) {
  const getSourceIconComponent = (sourceType: DataSourceType) => {
    switch (sourceType) {
      case DataSourceType.GOOGLE_DRIVE:
        return <Cloud className="w-5 h-5 text-blue-400" />;
      case DataSourceType.LOCAL_DIRECTORY:
        return <Folder className="w-5 h-5 text-yellow-400" />;
      case DataSourceType.LOCAL_ZIP:
        return <Archive className="w-5 h-5 text-purple-400" />;
      case DataSourceType.INDIVIDUAL_UPLOAD:
        return <Upload className="w-5 h-5 text-green-400" />;
      default:
        return <Cloud className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="flex items-center justify-between p-3 bg-[#2a2a2a] rounded border border-[#444]">
      <div className="flex items-center space-x-3">
        {getSourceIconComponent(source.type)}
        <div>
          <div className="text-white text-sm font-medium">{source.name}</div>
          <div className="text-gray-400 text-xs">
            {source.isConnected ? (
              <span className="flex items-center space-x-1">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span>Connected</span>
                {source.fileCount && <span>• {source.fileCount} files</span>}
              </span>
            ) : (
              <span className="flex items-center space-x-1">
                <AlertCircle className="w-3 h-3 text-gray-400" />
                <span>Not connected</span>
              </span>
            )}
          </div>
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
        {source.isConnected ? (
          <>
            <button
              onClick={onDisconnect}
              disabled={isLoading}
              className="text-xs text-gray-400 hover:text-white disabled:opacity-50"
            >
              Disconnect
            </button>
            <Settings className="w-4 h-4 text-gray-400 hover:text-white cursor-pointer" />
          </>
        ) : (
          <button
            onClick={onConnect}
            disabled={isLoading}
            className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:opacity-50"
          >
            Connect
          </button>
        )}
      </div>
    </div>
  );
}

// Unified File Browser Component with Enhanced Features
interface UnifiedFileBrowserProps {
  connectedSources: DataSource[];
  selectedFiles: Map<string, UniversalFileMetadata>;
  onFileSelect: (file: UniversalFileMetadata, selected: boolean) => void;
  searchQuery: string;
}

interface SortOption {
  key: 'name' | 'size' | 'modified' | 'source';
  label: string;
  direction: 'asc' | 'desc';
}

interface ViewMode {
  type: 'list' | 'grid';
  label: string;
}

function UnifiedFileBrowser({ connectedSources, selectedFiles, onFileSelect, searchQuery }: UnifiedFileBrowserProps) {
  const [sourceFiles, setSourceFiles] = useState<Map<DataSourceType, UniversalFileMetadata[]>>(new Map());
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const [activeSourceTab, setActiveSourceTab] = useState<DataSourceType | 'all'>('all');
  const [sortOption, setSortOption] = useState<SortOption>({ key: 'name', label: 'Name', direction: 'asc' });
  const [viewMode, setViewMode] = useState<ViewMode>({ type: 'list', label: 'List' });
  const [filterOptions, setFilterOptions] = useState({
    showProcessed: true,
    showPending: true,
    showFailed: true,
    sizeRange: { min: 0, max: Infinity },
    dateRange: { start: null as Date | null, end: null as Date | null }
  });

  useEffect(() => {
    loadSourceFiles();
  }, [connectedSources]);

  const loadSourceFiles = async () => {
    setIsLoading(true);
    try {
      const filesBySource = new Map<DataSourceType, UniversalFileMetadata[]>();
      
      for (const source of connectedSources) {
        try {
          const sourceTree = await browseSource(source.type);
          filesBySource.set(source.type, sourceTree.files);
        } catch (error) {
          console.error(`Failed to browse ${source.type}:`, error);
        }
      }
      
      setSourceFiles(filesBySource);
    } catch (error) {
      console.error('Failed to load source files:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleFolder = (folderId: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderId)) {
      newExpanded.delete(folderId);
    } else {
      newExpanded.add(folderId);
    }
    setExpandedFolders(newExpanded);
  };

  const sortFiles = (files: UniversalFileMetadata[]): UniversalFileMetadata[] => {
    return [...files].sort((a, b) => {
      let comparison = 0;
      
      switch (sortOption.key) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
        case 'modified':
          comparison = new Date(a.modified_time).getTime() - new Date(b.modified_time).getTime();
          break;
        case 'source':
          comparison = a.source_type.localeCompare(b.source_type);
          break;
      }
      
      return sortOption.direction === 'asc' ? comparison : -comparison;
    });
  };

  const filterFiles = (files: UniversalFileMetadata[]): UniversalFileMetadata[] => {
    return files.filter(file => {
      // Search query filter
      if (searchQuery && !file.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !file.source_path.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      // Size filter
      if (file.size < filterOptions.sizeRange.min || file.size > filterOptions.sizeRange.max) {
        return false;
      }

      // Date filter
      const fileDate = new Date(file.modified_time);
      if (filterOptions.dateRange.start && fileDate < filterOptions.dateRange.start) {
        return false;
      }
      if (filterOptions.dateRange.end && fileDate > filterOptions.dateRange.end) {
        return false;
      }

      return true;
    });
  };

  const getFilteredAndSortedFiles = (): UniversalFileMetadata[] => {
    let allFiles: UniversalFileMetadata[] = [];
    
    if (activeSourceTab === 'all') {
      allFiles = Array.from(sourceFiles.values()).flat();
    } else {
      allFiles = sourceFiles.get(activeSourceTab) || [];
    }
    
    return sortFiles(filterFiles(allFiles));
  };

  const getProcessingStatusIcon = (file: UniversalFileMetadata) => {
    const status = file.source_specific_metadata?.processing_status || 'not_processed';
    
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-yellow-400 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <RefreshCw className="w-6 h-6 animate-spin text-[#ff8c00]" />
        <span className="ml-2 text-gray-400">Loading files...</span>
      </div>
    );
  }

  const filteredFiles = getFilteredAndSortedFiles();

  return (
    <div className="space-y-4">
      {/* Source Tabs */}
      <div className="flex space-x-1 bg-[#2a2a2a] p-1 rounded">
        <button
          onClick={() => setActiveSourceTab('all')}
          className={`px-3 py-1 text-xs rounded flex items-center space-x-1 ${
            activeSourceTab === 'all' 
              ? 'bg-[#ff8c00] text-black' 
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Grid className="w-3 h-3" />
          <span>All Sources</span>
          <span className={`text-xs px-1 rounded ${
            activeSourceTab === 'all' ? 'bg-black text-[#ff8c00]' : 'bg-gray-600 text-white'
          }`}>
            {Array.from(sourceFiles.values()).flat().length}
          </span>
        </button>
        {connectedSources.map(source => (
          <button
            key={source.type}
            onClick={() => setActiveSourceTab(source.type)}
            className={`px-3 py-1 text-xs rounded flex items-center space-x-1 ${
              activeSourceTab === source.type 
                ? 'bg-[#ff8c00] text-black' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {getSourceIconComponent(source.type)}
            <span>{source.name}</span>
            <span className={`text-xs px-1 rounded ${
              activeSourceTab === source.type ? 'bg-black text-[#ff8c00]' : 'bg-gray-600 text-white'
            }`}>
              {sourceFiles.get(source.type)?.length || 0}
            </span>
          </button>
        ))}
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between bg-[#1a1a1a] p-3 rounded border border-[#444]">
        <div className="flex items-center space-x-2">
          {/* Sort Options */}
          <div className="flex items-center space-x-1">
            <button
              onClick={() => setSortOption(prev => ({ ...prev, direction: prev.direction === 'asc' ? 'desc' : 'asc' }))}
              className="p-1 text-gray-400 hover:text-white"
              title={`Sort ${sortOption.direction === 'asc' ? 'Descending' : 'Ascending'}`}
            >
              {sortOption.direction === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
            <select
              value={sortOption.key}
              onChange={(e) => setSortOption(prev => ({ ...prev, key: e.target.value as any }))}
              className="bg-[#2a2a2a] border border-[#444] text-white text-xs px-2 py-1 rounded"
            >
              <option value="name">Name</option>
              <option value="size">Size</option>
              <option value="modified">Modified</option>
              <option value="source">Source</option>
            </select>
          </div>

          {/* View Mode */}
          <div className="flex items-center space-x-1 border-l border-[#444] pl-2">
            <button
              onClick={() => setViewMode({ type: 'list', label: 'List' })}
              className={`p-1 ${viewMode.type === 'list' ? 'text-[#ff8c00]' : 'text-gray-400 hover:text-white'}`}
            >
              <List className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode({ type: 'grid', label: 'Grid' })}
              className={`p-1 ${viewMode.type === 'grid' ? 'text-[#ff8c00]' : 'text-gray-400 hover:text-white'}`}
            >
              <Grid className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            {filteredFiles.length} files
            {searchQuery && ` matching "${searchQuery}"`}
          </span>
          <button
            onClick={loadSourceFiles}
            className="p-1 text-gray-400 hover:text-white"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* File List/Grid */}
      <div className={`space-y-1 max-h-96 overflow-y-auto ${
        viewMode.type === 'grid' ? 'grid grid-cols-2 gap-2' : ''
      }`}>
        {filteredFiles.map(file => (
          <EnhancedFileItem
            key={file.file_id}
            file={file}
            isSelected={selectedFiles.has(file.file_id)}
            onSelect={(selected) => onFileSelect(file, selected)}
            viewMode={viewMode.type}
            showProcessingStatus={true}
          />
        ))}
        
        {filteredFiles.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            <FileText className="w-8 h-8 mx-auto mb-2" />
            <div>
              {searchQuery ? `No files match "${searchQuery}"` : 'No files available'}
            </div>
            {activeSourceTab !== 'all' && (
              <div className="text-xs mt-1">
                Try switching to "All Sources" or connecting more data sources
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hierarchical Folder Navigation (for applicable sources) */}
      {activeSourceTab !== 'all' && connectedSources.find(s => s.type === activeSourceTab)?.capabilities.canBrowse && (
        <HierarchicalFolderBrowser
          sourceType={activeSourceTab as DataSourceType}
          expandedFolders={expandedFolders}
          onToggleFolder={toggleFolder}
          onFileSelect={onFileSelect}
          selectedFiles={selectedFiles}
        />
      )}
    </div>
  );
}

// Enhanced File Item Component with Processing Status
interface EnhancedFileItemProps {
  file: UniversalFileMetadata;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  viewMode: 'list' | 'grid';
  showProcessingStatus: boolean;
}

function EnhancedFileItem({ file, isSelected, onSelect, viewMode, showProcessingStatus }: EnhancedFileItemProps) {
  const processingStatus = file.source_specific_metadata?.processing_status || 'not_processed';
  const qualityScore = file.source_specific_metadata?.quality_score;
  const chunks = file.source_specific_metadata?.chunks;
  const lastProcessed = file.source_specific_metadata?.last_processed;

  const getProcessingStatusBadge = () => {
    switch (processingStatus) {
      case 'completed':
        return (
          <div className="flex items-center space-x-1 text-xs text-green-400">
            <CheckCircle className="w-3 h-3" />
            <span>Processed</span>
            {qualityScore && <span>({Math.round(qualityScore * 100)}%)</span>}
          </div>
        );
      case 'processing':
        return (
          <div className="flex items-center space-x-1 text-xs text-yellow-400">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span>Processing</span>
          </div>
        );
      case 'failed':
        return (
          <div className="flex items-center space-x-1 text-xs text-red-400">
            <AlertCircle className="w-3 h-3" />
            <span>Failed</span>
          </div>
        );
      default:
        return (
          <div className="flex items-center space-x-1 text-xs text-gray-400">
            <Clock className="w-3 h-3" />
            <span>Pending</span>
          </div>
        );
    }
  };

  if (viewMode === 'grid') {
    return (
      <div className={`p-3 rounded border cursor-pointer transition-all ${
        isSelected 
          ? 'border-[#ff8c00] bg-[#ff8c00]/10' 
          : 'border-[#333] hover:border-[#444] hover:bg-[#2a2a2a]'
      }`}>
        <div className="flex items-start justify-between mb-2">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => onSelect(e.target.checked)}
            className="w-4 h-4 text-[#ff8c00] bg-[#1a1a1a] border-[#444] rounded focus:ring-[#ff8c00]"
          />
          {showProcessingStatus && getProcessingStatusBadge()}
        </div>
        
        <div className="mb-2">
          <div className="flex items-center space-x-2 mb-1">
            <FileText className="w-4 h-4 text-gray-400" />
            <span className="text-sm">{getSourceIcon(file.source_type)}</span>
          </div>
          <div className="text-white text-sm font-medium truncate" title={file.name}>
            {file.name}
          </div>
        </div>
        
        <div className="space-y-1 text-xs text-gray-400">
          <div>{formatFileSize(file.size)}</div>
          <div>{getSourceName(file.source_type)}</div>
          <div>{new Date(file.modified_time).toLocaleDateString()}</div>
          {chunks && <div>{chunks} chunks</div>}
        </div>
        
        <div className="flex items-center justify-between mt-2">
          {file.access_url && (
            <button className="text-gray-400 hover:text-white">
              <ExternalLink className="w-3 h-3" />
            </button>
          )}
          <button className="text-gray-400 hover:text-white">
            <MoreHorizontal className="w-3 h-3" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-3 p-3 rounded border cursor-pointer transition-all ${
      isSelected 
        ? 'border-[#ff8c00] bg-[#ff8c00]/10' 
        : 'border-[#333] hover:border-[#444] hover:bg-[#2a2a2a]'
    }`}>
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => onSelect(e.target.checked)}
        className="w-4 h-4 text-[#ff8c00] bg-[#1a1a1a] border-[#444] rounded focus:ring-[#ff8c00]"
      />
      
      <div className="flex items-center space-x-2">
        <FileText className="w-4 h-4 text-gray-400" />
        <span className="text-sm">{getSourceIcon(file.source_type)}</span>
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className="text-white text-sm font-medium truncate">{file.name}</span>
          {showProcessingStatus && getProcessingStatusBadge()}
        </div>
        <div className="flex items-center space-x-3 text-xs text-gray-400 mt-1">
          <span>{formatFileSize(file.size)}</span>
          <span>•</span>
          <span>{getSourceName(file.source_type)}</span>
          <span>•</span>
          <span>{new Date(file.modified_time).toLocaleDateString()}</span>
          {chunks && (
            <>
              <span>•</span>
              <span>{chunks} chunks</span>
            </>
          )}
          {lastProcessed && (
            <>
              <span>•</span>
              <span>Processed {new Date(lastProcessed).toLocaleDateString()}</span>
            </>
          )}
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
        {file.access_url && (
          <button className="text-gray-400 hover:text-white p-1">
            <ExternalLink className="w-4 h-4" />
          </button>
        )}
        <button className="text-gray-400 hover:text-white p-1">
          <Eye className="w-4 h-4" />
        </button>
        <button className="text-gray-400 hover:text-white p-1">
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// Hierarchical Folder Browser Component
interface HierarchicalFolderBrowserProps {
  sourceType: DataSourceType;
  expandedFolders: Set<string>;
  onToggleFolder: (folderId: string) => void;
  onFileSelect: (file: UniversalFileMetadata, selected: boolean) => void;
  selectedFiles: Map<string, UniversalFileMetadata>;
}

function HierarchicalFolderBrowser({ 
  sourceType, 
  expandedFolders, 
  onToggleFolder, 
  onFileSelect, 
  selectedFiles 
}: HierarchicalFolderBrowserProps) {
  const [folderTree, setFolderTree] = useState<FileTreeNode[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadFolderStructure();
  }, [sourceType]);

  const loadFolderStructure = async () => {
    setIsLoading(true);
    try {
      const sourceTree = await browseSource(sourceType);
      setFolderTree(sourceTree.folders);
    } catch (error) {
      console.error(`Failed to load folder structure for ${sourceType}:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderFolderNode = (node: FileTreeNode, depth: number = 0) => {
    const isExpanded = expandedFolders.has(node.id);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.id} className="select-none">
        <div 
          className={`flex items-center space-x-2 p-2 hover:bg-[#2a2a2a] rounded cursor-pointer`}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => hasChildren && onToggleFolder(node.id)}
        >
          {hasChildren ? (
            isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )
          ) : (
            <div className="w-4 h-4" />
          )}
          
          {node.type === 'folder' ? (
            <Folder className="w-4 h-4 text-yellow-400" />
          ) : (
            <FileText className="w-4 h-4 text-gray-400" />
          )}
          
          <span className="text-sm text-white">{node.name}</span>
          
          {node.type === 'folder' && node.children && (
            <span className="text-xs text-gray-400">
              ({node.children.length} items)
            </span>
          )}
        </div>
        
        {isExpanded && hasChildren && (
          <div>
            {node.children!.map(child => renderFolderNode(child, depth + 1))}
          </div>
        )}
        
        {node.type === 'file' && node.metadata && (
          <div style={{ paddingLeft: `${(depth + 1) * 16 + 8}px` }}>
            <EnhancedFileItem
              file={node.metadata}
              isSelected={selectedFiles.has(node.metadata.file_id)}
              onSelect={(selected) => onFileSelect(node.metadata!, selected)}
              viewMode="list"
              showProcessingStatus={true}
            />
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="bg-[#1a1a1a] p-4 rounded border border-[#444]">
        <div className="flex items-center space-x-2 text-gray-400">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading folder structure...</span>
        </div>
      </div>
    );
  }

  if (folderTree.length === 0) {
    return (
      <div className="bg-[#1a1a1a] p-4 rounded border border-[#444]">
        <div className="text-center text-gray-400">
          <Folder className="w-8 h-8 mx-auto mb-2" />
          <div className="text-sm">No folder structure available</div>
          <div className="text-xs mt-1">
            This source type may not support hierarchical browsing
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#1a1a1a] p-4 rounded border border-[#444]">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-[#ff8c00] font-medium flex items-center space-x-2">
          <HardDrive className="w-4 h-4" />
          <span>Folder Structure - {getSourceName(sourceType)}</span>
        </h4>
        <button
          onClick={loadFolderStructure}
          className="text-gray-400 hover:text-white p-1"
          title="Refresh folder structure"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      
      <div className="max-h-64 overflow-y-auto space-y-1">
        {folderTree.map(node => renderFolderNode(node))}
      </div>
    </div>
  );
}

// Helper function to get source icon component
function getSourceIconComponent(sourceType: DataSourceType) {
  switch (sourceType) {
    case DataSourceType.GOOGLE_DRIVE:
      return <Cloud className="w-4 h-4 text-blue-400" />;
    case DataSourceType.LOCAL_DIRECTORY:
      return <Folder className="w-4 h-4 text-yellow-400" />;
    case DataSourceType.LOCAL_ZIP:
      return <Archive className="w-4 h-4 text-purple-400" />;
    case DataSourceType.INDIVIDUAL_UPLOAD:
      return <Upload className="w-4 h-4 text-green-400" />;
    case DataSourceType.AWS_S3:
      return <Cloud className="w-4 h-4 text-orange-400" />;
    case DataSourceType.AZURE_BLOB:
      return <Cloud className="w-4 h-4 text-blue-500" />;
    case DataSourceType.GOOGLE_CLOUD_STORAGE:
      return <Cloud className="w-4 h-4 text-red-400" />;
    default:
      return <Cloud className="w-4 h-4 text-gray-400" />;
  }
}

// Processing Job Card Component
interface ProcessingJobCardProps {
  job: MultiSourceIngestionJob;
  onCancel: () => void;
}

function ProcessingJobCard({ job, onCancel }: ProcessingJobCardProps) {
  const progressPercentage = job.totalFiles > 0 ? (job.processedFiles / job.totalFiles) * 100 : 0;

  return (
    <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className={`w-2 h-2 rounded-full ${
            job.status === 'running' ? 'bg-yellow-400' :
            job.status === 'completed' ? 'bg-green-400' :
            job.status === 'failed' ? 'bg-red-400' : 'bg-gray-400'
          }`} />
          <span className="text-white text-sm font-medium">Job {job.id}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`text-xs ${getStatusColor(job.status)}`}>
            {job.status.toUpperCase()}
          </span>
          {job.status === 'running' && (
            <button
              onClick={onCancel}
              className="text-red-400 hover:text-red-300"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      
      <div className="mb-2">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{job.processedFiles} / {job.totalFiles} files</span>
          <span>{Math.round(progressPercentage)}%</span>
        </div>
        <div className="bg-gray-700 rounded-full h-2">
          <div 
            className="bg-[#ff8c00] h-2 rounded-full transition-all duration-300"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>
      
      <div className="text-xs text-gray-400">
        Started: {new Date(job.startedAt).toLocaleString()}
        {job.completedAt && (
          <span> • Completed: {new Date(job.completedAt).toLocaleString()}</span>
        )}
      </div>
    </div>
  );
}

// Enhanced Processing Job Card Component with Job Management
interface EnhancedProcessingJobCardProps {
  job: MultiSourceIngestionJob;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  onRetry: () => void;
}

function EnhancedProcessingJobCard({ job, onCancel, onPause, onResume, onRetry }: EnhancedProcessingJobCardProps) {
  const [showDetails, setShowDetails] = useState(false);
  const progressPercentage = job.totalFiles > 0 ? (job.processedFiles / job.totalFiles) * 100 : 0;

  const getJobActions = () => {
    switch (job.status) {
      case 'running':
        return (
          <>
            <button
              onClick={onPause}
              className="text-yellow-400 hover:text-yellow-300 p-1"
              title="Pause job"
            >
              <Clock className="w-4 h-4" />
            </button>
            <button
              onClick={onCancel}
              className="text-red-400 hover:text-red-300 p-1"
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
            className="text-blue-400 hover:text-blue-300 p-1"
            title="Retry job"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        );
      case 'pending':
        return (
          <button
            onClick={onCancel}
            className="text-red-400 hover:text-red-300 p-1"
            title="Cancel job"
          >
            <X className="w-4 h-4" />
          </button>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <span className={`w-3 h-3 rounded-full ${
            job.status === 'running' ? 'bg-yellow-400 animate-pulse' :
            job.status === 'completed' ? 'bg-green-400' :
            job.status === 'failed' ? 'bg-red-400' : 'bg-gray-400'
          }`} />
          <div>
            <div className="text-white text-sm font-medium">Multi-Source Job {job.id.slice(-8)}</div>
            <div className="text-xs text-gray-400">
              {job.sourceProgress.length} sources • {job.totalFiles} files
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`text-xs font-medium ${getStatusColor(job.status)}`}>
            {job.status.toUpperCase()}
          </span>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-gray-400 hover:text-white p-1"
            title="Toggle details"
          >
            {showDetails ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
          {getJobActions()}
        </div>
      </div>
      
      {/* Overall Progress */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{job.processedFiles} / {job.totalFiles} files processed</span>
          <span>{Math.round(progressPercentage)}%</span>
        </div>
        <div className="bg-gray-700 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${
              job.status === 'failed' ? 'bg-red-400' :
              job.status === 'completed' ? 'bg-green-400' : 'bg-[#ff8c00]'
            }`}
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>

      {/* Source-wise Progress */}
      {showDetails && (
        <div className="space-y-3 border-t border-[#333] pt-3">
          {job.sourceProgress.map(sourceProgress => (
            <div key={sourceProgress.sourceType} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getSourceIconComponent(sourceProgress.sourceType)}
                  <span className="text-sm text-white">{getSourceName(sourceProgress.sourceType)}</span>
                </div>
                <span className="text-xs text-gray-400">
                  {sourceProgress.completed} / {sourceProgress.total}
                </span>
              </div>
              
              <div className="bg-gray-700 rounded-full h-1">
                <div 
                  className="bg-[#ff8c00] h-1 rounded-full transition-all duration-300"
                  style={{ width: `${(sourceProgress.completed / sourceProgress.total) * 100}%` }}
                />
              </div>

              {/* Individual File Progress */}
              {sourceProgress.files.slice(0, 3).map(file => (
                <div key={file.id} className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <FileText className="w-3 h-3 text-gray-400" />
                    <span className="text-gray-300 truncate max-w-[200px]">{file.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`${getStatusColor(file.status)}`}>
                      {file.status === 'completed' ? '✓' :
                       file.status === 'failed' ? '✗' :
                       file.status === 'processing' ? '⟳' : '○'}
                    </span>
                    <span className="text-gray-400">{file.progress}%</span>
                  </div>
                </div>
              ))}
              
              {sourceProgress.files.length > 3 && (
                <div className="text-xs text-gray-400 text-center">
                  +{sourceProgress.files.length - 3} more files
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      
      <div className="flex justify-between text-xs text-gray-400 mt-3 pt-2 border-t border-[#333]">
        <span>Started: {new Date(job.startedAt).toLocaleString()}</span>
        {job.completedAt && (
          <span>Completed: {new Date(job.completedAt).toLocaleString()}</span>
        )}
        {job.error && (
          <span className="text-red-400">Error: {job.error}</span>
        )}
      </div>
    </div>
  );
}

// Batch Configuration Modal Component
interface BatchConfigurationModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedFiles: Map<string, UniversalFileMetadata>;
  processingOptions: Map<DataSourceType, ProcessingOptions>;
  onUpdateOptions: (options: Map<DataSourceType, ProcessingOptions>) => void;
  onStartProcessing: () => void;
  estimatedTime: number;
}

function BatchConfigurationModal({
  isOpen,
  onClose,
  selectedFiles,
  processingOptions,
  onUpdateOptions,
  onStartProcessing,
  estimatedTime
}: BatchConfigurationModalProps) {
  const [localOptions, setLocalOptions] = useState<Map<DataSourceType, ProcessingOptions>>(new Map(processingOptions));

  if (!isOpen) return null;

  // Group files by source type
  const filesBySource = new Map<DataSourceType, UniversalFileMetadata[]>();
  selectedFiles.forEach(file => {
    const existing = filesBySource.get(file.source_type) || [];
    existing.push(file);
    filesBySource.set(file.source_type, existing);
  });

  const updateSourceOptions = (sourceType: DataSourceType, options: ProcessingOptions) => {
    const newOptions = new Map(localOptions);
    newOptions.set(sourceType, options);
    setLocalOptions(newOptions);
  };

  const handleSaveAndProcess = () => {
    onUpdateOptions(localOptions);
    onClose();
    onStartProcessing();
  };

  const formatEstimatedTime = (seconds: number): string => {
    if (seconds < 60) return `~${seconds}s`;
    if (seconds < 3600) return `~${Math.ceil(seconds / 60)}m`;
    return `~${Math.ceil(seconds / 3600)}h ${Math.ceil((seconds % 3600) / 60)}m`;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[#1a1a1a] border border-[#444] rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-[#ff8c00] text-lg font-medium">Batch Processing Configuration</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-6">
          {/* Processing Summary */}
          <div className="bg-[#2a2a2a] p-4 rounded border border-[#333]">
            <h3 className="text-white font-medium mb-2">Processing Summary</h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Total Files:</span>
                <span className="text-white ml-2">{selectedFiles.size}</span>
              </div>
              <div>
                <span className="text-gray-400">Sources:</span>
                <span className="text-white ml-2">{filesBySource.size}</span>
              </div>
              <div>
                <span className="text-gray-400">Estimated Time:</span>
                <span className="text-[#ff8c00] ml-2">{formatEstimatedTime(estimatedTime)}</span>
              </div>
            </div>
          </div>

          {/* Per-Source Configuration */}
          {Array.from(filesBySource.entries()).map(([sourceType, files]) => (
            <SourceProcessingConfig
              key={sourceType}
              sourceType={sourceType}
              files={files}
              options={localOptions.get(sourceType) || {
                chunkSize: 1000,
                embeddingModel: 'text-embedding-3-large',
                preserveMath: true,
                category: 'Multi-Source Batch',
                tags: ['batch', sourceType]
              }}
              onUpdateOptions={(options) => updateSourceOptions(sourceType, options)}
            />
          ))}
        </div>

        <div className="flex items-center justify-between mt-6 pt-4 border-t border-[#333]">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white"
          >
            Cancel
          </button>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => {
                onUpdateOptions(localOptions);
                onClose();
              }}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500"
            >
              Save Configuration
            </button>
            <button
              onClick={handleSaveAndProcess}
              className="px-4 py-2 bg-[#ff8c00] text-black rounded hover:bg-orange-600 flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Start Processing</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Source Processing Configuration Component
interface SourceProcessingConfigProps {
  sourceType: DataSourceType;
  files: UniversalFileMetadata[];
  options: ProcessingOptions;
  onUpdateOptions: (options: ProcessingOptions) => void;
}

function SourceProcessingConfig({ sourceType, files, options, onUpdateOptions }: SourceProcessingConfigProps) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);

  const updateOption = (key: keyof ProcessingOptions, value: any) => {
    onUpdateOptions({ ...options, [key]: value });
  };

  const addTag = (tag: string) => {
    if (tag && !options.tags?.includes(tag)) {
      updateOption('tags', [...(options.tags || []), tag]);
    }
  };

  const removeTag = (tagToRemove: string) => {
    updateOption('tags', options.tags?.filter(tag => tag !== tagToRemove) || []);
  };

  return (
    <div className="bg-[#2a2a2a] p-4 rounded border border-[#333]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {getSourceIconComponent(sourceType)}
          <div>
            <h3 className="text-white font-medium">{getSourceName(sourceType)}</h3>
            <div className="text-xs text-gray-400">
              {files.length} files • {formatFileSize(totalSize)}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Processing Options */}
        <div className="space-y-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Chunk Size</label>
            <select
              value={options.chunkSize || 1000}
              onChange={(e) => updateOption('chunkSize', parseInt(e.target.value))}
              className="w-full bg-[#1a1a1a] border border-[#444] text-white text-sm px-3 py-2 rounded"
            >
              <option value={500}>Small (500 tokens)</option>
              <option value={1000}>Medium (1000 tokens)</option>
              <option value={1500}>Large (1500 tokens)</option>
              <option value={2000}>Extra Large (2000 tokens)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Embedding Model</label>
            <select
              value={options.embeddingModel || 'text-embedding-3-large'}
              onChange={(e) => updateOption('embeddingModel', e.target.value)}
              className="w-full bg-[#1a1a1a] border border-[#444] text-white text-sm px-3 py-2 rounded"
            >
              <option value="text-embedding-3-large">OpenAI Large (General)</option>
              <option value="text-embedding-3-small">OpenAI Small (Fast)</option>
              <option value="bge-large-en-v1.5">BGE Large (Financial)</option>
              <option value="all-mpnet-base-v2">MPNet (Mathematical)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id={`preserve-math-${sourceType}`}
              checked={options.preserveMath || false}
              onChange={(e) => updateOption('preserveMath', e.target.checked)}
              className="w-4 h-4 text-[#ff8c00] bg-[#1a1a1a] border-[#444] rounded"
            />
            <label htmlFor={`preserve-math-${sourceType}`} className="text-sm text-gray-400">
              Preserve Mathematical Notation
            </label>
          </div>
        </div>

        {/* Metadata Options */}
        <div className="space-y-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Category</label>
            <input
              type="text"
              value={options.category || ''}
              onChange={(e) => updateOption('category', e.target.value)}
              placeholder="e.g., Research Papers, Financial Reports"
              className="w-full bg-[#1a1a1a] border border-[#444] text-white text-sm px-3 py-2 rounded"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Tags</label>
            <div className="flex flex-wrap gap-1 mb-2">
              {options.tags?.map(tag => (
                <span
                  key={tag}
                  className="inline-flex items-center space-x-1 px-2 py-1 bg-[#ff8c00] text-black text-xs rounded"
                >
                  <span>{tag}</span>
                  <button
                    onClick={() => removeTag(tag)}
                    className="hover:bg-orange-600 rounded"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
            <div className="flex space-x-2">
              <input
                type="text"
                placeholder="Add tag..."
                className="flex-1 bg-[#1a1a1a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    addTag((e.target as HTMLInputElement).value);
                    (e.target as HTMLInputElement).value = '';
                  }
                }}
              />
              <button
                onClick={() => {
                  const input = document.querySelector(`input[placeholder="Add tag..."]`) as HTMLInputElement;
                  if (input?.value) {
                    addTag(input.value);
                    input.value = '';
                  }
                }}
                className="px-3 py-2 bg-gray-600 text-white text-sm rounded hover:bg-gray-500"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
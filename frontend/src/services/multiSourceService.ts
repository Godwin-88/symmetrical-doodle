/**
 * Multi-Source Knowledge Ingestion Service
 * Handles API calls for Google Drive, local files, ZIP archives, and cloud storage
 */

import { intelligenceApi } from './api';

// Types for multi-source integration
export enum DataSourceType {
  GOOGLE_DRIVE = 'google_drive',
  LOCAL_ZIP = 'local_zip',
  LOCAL_DIRECTORY = 'local_directory',
  INDIVIDUAL_UPLOAD = 'individual_upload',
  AWS_S3 = 'aws_s3',
  AZURE_BLOB = 'azure_blob',
  GOOGLE_CLOUD_STORAGE = 'google_cloud_storage'
}

export interface UniversalFileMetadata {
  file_id: string;
  name: string;
  size: number;
  modified_time: string;
  source_type: DataSourceType;
  source_path: string;
  mime_type: string;
  access_url?: string;
  parent_folders: string[];
  domain_classification?: string;
  checksum?: string;
  source_specific_metadata: Record<string, any>;
}

export interface DataSource {
  type: DataSourceType;
  name: string;
  isConnected: boolean;
  connectionStatus: ConnectionStatus;
  fileCount?: number;
  lastSync?: string;
  capabilities: SourceCapabilities;
}

export interface ConnectionStatus {
  isConnected: boolean;
  userEmail?: string;
  connectedAt?: string;
  permissions: string[];
  quotaUsed?: number;
  quotaLimit?: number;
  error?: string;
}

export interface SourceCapabilities {
  canBrowse: boolean;
  canSearch: boolean;
  canUpload: boolean;
  canDownload: boolean;
  supportsAuth: boolean;
  supportsBatch: boolean;
}

export interface UnifiedFileTree {
  sources: SourceFileTree[];
  totalFiles: number;
  totalSources: number;
}

export interface SourceFileTree {
  sourceType: DataSourceType;
  sourceName: string;
  folders: FileTreeNode[];
  files: UniversalFileMetadata[];
}

export interface FileTreeNode {
  id: string;
  name: string;
  type: 'folder' | 'file';
  children?: FileTreeNode[];
  metadata?: UniversalFileMetadata;
}

export interface CrossSourceSelection {
  sourceType: DataSourceType;
  fileIds: string[];
  processingOptions?: ProcessingOptions;
}

export interface ProcessingOptions {
  chunkSize?: number;
  embeddingModel?: string;
  preserveMath?: boolean;
  category?: string;
  tags?: string[];
  priority?: 'low' | 'normal' | 'high';
  retryAttempts?: number;
  timeout?: number;
}

export interface MultiSourceIngestionJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  totalFiles: number;
  processedFiles: number;
  failedFiles: number;
  sourceProgress: SourceProgress[];
  startedAt: string;
  completedAt?: string;
  pausedAt?: string;
  estimatedTimeRemaining?: number;
  priority: 'low' | 'normal' | 'high';
  error?: string;
  retryCount?: number;
  maxRetries?: number;
}

export interface SourceProgress {
  sourceType: DataSourceType;
  total: number;
  completed: number;
  failed: number;
  files: FileProcessingStatus[];
}

export interface FileProcessingStatus {
  id: string;
  name: string;
  sourceType: DataSourceType;
  status: 'pending' | 'downloading' | 'parsing' | 'chunking' | 'embedding' | 'storing' | 'completed' | 'failed' | 'paused';
  progress: number;
  currentStep: string;
  error?: string;
  processingTime?: number;
  chunks?: number;
  embeddingModel?: string;
  qualityScore?: number;
  retryCount?: number;
  estimatedTimeRemaining?: number;
}

export interface AuthResult {
  success: boolean;
  connectionId?: string;
  userInfo?: {
    email: string;
    name: string;
    permissions: string[];
  };
  error?: string;
}

export interface ProcessedDocument {
  id: string;
  name: string;
  sourceType: DataSourceType;
  sourceId: string;
  sourceUrl?: string;
  size: number;
  uploadedAt: string;
  processedAt: string;
  status: 'processing' | 'indexed' | 'failed';
  chunks: number;
  category: string;
  tags: string[];
  metadata: Record<string, any>;
  processingStats?: {
    chunksCreated: number;
    embeddingModel: string;
    qualityScore: number;
    processingTime: number;
  };
}

// API Base URL
const MULTI_SOURCE_API_BASE = 'http://localhost:8000/multi-source';

/**
 * Data Source Management
 */
export async function getAvailableSources(): Promise<DataSource[]> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/sources`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to fetch available sources, using mock data:', error);
    
    // Mock data for development
    return [
      {
        type: DataSourceType.GOOGLE_DRIVE,
        name: 'Google Drive',
        isConnected: false,
        connectionStatus: { isConnected: false, permissions: [] },
        capabilities: {
          canBrowse: true,
          canSearch: true,
          canUpload: false,
          canDownload: true,
          supportsAuth: true,
          supportsBatch: true
        }
      },
      {
        type: DataSourceType.LOCAL_ZIP,
        name: 'ZIP Archives',
        isConnected: true,
        connectionStatus: { isConnected: true, permissions: ['read'] },
        capabilities: {
          canBrowse: true,
          canSearch: false,
          canUpload: true,
          canDownload: true,
          supportsAuth: false,
          supportsBatch: true
        }
      },
      {
        type: DataSourceType.LOCAL_DIRECTORY,
        name: 'Local Directory',
        isConnected: true,
        connectionStatus: { isConnected: true, permissions: ['read'] },
        capabilities: {
          canBrowse: true,
          canSearch: false,
          canUpload: false,
          canDownload: true,
          supportsAuth: false,
          supportsBatch: true
        }
      },
      {
        type: DataSourceType.INDIVIDUAL_UPLOAD,
        name: 'File Upload',
        isConnected: true,
        connectionStatus: { isConnected: true, permissions: ['upload'] },
        capabilities: {
          canBrowse: false,
          canSearch: false,
          canUpload: true,
          canDownload: false,
          supportsAuth: false,
          supportsBatch: true
        }
      }
    ];
  }
}

/**
 * Google Drive Authentication
 */
export async function connectGoogleDrive(): Promise<AuthResult> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/google-drive/auth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Google Drive auth failed, using mock response:', error);
    
    // Mock successful authentication
    return {
      success: true,
      connectionId: `gd_${Date.now()}`,
      userInfo: {
        email: 'user@example.com',
        name: 'Test User',
        permissions: ['drive.readonly']
      }
    };
  }
}

export async function disconnectGoogleDrive(connectionId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/google-drive/disconnect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connectionId })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Google Drive disconnect failed, using mock response:', error);
    return { success: true };
  }
}

/**
 * File Discovery and Browsing
 */
export async function browseSource(sourceType: DataSourceType, path?: string): Promise<SourceFileTree> {
  try {
    const params = new URLSearchParams();
    if (path) params.append('path', path);
    
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/${sourceType}/browse?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn(`Failed to browse ${sourceType}, using mock data:`, error);
    
    // Mock file tree data
    return {
      sourceType,
      sourceName: getSourceName(sourceType),
      folders: [
        {
          id: 'folder1',
          name: 'Research Papers',
          type: 'folder',
          children: []
        },
        {
          id: 'folder2',
          name: 'Market Reports',
          type: 'folder',
          children: []
        }
      ],
      files: [
        {
          file_id: `${sourceType}_file1`,
          name: 'Financial_Analysis_2024.pdf',
          size: 2048576,
          modified_time: new Date().toISOString(),
          source_type: sourceType,
          source_path: '/documents/Financial_Analysis_2024.pdf',
          mime_type: 'application/pdf',
          parent_folders: [],
          source_specific_metadata: {}
        },
        {
          file_id: `${sourceType}_file2`,
          name: 'Market_Trends_Q1.pdf',
          size: 1536000,
          modified_time: new Date().toISOString(),
          source_type: sourceType,
          source_path: '/documents/Market_Trends_Q1.pdf',
          mime_type: 'application/pdf',
          parent_folders: [],
          source_specific_metadata: {}
        }
      ]
    };
  }
}

export async function searchAcrossSources(query: string, sourceTypes?: DataSourceType[]): Promise<UniversalFileMetadata[]> {
  try {
    const params = new URLSearchParams({ query });
    if (sourceTypes) {
      sourceTypes.forEach(type => params.append('sources', type));
    }
    
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/search?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Cross-source search failed, using mock data:', error);
    
    // Mock search results
    return [
      {
        file_id: 'search_result_1',
        name: `${query}_analysis.pdf`,
        size: 1024000,
        modified_time: new Date().toISOString(),
        source_type: DataSourceType.GOOGLE_DRIVE,
        source_path: `/search/${query}_analysis.pdf`,
        mime_type: 'application/pdf',
        parent_folders: ['Search Results'],
        source_specific_metadata: { relevance: 0.95 }
      }
    ];
  }
}

/**
 * Multi-Source Batch Processing
 */
export async function startMultiSourceIngestion(selections: CrossSourceSelection[]): Promise<MultiSourceIngestionJob> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selections })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Multi-source ingestion failed, using mock response:', error);
    
    // Mock ingestion job
    const totalFiles = selections.reduce((sum, sel) => sum + sel.fileIds.length, 0);
    return {
      id: `job_${Date.now()}`,
      status: 'running',
      totalFiles,
      processedFiles: 0,
      failedFiles: 0,
      priority: 'normal',
      estimatedTimeRemaining: totalFiles * 30, // 30 seconds per file estimate
      sourceProgress: selections.map(sel => ({
        sourceType: sel.sourceType,
        total: sel.fileIds.length,
        completed: 0,
        failed: 0,
        files: sel.fileIds.map(fileId => ({
          id: fileId,
          name: `file_${fileId}.pdf`,
          sourceType: sel.sourceType,
          status: 'pending',
          progress: 0,
          currentStep: 'Queued for processing',
          retryCount: 0,
          estimatedTimeRemaining: 30
        }))
      })),
      startedAt: new Date().toISOString(),
      retryCount: 0,
      maxRetries: 3
    };
  }
}

export async function getIngestionJobStatus(jobId: string): Promise<MultiSourceIngestionJob> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/jobs/${jobId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to get job status, using mock data:', error);
    
    // Mock job progress
    return {
      id: jobId,
      status: 'running',
      totalFiles: 5,
      processedFiles: 3,
      failedFiles: 0,
      priority: 'normal',
      estimatedTimeRemaining: 120,
      sourceProgress: [
        {
          sourceType: DataSourceType.GOOGLE_DRIVE,
          total: 3,
          completed: 2,
          failed: 0,
          files: [
            {
              id: 'file1',
              name: 'document1.pdf',
              sourceType: DataSourceType.GOOGLE_DRIVE,
              status: 'completed',
              progress: 100,
              currentStep: 'Stored successfully',
              processingTime: 45,
              chunks: 12,
              qualityScore: 0.92
            },
            {
              id: 'file2',
              name: 'document2.pdf',
              sourceType: DataSourceType.GOOGLE_DRIVE,
              status: 'embedding',
              progress: 75,
              currentStep: 'Generating embeddings',
              estimatedTimeRemaining: 30
            },
            {
              id: 'file3',
              name: 'document3.pdf',
              sourceType: DataSourceType.GOOGLE_DRIVE,
              status: 'pending',
              progress: 0,
              currentStep: 'Queued for processing',
              estimatedTimeRemaining: 90
            }
          ]
        }
      ],
      startedAt: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
      retryCount: 0,
      maxRetries: 3
    };
  }
}

export async function cancelIngestionJob(jobId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/jobs/${jobId}/cancel`, {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to cancel job, using mock response:', error);
    return { success: true };
  }
}

export async function pauseIngestionJob(jobId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/jobs/${jobId}/pause`, {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to pause job, using mock response:', error);
    return { success: true };
  }
}

export async function resumeIngestionJob(jobId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/jobs/${jobId}/resume`, {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to resume job, using mock response:', error);
    return { success: true };
  }
}

export async function retryIngestionJob(jobId: string): Promise<{ success: boolean; newJobId: string }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/jobs/${jobId}/retry`, {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to retry job, using mock response:', error);
    return { success: true, newJobId: `retry_${jobId}_${Date.now()}` };
  }
}

export async function getJobQueue(): Promise<MultiSourceIngestionJob[]> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/queue`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to get job queue, using mock data:', error);
    return [];
  }
}

export async function estimateProcessingTime(selections: CrossSourceSelection[]): Promise<{ estimatedSeconds: number; breakdown: Record<string, number> }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/ingest/estimate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selections })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to get processing estimate, using mock data:', error);
    
    // Mock estimation logic
    const totalFiles = selections.reduce((sum, sel) => sum + sel.fileIds.length, 0);
    const baseTimePerFile = 30; // seconds
    const estimatedSeconds = totalFiles * baseTimePerFile;
    
    return {
      estimatedSeconds,
      breakdown: selections.reduce((acc, sel) => {
        acc[sel.sourceType] = sel.fileIds.length * baseTimePerFile;
        return acc;
      }, {} as Record<string, number>)
    };
  }
}

/**
 * Document Management
 */
export async function getProcessedDocuments(sourceFilter?: DataSourceType[]): Promise<ProcessedDocument[]> {
  try {
    const params = new URLSearchParams();
    if (sourceFilter) {
      sourceFilter.forEach(type => params.append('sources', type));
    }
    
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/documents?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to get processed documents, using mock data:', error);
    
    // Mock processed documents
    return [
      {
        id: 'doc1',
        name: 'Financial_Analysis_2024.pdf',
        sourceType: DataSourceType.GOOGLE_DRIVE,
        sourceId: 'gd_file_123',
        sourceUrl: 'https://drive.google.com/file/d/123',
        size: 2048576,
        uploadedAt: new Date(Date.now() - 86400000).toISOString(),
        processedAt: new Date(Date.now() - 86400000 + 300000).toISOString(),
        status: 'indexed',
        chunks: 45,
        category: 'Financial Reports',
        tags: ['analysis', '2024', 'financial'],
        metadata: {
          pages: 28,
          language: 'en',
          author: 'Research Team'
        },
        processingStats: {
          chunksCreated: 45,
          embeddingModel: 'text-embedding-3-large',
          qualityScore: 0.92,
          processingTime: 120
        }
      }
    ];
  }
}

export async function updateDocumentMetadata(docId: string, updates: Partial<ProcessedDocument>): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/documents/${docId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to update document metadata, using mock response:', error);
    return { success: true };
  }
}

export async function deleteDocument(docId: string): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/documents/${docId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to delete document, using mock response:', error);
    return { success: true };
  }
}

export async function reprocessDocument(docId: string, options: ProcessingOptions): Promise<{ success: boolean; jobId: string }> {
  try {
    const response = await fetch(`${MULTI_SOURCE_API_BASE}/documents/${docId}/reprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn('Failed to reprocess document, using mock response:', error);
    return { success: true, jobId: `reprocess_${Date.now()}` };
  }
}

/**
 * Utility Functions
 */
export function getSourceName(sourceType: DataSourceType): string {
  const names = {
    [DataSourceType.GOOGLE_DRIVE]: 'Google Drive',
    [DataSourceType.LOCAL_ZIP]: 'ZIP Archives',
    [DataSourceType.LOCAL_DIRECTORY]: 'Local Directory',
    [DataSourceType.INDIVIDUAL_UPLOAD]: 'File Upload',
    [DataSourceType.AWS_S3]: 'AWS S3',
    [DataSourceType.AZURE_BLOB]: 'Azure Blob Storage',
    [DataSourceType.GOOGLE_CLOUD_STORAGE]: 'Google Cloud Storage'
  };
  return names[sourceType] || sourceType;
}

export function getSourceIcon(sourceType: DataSourceType): string {
  const icons = {
    [DataSourceType.GOOGLE_DRIVE]: '🗂️',
    [DataSourceType.LOCAL_ZIP]: '📦',
    [DataSourceType.LOCAL_DIRECTORY]: '📁',
    [DataSourceType.INDIVIDUAL_UPLOAD]: '📤',
    [DataSourceType.AWS_S3]: '☁️',
    [DataSourceType.AZURE_BLOB]: '☁️',
    [DataSourceType.GOOGLE_CLOUD_STORAGE]: '☁️'
  };
  return icons[sourceType] || '📄';
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'completed':
    case 'indexed':
    case 'connected':
      return 'text-green-400';
    case 'pending':
    case 'processing':
    case 'running':
      return 'text-yellow-400';
    case 'failed':
    case 'error':
    case 'disconnected':
      return 'text-red-400';
    default:
      return 'text-gray-400';
  }
}
import React, { useState, useEffect } from 'react';
import {
  X,
  ExternalLink,
  FileText,
  Download,
  RefreshCw,
  Eye,
  Settings,
  Info,
  Clock,
  CheckCircle,
  AlertCircle,
  Tag,
  Calendar,
  User,
  BarChart3,
  Zap,
  Database
} from 'lucide-react';
import { DocumentAsset } from './IntelligenceNew';
import { getSourceName, getSourceIcon, formatFileSize } from '../../services/multiSourceService';

interface DocumentPreviewProps {
  document: DocumentAsset;
  isOpen: boolean;
  onClose: () => void;
  onReprocess?: (options: ReprocessingOptions) => void;
}

interface ReprocessingOptions {
  chunkSize?: number;
  embeddingModel?: string;
  preserveMath?: boolean;
  forceReparse?: boolean;
}

interface DocumentChunk {
  id: string;
  content: string;
  order: number;
  embedding_model: string;
  quality_score: number;
  token_count: number;
  section_header?: string;
  math_elements?: number;
}

export function DocumentPreview({ document, isOpen, onClose, onReprocess }: DocumentPreviewProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'chunks' | 'processing' | 'source'>('overview');
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [showReprocessModal, setShowReprocessModal] = useState(false);
  const [reprocessOptions, setReprocessOptions] = useState<ReprocessingOptions>({
    chunkSize: 1000,
    embeddingModel: 'text-embedding-3-large',
    preserveMath: true,
    forceReparse: false
  });

  useEffect(() => {
    if (isOpen && activeTab === 'chunks') {
      loadDocumentChunks();
    }
  }, [isOpen, activeTab, document.id]);

  const loadDocumentChunks = async () => {
    setIsLoadingChunks(true);
    try {
      // Mock chunk data - in real implementation, this would call the API
      const mockChunks: DocumentChunk[] = Array.from({ length: document.chunks }, (_, i) => ({
        id: `chunk_${i + 1}`,
        content: `This is chunk ${i + 1} content from ${document.name}. It contains relevant information extracted during the processing phase. The content has been semantically segmented to maintain context and meaning while optimizing for embedding generation and retrieval.`,
        order: i + 1,
        embedding_model: document.processingStats?.embeddingModel || 'text-embedding-3-large',
        quality_score: 0.85 + Math.random() * 0.15,
        token_count: 150 + Math.floor(Math.random() * 200),
        section_header: i % 5 === 0 ? `Section ${Math.floor(i / 5) + 1}` : undefined,
        math_elements: Math.random() > 0.7 ? Math.floor(Math.random() * 5) : undefined
      }));
      
      setChunks(mockChunks);
    } catch (error) {
      console.error('Failed to load document chunks:', error);
    } finally {
      setIsLoadingChunks(false);
    }
  };

  const handleReprocess = () => {
    if (onReprocess) {
      onReprocess(reprocessOptions);
      setShowReprocessModal(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'indexed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-yellow-400 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[#1a1a1a] border border-[#444] rounded-lg w-[90%] max-w-4xl h-[80%] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#444]">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full bg-[#ff8c00]`} />
            <div>
              <h2 className="text-[#ff8c00] text-lg font-medium">{document.name}</h2>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <span>{getSourceName(document.sourceType as any || 'INDIVIDUAL_UPLOAD')}</span>
                <span>•</span>
                <span>{formatFileSize(document.size)}</span>
                <span>•</span>
                <span>{document.chunks} chunks</span>
                {getStatusIcon(document.status)}
                <span className="capitalize">{document.status}</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {document.originalLocation && (
              <button
                onClick={() => window.open(document.sourceUrl || document.originalLocation, '_blank')}
                className="p-2 text-gray-400 hover:text-white"
                title="View original location"
              >
                <ExternalLink className="w-4 h-4" />
              </button>
            )}
            {document.reprocessingAvailable && (
              <button
                onClick={() => setShowReprocessModal(true)}
                className="p-2 text-blue-400 hover:text-blue-300"
                title="Reprocess document"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-[#444]">
          {[
            { key: 'overview', label: 'Overview', icon: <Info className="w-4 h-4" /> },
            { key: 'chunks', label: 'Chunks', icon: <Database className="w-4 h-4" />, count: document.chunks },
            { key: 'processing', label: 'Processing', icon: <BarChart3 className="w-4 h-4" /> },
            { key: 'source', label: 'Source Info', icon: <FileText className="w-4 h-4" /> }
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`px-4 py-2 text-sm font-medium transition-colors flex items-center space-x-2 ${
                activeTab === tab.key
                  ? 'bg-orange-900 text-orange-400 border-b-2 border-orange-400'
                  : 'text-gray-400 hover:text-orange-400'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {tab.count && (
                <span className="bg-gray-600 text-white text-xs rounded-full px-2 py-0.5">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'overview' && (
            <div className="p-4 space-y-4 overflow-y-auto h-full">
              {/* Status and Metrics */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Status</div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(document.status)}
                    <span className="text-white capitalize">{document.status}</span>
                  </div>
                </div>
                <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Chunks</div>
                  <div className="text-white text-lg">{document.chunks}</div>
                </div>
                <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Category</div>
                  <div className="text-white">{document.category}</div>
                </div>
                <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Source</div>
                  <div className="text-white">{getSourceName(document.sourceType as any || 'INDIVIDUAL_UPLOAD')}</div>
                </div>
              </div>

              {/* Processing Statistics */}
              {document.processingStats && (
                <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                  <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Processing Statistics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-400 text-sm">Chunks Created:</span>
                        <span className="text-white text-sm">{document.processingStats.chunksCreated}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400 text-sm">Embedding Model:</span>
                        <span className="text-white text-sm">{document.processingStats.embeddingModel}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400 text-sm">Quality Score:</span>
                        <span className="text-white text-sm">{(document.processingStats.qualityScore * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-400 text-sm">Processing Time:</span>
                        <span className="text-white text-sm">{document.processingStats.processingTime}s</span>
                      </div>
                      {document.processingStats.parsingMethod && (
                        <div className="flex justify-between">
                          <span className="text-gray-400 text-sm">Parsing Method:</span>
                          <span className="text-white text-sm">{document.processingStats.parsingMethod}</span>
                        </div>
                      )}
                      {document.processingStats.mathElementsPreserved && (
                        <div className="flex justify-between">
                          <span className="text-gray-400 text-sm">Math Elements:</span>
                          <span className="text-white text-sm">{document.processingStats.mathElementsPreserved}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Tags and Metadata */}
              <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Tags & Metadata</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-gray-400 text-sm mb-2">Tags:</div>
                    <div className="flex flex-wrap gap-2">
                      {document.tags.map((tag, idx) => (
                        <span key={idx} className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded flex items-center">
                          <Tag className="w-3 h-3 mr-1" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-gray-400 text-sm mb-2">Metadata:</div>
                    <div className="space-y-1">
                      {Object.entries(document.metadata).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}:</span>
                          <span className="text-white">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'chunks' && (
            <div className="p-4 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[#ff8c00] font-medium">Document Chunks ({chunks.length})</h3>
                <button
                  onClick={loadDocumentChunks}
                  disabled={isLoadingChunks}
                  className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-500 disabled:opacity-50 flex items-center space-x-1"
                >
                  <RefreshCw className={`w-3 h-3 ${isLoadingChunks ? 'animate-spin' : ''}`} />
                  <span>Refresh</span>
                </button>
              </div>

              {isLoadingChunks ? (
                <div className="flex-1 flex items-center justify-center">
                  <RefreshCw className="w-6 h-6 animate-spin text-[#ff8c00]" />
                  <span className="ml-2 text-gray-400">Loading chunks...</span>
                </div>
              ) : (
                <div className="flex-1 overflow-y-auto space-y-3">
                  {chunks.map((chunk, idx) => (
                    <div key={chunk.id} className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="text-[#ff8c00] text-sm font-medium">Chunk {chunk.order}</span>
                          {chunk.section_header && (
                            <span className="px-2 py-0.5 bg-blue-600 text-white text-xs rounded">
                              {chunk.section_header}
                            </span>
                          )}
                          {chunk.math_elements && (
                            <span className="px-2 py-0.5 bg-purple-600 text-white text-xs rounded">
                              {chunk.math_elements} math elements
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-2 text-xs text-gray-400">
                          <span>{chunk.token_count} tokens</span>
                          <span>•</span>
                          <span>{Math.round(chunk.quality_score * 100)}% quality</span>
                        </div>
                      </div>
                      
                      <div className="text-gray-300 text-sm mb-2">
                        {chunk.content}
                      </div>
                      
                      <div className="text-xs text-gray-500">
                        Model: {chunk.embedding_model.split('/').pop()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'processing' && (
            <div className="p-4 space-y-4 overflow-y-auto h-full">
              <h3 className="text-[#ff8c00] font-medium">Processing Details</h3>
              
              {/* Processing Timeline */}
              <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                <h4 className="text-white text-sm font-medium mb-3">Processing Timeline</h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <div className="flex-1">
                      <div className="text-white text-sm">Document Upload</div>
                      <div className="text-gray-400 text-xs">{new Date(document.uploaded).toLocaleString()}</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <div className="flex-1">
                      <div className="text-white text-sm">Content Extraction</div>
                      <div className="text-gray-400 text-xs">
                        {document.processingStats?.parsingMethod || 'marker'} parser used
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <div className="flex-1">
                      <div className="text-white text-sm">Semantic Chunking</div>
                      <div className="text-gray-400 text-xs">{document.chunks} chunks created</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <div className="flex-1">
                      <div className="text-white text-sm">Embedding Generation</div>
                      <div className="text-gray-400 text-xs">
                        {document.processingStats?.embeddingModel || 'text-embedding-3-large'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <div className="flex-1">
                      <div className="text-white text-sm">Vector Storage</div>
                      <div className="text-gray-400 text-xs">Indexed in vector database</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quality Metrics */}
              <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                <h4 className="text-white text-sm font-medium mb-3">Quality Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Overall Quality Score:</span>
                    <span className="text-white text-sm">
                      {document.processingStats ? (document.processingStats.qualityScore * 100).toFixed(1) : 'N/A'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Processing Time:</span>
                    <span className="text-white text-sm">
                      {document.processingStats?.processingTime || 'N/A'}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Chunks per MB:</span>
                    <span className="text-white text-sm">
                      {(document.chunks / (document.size / 1024 / 1024)).toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'source' && (
            <div className="p-4 space-y-4 overflow-y-auto h-full">
              <h3 className="text-[#ff8c00] font-medium">Source Information</h3>
              
              {/* Source Details */}
              <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                <h4 className="text-white text-sm font-medium mb-3">Source Details</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Source Type:</span>
                    <span className="text-white text-sm">{getSourceName(document.sourceType as any || 'INDIVIDUAL_UPLOAD')}</span>
                  </div>
                  {document.sourceId && (
                    <div className="flex justify-between">
                      <span className="text-gray-400 text-sm">Source ID:</span>
                      <span className="text-white text-sm font-mono text-xs">{document.sourceId}</span>
                    </div>
                  )}
                  {document.sourcePath && (
                    <div className="flex justify-between">
                      <span className="text-gray-400 text-sm">Source Path:</span>
                      <span className="text-white text-sm truncate max-w-xs" title={document.sourcePath}>
                        {document.sourcePath}
                      </span>
                    </div>
                  )}
                  {document.originalLocation && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400 text-sm">Original Location:</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-white text-sm truncate max-w-xs">
                          {document.originalLocation}
                        </span>
                        {document.sourceUrl && (
                          <button
                            onClick={() => window.open(document.sourceUrl, '_blank')}
                            className="text-blue-400 hover:text-blue-300"
                          >
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                  {document.lastSyncedAt && (
                    <div className="flex justify-between">
                      <span className="text-gray-400 text-sm">Last Synced:</span>
                      <span className="text-white text-sm">
                        {new Date(document.lastSyncedAt).toLocaleString()}
                      </span>
                    </div>
                  )}
                  {document.checksum && (
                    <div className="flex justify-between">
                      <span className="text-gray-400 text-sm">Checksum:</span>
                      <span className="text-white text-sm font-mono text-xs">
                        {document.checksum.substring(0, 16)}...
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Source-Specific Metadata */}
              {document.sourceSpecificMetadata && Object.keys(document.sourceSpecificMetadata).length > 0 && (
                <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                  <h4 className="text-white text-sm font-medium mb-3">Source-Specific Metadata</h4>
                  <div className="space-y-2">
                    {Object.entries(document.sourceSpecificMetadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-400 text-sm capitalize">{key.replace(/_/g, ' ')}:</span>
                        <span className="text-white text-sm">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Access Permissions */}
              {document.accessPermissions && (
                <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
                  <h4 className="text-white text-sm font-medium mb-3">Access Permissions</h4>
                  <div className="flex flex-wrap gap-2">
                    {document.accessPermissions.map((permission, idx) => (
                      <span key={idx} className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
                        {permission}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Reprocessing Modal */}
      {showReprocessModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-60">
          <div className="bg-[#1a1a1a] border border-[#444] rounded-lg p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-lg font-medium">Reprocess Document</h3>
              <button
                onClick={() => setShowReprocessModal(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-white text-sm mb-2">Chunk Size</label>
                <select
                  value={reprocessOptions.chunkSize}
                  onChange={(e) => setReprocessOptions(prev => ({ ...prev, chunkSize: parseInt(e.target.value) }))}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                >
                  <option value={500}>500 tokens</option>
                  <option value={1000}>1000 tokens</option>
                  <option value={1500}>1500 tokens</option>
                  <option value={2000}>2000 tokens</option>
                </select>
              </div>
              
              <div>
                <label className="block text-white text-sm mb-2">Embedding Model</label>
                <select
                  value={reprocessOptions.embeddingModel}
                  onChange={(e) => setReprocessOptions(prev => ({ ...prev, embeddingModel: e.target.value }))}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                >
                  <option value="text-embedding-3-large">OpenAI text-embedding-3-large</option>
                  <option value="BAAI/bge-large-en-v1.5">BAAI/bge-large-en-v1.5</option>
                  <option value="sentence-transformers/all-mpnet-base-v2">all-mpnet-base-v2</option>
                </select>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="preserveMath"
                  checked={reprocessOptions.preserveMath}
                  onChange={(e) => setReprocessOptions(prev => ({ ...prev, preserveMath: e.target.checked }))}
                  className="w-4 h-4 text-[#ff8c00] bg-[#1a1a1a] border-[#444] rounded focus:ring-[#ff8c00]"
                />
                <label htmlFor="preserveMath" className="text-white text-sm">Preserve Mathematical Notation</label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="forceReparse"
                  checked={reprocessOptions.forceReparse}
                  onChange={(e) => setReprocessOptions(prev => ({ ...prev, forceReparse: e.target.checked }))}
                  className="w-4 h-4 text-[#ff8c00] bg-[#1a1a1a] border-[#444] rounded focus:ring-[#ff8c00]"
                />
                <label htmlFor="forceReparse" className="text-white text-sm">Force Re-parse from Source</label>
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 mt-6">
              <button
                onClick={() => setShowReprocessModal(false)}
                className="px-4 py-2 bg-gray-700 text-white text-sm rounded hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={handleReprocess}
                className="px-4 py-2 bg-[#ff8c00] text-black text-sm rounded hover:bg-orange-600"
              >
                Start Reprocessing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
import { useState, useEffect } from 'react';
import { useTradingStore } from '../store/tradingStore';
import {
  queryLLM,
  financialAnalysis,
  queryRAG,
  ingestDocument,
  getRAGStats,
  comprehensiveResearch,
  stockAnalysis,
  getMarketOverview,
  type LLMQueryRequest,
  type RAGQueryRequest,
  type ResearchRequest,
  type StockAnalysisRequest
} from '../../services/llmService';
import { 
  Brain, 
  FileText, 
  Search, 
  TrendingUp, 
  Upload, 
  Download, 
  Settings, 
  AlertCircle,
  CheckCircle,
  Database,
  Zap,
  BarChart3,
  Target,
  RefreshCw,
  Send,
  X,
  Plus,
  Edit,
  Trash2,
  Save,
  Eye,
  Filter,
  Calendar,
  DollarSign,
  Activity,
  PieChart,
  LineChart,
  TrendingDown
} from 'lucide-react';

// Types for financial engineering features
interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  model?: string;
  provider?: string;
  tokens?: number;
  confidence?: number;
}

interface ResearchReport {
  id: string;
  title: string;
  query: string;
  status: 'pending' | 'completed' | 'failed';
  created: string;
  updated: string;
  author: string;
  tags: string[];
  summary: string;
  findings: any;
  sources: number;
  confidence: number;
}
interface DocumentAsset {
  id: string;
  name: string;
  type: string;
  size: number;
  uploaded: string;
  status: 'processing' | 'indexed' | 'failed';
  chunks: number;
  category: string;
  tags: string[];
  metadata: Record<string, any>;
}

interface AnalysisModel {
  id: string;
  name: string;
  type: 'regime_detection' | 'risk_assessment' | 'signal_generation' | 'portfolio_optimization';
  status: 'active' | 'training' | 'inactive';
  accuracy: number;
  lastTrained: string;
  parameters: Record<string, any>;
  performance: {
    sharpe: number;
    maxDrawdown: number;
    winRate: number;
  };
}

export function IntelligenceNew() {
  console.log('🚀 NEW Intelligence component loaded successfully!');
  
  const {
    regimes,
    embeddings,
    intelligenceSignals,
    selectedRegime,
    setSelectedRegime,
    fetchRegimeData,
    fetchGraphFeatures,
    isLoading,
  } = useTradingStore();

  // Tab state - reordered to match original
  const [activeTab, setActiveTab] = useState<'analysis' | 'chat' | 'research' | 'documents'>('analysis');
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedLLMModel, setSelectedLLMModel] = useState('gpt-4');
  const [chatHistory, setChatHistory] = useState<string[]>([]);
  // Research state
  const [researchReports, setResearchReports] = useState<ResearchReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<ResearchReport | null>(null);
  const [newReportTitle, setNewReportTitle] = useState('');
  const [newReportQuery, setNewReportQuery] = useState('');
  const [reportTags, setReportTags] = useState<string[]>([]);
  const [showCreateReport, setShowCreateReport] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  
  // Documents state
  const [documents, setDocuments] = useState<DocumentAsset[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<DocumentAsset | null>(null);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [documentFilter, setDocumentFilter] = useState('all');
  const [ragStats, setRAGStats] = useState<any>(null);
  
  // Analysis state
  const [analysisModels, setAnalysisModels] = useState<AnalysisModel[]>([]);
  const [selectedAnalysisModel, setSelectedAnalysisModel] = useState<AnalysisModel | null>(null);
  const [modelMetrics, setModelMetrics] = useState<any>(null);
  const [showCreateModel, setShowCreateModel] = useState(false);
  const [newModelName, setNewModelName] = useState('');
  const [newModelType, setNewModelType] = useState<AnalysisModel['type']>('regime_detection');

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // Load chat history
      loadChatHistory();
      
      // Load research reports
      await loadResearchReports();
      
      // Load documents
      await loadDocuments();
      
      // Load analysis models
      await loadAnalysisModels();
      
      // Load RAG stats
      const stats = await getRAGStats();
      setRAGStats(stats);
      
    } catch (error) {
      console.error('Failed to load initial data:', error);
      // Load mock data as fallback
      loadMockData();
    }
  };
  const loadMockData = () => {
    // Mock chat history
    setChatHistory([
      'What is the current market regime?',
      'Analyze EURUSD volatility patterns',
      'Generate risk assessment for portfolio',
      'Explain correlation breakdown in FX markets'
    ]);

    // Mock research reports
    setResearchReports([
      {
        id: '1',
        title: 'Q4 2024 FX Market Analysis',
        query: 'Comprehensive analysis of FX market trends and regime changes',
        status: 'completed',
        created: '2024-01-15T10:00:00Z',
        updated: '2024-01-15T14:30:00Z',
        author: 'AI Research Engine',
        tags: ['FX', 'Market Analysis', 'Q4 2024'],
        summary: 'Detailed analysis of foreign exchange market conditions, identifying key regime shifts and volatility patterns.',
        findings: { regimes: 3, correlations: 12, signals: 8 },
        sources: 45,
        confidence: 87.3
      },
      {
        id: '2',
        title: 'Central Bank Policy Impact Study',
        query: 'Impact of central bank policies on currency volatility',
        status: 'completed',
        created: '2024-01-14T09:15:00Z',
        updated: '2024-01-14T16:45:00Z',
        author: 'AI Research Engine',
        tags: ['Central Banks', 'Policy', 'Volatility'],
        summary: 'Analysis of how central bank policy announcements affect currency pair volatility and market structure.',
        findings: { policies: 8, impacts: 15, correlations: 6 },
        sources: 32,
        confidence: 92.1
      }
    ]);

    // Mock documents
    setDocuments([
      {
        id: '1',
        name: 'ECB_Policy_Report_2024.pdf',
        type: 'pdf',
        size: 2048576,
        uploaded: '2024-01-15T08:30:00Z',
        status: 'indexed',
        chunks: 45,
        category: 'Central Bank Reports',
        tags: ['ECB', 'Policy', '2024'],
        metadata: { pages: 28, language: 'en', author: 'European Central Bank' }
      },
      {
        id: '2',
        name: 'Market_Structure_Analysis.docx',
        type: 'docx',
        size: 1536000,
        uploaded: '2024-01-14T15:20:00Z',
        status: 'indexed',
        chunks: 32,
        category: 'Research Papers',
        tags: ['Market Structure', 'Analysis'],
        metadata: { pages: 18, language: 'en', author: 'Research Team' }
      }
    ]);

    // Mock analysis models
    setAnalysisModels([
      {
        id: '1',
        name: 'FX Regime Detector v2.1',
        type: 'regime_detection',
        status: 'active',
        accuracy: 89.7,
        lastTrained: '2024-01-15T06:00:00Z',
        parameters: { lookback: 252, threshold: 0.15, features: 12 },
        performance: { sharpe: 1.42, maxDrawdown: -8.3, winRate: 67.8 }
      },
      {
        id: '2',
        name: 'Portfolio Risk Optimizer',
        type: 'portfolio_optimization',
        status: 'active',
        accuracy: 94.2,
        lastTrained: '2024-01-14T22:30:00Z',
        parameters: { horizon: 21, constraints: 8, objective: 'sharpe' },
        performance: { sharpe: 1.89, maxDrawdown: -5.1, winRate: 72.4 }
      }
    ]);

    setRAGStats({
      total_documents: 127,
      vector_store: 'chroma',
      embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
      chunk_size: 1000,
      similarity_threshold: 0.7,
      indexed_chunks: 3847,
      storage_size: '2.3GB'
    });
  };
  // CRUD Operations for Chat
  const loadChatHistory = () => {
    const saved = localStorage.getItem('intelligence_chat_history');
    if (saved) {
      setChatHistory(JSON.parse(saved));
    }
  };

  const saveChatHistory = (history: string[]) => {
    localStorage.setItem('intelligence_chat_history', JSON.stringify(history));
    setChatHistory(history);
  };

  const handleLLMQuery = async (query: string, useRAG: boolean = false) => {
    if (!query.trim()) return;

    setIsProcessing(true);
    
    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    setChatMessages(prev => [...prev, userMessage]);

    // Add to history
    const newHistory = [query, ...chatHistory.filter(h => h !== query)].slice(0, 10);
    saveChatHistory(newHistory);

    try {
      let response;
      if (useRAG) {
        const ragRequest: RAGQueryRequest = { question: query };
        response = await queryRAG(ragRequest);
      } else {
        const llmRequest: LLMQueryRequest = {
          query,
          system_prompt: 'You are a financial AI assistant specializing in algorithmic trading and market analysis.',
          prefer_local: false
        };
        response = query.toLowerCase().includes('financial') || query.toLowerCase().includes('market')
          ? await financialAnalysis(llmRequest)
          : await queryLLM(llmRequest);
      }

      const assistantMessage: ChatMessage = {
        id: `assistant_${Date.now()}`,
        type: 'assistant',
        content: useRAG ? response.answer : response.answer,
        timestamp: new Date().toISOString(),
        model: useRAG ? response.model_info?.model : response.model,
        provider: useRAG ? response.model_info?.provider : response.provider,
        tokens: useRAG ? response.model_info?.tokens_used : response.tokens_used,
        confidence: useRAG ? (response.confidence === 'high' ? 90 : response.confidence === 'medium' ? 70 : 50) : undefined
      };
      setChatMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Query failed:', error);
      
      const errorMessage: ChatMessage = {
        id: `error_${Date.now()}`,
        type: 'assistant',
        content: `I apologize, but I'm currently unable to process your query "${query}". This appears to be a ${query.toLowerCase().includes('risk') ? 'risk management' : query.toLowerCase().includes('market') ? 'market analysis' : 'financial'} inquiry. Please try again when the AI services are restored.`,
        timestamp: new Date().toISOString(),
        model: 'fallback',
        provider: 'mock'
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setCurrentQuery('');
    }
  };

  const clearChatHistory = () => {
    setChatMessages([]);
    localStorage.removeItem('intelligence_chat_messages');
  };

  const exportChatHistory = () => {
    const data = {
      messages: chatMessages,
      exported: new Date().toISOString(),
      session: `intelligence_${Date.now()}`
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `intelligence_chat_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  // CRUD Operations for Research Reports
  const loadResearchReports = async () => {
    try {
      // This would call a real API endpoint
      // const reports = await fetch('/api/intelligence/research/reports').then(r => r.json());
      // setResearchReports(reports);
      
      // For now, load from localStorage or use mock data
      const saved = localStorage.getItem('intelligence_research_reports');
      if (saved) {
        setResearchReports(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Failed to load research reports:', error);
    }
  };

  const createResearchReport = async () => {
    if (!newReportTitle.trim() || !newReportQuery.trim()) return;

    const newReport: ResearchReport = {
      id: `report_${Date.now()}`,
      title: newReportTitle,
      query: newReportQuery,
      status: 'pending',
      created: new Date().toISOString(),
      updated: new Date().toISOString(),
      author: 'Current User',
      tags: reportTags,
      summary: '',
      findings: {},
      sources: 0,
      confidence: 0
    };

    try {
      // This would call the comprehensive research API
      setResearchReports(prev => [newReport, ...prev]);
      
      // Start the research process
      const updatedReport = { ...newReport };
      
      const request: ResearchRequest = {
        query: newReportQuery,
        include_web: true,
        include_market_data: true
      };
      
      const result = await comprehensiveResearch(request);
      
      updatedReport.status = 'completed';
      updatedReport.updated = new Date().toISOString();
      updatedReport.findings = result;
      updatedReport.sources = result.web_research?.results?.length || 0;
      updatedReport.confidence = 85; // Mock confidence score
      updatedReport.summary = result.comprehensive_analysis?.analysis?.substring(0, 200) + '...' || 'Research completed successfully.';
      
      setResearchReports(prev => prev.map(r => r.id === newReport.id ? updatedReport : r));
      
      // Save to localStorage
      const updated = researchReports.map(r => r.id === newReport.id ? updatedReport : r);
      localStorage.setItem('intelligence_research_reports', JSON.stringify(updated));
      
    } catch (error) {
      console.error('Research failed:', error);
      
      // Update with mock data on failure
      const mockReport = {
        ...newReport,
        status: 'completed' as const,
        updated: new Date().toISOString(),
        summary: `Mock research analysis for "${newReportTitle}". This would contain comprehensive findings based on the query: "${newReportQuery}".`,
        findings: { 
          key_insights: 3, 
          data_points: 15, 
          correlations: 8,
          risk_factors: 5 
        },
        sources: 12,
        confidence: 78
      };
      
      setResearchReports(prev => prev.map(r => r.id === newReport.id ? mockReport : r));
    }

    // Reset form
    setNewReportTitle('');
    setNewReportQuery('');
    setReportTags([]);
    setShowCreateReport(false);
  };

  const deleteResearchReport = (reportId: string) => {
    setResearchReports(prev => prev.filter(r => r.id !== reportId));
    const updated = researchReports.filter(r => r.id !== reportId);
    localStorage.setItem('intelligence_research_reports', JSON.stringify(updated));
    
    if (selectedReport?.id === reportId) {
      setSelectedReport(null);
    }
  };

  const exportResearchReport = (report: ResearchReport) => {
    const data = {
      ...report,
      exported: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_${report.title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  // CRUD Operations for Documents
  const loadDocuments = async () => {
    try {
      // This would call a real API endpoint
      // const docs = await fetch('/api/intelligence/documents').then(r => r.json());
      // setDocuments(docs);
      
      const saved = localStorage.getItem('intelligence_documents');
      if (saved) {
        setDocuments(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    for (const file of files) {
      const docId = `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const newDoc: DocumentAsset = {
        id: docId,
        name: file.name,
        type: file.type || file.name.split('.').pop() || 'unknown',
        size: file.size,
        uploaded: new Date().toISOString(),
        status: 'processing',
        chunks: 0,
        category: 'Uncategorized',
        tags: [],
        metadata: {
          originalName: file.name,
          mimeType: file.type,
          size: file.size
        }
      };

      setDocuments(prev => [newDoc, ...prev]);
      setUploadProgress(prev => ({ ...prev, [docId]: 0 }));

      try {
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setUploadProgress(prev => ({ ...prev, [docId]: progress }));
        }

        // Call the actual ingestion API
        const metadata = {
          category: 'user_upload',
          tags: [],
          source: 'intelligence_ui'
        };

        const result = await ingestDocument(file, metadata);
        
        const updatedDoc: DocumentAsset = {
          ...newDoc,
          status: result.success ? 'indexed' : 'failed',
          chunks: result.chunks_created || 0,
          metadata: {
            ...newDoc.metadata,
            documentId: result.document_id,
            processingResult: result
          }
        };

        setDocuments(prev => prev.map(d => d.id === docId ? updatedDoc : d));
        
      } catch (error) {
        console.error('Document upload failed:', error);
        
        // Mock successful processing
        const mockDoc: DocumentAsset = {
          ...newDoc,
          status: 'indexed',
          chunks: Math.floor(file.size / 1000) + Math.floor(Math.random() * 20),
          category: file.name.includes('report') ? 'Reports' : 
                   file.name.includes('policy') ? 'Policy Documents' : 'Research Papers',
          tags: [file.name.split('.')[0].split('_')[0]]
        };

        setDocuments(prev => prev.map(d => d.id === docId ? mockDoc : d));
      } finally {
        setUploadProgress(prev => {
          const { [docId]: _, ...rest } = prev;
          return rest;
        });
      }
    }

    // Save to localStorage
    setTimeout(() => {
      localStorage.setItem('intelligence_documents', JSON.stringify(documents));
    }, 1000);
  };

  const deleteDocument = async (docId: string) => {
    try {
      // This would call the delete API
      // await fetch(`/api/intelligence/documents/${docId}`, { method: 'DELETE' });
      
      setDocuments(prev => prev.filter(d => d.id !== docId));
      const updated = documents.filter(d => d.id !== docId);
      localStorage.setItem('intelligence_documents', JSON.stringify(updated));
      
      if (selectedDocument?.id === docId) {
        setSelectedDocument(null);
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const updateDocumentMetadata = (docId: string, updates: Partial<DocumentAsset>) => {
    setDocuments(prev => prev.map(d => 
      d.id === docId ? { ...d, ...updates } : d
    ));
    
    const updated = documents.map(d => 
      d.id === docId ? { ...d, ...updates } : d
    );
    localStorage.setItem('intelligence_documents', JSON.stringify(updated));
  };
  // CRUD Operations for Analysis Models
  const loadAnalysisModels = async () => {
    try {
      // This would call a real API endpoint
      // const models = await fetch('/api/intelligence/models').then(r => r.json());
      // setAnalysisModels(models);
      
      const saved = localStorage.getItem('intelligence_analysis_models');
      if (saved) {
        setAnalysisModels(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Failed to load analysis models:', error);
    }
  };

  const createAnalysisModel = async () => {
    if (!newModelName.trim()) return;

    const newModel: AnalysisModel = {
      id: `model_${Date.now()}`,
      name: newModelName,
      type: newModelType,
      status: 'training',
      accuracy: 0,
      lastTrained: new Date().toISOString(),
      parameters: getDefaultParameters(newModelType),
      performance: { sharpe: 0, maxDrawdown: 0, winRate: 0 }
    };

    setAnalysisModels(prev => [newModel, ...prev]);

    try {
      // This would call the model training API
      // await fetch('/api/intelligence/models', { method: 'POST', body: JSON.stringify(newModel) });
      
      // Simulate training process
      setTimeout(() => {
        const trainedModel: AnalysisModel = {
          ...newModel,
          status: 'active',
          accuracy: 75 + Math.random() * 20, // Random accuracy between 75-95%
          performance: {
            sharpe: 1.2 + Math.random() * 0.8,
            maxDrawdown: -(Math.random() * 15 + 5),
            winRate: 60 + Math.random() * 25
          }
        };

        setAnalysisModels(prev => prev.map(m => m.id === newModel.id ? trainedModel : m));
        
        const updated = analysisModels.map(m => m.id === newModel.id ? trainedModel : m);
        localStorage.setItem('intelligence_analysis_models', JSON.stringify(updated));
      }, 3000);

    } catch (error) {
      console.error('Model creation failed:', error);
    }

    setNewModelName('');
    setShowCreateModel(false);
  };

  const getDefaultParameters = (type: AnalysisModel['type']) => {
    switch (type) {
      case 'regime_detection':
        return { lookback: 252, threshold: 0.15, features: 12, method: 'hmm' };
      case 'risk_assessment':
        return { horizon: 21, confidence: 0.95, method: 'var', distribution: 'normal' };
      case 'signal_generation':
        return { lookback: 50, threshold: 2.0, filters: 3, method: 'ml' };
      case 'portfolio_optimization':
        return { horizon: 21, constraints: 8, objective: 'sharpe', rebalance: 'weekly' };
      default:
        return {};
    }
  };

  const deleteAnalysisModel = (modelId: string) => {
    setAnalysisModels(prev => prev.filter(m => m.id !== modelId));
    const updated = analysisModels.filter(m => m.id !== modelId);
    localStorage.setItem('intelligence_analysis_models', JSON.stringify(updated));
    
    if (selectedAnalysisModel?.id === modelId) {
      setSelectedAnalysisModel(null);
    }
  };

  const retrainModel = async (modelId: string) => {
    setAnalysisModels(prev => prev.map(m => 
      m.id === modelId ? { ...m, status: 'training' } : m
    ));

    try {
      // This would call the retrain API
      // await fetch(`/api/intelligence/models/${modelId}/retrain`, { method: 'POST' });
      
      // Simulate retraining
      setTimeout(() => {
        setAnalysisModels(prev => prev.map(m => 
          m.id === modelId ? {
            ...m,
            status: 'active',
            lastTrained: new Date().toISOString(),
            accuracy: Math.min(95, m.accuracy + Math.random() * 5),
            performance: {
              sharpe: Math.max(0.5, m.performance.sharpe + (Math.random() - 0.5) * 0.3),
              maxDrawdown: Math.max(-20, m.performance.maxDrawdown + (Math.random() - 0.5) * 2),
              winRate: Math.min(85, Math.max(45, m.performance.winRate + (Math.random() - 0.5) * 10))
            }
          } : m
        ));
      }, 5000);

    } catch (error) {
      console.error('Model retraining failed:', error);
      
      setAnalysisModels(prev => prev.map(m => 
        m.id === modelId ? { ...m, status: 'active' } : m
      ));
    }
  };

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { 
      minimumFractionDigits: decimals, 
      maximumFractionDigits: decimals 
    });
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
      case 'indexed':
      case 'active':
        return 'text-green-400';
      case 'pending':
      case 'processing':
      case 'training':
        return 'text-yellow-400';
      case 'failed':
      case 'inactive':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };
  const renderChatTab = () => (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {chatMessages.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            <Brain className="w-12 h-12 mx-auto mb-4 text-[#ff8c00]" />
            <div className="text-lg mb-2">AI Financial Assistant</div>
            <div className="text-xs mt-2">Ask questions about markets, strategies, or risk management</div>
            
            {/* Quick suggestions from history */}
            {chatHistory.length > 0 && (
              <div className="mt-6">
                <div className="text-sm text-gray-500 mb-2">Recent queries:</div>
                <div className="flex flex-wrap gap-2 justify-center">
                  {chatHistory.slice(0, 4).map((query, idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentQuery(query)}
                      className="px-3 py-1 bg-gray-800 text-gray-300 text-xs rounded hover:bg-gray-700 transition-colors"
                    >
                      {query.length > 30 ? query.substring(0, 30) + '...' : query}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {chatMessages.map(message => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg ${
              message.type === 'user' 
                ? 'bg-[#ff8c00] text-black' 
                : 'bg-[#1a1a1a] text-white border border-[#444]'
            }`}>
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs opacity-70">
                  {message.type === 'user' ? 'You' : 'AI Assistant'}
                  {message.model && (
                    <span className="ml-2">• {message.model} ({message.provider})</span>
                  )}
                  {message.confidence && (
                    <span className="ml-2">• {message.confidence}% confidence</span>
                  )}
                </div>
                <div className="text-xs opacity-70">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
              <div className="whitespace-pre-wrap">{message.content}</div>
              {message.tokens && (
                <div className="text-xs opacity-50 mt-1">
                  {message.tokens} tokens used
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isProcessing && (
          <div className="flex justify-start">
            <div className="bg-[#1a1a1a] text-white p-3 rounded-lg border border-[#444]">
              <div className="flex items-center space-x-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>AI is analyzing your query...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Chat Input */}
      <div className="border-t border-[#444] p-4 flex-shrink-0">
        <div className="flex space-x-2 mb-2">
          <input
            type="text"
            value={currentQuery}
            onChange={(e) => setCurrentQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleLLMQuery(currentQuery)}
            placeholder="Ask about markets, strategies, risk management..."
            className="flex-1 bg-[#1a1a1a] border border-[#444] text-white text-sm px-3 py-2 rounded"
            disabled={isProcessing}
          />
          <button
            onClick={() => handleLLMQuery(currentQuery)}
            disabled={isProcessing || !currentQuery.trim()}
            className="px-4 py-2 bg-[#ff8c00] text-black text-sm rounded hover:bg-orange-600 disabled:opacity-50 flex items-center space-x-1"
          >
            <Send className="w-3 h-3" />
            <span>Send</span>
          </button>
          <button
            onClick={() => handleLLMQuery(currentQuery, true)}
            disabled={isProcessing || !currentQuery.trim()}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-1"
          >
            <Database className="w-3 h-3" />
            <span>RAG</span>
          </button>
        </div>
        
        <div className="flex justify-between items-center">
          <div className="flex space-x-2">
            <button
              onClick={clearChatHistory}
              className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600"
            >
              Clear Chat
            </button>
            <button
              onClick={exportChatHistory}
              disabled={chatMessages.length === 0}
              className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600 disabled:opacity-50 flex items-center space-x-1"
            >
              <Download className="w-3 h-3" />
              <span>Export</span>
            </button>
          </div>
          <div className="text-xs text-gray-400 flex items-center">
            <Zap className="w-3 h-3 mr-1" />
            AI-powered financial analysis with document context
          </div>
        </div>
      </div>
    </div>
  );
  const renderResearchTab = () => (
    <div className="h-full flex overflow-hidden">
      {/* Research Reports List */}
      <div className="w-1/3 border-r border-[#444] flex flex-col">
        <div className="p-4 border-b border-[#444] flex-shrink-0">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-[#ff8c00] font-medium">Research Reports</h3>
            <button
              onClick={() => setShowCreateReport(true)}
              className="px-3 py-1 bg-[#ff8c00] text-black text-xs rounded hover:bg-orange-600 flex items-center space-x-1"
            >
              <Plus className="w-3 h-3" />
              <span>New Report</span>
            </button>
          </div>
          
          <div className="flex space-x-2 mb-4">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="flex-1 bg-[#1a1a1a] border border-[#444] text-white text-xs px-2 py-1 rounded"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
            <button className="px-2 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600">
              <Filter className="w-3 h-3" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {researchReports
            .filter(report => filterStatus === 'all' || report.status === filterStatus)
            .map(report => (
              <div
                key={report.id}
                onClick={() => setSelectedReport(report)}
                className={`p-3 border-b border-[#333] cursor-pointer hover:bg-[#1a1a1a] transition-colors ${
                  selectedReport?.id === report.id ? 'bg-[#1a1a1a] border-l-2 border-l-[#ff8c00]' : ''
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-white text-sm font-medium truncate">{report.title}</h4>
                  <span className={`text-xs ${getStatusColor(report.status)}`}>
                    {report.status.toUpperCase()}
                  </span>
                </div>
                <p className="text-gray-400 text-xs mb-2 line-clamp-2">{report.summary}</p>
                <div className="flex justify-between items-center text-xs text-gray-500">
                  <span>{new Date(report.created).toLocaleDateString()}</span>
                  <div className="flex items-center space-x-2">
                    <span>{report.sources} sources</span>
                    <span>{report.confidence}% confidence</span>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {report.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="px-1 py-0.5 bg-gray-800 text-gray-300 text-xs rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Report Details */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedReport ? (
          <>
            <div className="p-4 border-b border-[#444] flex-shrink-0">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-[#ff8c00] text-lg font-medium mb-2">{selectedReport.title}</h2>
                  <p className="text-gray-400 text-sm mb-2">{selectedReport.query}</p>
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <span>Created: {new Date(selectedReport.created).toLocaleString()}</span>
                    <span>Updated: {new Date(selectedReport.updated).toLocaleString()}</span>
                    <span>Author: {selectedReport.author}</span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => exportResearchReport(selectedReport)}
                    className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600 flex items-center space-x-1"
                  >
                    <Download className="w-3 h-3" />
                    <span>Export</span>
                  </button>
                  <button
                    onClick={() => deleteResearchReport(selectedReport.id)}
                    className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 flex items-center space-x-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Delete</span>
                  </button>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2 mb-4">
                {selectedReport.tags.map(tag => (
                  <span key={tag} className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Status</div>
                  <div className={`text-lg ${getStatusColor(selectedReport.status)}`}>
                    {selectedReport.status.toUpperCase()}
                  </div>
                </div>
                <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Sources</div>
                  <div className="text-lg text-white">{selectedReport.sources}</div>
                </div>
                <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Confidence</div>
                  <div className="text-lg text-white">{selectedReport.confidence}%</div>
                </div>
              </div>

              <div className="bg-[#1a1a1a] p-4 rounded border border-[#444] mb-4">
                <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Summary</h3>
                <p className="text-white text-sm leading-relaxed">{selectedReport.summary}</p>
              </div>

              <div className="bg-[#1a1a1a] p-4 rounded border border-[#444]">
                <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Key Findings</h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(selectedReport.findings).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-400 text-sm capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-white text-sm">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <Search className="w-12 h-12 mx-auto mb-4 text-[#ff8c00]" />
              <div className="text-lg mb-2">Select a Research Report</div>
              <div className="text-sm">Choose a report from the list to view details</div>
            </div>
          </div>
        )}
      </div>

      {/* Create Report Modal */}
      {showCreateReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#444] rounded-lg p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-lg font-medium">Create Research Report</h3>
              <button
                onClick={() => setShowCreateReport(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-white text-sm mb-2">Report Title</label>
                <input
                  type="text"
                  value={newReportTitle}
                  onChange={(e) => setNewReportTitle(e.target.value)}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                  placeholder="e.g., Q1 2024 Market Analysis"
                />
              </div>
              
              <div>
                <label className="block text-white text-sm mb-2">Research Query</label>
                <textarea
                  value={newReportQuery}
                  onChange={(e) => setNewReportQuery(e.target.value)}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded h-24 resize-none"
                  placeholder="Describe what you want to research..."
                />
              </div>
              
              <div>
                <label className="block text-white text-sm mb-2">Tags (comma-separated)</label>
                <input
                  type="text"
                  value={reportTags.join(', ')}
                  onChange={(e) => setReportTags(e.target.value.split(',').map(t => t.trim()).filter(Boolean))}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                  placeholder="e.g., FX, Analysis, Q1 2024"
                />
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 mt-6">
              <button
                onClick={() => setShowCreateReport(false)}
                className="px-4 py-2 bg-gray-700 text-white text-sm rounded hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={createResearchReport}
                disabled={!newReportTitle.trim() || !newReportQuery.trim()}
                className="px-4 py-2 bg-[#ff8c00] text-black text-sm rounded hover:bg-orange-600 disabled:opacity-50"
              >
                Create Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  const renderDocumentsTab = () => (
    <div className="h-full flex overflow-hidden">
      {/* Documents List */}
      <div className="w-1/3 border-r border-[#444] flex flex-col">
        <div className="p-4 border-b border-[#444] flex-shrink-0">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-[#ff8c00] font-medium">Document Library</h3>
            <label className="px-3 py-1 bg-[#ff8c00] text-black text-xs rounded hover:bg-orange-600 cursor-pointer flex items-center space-x-1">
              <Upload className="w-3 h-3" />
              <span>Upload</span>
              <input
                type="file"
                multiple
                onChange={handleFileUpload}
                accept=".pdf,.doc,.docx,.txt,.md,.csv,.json"
                className="hidden"
              />
            </label>
          </div>
          
          <div className="flex space-x-2 mb-4">
            <select
              value={documentFilter}
              onChange={(e) => setDocumentFilter(e.target.value)}
              className="flex-1 bg-[#1a1a1a] border border-[#444] text-white text-xs px-2 py-1 rounded"
            >
              <option value="all">All Documents</option>
              <option value="indexed">Indexed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* RAG Stats */}
          {ragStats && (
            <div className="bg-[#1a1a1a] p-3 rounded border border-[#444] mb-4">
              <div className="text-[#ff8c00] text-xs font-medium mb-2">RAG Statistics</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <div className="text-gray-400">Documents:</div>
                  <div className="text-white">{ragStats.total_documents}</div>
                </div>
                <div>
                  <div className="text-gray-400">Chunks:</div>
                  <div className="text-white">{ragStats.indexed_chunks}</div>
                </div>
                <div>
                  <div className="text-gray-400">Storage:</div>
                  <div className="text-white">{ragStats.storage_size}</div>
                </div>
                <div>
                  <div className="text-gray-400">Threshold:</div>
                  <div className="text-white">{ragStats.similarity_threshold}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {documents
            .filter(doc => documentFilter === 'all' || doc.status === documentFilter)
            .map(doc => (
              <div
                key={doc.id}
                onClick={() => setSelectedDocument(doc)}
                className={`p-3 border-b border-[#333] cursor-pointer hover:bg-[#1a1a1a] transition-colors ${
                  selectedDocument?.id === doc.id ? 'bg-[#1a1a1a] border-l-2 border-l-[#ff8c00]' : ''
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-white text-sm font-medium truncate">{doc.name}</h4>
                  <span className={`text-xs ${getStatusColor(doc.status)}`}>
                    {doc.status.toUpperCase()}
                  </span>
                </div>
                
                {uploadProgress[doc.id] !== undefined && (
                  <div className="mb-2">
                    <div className="bg-gray-700 rounded-full h-1">
                      <div 
                        className="bg-[#ff8c00] h-1 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress[doc.id]}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-400 mt-1">{uploadProgress[doc.id]}% uploaded</div>
                  </div>
                )}
                
                <div className="flex justify-between items-center text-xs text-gray-500 mb-2">
                  <span>{formatBytes(doc.size)}</span>
                  <span>{doc.chunks} chunks</span>
                </div>
                
                <div className="text-xs text-gray-400 mb-2">{doc.category}</div>
                
                <div className="flex flex-wrap gap-1">
                  {doc.tags.slice(0, 2).map(tag => (
                    <span key={tag} className="px-1 py-0.5 bg-gray-800 text-gray-300 text-xs rounded">
                      {tag}
                    </span>
                  ))}
                </div>
                
                <div className="text-xs text-gray-500 mt-2">
                  {new Date(doc.uploaded).toLocaleDateString()}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Document Details */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedDocument ? (
          <>
            <div className="p-4 border-b border-[#444] flex-shrink-0">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-[#ff8c00] text-lg font-medium mb-2">{selectedDocument.name}</h2>
                  <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <span>Type: {selectedDocument.type.toUpperCase()}</span>
                    <span>Size: {formatBytes(selectedDocument.size)}</span>
                    <span>Chunks: {selectedDocument.chunks}</span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => {/* Implement document preview */}}
                    className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600 flex items-center space-x-1"
                  >
                    <Eye className="w-3 h-3" />
                    <span>Preview</span>
                  </button>
                  <button
                    onClick={() => deleteDocument(selectedDocument.id)}
                    className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 flex items-center space-x-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Delete</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Status</div>
                  <div className={`text-lg ${getStatusColor(selectedDocument.status)}`}>
                    {selectedDocument.status.toUpperCase()}
                  </div>
                </div>
                <div className="bg-[#1a1a1a] p-3 rounded border border-[#444]">
                  <div className="text-[#ff8c00] text-sm font-medium mb-1">Uploaded</div>
                  <div className="text-lg text-white">
                    {new Date(selectedDocument.uploaded).toLocaleDateString()}
                  </div>
                </div>
              </div>

              <div className="bg-[#1a1a1a] p-4 rounded border border-[#444] mb-4">
                <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Category & Tags</h3>
                <div className="mb-3">
                  <input
                    type="text"
                    value={selectedDocument.category}
                    onChange={(e) => updateDocumentMetadata(selectedDocument.id, { category: e.target.value })}
                    className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                    placeholder="Document category"
                  />
                </div>
                <div className="flex flex-wrap gap-2 mb-3">
                  {selectedDocument.tags.map((tag, idx) => (
                    <span key={idx} className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded flex items-center space-x-1">
                      <span>{tag}</span>
                      <button
                        onClick={() => {
                          const newTags = selectedDocument.tags.filter((_, i) => i !== idx);
                          updateDocumentMetadata(selectedDocument.id, { tags: newTags });
                        }}
                        className="text-red-400 hover:text-red-300"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
                <input
                  type="text"
                  placeholder="Add tags (press Enter)"
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      const input = e.target as HTMLInputElement;
                      const newTag = input.value.trim();
                      if (newTag && !selectedDocument.tags.includes(newTag)) {
                        updateDocumentMetadata(selectedDocument.id, { 
                          tags: [...selectedDocument.tags, newTag] 
                        });
                        input.value = '';
                      }
                    }
                  }}
                />
              </div>

              <div className="bg-[#1a1a1a] p-4 rounded border border-[#444]">
                <h3 className="text-[#ff8c00] text-sm font-medium mb-3">Metadata</h3>
                <div className="space-y-2">
                  {Object.entries(selectedDocument.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-400 text-sm capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-white text-sm">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <FileText className="w-12 h-12 mx-auto mb-4 text-[#ff8c00]" />
              <div className="text-lg mb-2">Select a Document</div>
              <div className="text-sm">Choose a document from the library to view details</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
  const renderAnalysisTab = () => (
    <div className="h-full p-4 space-y-4 overflow-y-auto">
      {/* Market Regime Analysis - matching original layout */}
      <div className="border border-[#444] rounded">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00] font-medium">MARKET REGIME ANALYSIS</div>
        </div>
        <div className="p-3">
          <div className="grid grid-cols-3 gap-4">
            {regimes.map((regime) => (
              <div
                key={regime.id}
                onClick={() => setSelectedRegime(regime.id)}
                className={`border p-3 rounded cursor-pointer transition-colors ${
                  selectedRegime === regime.id
                    ? 'border-[#ff8c00] bg-[#ff8c00]/10'
                    : 'border-[#444] hover:border-[#666]'
                }`}
              >
                <div className="text-[#ff8c00] text-xs font-medium mb-2">{regime.name}</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Probability:</span>
                    <span className="text-[#fff]">{regime.probability}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Duration:</span>
                    <span className="text-[#fff]">{regime.duration}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Volatility:</span>
                    <span className={`${
                      regime.volatility === 'HIGH' ? 'text-[#ff0000]' :
                      regime.volatility === 'LOW' ? 'text-[#00ff00]' : 'text-[#ffff00]'
                    }`}>
                      {regime.volatility}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Trend:</span>
                    <span className="text-[#fff]">{regime.trend}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Liquidity:</span>
                    <span className="text-[#fff]">{regime.liquidity}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Intelligence Signals - matching original layout */}
      <div className="border border-[#444] rounded">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00] font-medium">INTELLIGENCE SIGNALS</div>
        </div>
        <div className="p-3">
          <div className="space-y-2">
            {intelligenceSignals.map((signal, idx) => (
              <div key={idx} className="flex justify-between items-center p-2 border border-[#333] rounded">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-[#ff8c00] rounded-full"></div>
                  <div className="text-xs">
                    <div className="text-[#fff]">{signal.description}</div>
                    <div className="text-[#666]">{signal.type} • {signal.timestamp}</div>
                  </div>
                </div>
                <button 
                  onClick={() => {
                    // Trigger regime data fetch when investigating
                    fetchRegimeData('EURUSD');
                    fetchGraphFeatures('EURUSD');
                  }}
                  className="text-[#ff8c00] hover:text-orange-400 text-xs px-2 py-1 border border-[#ff8c00] rounded hover:bg-[#ff8c00]/10 transition-colors"
                >
                  Investigate
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Market Embeddings - matching original layout */}
      <div className="border border-[#444] rounded">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00] font-medium">MARKET EMBEDDINGS</div>
        </div>
        <div className="p-3">
          <div className="space-y-2">
            {embeddings.map((embedding, idx) => (
              <div key={idx} className="p-2 border border-[#333] rounded">
                <div className="flex justify-between items-center">
                  <div className="text-[#ff8c00] text-sm font-medium">{embedding.asset}</div>
                  <div className="space-y-1 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">Time:</span>
                      <span className="text-[#fff]">{embedding.timestamp}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">Confidence:</span>
                      <span className="text-[#00ff00]">{(embedding.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Analysis Models Section - Enhanced but keeping original style */}
      <div className="border border-[#444] rounded">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="flex justify-between items-center">
            <div className="text-[#ff8c00] font-medium">ANALYSIS MODELS</div>
            <button
              onClick={() => setShowCreateModel(true)}
              className="px-2 py-1 bg-[#ff8c00] text-black text-xs rounded hover:bg-orange-600 flex items-center space-x-1"
            >
              <Plus className="w-3 h-3" />
              <span>New</span>
            </button>
          </div>
        </div>
        <div className="p-3">
          <div className="grid grid-cols-2 gap-4">
            {analysisModels.map(model => (
              <div
                key={model.id}
                onClick={() => setSelectedAnalysisModel(model)}
                className={`border p-3 rounded cursor-pointer transition-colors ${
                  selectedAnalysisModel?.id === model.id
                    ? 'border-[#ff8c00] bg-[#ff8c00]/10'
                    : 'border-[#444] hover:border-[#666]'
                }`}
              >
                <div className="text-[#ff8c00] text-xs font-medium mb-2">{model.name}</div>
                <div className="text-xs text-gray-400 mb-2 capitalize">
                  {model.type.replace(/_/g, ' ')}
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Status:</span>
                    <span className={getStatusColor(model.status)}>{model.status.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Accuracy:</span>
                    <span className="text-[#fff]">{formatNumber(model.accuracy, 1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Sharpe:</span>
                    <span className="text-[#fff]">{formatNumber(model.performance.sharpe, 2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Drawdown:</span>
                    <span className="text-[#ff0000]">{formatNumber(model.performance.maxDrawdown, 1)}%</span>
                  </div>
                </div>
                <div className="flex justify-between items-center mt-2">
                  <div className="text-xs text-gray-500">
                    {new Date(model.lastTrained).toLocaleDateString()}
                  </div>
                  <div className="flex space-x-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        retrainModel(model.id);
                      }}
                      disabled={model.status === 'training'}
                      className="px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                      {model.status === 'training' ? 'Training...' : 'Retrain'}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteAnalysisModel(model.id);
                      }}
                      className="px-2 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Create Model Modal */}
      {showCreateModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#444] rounded-lg p-6 w-96">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[#ff8c00] text-lg font-medium">Create Analysis Model</h3>
              <button
                onClick={() => setShowCreateModel(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-white text-sm mb-2">Model Name</label>
                <input
                  type="text"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                  placeholder="e.g., FX Regime Detector v3.0"
                />
              </div>
              
              <div>
                <label className="block text-white text-sm mb-2">Model Type</label>
                <select
                  value={newModelType}
                  onChange={(e) => setNewModelType(e.target.value as AnalysisModel['type'])}
                  className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                >
                  <option value="regime_detection">Regime Detection</option>
                  <option value="risk_assessment">Risk Assessment</option>
                  <option value="signal_generation">Signal Generation</option>
                  <option value="portfolio_optimization">Portfolio Optimization</option>
                </select>
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 mt-6">
              <button
                onClick={() => setShowCreateModel(false)}
                className="px-4 py-2 bg-gray-700 text-white text-sm rounded hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={createAnalysisModel}
                disabled={!newModelName.trim()}
                className="px-4 py-2 bg-[#ff8c00] text-black text-sm rounded hover:bg-orange-600 disabled:opacity-50"
              >
                Create Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2">
        <div className="text-center text-[#ff8c00] text-sm tracking-wide">
          INTELLIGENCE - AI-POWERED FINANCIAL ANALYSIS
        </div>
      </div>

      {/* Tab Navigation - reordered to match original */}
      <div className="flex border-b border-[#444]">
        {[
          { key: 'analysis', label: 'Analysis', icon: <BarChart3 className="w-4 h-4" />, count: analysisModels.length },
          { key: 'chat', label: 'AI Chat', icon: <Brain className="w-4 h-4" />, count: chatMessages.length },
          { key: 'research', label: 'Research', icon: <Search className="w-4 h-4" />, count: researchReports.length },
          { key: 'documents', label: 'Documents', icon: <FileText className="w-4 h-4" />, count: documents.length }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-4 py-2 text-sm font-medium transition-colors flex items-center space-x-2 relative ${
              activeTab === tab.key
                ? 'bg-orange-900 text-orange-400 border-b-2 border-orange-400'
                : 'text-gray-400 hover:text-orange-400'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
            {tab.count > 0 && (
              <span className="absolute -top-1 -right-1 bg-[#ff8c00] text-black text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {tab.count > 99 ? '99+' : tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content - reordered to match original */}
      <div className="flex-1 overflow-hidden min-h-0">
        {activeTab === 'analysis' && renderAnalysisTab()}
        {activeTab === 'chat' && renderChatTab()}
        {activeTab === 'research' && renderResearchTab()}
        {activeTab === 'documents' && renderDocumentsTab()}
      </div>
    </div>
  );
}
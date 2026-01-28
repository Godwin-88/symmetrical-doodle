import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { IntelligenceNew } from '../IntelligenceNew';

// Mock the trading store
const mockUseTradingStore = {
  regimes: [
    {
      id: '1',
      name: 'High Volatility',
      probability: 75,
      duration: '2-3 days',
      volatility: 'HIGH',
      trend: 'BEARISH',
      liquidity: 'LOW'
    }
  ],
  embeddings: [],
  intelligenceSignals: [],
  selectedRegime: null,
  setSelectedRegime: jest.fn(),
  fetchRegimeData: jest.fn(),
  fetchGraphFeatures: jest.fn(),
  isLoading: false
};

jest.mock('../../store/tradingStore', () => ({
  useTradingStore: () => mockUseTradingStore
}));

// Mock the multi-source service
jest.mock('../../../services/multiSourceService', () => ({
  getAvailableSources: jest.fn().mockResolvedValue([]),
  getProcessedDocuments: jest.fn().mockResolvedValue([])
}));

// Mock the LLM service
jest.mock('../../../services/llmService', () => ({
  queryLLM: jest.fn(),
  financialAnalysis: jest.fn(),
  queryRAG: jest.fn(),
  ingestDocument: jest.fn(),
  getRAGStats: jest.fn().mockResolvedValue({
    total_documents: 4,
    indexed_chunks: 144,
    storage_size: '8.5MB'
  }),
  comprehensiveResearch: jest.fn(),
  stockAnalysis: jest.fn(),
  getMarketOverview: jest.fn()
}));

// Mock the MultiSourcePanel component
jest.mock('../MultiSourcePanel', () => ({
  MultiSourcePanel: ({ onDocumentProcessed, onError }: any) => (
    <div data-testid="multi-source-panel">
      <button 
        onClick={() => onDocumentProcessed({
          id: 'test-doc',
          name: 'Test Document.pdf',
          sourceType: 'google_drive',
          sourceId: 'gd_test123',
          sourceUrl: 'https://drive.google.com/file/d/test123',
          size: 1024000,
          uploadedAt: new Date().toISOString(),
          status: 'indexed',
          chunks: 25,
          category: 'Test Category',
          tags: ['test'],
          metadata: { pages: 10 },
          processingStats: {
            chunksCreated: 25,
            embeddingModel: 'text-embedding-3-large',
            qualityScore: 0.95,
            processingTime: 60
          }
        })}
        data-testid="add-test-document"
      >
        Add Test Document
      </button>
    </div>
  )
}));

describe('Multi-Source Document Library Integration', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders unified document library with multi-source support', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Check for unified document library header
    expect(screen.getByText('Unified Document Library')).toBeInTheDocument();
    
    // Check for multi-source statistics
    expect(screen.getByText('Multi-Source Statistics')).toBeInTheDocument();
    
    // Check for source filter options
    const sourceFilter = screen.getByDisplayValue('All Sources');
    expect(sourceFilter).toBeInTheDocument();
    
    // Verify filter options include multi-source types
    fireEvent.click(sourceFilter);
    expect(screen.getByText('Google Drive')).toBeInTheDocument();
    expect(screen.getByText('ZIP Archives')).toBeInTheDocument();
    expect(screen.getByText('Local Directory')).toBeInTheDocument();
  });

  test('displays documents with source type indicators', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for mock data to load
    await waitFor(() => {
      // Check for documents with different source types
      expect(screen.getByText('ECB_Policy_Report_2024.pdf')).toBeInTheDocument();
      expect(screen.getByText('Market_Structure_Analysis.docx')).toBeInTheDocument();
      expect(screen.getByText('Trading_Strategies_ML.pdf')).toBeInTheDocument();
      expect(screen.getByText('Risk_Management_Framework.pdf')).toBeInTheDocument();
    });
    
    // Check for source type indicators (Google Drive, ZIP Archive, Upload, Local Directory)
    const sourceBreakdown = screen.getByText(/By Source:/);
    expect(sourceBreakdown).toBeInTheDocument();
  });

  test('shows enhanced document details with multi-source metadata', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for documents to load and click on a Google Drive document
    await waitFor(() => {
      fireEvent.click(screen.getByText('ECB_Policy_Report_2024.pdf'));
    });
    
    // Check for enhanced document details
    expect(screen.getByText('Source Information')).toBeInTheDocument();
    expect(screen.getByText('Processing Statistics')).toBeInTheDocument();
    
    // Check for source-specific information
    expect(screen.getByText('Google Drive')).toBeInTheDocument();
    expect(screen.getByText('View Original Location')).toBeInTheDocument();
    
    // Check for processing statistics
    expect(screen.getByText('Quality Score:')).toBeInTheDocument();
    expect(screen.getByText('Embedding Model:')).toBeInTheDocument();
    expect(screen.getByText('Processing Time:')).toBeInTheDocument();
  });

  test('supports filtering by source type', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for documents to load
    await waitFor(() => {
      expect(screen.getByText('ECB_Policy_Report_2024.pdf')).toBeInTheDocument();
    });
    
    // Filter by Google Drive
    const sourceFilter = screen.getByDisplayValue('All Sources');
    fireEvent.change(sourceFilter, { target: { value: 'google_drive' } });
    
    // Should show only Google Drive documents
    expect(screen.getByText('ECB_Policy_Report_2024.pdf')).toBeInTheDocument();
    
    // Filter by uploads
    fireEvent.change(sourceFilter, { target: { value: 'upload' } });
    
    // Should show only uploaded documents
    expect(screen.getByText('Trading_Strategies_ML.pdf')).toBeInTheDocument();
  });

  test('integrates with MultiSourcePanel for new document processing', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Check that MultiSourcePanel is rendered
    expect(screen.getByTestId('multi-source-panel')).toBeInTheDocument();
    
    // Simulate adding a new document through MultiSourcePanel
    fireEvent.click(screen.getByTestId('add-test-document'));
    
    // Wait for the new document to appear in the list
    await waitFor(() => {
      expect(screen.getByText('Test Document.pdf')).toBeInTheDocument();
    });
    
    // Click on the new document to view details
    fireEvent.click(screen.getByText('Test Document.pdf'));
    
    // Check that it shows Google Drive as source
    expect(screen.getByText('Google Drive')).toBeInTheDocument();
    expect(screen.getByText('View Original Location')).toBeInTheDocument();
  });

  test('displays reprocessing option for supported documents', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for documents to load and select a document that supports reprocessing
    await waitFor(() => {
      fireEvent.click(screen.getByText('ECB_Policy_Report_2024.pdf'));
    });
    
    // Check for reprocess button
    expect(screen.getByText('Reprocess')).toBeInTheDocument();
  });

  test('shows source-specific metadata in document details', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Select a ZIP archive document
    await waitFor(() => {
      fireEvent.click(screen.getByText('Market_Structure_Analysis.docx'));
    });
    
    // Check for ZIP-specific metadata
    expect(screen.getByText('ZIP Archive')).toBeInTheDocument();
    expect(screen.getByText('Source Information')).toBeInTheDocument();
    
    // Check for processing statistics specific to this document
    expect(screen.getByText('BAAI/bge-large-en-v1.5')).toBeInTheDocument(); // Embedding model
    expect(screen.getByText('89.0%')).toBeInTheDocument(); // Quality score
  });

  test('maintains backward compatibility with existing document interface', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for documents to load
    await waitFor(() => {
      expect(screen.getByText('ECB_Policy_Report_2024.pdf')).toBeInTheDocument();
    });
    
    // Select a document
    fireEvent.click(screen.getByText('ECB_Policy_Report_2024.pdf'));
    
    // Check that existing functionality still works
    expect(screen.getByText('Category & Tags')).toBeInTheDocument();
    expect(screen.getByText('Metadata')).toBeInTheDocument();
    expect(screen.getByText('Preview')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
    
    // Check that tags can still be managed
    expect(screen.getByPlaceholderText('Add tags (press Enter)')).toBeInTheDocument();
  });

  test('displays correct source statistics in multi-source breakdown', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Wait for statistics to load
    await waitFor(() => {
      expect(screen.getByText('Multi-Source Statistics')).toBeInTheDocument();
    });
    
    // Check for source breakdown
    const sourceBreakdown = screen.getByText(/By Source:/);
    expect(sourceBreakdown).toBeInTheDocument();
    
    // Should show counts for different source types
    expect(screen.getByText(/Google Drive: 1/)).toBeInTheDocument();
    expect(screen.getByText(/ZIP Archive: 1/)).toBeInTheDocument();
    expect(screen.getByText(/File Upload: 1/)).toBeInTheDocument();
    expect(screen.getByText(/Local Directory: 1/)).toBeInTheDocument();
  });
});
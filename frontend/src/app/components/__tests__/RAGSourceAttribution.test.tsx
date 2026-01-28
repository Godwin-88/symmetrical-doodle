import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { IntelligenceNew } from '../IntelligenceNew';
import * as llmService from '../../../services/llmService';

// Mock the LLM service
jest.mock('../../../services/llmService');
const mockLlmService = llmService as jest.Mocked<typeof llmService>;

// Mock the trading store
jest.mock('../../store/tradingStore', () => ({
  useTradingStore: () => ({
    regimes: [],
    embeddings: [],
    intelligenceSignals: [],
    selectedRegime: null,
    setSelectedRegime: jest.fn(),
    fetchRegimeData: jest.fn(),
    fetchGraphFeatures: jest.fn(),
    isLoading: false,
  })
}));

// Mock the MultiSourcePanel component
jest.mock('../MultiSourcePanel', () => ({
  MultiSourcePanel: ({ onDocumentProcessed }: any) => (
    <div data-testid="multi-source-panel">
      <button 
        onClick={() => onDocumentProcessed?.({
          id: 'test-doc',
          name: 'Test Document.pdf',
          sourceType: 'google_drive',
          sourceId: 'gd_123',
          sourceUrl: 'https://drive.google.com/file/d/123',
          size: 1024000,
          uploadedAt: new Date().toISOString(),
          status: 'indexed',
          chunks: 25,
          category: 'Research',
          tags: ['test'],
          metadata: {}
        })}
      >
        Add Test Document
      </button>
    </div>
  )
}));

describe('RAG Integration with Source Attribution', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock enhanced RAG response with comprehensive source attribution
    mockLlmService.queryRAG.mockResolvedValue({
      answer: `Enhanced RAG Analysis with Multi-Source Attribution:

Based on comprehensive analysis of documents from multiple sources:

**Key Findings:**
• Cross-source validation confirms consistent patterns across Google Drive, local archives, and uploaded documents
• Primary insights from Google Drive document "Financial Analysis 2024" (94% quality score)
• Supporting evidence from ZIP archive "Market Trends Report" and uploaded "Risk Framework"
• Analysis incorporates 3 high-quality sources with average similarity of 85.7%

**Source Attribution:**
• Google Drive: 1 document (33.3%)
• Local ZIP Archive: 1 document (33.3%)
• Direct Upload: 1 document (33.3%)

**Confidence Assessment:**
High confidence based on cross-source validation and quality scores above 85%.`,
      sources: [
        {
          file_path: 'research/financial_analysis_2024.pdf',
          similarity: 0.94,
          metadata: { 
            type: 'research', 
            date: '2024-01-15',
            pages: 28,
            author: 'Research Team'
          },
          source_type: 'google_drive',
          source_id: 'gd_1BxYz3456789',
          source_url: 'https://drive.google.com/file/d/1BxYz3456789/view',
          original_location: 'https://drive.google.com/file/d/1BxYz3456789/view',
          document_title: 'Financial Analysis 2024',
          chunk_content: 'Based on comprehensive analysis of market data from multiple sources...',
          chunk_order: 3,
          processing_stats: {
            embedding_model: 'text-embedding-3-large',
            quality_score: 0.94,
            chunks_created: 45
          }
        },
        {
          file_path: 'reports/market_trends.pdf',
          similarity: 0.87,
          metadata: { 
            type: 'report', 
            date: '2024-01-10',
            category: 'Market Analysis'
          },
          source_type: 'local_zip',
          source_id: 'research_papers_2024.zip/market_trends.pdf',
          original_location: 'research_papers_2024.zip',
          document_title: 'Market Trends Report',
          chunk_content: 'The analysis of market conditions reveals significant trends...',
          chunk_order: 1,
          processing_stats: {
            embedding_model: 'BAAI/bge-large-en-v1.5',
            quality_score: 0.87,
            chunks_created: 32
          }
        },
        {
          file_path: 'uploads/risk_framework.pdf',
          similarity: 0.76,
          metadata: { 
            type: 'framework', 
            date: '2024-01-08',
            category: 'Risk Management'
          },
          source_type: 'upload',
          source_id: 'upload_20240108_001',
          original_location: 'Direct Upload',
          document_title: 'Risk Framework',
          chunk_content: 'The framework provides comprehensive guidelines for risk assessment...',
          chunk_order: 2,
          processing_stats: {
            embedding_model: 'sentence-transformers/all-mpnet-base-v2',
            quality_score: 0.91,
            chunks_created: 28
          }
        }
      ],
      confidence: 'high',
      model_info: {
        model: 'gpt-4-enhanced',
        provider: 'openai',
        tokens_used: 350,
        response_time: 1.2
      },
      retrieval_info: {
        documents_found: 8,
        avg_similarity: 0.857,
        sources_by_type: {
          'google_drive': 3,
          'local_zip': 2,
          'upload': 3
        },
        total_chunks_searched: 156
      },
      source_attribution: {
        total_sources: 3,
        source_breakdown: {
          'google_drive': 1,
          'local_zip': 1,
          'upload': 1
        },
        primary_source: {
          name: 'Financial Analysis 2024',
          type: 'google_drive',
          url: 'https://drive.google.com/file/d/1BxYz3456789/view',
          confidence: 0.94
        },
        cross_source_validation: true
      }
    });
  });

  test('displays comprehensive source attribution in RAG responses', async () => {
    render(<IntelligenceNew />);
    
    // Switch to chat tab
    fireEvent.click(screen.getByText('AI Chat'));
    
    // Enter a query
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'What are the current market trends?' } });
    
    // Click RAG button
    fireEvent.click(screen.getByText('RAG'));
    
    // Wait for response
    await waitFor(() => {
      expect(screen.getByText(/Enhanced RAG Analysis with Multi-Source Attribution/)).toBeInTheDocument();
    });
    
    // Check that source attribution is displayed
    expect(screen.getByText(/📚 Sources \(3\)/)).toBeInTheDocument();
    expect(screen.getByText(/Multi-Source Analysis/)).toBeInTheDocument();
    expect(screen.getByText(/Cross-validated/)).toBeInTheDocument();
    
    // Check source breakdown
    expect(screen.getByText(/Google Drive: 1/)).toBeInTheDocument();
    expect(screen.getByText(/ZIP Archive: 1/)).toBeInTheDocument();
    expect(screen.getByText(/File Upload: 1/)).toBeInTheDocument();
    
    // Check primary source
    expect(screen.getByText(/Primary: Financial Analysis 2024/)).toBeInTheDocument();
    expect(screen.getByText(/94% confidence/)).toBeInTheDocument();
    
    // Check individual sources
    expect(screen.getByText('Financial Analysis 2024')).toBeInTheDocument();
    expect(screen.getByText('Market Trends Report')).toBeInTheDocument();
    expect(screen.getByText('Risk Framework')).toBeInTheDocument();
    
    // Check retrieval statistics
    expect(screen.getByText(/Documents searched: 8/)).toBeInTheDocument();
    expect(screen.getByText(/Avg similarity: 86%/)).toBeInTheDocument();
    expect(screen.getByText(/156 chunks analyzed/)).toBeInTheDocument();
  });

  test('displays source-specific information for each document', async () => {
    render(<IntelligenceNew />);
    
    // Switch to chat tab and perform RAG query
    fireEvent.click(screen.getByText('AI Chat'));
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'Analyze financial trends' } });
    fireEvent.click(screen.getByText('RAG'));
    
    await waitFor(() => {
      expect(screen.getByText(/📚 Sources/)).toBeInTheDocument();
    });
    
    // Check that each source shows its specific information
    const sources = screen.getAllByText(/text-embedding-3-large|BAAI\/bge-large-en-v1.5|sentence-transformers/);
    expect(sources.length).toBeGreaterThan(0);
    
    // Check quality scores are displayed
    expect(screen.getByText(/Quality: 94%/)).toBeInTheDocument();
    expect(screen.getByText(/Quality: 87%/)).toBeInTheDocument();
    expect(screen.getByText(/Quality: 91%/)).toBeInTheDocument();
    
    // Check chunk content previews
    expect(screen.getByText(/Based on comprehensive analysis of market data/)).toBeInTheDocument();
    expect(screen.getByText(/The analysis of market conditions reveals/)).toBeInTheDocument();
    expect(screen.getByText(/The framework provides comprehensive guidelines/)).toBeInTheDocument();
  });

  test('provides external links to original document locations', async () => {
    render(<IntelligenceNew />);
    
    // Switch to chat tab and perform RAG query
    fireEvent.click(screen.getByText('AI Chat'));
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'Show me research data' } });
    fireEvent.click(screen.getByText('RAG'));
    
    await waitFor(() => {
      expect(screen.getByText(/📚 Sources/)).toBeInTheDocument();
    });
    
    // Check for external link buttons
    const externalLinks = screen.getAllByTitle('View original document');
    expect(externalLinks.length).toBeGreaterThan(0);
    
    // Verify the links would open the correct URLs
    // Note: In a real test, you might want to mock window.open and verify it's called with the right URL
  });

  test('shows cross-source validation indicator when multiple source types are used', async () => {
    render(<IntelligenceNew />);
    
    // Switch to chat tab and perform RAG query
    fireEvent.click(screen.getByText('AI Chat'));
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'Cross-validate market analysis' } });
    fireEvent.click(screen.getByText('RAG'));
    
    await waitFor(() => {
      expect(screen.getByText(/Cross-validated/)).toBeInTheDocument();
    });
    
    // Check that the cross-validation indicator is present
    const crossValidatedElement = screen.getByText(/Cross-validated/);
    expect(crossValidatedElement).toBeInTheDocument();
    
    // Check that it shows the checkmark icon (via class or test id)
    const parentElement = crossValidatedElement.closest('span');
    expect(parentElement).toHaveClass('text-green-400');
  });

  test('handles RAG responses without sources gracefully', async () => {
    // Mock a response with no sources
    mockLlmService.queryRAG.mockResolvedValue({
      answer: 'I don\'t have specific documents to reference for this query.',
      sources: [],
      confidence: 'low',
      model_info: {
        model: 'gpt-4',
        provider: 'openai',
        tokens_used: 50,
        response_time: 0.3
      },
      retrieval_info: {
        documents_found: 0,
        avg_similarity: 0.0,
        sources_by_type: {},
        total_chunks_searched: 0
      },
      source_attribution: {
        total_sources: 0,
        source_breakdown: {},
        primary_source: null,
        cross_source_validation: false
      }
    });

    render(<IntelligenceNew />);
    
    // Switch to chat tab and perform RAG query
    fireEvent.click(screen.getByText('AI Chat'));
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'Unknown topic query' } });
    fireEvent.click(screen.getByText('RAG'));
    
    await waitFor(() => {
      expect(screen.getByText(/I don't have specific documents/)).toBeInTheDocument();
    });
    
    // Should not show source attribution section when no sources
    expect(screen.queryByText(/📚 Sources/)).not.toBeInTheDocument();
  });

  test('integrates with multi-source document processing', async () => {
    render(<IntelligenceNew />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    // Add a test document via the MultiSourcePanel
    fireEvent.click(screen.getByText('Add Test Document'));
    
    // Verify the document appears in the list
    await waitFor(() => {
      expect(screen.getByText('Test Document.pdf')).toBeInTheDocument();
    });
    
    // Switch back to chat and perform RAG query
    fireEvent.click(screen.getByText('AI Chat'));
    const input = screen.getByPlaceholderText(/Ask about markets/);
    fireEvent.change(input, { target: { value: 'Query about test document' } });
    fireEvent.click(screen.getByText('RAG'));
    
    // Verify RAG can access the newly processed document
    await waitFor(() => {
      expect(mockLlmService.queryRAG).toHaveBeenCalledWith({
        question: 'Query about test document'
      });
    });
  });
});
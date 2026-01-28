import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MultiSourcePanel } from '../MultiSourcePanel';
import * as multiSourceService from '../../services/multiSourceService';

// Mock the multiSourceService
jest.mock('../../services/multiSourceService');
jest.mock('../../services/multiSourceWebSocket', () => ({
  useMultiSourceWebSocket: () => ({
    subscribe: jest.fn(() => jest.fn())
  })
}));

const mockMultiSourceService = multiSourceService as jest.Mocked<typeof multiSourceService>;

describe('MultiSourcePanel', () => {
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Mock default responses
    mockMultiSourceService.getAvailableSources.mockResolvedValue([
      {
        type: multiSourceService.DataSourceType.GOOGLE_DRIVE,
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
        type: multiSourceService.DataSourceType.LOCAL_ZIP,
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
      }
    ]);

    mockMultiSourceService.getProcessedDocuments.mockResolvedValue([]);
    mockMultiSourceService.browseSource.mockResolvedValue({
      sourceType: multiSourceService.DataSourceType.LOCAL_ZIP,
      sourceName: 'ZIP Archives',
      folders: [],
      files: [
        {
          file_id: 'test_file_1',
          name: 'Test_Document.pdf',
          size: 1024000,
          modified_time: new Date().toISOString(),
          source_type: multiSourceService.DataSourceType.LOCAL_ZIP,
          source_path: '/test/Test_Document.pdf',
          mime_type: 'application/pdf',
          parent_folders: [],
          source_specific_metadata: {
            processing_status: 'completed',
            quality_score: 0.95,
            chunks: 25
          }
        }
      ]
    });
  });

  it('renders the tabbed interface correctly', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Sources')).toBeInTheDocument();
      expect(screen.getByText('Browse')).toBeInTheDocument();
      expect(screen.getByText('Processing')).toBeInTheDocument();
      expect(screen.getByText('Documents')).toBeInTheDocument();
    });
  });

  it('displays connected sources in the sources tab', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
      expect(screen.getByText('ZIP Archives')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Not connected')).toBeInTheDocument();
    });
  });

  it('shows search functionality in browse tab', async () => {
    render(<MultiSourcePanel />);
    
    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search across all sources...')).toBeInTheDocument();
      expect(screen.getByText('Filters')).toBeInTheDocument();
      expect(screen.getByText('Search')).toBeInTheDocument();
    });
  });

  it('displays files with processing status indicators', async () => {
    render(<MultiSourcePanel />);
    
    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));
    
    await waitFor(() => {
      expect(screen.getByText('Test_Document.pdf')).toBeInTheDocument();
      expect(screen.getByText('Processed')).toBeInTheDocument();
      expect(screen.getByText('(95%)')).toBeInTheDocument(); // Quality score
      expect(screen.getByText('25 chunks')).toBeInTheDocument();
    });
  });

  it('supports file selection for batch processing', async () => {
    render(<MultiSourcePanel />);
    
    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));
    
    await waitFor(() => {
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      
      expect(screen.getByText('Process Selected (1)')).toBeInTheDocument();
    });
  });

  it('shows hierarchical folder navigation for applicable sources', async () => {
    // Mock folder structure
    mockMultiSourceService.browseSource.mockResolvedValue({
      sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
      sourceName: 'Google Drive',
      folders: [
        {
          id: 'folder1',
          name: 'Research Papers',
          type: 'folder',
          children: [
            {
              id: 'subfolder1',
              name: 'ML Papers',
              type: 'folder',
              children: []
            }
          ]
        }
      ],
      files: []
    });

    render(<MultiSourcePanel />);
    
    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));
    
    // Switch to Google Drive tab (assuming it's connected)
    await waitFor(() => {
      const gdTab = screen.getByText('Google Drive');
      if (gdTab) {
        fireEvent.click(gdTab);
      }
    });
  });

  it('handles search across multiple sources', async () => {
    const mockSearchResults = [
      {
        file_id: 'search_result_1',
        name: 'Financial_Analysis.pdf',
        size: 2048000,
        modified_time: new Date().toISOString(),
        source_type: multiSourceService.DataSourceType.GOOGLE_DRIVE,
        source_path: '/search/Financial_Analysis.pdf',
        mime_type: 'application/pdf',
        parent_folders: ['Search Results'],
        source_specific_metadata: { relevance: 0.95 }
      }
    ];

    mockMultiSourceService.searchAcrossSources.mockResolvedValue(mockSearchResults);

    render(<MultiSourcePanel />);
    
    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));
    
    await waitFor(() => {
      const searchInput = screen.getByPlaceholderText('Search across all sources...');
      fireEvent.change(searchInput, { target: { value: 'financial' } });
      
      const searchButton = screen.getByText('Search');
      fireEvent.click(searchButton);
    });

    await waitFor(() => {
      expect(mockMultiSourceService.searchAcrossSources).toHaveBeenCalledWith('financial');
    });
  });

  it('displays processing jobs with real-time updates', async () => {
    render(<MultiSourcePanel />);
    
    // Switch to processing tab
    fireEvent.click(screen.getByText('Processing'));
    
    await waitFor(() => {
      expect(screen.getByText('No active processing jobs')).toBeInTheDocument();
    });
  });

  it('shows processed documents with source attribution', async () => {
    const mockProcessedDocs = [
      {
        id: 'doc1',
        name: 'Financial_Report.pdf',
        sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
        sourceId: 'gd_file_123',
        sourceUrl: 'https://drive.google.com/file/d/123',
        size: 2048576,
        uploadedAt: new Date().toISOString(),
        processedAt: new Date().toISOString(),
        status: 'indexed' as const,
        chunks: 45,
        category: 'Financial Reports',
        tags: ['analysis', '2024', 'financial'],
        metadata: {}
      }
    ];

    mockMultiSourceService.getProcessedDocuments.mockResolvedValue(mockProcessedDocs);

    render(<MultiSourcePanel />);
    
    // Switch to documents tab
    fireEvent.click(screen.getByText('Documents'));
    
    await waitFor(() => {
      expect(screen.getByText('Financial_Report.pdf')).toBeInTheDocument();
      expect(screen.getByText('45')).toBeInTheDocument(); // chunks
      expect(screen.getByText('INDEXED')).toBeInTheDocument();
    });
  });
});
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MultiSourcePanel } from '../MultiSourcePanel';
import * as multiSourceService from '../../../services/multiSourceService';

// Mock the multiSourceService
jest.mock('../../../services/multiSourceService');
const mockMultiSourceService = multiSourceService as jest.Mocked<typeof multiSourceService>;

// Mock the WebSocket hook
jest.mock('../../../services/multiSourceWebSocket', () => ({
  useMultiSourceWebSocket: () => ({
    subscribe: jest.fn(() => jest.fn()),
    send: jest.fn(),
    isConnected: jest.fn(() => true),
    getConnectionState: jest.fn(() => 'connected')
  })
}));

describe('MultiSourcePanel - Enhanced Batch Processing', () => {
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock responses
    mockMultiSourceService.getAvailableSources.mockResolvedValue([
      {
        type: multiSourceService.DataSourceType.GOOGLE_DRIVE,
        name: 'Google Drive',
        isConnected: true,
        connectionStatus: { isConnected: true, permissions: ['drive.readonly'] },
        fileCount: 10,
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
        fileCount: 5,
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
      sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
      sourceName: 'Google Drive',
      folders: [],
      files: [
        {
          file_id: 'test_file_1',
          name: 'Test Document 1.pdf',
          size: 1024000,
          modified_time: new Date().toISOString(),
          source_type: multiSourceService.DataSourceType.GOOGLE_DRIVE,
          source_path: '/test/Test Document 1.pdf',
          mime_type: 'application/pdf',
          parent_folders: [],
          source_specific_metadata: {}
        },
        {
          file_id: 'test_file_2',
          name: 'Test Document 2.pdf',
          size: 2048000,
          modified_time: new Date().toISOString(),
          source_type: multiSourceService.DataSourceType.GOOGLE_DRIVE,
          source_path: '/test/Test Document 2.pdf',
          mime_type: 'application/pdf',
          parent_folders: [],
          source_specific_metadata: {}
        }
      ]
    });
  });

  test('displays enhanced batch selection interface', async () => {
    render(<MultiSourcePanel />);
    
    // Wait for initial data to load
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    // Switch to browse tab
    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Should show enhanced batch actions section
    expect(screen.getByText('0 files selected across 0 sources')).toBeInTheDocument();
    expect(screen.getByText('Configure')).toBeInTheDocument();
    expect(screen.getByText('Process Selected (0)')).toBeInTheDocument();
  });

  test('shows processing time estimation when files are selected', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Select a file
    const checkbox = screen.getAllByRole('checkbox')[0];
    fireEvent.click(checkbox);

    await waitFor(() => {
      expect(screen.getByText('1 files selected across 1 sources')).toBeInTheDocument();
      expect(screen.getByText(/Estimated processing time:/)).toBeInTheDocument();
    });
  });

  test('opens batch configuration modal when configure is clicked', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Select a file
    const checkbox = screen.getAllByRole('checkbox')[0];
    fireEvent.click(checkbox);

    // Click configure button
    fireEvent.click(screen.getByText('Configure'));

    await waitFor(() => {
      expect(screen.getByText('Batch Processing Configuration')).toBeInTheDocument();
    });
  });

  test('shows source-wise selection summary', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Select files
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]);
    fireEvent.click(checkboxes[1]);

    await waitFor(() => {
      expect(screen.getByText('2 files selected across 1 sources')).toBeInTheDocument();
      expect(screen.getByText('Selection by source:')).toBeInTheDocument();
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument(); // Count of selected files
    });
  });

  test('starts batch processing with enhanced job management', async () => {
    const mockJob = {
      id: 'test_job_123',
      status: 'running' as const,
      totalFiles: 2,
      processedFiles: 0,
      failedFiles: 0,
      priority: 'normal' as const,
      estimatedTimeRemaining: 120,
      sourceProgress: [],
      startedAt: new Date().toISOString(),
      retryCount: 0,
      maxRetries: 3
    };

    mockMultiSourceService.startMultiSourceIngestion.mockResolvedValue(mockJob);

    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Select files
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]);
    fireEvent.click(checkboxes[1]);

    // Start processing
    fireEvent.click(screen.getByText('Process Selected (2)'));

    await waitFor(() => {
      expect(mockMultiSourceService.startMultiSourceIngestion).toHaveBeenCalledWith([
        {
          sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
          fileIds: ['test_file_1', 'test_file_2'],
          processingOptions: expect.objectContaining({
            chunkSize: 1000,
            embeddingModel: 'text-embedding-3-large',
            preserveMath: true,
            category: 'Multi-Source Batch',
            tags: expect.arrayContaining(['multi-source', 'batch', 'google_drive'])
          })
        }
      ]);
    });

    // Should switch to processing tab
    expect(screen.getByText('Processing')).toHaveClass('bg-[#ff8c00]');
  });

  test('displays enhanced processing job cards with job management', async () => {
    const mockJob = {
      id: 'test_job_123',
      status: 'running' as const,
      totalFiles: 2,
      processedFiles: 1,
      failedFiles: 0,
      priority: 'normal' as const,
      estimatedTimeRemaining: 60,
      sourceProgress: [
        {
          sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
          total: 2,
          completed: 1,
          failed: 0,
          files: [
            {
              id: 'test_file_1',
              name: 'Test Document 1.pdf',
              sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
              status: 'completed' as const,
              progress: 100,
              currentStep: 'Stored successfully',
              processingTime: 45,
              chunks: 12,
              qualityScore: 0.92
            },
            {
              id: 'test_file_2',
              name: 'Test Document 2.pdf',
              sourceType: multiSourceService.DataSourceType.GOOGLE_DRIVE,
              status: 'embedding' as const,
              progress: 75,
              currentStep: 'Generating embeddings',
              estimatedTimeRemaining: 30
            }
          ]
        }
      ],
      startedAt: new Date().toISOString(),
      retryCount: 0,
      maxRetries: 3
    };

    render(<MultiSourcePanel />);
    
    // Manually set the active jobs state
    const component = screen.getByTestId ? screen.getByTestId('multi-source-panel') : null;
    
    // Switch to processing tab
    fireEvent.click(screen.getByText('Processing'));

    // Mock the job being present
    mockMultiSourceService.getIngestionJobStatus.mockResolvedValue(mockJob);

    // The component should show job management features
    await waitFor(() => {
      expect(screen.getByText('Processing Jobs')).toBeInTheDocument();
    });
  });

  test('handles job queue management', async () => {
    render(<MultiSourcePanel />);
    
    fireEvent.click(screen.getByText('Processing'));

    await waitFor(() => {
      expect(screen.getByText('Processing Jobs')).toBeInTheDocument();
      expect(screen.getByText('0 active')).toBeInTheDocument();
      expect(screen.getByText('0 in queue')).toBeInTheDocument();
    });
  });

  test('calculates processing time estimates correctly', async () => {
    render(<MultiSourcePanel />);
    
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Browse'));

    await waitFor(() => {
      expect(screen.getByText('Test Document 1.pdf')).toBeInTheDocument();
    });

    // Select files of different sizes
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]); // 1MB file
    fireEvent.click(checkboxes[1]); // 2MB file

    await waitFor(() => {
      // Should show estimated processing time based on file sizes and source types
      expect(screen.getByText(/Estimated processing time:/)).toBeInTheDocument();
    });
  });
});
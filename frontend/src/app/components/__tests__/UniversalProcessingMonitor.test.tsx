import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { UniversalProcessingMonitor } from '../UniversalProcessingMonitor';
import { MultiSourceIngestionJob, DataSourceType } from '../../../services/multiSourceService';

// Mock the WebSocket hook
jest.mock('../../../services/multiSourceWebSocket', () => ({
  useMultiSourceWebSocket: () => ({
    subscribe: jest.fn(() => jest.fn()),
    subscribeToJob: jest.fn(() => jest.fn()),
    subscribeToSource: jest.fn(() => jest.fn()),
    cancelJob: jest.fn(),
    pauseJob: jest.fn(),
    resumeJob: jest.fn(),
    retryFile: jest.fn(),
    isConnected: jest.fn(() => true),
    getConnectionState: jest.fn(() => 'connected'),
    getConnectionQuality: jest.fn(() => 'excellent'),
    getQueuedMessageCount: jest.fn(() => 0)
  })
}));

describe('UniversalProcessingMonitor', () => {
  const mockJobs: MultiSourceIngestionJob[] = [
    {
      id: 'job-123',
      status: 'running',
      totalFiles: 5,
      processedFiles: 2,
      failedFiles: 0,
      priority: 'normal',
      estimatedTimeRemaining: 300,
      sourceProgress: [
        {
          sourceType: DataSourceType.GOOGLE_DRIVE,
          total: 3,
          completed: 1,
          failed: 0,
          files: [
            {
              id: 'file-1',
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
              id: 'file-2',
              name: 'document2.pdf',
              sourceType: DataSourceType.GOOGLE_DRIVE,
              status: 'embedding',
              progress: 75,
              currentStep: 'Generating embeddings',
              estimatedTimeRemaining: 30
            },
            {
              id: 'file-3',
              name: 'document3.pdf',
              sourceType: DataSourceType.GOOGLE_DRIVE,
              status: 'pending',
              progress: 0,
              currentStep: 'Queued for processing',
              estimatedTimeRemaining: 90
            }
          ]
        },
        {
          sourceType: DataSourceType.LOCAL_ZIP,
          total: 2,
          completed: 1,
          failed: 0,
          files: [
            {
              id: 'file-4',
              name: 'archive_doc1.pdf',
              sourceType: DataSourceType.LOCAL_ZIP,
              status: 'completed',
              progress: 100,
              currentStep: 'Stored successfully',
              processingTime: 32,
              chunks: 8,
              qualityScore: 0.88
            },
            {
              id: 'file-5',
              name: 'archive_doc2.pdf',
              sourceType: DataSourceType.LOCAL_ZIP,
              status: 'parsing',
              progress: 25,
              currentStep: 'Parsing PDF content',
              estimatedTimeRemaining: 60
            }
          ]
        }
      ],
      startedAt: new Date(Date.now() - 300000).toISOString(),
      retryCount: 0,
      maxRetries: 3
    }
  ];

  const mockProps = {
    jobs: mockJobs,
    onJobCancel: jest.fn(),
    onJobPause: jest.fn(),
    onJobResume: jest.fn(),
    onJobRetry: jest.fn(),
    onFileRetry: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders empty state when no jobs are provided', () => {
    render(<UniversalProcessingMonitor {...mockProps} jobs={[]} />);
    
    expect(screen.getByText('No Active Processing Jobs')).toBeInTheDocument();
    expect(screen.getByText('Select files and start batch processing to monitor progress here')).toBeInTheDocument();
  });

  it('displays connection status correctly', () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    expect(screen.getByText('Real-time Monitor')).toBeInTheDocument();
    expect(screen.getByText('Connected')).toBeInTheDocument();
  });

  it('renders job cards with correct information', () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Check job header information
    expect(screen.getByText('Multi-Source Job job-123')).toBeInTheDocument();
    expect(screen.getByText('2 sources • 5 files')).toBeInTheDocument();
    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    
    // Check progress information
    expect(screen.getByText('2 / 5 files processed')).toBeInTheDocument();
    expect(screen.getByText('40%')).toBeInTheDocument();
  });

  it('expands and collapses job details', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Initially collapsed
    expect(screen.queryByText('Google Drive')).not.toBeInTheDocument();
    
    // Click to expand
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    // Should show source details
    await waitFor(() => {
      expect(screen.getByText('Google Drive')).toBeInTheDocument();
      expect(screen.getByText('ZIP Archives')).toBeInTheDocument();
    });
  });

  it('displays source-wise progress correctly', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Expand job details
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    await waitFor(() => {
      // Check Google Drive source progress
      expect(screen.getByText('1 / 3')).toBeInTheDocument();
      
      // Check ZIP Archives source progress
      expect(screen.getByText('1 / 2')).toBeInTheDocument();
    });
  });

  it('shows individual file progress when source is expanded', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Expand job details
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    await waitFor(() => {
      // Click on Google Drive source to expand
      const googleDriveSection = screen.getByText('Google Drive');
      fireEvent.click(googleDriveSection);
    });
    
    await waitFor(() => {
      // Should show individual files
      expect(screen.getByText('document1.pdf')).toBeInTheDocument();
      expect(screen.getByText('document2.pdf')).toBeInTheDocument();
      expect(screen.getByText('document3.pdf')).toBeInTheDocument();
    });
  });

  it('handles job control actions', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Find pause button
    const pauseButton = screen.getByRole('button', { name: /pause job/i });
    fireEvent.click(pauseButton);
    
    expect(mockProps.onJobPause).toHaveBeenCalledWith('job-123');
  });

  it('displays detailed phase progress when enabled', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Expand job and source details
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    await waitFor(() => {
      const googleDriveSection = screen.getByText('Google Drive');
      fireEvent.click(googleDriveSection);
    });
    
    // Should show processing phases for files
    await waitFor(() => {
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('EMBEDDING')).toBeInTheDocument();
    });
  });

  it('shows error information for failed files', () => {
    const jobsWithError: MultiSourceIngestionJob[] = [
      {
        ...mockJobs[0],
        failedFiles: 1,
        sourceProgress: [
          {
            ...mockJobs[0].sourceProgress[0],
            failed: 1,
            files: [
              ...mockJobs[0].sourceProgress[0].files,
              {
                id: 'file-error',
                name: 'error_document.pdf',
                sourceType: DataSourceType.GOOGLE_DRIVE,
                status: 'failed',
                progress: 50,
                currentStep: 'Failed during parsing',
                error: 'Corrupted PDF file',
                retryCount: 1
              }
            ]
          }
        ]
      }
    ];

    render(<UniversalProcessingMonitor {...mockProps} jobs={jobsWithError} />);
    
    // Should show error indicator
    expect(screen.getByText('(1 failed)')).toBeInTheDocument();
  });

  it('handles file retry actions', async () => {
    const jobsWithError: MultiSourceIngestionJob[] = [
      {
        ...mockJobs[0],
        sourceProgress: [
          {
            ...mockJobs[0].sourceProgress[0],
            files: [
              {
                id: 'file-error',
                name: 'error_document.pdf',
                sourceType: DataSourceType.GOOGLE_DRIVE,
                status: 'failed',
                progress: 50,
                currentStep: 'Failed during parsing',
                error: 'Corrupted PDF file',
                retryCount: 1
              }
            ]
          }
        ]
      }
    ];

    render(<UniversalProcessingMonitor {...mockProps} jobs={jobsWithError} />);
    
    // Expand to show file details
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    await waitFor(() => {
      const googleDriveSection = screen.getByText('Google Drive');
      fireEvent.click(googleDriveSection);
    });
    
    // Find and click retry button
    await waitFor(() => {
      const retryButton = screen.getByRole('button', { name: /retry file processing/i });
      fireEvent.click(retryButton);
    });
    
    expect(mockProps.onFileRetry).toHaveBeenCalledWith('job-123', 'file-error');
  });

  it('displays processing statistics for completed files', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Expand job and source details
    const expandButton = screen.getByRole('button', { name: /toggle details/i });
    fireEvent.click(expandButton);
    
    await waitFor(() => {
      const googleDriveSection = screen.getByText('Google Drive');
      fireEvent.click(googleDriveSection);
    });
    
    // Should show processing statistics
    await waitFor(() => {
      expect(screen.getByText('12')).toBeInTheDocument(); // chunks
      expect(screen.getByText('92%')).toBeInTheDocument(); // quality score
    });
  });

  it('toggles detailed progress view', () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Find the toggle button for detailed progress
    const toggleButton = screen.getByRole('button', { name: /toggle detailed progress/i });
    fireEvent.click(toggleButton);
    
    // The detailed progress should be toggled
    // This would affect the display of phase-level progress
  });

  it('handles WebSocket connection quality indicators', () => {
    // Mock different connection qualities
    const { useMultiSourceWebSocket } = require('../../../services/multiSourceWebSocket');
    useMultiSourceWebSocket.mockReturnValue({
      ...useMultiSourceWebSocket(),
      getConnectionQuality: jest.fn(() => 'poor'),
      isConnected: jest.fn(() => true)
    });

    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Should show connection status
    expect(screen.getByText('Connected')).toBeInTheDocument();
  });

  it('formats time remaining correctly', () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    // Should show estimated time remaining
    expect(screen.getByText(/~5m remaining/)).toBeInTheDocument();
  });

  it('handles job cancellation', async () => {
    render(<UniversalProcessingMonitor {...mockProps} />);
    
    const cancelButton = screen.getByRole('button', { name: /cancel job/i });
    fireEvent.click(cancelButton);
    
    expect(mockProps.onJobCancel).toHaveBeenCalledWith('job-123');
  });

  it('displays priority indicators for high priority jobs', () => {
    const highPriorityJobs: MultiSourceIngestionJob[] = [
      {
        ...mockJobs[0],
        priority: 'high'
      }
    ];

    render(<UniversalProcessingMonitor {...mockProps} jobs={highPriorityJobs} />);
    
    expect(screen.getByText('HIGH')).toBeInTheDocument();
  });

  it('shows paused job status and resume option', () => {
    const pausedJobs: MultiSourceIngestionJob[] = [
      {
        ...mockJobs[0],
        status: 'paused'
      }
    ];

    render(<UniversalProcessingMonitor {...mockProps} jobs={pausedJobs} />);
    
    expect(screen.getByText('PAUSED')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /resume job/i })).toBeInTheDocument();
  });
});
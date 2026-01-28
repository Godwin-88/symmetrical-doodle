/**
 * Enhanced WebSocket service for real-time multi-source processing updates
 * Supports detailed progress monitoring across all processing phases
 */

export interface ProcessingUpdate {
  jobId: string;
  type: 'job_status' | 'file_progress' | 'phase_progress' | 'source_progress' | 'error' | 'completion' | 'cancellation' | 'retry';
  data: any;
  timestamp: string;
  sourceType?: string;
  fileId?: string;
  phase?: ProcessingPhase;
}

export interface ProcessingPhase {
  name: 'downloading' | 'parsing' | 'chunking' | 'embedding' | 'storing';
  displayName: string;
  progress: number;
  status: 'pending' | 'active' | 'completed' | 'failed';
  startTime?: string;
  endTime?: string;
  error?: string;
  metrics?: {
    bytesProcessed?: number;
    chunksCreated?: number;
    embeddingsGenerated?: number;
    qualityScore?: number;
  };
}

export interface DetailedFileProgress {
  fileId: string;
  fileName: string;
  sourceType: string;
  overallProgress: number;
  currentPhase: ProcessingPhase;
  phases: ProcessingPhase[];
  estimatedTimeRemaining: number;
  processingStartTime: string;
  error?: {
    phase: string;
    message: string;
    retryable: boolean;
    retryCount: number;
    maxRetries: number;
  };
}

export interface SourceProgressUpdate {
  sourceType: string;
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  activeFiles: number;
  overallProgress: number;
  currentPhaseDistribution: Record<string, number>;
  averageProcessingTime: number;
  estimatedTimeRemaining: number;
  errors: Array<{
    fileId: string;
    fileName: string;
    phase: string;
    message: string;
    retryable: boolean;
  }>;
}

export type ProcessingUpdateHandler = (update: ProcessingUpdate) => void;

class MultiSourceWebSocketService {
  private ws: WebSocket | null = null;
  private handlers: Set<ProcessingUpdateHandler> = new Set();
  private jobHandlers: Map<string, Set<ProcessingUpdateHandler>> = new Map();
  private sourceHandlers: Map<string, Set<ProcessingUpdateHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected' = 'disconnected';
  private messageQueue: any[] = [];

  constructor() {
    this.connect();
  }

  private connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isConnecting = true;

    try {
      // Use secure WebSocket in production, regular WebSocket in development
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/multi-source`;
      
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('Multi-source WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.connectionQuality = 'excellent';
        this.startHeartbeat();
        this.flushMessageQueue();
      };

      this.ws.onmessage = (event) => {
        try {
          const update: ProcessingUpdate = JSON.parse(event.data);
          
          // Handle heartbeat responses
          if (update.type === 'heartbeat') {
            this.updateConnectionQuality(Date.now() - parseInt(update.data.timestamp));
            return;
          }
          
          this.notifyHandlers(update);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('Multi-source WebSocket disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.ws = null;
        this.connectionQuality = 'disconnected';
        this.stopHeartbeat();

        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('Multi-source WebSocket error:', error);
        this.isConnecting = false;
        this.connectionQuality = 'poor';
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'heartbeat', timestamp: Date.now().toString() });
      }
    }, 30000); // Send heartbeat every 30 seconds
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private updateConnectionQuality(latency: number) {
    if (latency < 100) {
      this.connectionQuality = 'excellent';
    } else if (latency < 500) {
      this.connectionQuality = 'good';
    } else {
      this.connectionQuality = 'poor';
    }
  }

  private flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  private scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private notifyHandlers(update: ProcessingUpdate) {
    // Notify global handlers
    this.handlers.forEach(handler => {
      try {
        handler(update);
      } catch (error) {
        console.error('Error in processing update handler:', error);
      }
    });

    // Notify job-specific handlers
    if (update.jobId) {
      const jobHandlers = this.jobHandlers.get(update.jobId);
      if (jobHandlers) {
        jobHandlers.forEach(handler => {
          try {
            handler(update);
          } catch (error) {
            console.error('Error in job-specific handler:', error);
          }
        });
      }
    }

    // Notify source-specific handlers
    if (update.sourceType) {
      const sourceHandlers = this.sourceHandlers.get(update.sourceType);
      if (sourceHandlers) {
        sourceHandlers.forEach(handler => {
          try {
            handler(update);
          } catch (error) {
            console.error('Error in source-specific handler:', error);
          }
        });
      }
    }
  }

  public subscribe(handler: ProcessingUpdateHandler): () => void {
    this.handlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.handlers.delete(handler);
    };
  }

  public subscribeToJob(jobId: string, handler: ProcessingUpdateHandler): () => void {
    if (!this.jobHandlers.has(jobId)) {
      this.jobHandlers.set(jobId, new Set());
    }
    
    const jobHandlers = this.jobHandlers.get(jobId)!;
    jobHandlers.add(handler);

    // Return unsubscribe function
    return () => {
      jobHandlers.delete(handler);
      if (jobHandlers.size === 0) {
        this.jobHandlers.delete(jobId);
      }
    };
  }

  public subscribeToSource(sourceType: string, handler: ProcessingUpdateHandler): () => void {
    if (!this.sourceHandlers.has(sourceType)) {
      this.sourceHandlers.set(sourceType, new Set());
    }
    
    const sourceHandlers = this.sourceHandlers.get(sourceType)!;
    sourceHandlers.add(handler);

    // Return unsubscribe function
    return () => {
      sourceHandlers.delete(handler);
      if (sourceHandlers.size === 0) {
        this.sourceHandlers.delete(sourceType);
      }
    };
  }

  public send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for when connection is restored
      this.messageQueue.push(message);
      console.warn('WebSocket not connected, message queued:', message);
    }
  }

  public cancelJob(jobId: string): void {
    this.send({
      type: 'cancel_job',
      jobId,
      timestamp: new Date().toISOString()
    });
  }

  public pauseJob(jobId: string): void {
    this.send({
      type: 'pause_job',
      jobId,
      timestamp: new Date().toISOString()
    });
  }

  public resumeJob(jobId: string): void {
    this.send({
      type: 'resume_job',
      jobId,
      timestamp: new Date().toISOString()
    });
  }

  public retryFile(jobId: string, fileId: string): void {
    this.send({
      type: 'retry_file',
      jobId,
      fileId,
      timestamp: new Date().toISOString()
    });
  }

  public disconnect() {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.handlers.clear();
    this.jobHandlers.clear();
    this.sourceHandlers.clear();
    this.messageQueue = [];
  }

  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'unknown';
    }
  }

  public getConnectionQuality(): 'excellent' | 'good' | 'poor' | 'disconnected' {
    return this.connectionQuality;
  }

  public getQueuedMessageCount(): number {
    return this.messageQueue.length;
  }
}

// Create singleton instance
export const multiSourceWebSocket = new MultiSourceWebSocketService();

// Hook for React components
export function useMultiSourceWebSocket() {
  return {
    subscribe: multiSourceWebSocket.subscribe.bind(multiSourceWebSocket),
    subscribeToJob: multiSourceWebSocket.subscribeToJob.bind(multiSourceWebSocket),
    subscribeToSource: multiSourceWebSocket.subscribeToSource.bind(multiSourceWebSocket),
    send: multiSourceWebSocket.send.bind(multiSourceWebSocket),
    cancelJob: multiSourceWebSocket.cancelJob.bind(multiSourceWebSocket),
    pauseJob: multiSourceWebSocket.pauseJob.bind(multiSourceWebSocket),
    resumeJob: multiSourceWebSocket.resumeJob.bind(multiSourceWebSocket),
    retryFile: multiSourceWebSocket.retryFile.bind(multiSourceWebSocket),
    isConnected: multiSourceWebSocket.isConnected.bind(multiSourceWebSocket),
    getConnectionState: multiSourceWebSocket.getConnectionState.bind(multiSourceWebSocket),
    getConnectionQuality: multiSourceWebSocket.getConnectionQuality.bind(multiSourceWebSocket),
    getQueuedMessageCount: multiSourceWebSocket.getQueuedMessageCount.bind(multiSourceWebSocket)
  };
}

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    multiSourceWebSocket.disconnect();
  });
}
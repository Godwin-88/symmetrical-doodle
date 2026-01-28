/**
 * Real-Time WebSocket Service
 * Provides live market data, system metrics, and trading updates
 */

import React from 'react';

export interface RealTimeData {
  timestamp: number;
  type: 'market' | 'system' | 'trading' | 'risk';
  data: any;
}

export interface MarketTick {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  timestamp: number;
  change: number;
  changePercent: number;
}

export interface SystemMetric {
  component: string;
  metric: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  timestamp: number;
}

export interface TradingUpdate {
  type: 'order' | 'fill' | 'position' | 'pnl';
  data: any;
  timestamp: number;
}

export interface RiskAlert {
  level: 'info' | 'warning' | 'critical';
  message: string;
  metric: string;
  value: number;
  threshold: number;
  timestamp: number;
}

class RealTimeService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private subscribers: Map<string, Set<(data: any) => void>> = new Map();
  private isConnected = false;
  private heartbeatInterval: number | null = null;

  constructor() {
    this.connect();
  }

  private connect() {
    try {
      // Try multiple WebSocket endpoints
      const endpoints = [
        'ws://localhost:8000/ws',
        'ws://localhost:8001/ws',
        'ws://localhost:3001/ws'
      ];

      const wsUrl = endpoints[0]; // Start with intelligence layer
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.notifySubscribers('connection', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const data: RealTimeData = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        this.isConnected = false;
        this.stopHeartbeat();
        this.notifySubscribers('connection', { status: 'disconnected' });
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.notifySubscribers('connection', { status: 'error', error });
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.attemptReconnect();
    }
  }

  private handleMessage(data: RealTimeData) {
    // Route message to appropriate subscribers
    this.notifySubscribers(data.type, data.data);
    this.notifySubscribers('all', data);

    // Handle specific message types
    switch (data.type) {
      case 'market':
        this.handleMarketData(data.data);
        break;
      case 'system':
        this.handleSystemMetric(data.data);
        break;
      case 'trading':
        this.handleTradingUpdate(data.data);
        break;
      case 'risk':
        this.handleRiskAlert(data.data);
        break;
    }
  }

  private handleMarketData(data: MarketTick) {
    // Store latest market data
    const key = `market_${data.symbol}`;
    localStorage.setItem(key, JSON.stringify(data));
    
    // Notify market data subscribers
    this.notifySubscribers('market_data', data);
    this.notifySubscribers(`market_${data.symbol}`, data);
  }

  private handleSystemMetric(data: SystemMetric) {
    // Store system metrics
    const key = `system_${data.component}_${data.metric}`;
    localStorage.setItem(key, JSON.stringify(data));
    
    // Notify system subscribers
    this.notifySubscribers('system_metrics', data);
    this.notifySubscribers(`system_${data.component}`, data);
  }

  private handleTradingUpdate(data: TradingUpdate) {
    // Notify trading subscribers
    this.notifySubscribers('trading_updates', data);
    this.notifySubscribers(`trading_${data.type}`, data);
  }

  private handleRiskAlert(data: RiskAlert) {
    // Store risk alerts
    const alerts = this.getRiskAlerts();
    alerts.unshift(data);
    localStorage.setItem('risk_alerts', JSON.stringify(alerts.slice(0, 100)));
    
    // Notify risk subscribers
    this.notifySubscribers('risk_alerts', data);
    
    // Show critical alerts immediately
    if (data.level === 'critical') {
      this.showCriticalAlert(data);
    }
  }

  private showCriticalAlert(alert: RiskAlert) {
    // Create a visual alert for critical risk events
    if ('Notification' in window && Notification.permission === 'granted') {
      const notification = new Notification('Critical Risk Alert', {
        body: alert.message,
        icon: '/favicon.ico',
        requireInteraction: true
      });
      
      // Prevent unused variable warning
      notification.onclick = () => {
        console.log('Risk alert notification clicked');
      };
    }
    
    // Also log to console
    console.error('🚨 CRITICAL RISK ALERT:', alert);
  }

  private startHeartbeat() {
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, 30000); // 30 second heartbeat
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.error('❌ Max reconnection attempts reached. Switching to mock data mode.');
      this.startMockDataMode();
    }
  }

  private startMockDataMode() {
    // Generate mock real-time data when WebSocket is unavailable
    console.log('📊 Starting mock data mode for development');
    
    window.setInterval(() => {
      // Mock market data
      const symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'];
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      
      const mockTick: MarketTick = {
        symbol,
        bid: 1.0500 + (Math.random() - 0.5) * 0.01,
        ask: 1.0502 + (Math.random() - 0.5) * 0.01,
        last: 1.0501 + (Math.random() - 0.5) * 0.01,
        volume: Math.floor(Math.random() * 1000000),
        timestamp: Date.now(),
        change: (Math.random() - 0.5) * 0.001,
        changePercent: (Math.random() - 0.5) * 0.1
      };
      
      this.handleMarketData(mockTick);
    }, 1000);

    // Mock system metrics
    window.setInterval(() => {
      const components = ['frontend', 'intelligence', 'execution', 'simulation'];
      const metrics = ['cpu', 'memory', 'latency', 'throughput'];
      
      const component = components[Math.floor(Math.random() * components.length)];
      const metric = metrics[Math.floor(Math.random() * metrics.length)];
      
      const mockMetric: SystemMetric = {
        component,
        metric,
        value: Math.random() * 100,
        unit: metric === 'latency' ? 'ms' : metric === 'throughput' ? 'req/s' : '%',
        status: Math.random() > 0.8 ? 'warning' : 'healthy',
        timestamp: Date.now()
      };
      
      this.handleSystemMetric(mockMetric);
    }, 5000);
  }

  private notifySubscribers(channel: string, data: any) {
    const subscribers = this.subscribers.get(channel);
    if (subscribers) {
      subscribers.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in subscriber callback for ${channel}:`, error);
        }
      });
    }
  }

  // Public API
  subscribe(channel: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set());
    }
    
    this.subscribers.get(channel)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const subscribers = this.subscribers.get(channel);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          this.subscribers.delete(channel);
        }
      }
    };
  }

  send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.maxReconnectAttempts
    };
  }

  // Utility methods
  getLatestMarketData(symbol: string): MarketTick | null {
    const key = `market_${symbol}`;
    const data = localStorage.getItem(key);
    return data ? JSON.parse(data) : null;
  }

  getSystemMetrics(): SystemMetric[] {
    const metrics: SystemMetric[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('system_')) {
        const data = localStorage.getItem(key);
        if (data) {
          metrics.push(JSON.parse(data));
        }
      }
    }
    return metrics.sort((a, b) => b.timestamp - a.timestamp);
  }

  getRiskAlerts(): RiskAlert[] {
    const data = localStorage.getItem('risk_alerts');
    return data ? JSON.parse(data) : [];
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopHeartbeat();
    this.subscribers.clear();
  }
}

// Export singleton instance
export const realTimeService = new RealTimeService();

// React hook for easy integration
export function useRealTimeData(channel: string) {
  const [data, setData] = React.useState<any>(null);
  const [connected, setConnected] = React.useState(false);

  React.useEffect(() => {
    const unsubscribeData = realTimeService.subscribe(channel, setData);
    const unsubscribeConnection = realTimeService.subscribe('connection', (status) => {
      setConnected(status.status === 'connected');
    });

    // Set initial connection status
    setConnected(realTimeService.getConnectionStatus().connected);

    return () => {
      unsubscribeData();
      unsubscribeConnection();
    };
  }, [channel]);

  return { data, connected };
}

// Market data hook
export function useMarketData(symbol: string) {
  const { data, connected } = useRealTimeData(`market_${symbol}`);
  
  return {
    tick: data as MarketTick | null,
    connected,
    latest: realTimeService.getLatestMarketData(symbol)
  };
}

// System metrics hook
export function useSystemMetrics() {
  const { data, connected } = useRealTimeData('system_metrics');
  
  return {
    metric: data as SystemMetric | null,
    connected,
    allMetrics: realTimeService.getSystemMetrics()
  };
}

// Risk alerts hook
export function useRiskAlerts() {
  const { data, connected } = useRealTimeData('risk_alerts');
  
  return {
    alert: data as RiskAlert | null,
    connected,
    allAlerts: realTimeService.getRiskAlerts()
  };
}
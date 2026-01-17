import { useState } from 'react';
import MockDataDemo from './MockDataDemo';

export function System() {
  const [activeTab, setActiveTab] = useState<'overview' | 'demo'>('overview');

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  // Mock system data
  const components = [
    { name: 'EXECUTION CORE (RUST)', status: 'HEALTHY', uptime: 99.98, cpu: 12.5, memory: 245, latency: 2 },
    { name: 'INTELLIGENCE LAYER (PYTHON)', status: 'HEALTHY', uptime: 99.95, cpu: 34.2, memory: 1024, latency: 15 },
    { name: 'FRONTEND (REACT)', status: 'HEALTHY', uptime: 100.00, cpu: 5.3, memory: 128, latency: 1 },
    { name: 'POSTGRES + PGVECTOR', status: 'HEALTHY', uptime: 99.99, cpu: 8.7, memory: 512, latency: 3 },
    { name: 'NEO4J + GDS', status: 'HEALTHY', uptime: 99.97, cpu: 15.4, memory: 768, latency: 5 },
    { name: 'REDIS CACHE', status: 'HEALTHY', uptime: 100.00, cpu: 2.1, memory: 64, latency: 1 },
  ];

  const systemMetrics = {
    totalCpu: 78.2,
    totalMemory: 2741,
    maxMemory: 8192,
    diskUsage: 45.3,
    networkIn: 125.5,
    networkOut: 89.3,
  };

  const logs = [
    { time: '14:30:45', level: 'INFO', component: 'EXEC-CORE', message: 'ORDER FILLED: EUR/USD BUY 50000 @ 1.0845' },
    { time: '14:30:42', level: 'INFO', component: 'INTELLIGENCE', message: 'REGIME INFERENCE COMPLETED: LOW_VOL_TRENDING (CONF: 0.87)' },
    { time: '14:30:38', level: 'WARN', component: 'EXEC-CORE', message: 'HIGH LATENCY DETECTED: MT5 ADAPTER (25ms)' },
    { time: '14:30:35', level: 'INFO', component: 'RISK-MGR', message: 'RISK CHECK PASSED: UTILIZATION 47.3%' },
    { time: '14:30:30', level: 'ERROR', component: 'EXEC-CORE', message: 'ORDER REJECTED: INSUFFICIENT MARGIN' },
  ];

  const configuration = {
    EXECUTION_MODE: 'SHADOW',
    RISK_LIMITS_ENABLED: 'TRUE',
    MAX_POSITION_SIZE: '100000',
    MAX_DRAWDOWN: '5.0%',
    DAILY_LOSS_LIMIT: '10000',
    KILL_SWITCH_ENABLED: 'TRUE',
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs overflow-hidden">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2">
        <div className="text-center text-[#ff8c00] text-sm tracking-wide">
          SYSTEM MANAGEMENT - HEALTH & CONFIGURATION
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-600">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'overview'
              ? 'bg-orange-900 text-orange-400 border-b-2 border-orange-400'
              : 'text-gray-400 hover:text-orange-400'
          }`}
        >
          System Overview
        </button>
        <button
          onClick={() => setActiveTab('demo')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'demo'
              ? 'bg-orange-900 text-orange-400 border-b-2 border-orange-400'
              : 'text-gray-400 hover:text-orange-400'
          }`}
        >
          Mock Data Demo
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'overview' ? (
          <SystemOverview 
            components={components}
            systemMetrics={systemMetrics}
            logs={logs}
            configuration={configuration}
            formatNumber={formatNumber}
          />
        ) : (
          <MockDataDemo />
        )}
      </div>
    </div>
  );
}

// System Overview Component
function SystemOverview({ 
  components, 
  systemMetrics, 
  logs, 
  configuration, 
  formatNumber 
}: {
  components: any[];
  systemMetrics: any;
  logs: any[];
  configuration: any;
  formatNumber: (num: number, decimals?: number) => string;
}) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="p-4 space-y-4">
        {/* Component Health */}
        <div className="border border-[#444]">
          <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
            <div className="text-[#ff8c00]">COMPONENT HEALTH</div>
          </div>
          <table className="w-full">
            <thead>
              <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                <th className="px-3 py-2 text-left border-b border-[#444]">COMPONENT</th>
                <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                <th className="px-3 py-2 text-right border-b border-[#444]">UPTIME</th>
                <th className="px-3 py-2 text-right border-b border-[#444]">CPU %</th>
                <th className="px-3 py-2 text-right border-b border-[#444]">MEMORY (MB)</th>
                <th className="px-3 py-2 text-right border-b border-[#444]">LATENCY (MS)</th>
              </tr>
            </thead>
            <tbody>
              {components.map((comp, idx) => (
                <tr key={idx} className="border-b border-[#222]">
                  <td className="px-3 py-2 text-[#00ff00]">{comp.name}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={comp.status === 'HEALTHY' ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {comp.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-[#00ff00]">{formatNumber(comp.uptime)}%</td>
                  <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(comp.cpu, 1)}%</td>
                  <td className="px-3 py-2 text-right text-[#fff]">{comp.memory}</td>
                  <td className="px-3 py-2 text-right text-[#00ff00]">{comp.latency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-3 gap-4">
          <div className="border border-[#444] p-3">
            <div className="text-[#ff8c00] mb-3">RESOURCE USAGE</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">TOTAL CPU</span>
                <span className="text-[#00ff00]">{formatNumber(systemMetrics.totalCpu, 1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">TOTAL MEMORY</span>
                <span className="text-[#00ff00]">
                  {systemMetrics.totalMemory}MB / {systemMetrics.maxMemory}MB
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DISK USAGE</span>
                <span className="text-[#00ff00]">{formatNumber(systemMetrics.diskUsage, 1)}%</span>
              </div>
            </div>
          </div>

          <div className="border border-[#444] p-3">
            <div className="text-[#ff8c00] mb-3">NETWORK</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">INBOUND</span>
                <span className="text-[#00ff00]">{formatNumber(systemMetrics.networkIn, 1)} MB/s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">OUTBOUND</span>
                <span className="text-[#00ff00]">{formatNumber(systemMetrics.networkOut, 1)} MB/s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">CONNECTIONS</span>
                <span className="text-[#00ff00]">24 ACTIVE</span>
              </div>
            </div>
          </div>

          <div className="border border-[#444] p-3">
            <div className="text-[#ff8c00] mb-3">SYSTEM STATUS</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">OVERALL</span>
                <span className="text-[#00ff00]">OPERATIONAL</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">UPTIME</span>
                <span className="text-[#00ff00]">7D 12H 34M</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">LAST RESTART</span>
                <span className="text-[#fff]">2024-01-08</span>
              </div>
            </div>
          </div>
        </div>

        {/* System Configuration */}
        <div className="border border-[#444]">
          <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
            <div className="text-[#ff8c00]">SYSTEM CONFIGURATION</div>
          </div>
          <div className="p-3">
            <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-[10px]">
              {Object.entries(configuration).map(([key, value], idx) => (
                <div key={idx} className="flex justify-between">
                  <span className="text-[#666]">{key}</span>
                  <span className={`
                    ${String(value) === 'TRUE' || String(value) === 'SHADOW' ? 'text-[#00ff00]' : ''}
                    ${String(value) === 'FALSE' ? 'text-[#ff0000]' : ''}
                    ${!['TRUE', 'FALSE', 'SHADOW'].includes(String(value)) ? 'text-[#fff]' : ''}
                  `}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* System Logs */}
        <div className="border border-[#444]">
          <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
            <div className="text-[#ff8c00]">SYSTEM LOGS (LAST 5 ENTRIES)</div>
          </div>
          <table className="w-full">
            <thead>
              <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                <th className="px-3 py-2 text-left border-b border-[#444]">TIME</th>
                <th className="px-3 py-2 text-center border-b border-[#444]">LEVEL</th>
                <th className="px-3 py-2 text-left border-b border-[#444]">COMPONENT</th>
                <th className="px-3 py-2 text-left border-b border-[#444]">MESSAGE</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, idx) => (
                <tr key={idx} className="border-b border-[#222]">
                  <td className="px-3 py-2 text-[#666]">{log.time}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`
                      ${log.level === 'INFO' ? 'text-[#00ff00]' : ''}
                      ${log.level === 'WARN' ? 'text-[#ffff00]' : ''}
                      ${log.level === 'ERROR' ? 'text-[#ff0000]' : ''}
                    `}>
                      {log.level}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-[#00ff00]">{log.component}</td>
                  <td className="px-3 py-2 text-[#fff]">{log.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* System Actions */}
        <div className="grid grid-cols-4 gap-2">
          <button className="py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
            RESTART SERVICES
          </button>
          <button className="py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors">
            VIEW FULL LOGS
          </button>
          <button className="py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors">
            EDIT CONFIG
          </button>
          <button className="py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors">
            EMERGENCY HALT
          </button>
        </div>

        {/* System Alerts */}
        <div className="border border-[#444] p-3">
          <div className="text-[#ff8c00] mb-3">SYSTEM ALERTS</div>
          <div className="space-y-1 text-[10px]">
            <div className="text-[#00ff00]">ALL SYSTEMS OPERATIONAL</div>
            <div className="text-[#666]">NO CRITICAL ALERTS</div>
            <div className="text-[#666]">LAST HEALTH CHECK: 14:30:45 UTC</div>
          </div>
        </div>
      </div>
    </div>
  );
}
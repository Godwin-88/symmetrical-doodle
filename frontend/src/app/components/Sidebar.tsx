import React from 'react';
import { 
  BarChart3, 
  Database, 
  TrendingUp, 
  Brain, 
  Target, 
  Play, 
  Shield, 
  Zap, 
  Activity,
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useTradingStore, type Domain } from '../store/tradingStore';

interface SidebarItem {
  key: Domain;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const sidebarItems: SidebarItem[] = [
  {
    key: 'DASH',
    label: 'Dashboard',
    icon: <BarChart3 className="w-5 h-5" />,
    description: 'System overview and real-time monitoring'
  },
  {
    key: 'WORK',
    label: 'Data Workspace',
    icon: <Database className="w-5 h-5" />,
    description: 'Advanced analytics and data exploration'
  },
  {
    key: 'MLOPS',
    label: 'MLOps',
    icon: <Brain className="w-5 h-5" />,
    description: 'ML model operations and registry'
  },
  {
    key: 'MKTS',
    label: 'Markets',
    icon: <TrendingUp className="w-5 h-5" />,
    description: 'Live market data and analysis'
  },
  {
    key: 'INTL',
    label: 'Intelligence',
    icon: <Zap className="w-5 h-5" />,
    description: 'Intelligence layer and regime detection'
  },
  {
    key: 'STRT',
    label: 'Strategies',
    icon: <Target className="w-5 h-5" />,
    description: 'Strategy catalog and management'
  },
  {
    key: 'SIMU',
    label: 'Simulation',
    icon: <Play className="w-5 h-5" />,
    description: 'Backtesting and experiments'
  },
  {
    key: 'PORT',
    label: 'Portfolio',
    icon: <Shield className="w-5 h-5" />,
    description: 'Risk management and positions'
  },
  {
    key: 'EXEC',
    label: 'Execution',
    icon: <Activity className="w-5 h-5" />,
    description: 'Order management and execution'
  },
  {
    key: 'SYST',
    label: 'System',
    icon: <Settings className="w-5 h-5" />,
    description: 'System health and configuration'
  },
];

const Sidebar: React.FC = () => {
  const { 
    currentDomain, 
    setCurrentDomain, 
    systemStatus, 
    emergencyControls,
    sidebarCollapsed,
    toggleSidebar
  } = useTradingStore();

  const getItemStatus = (key: Domain) => {
    // Add status indicators based on domain
    switch (key) {
      case 'SYST':
        return systemStatus === 'OPERATIONAL' ? 'healthy' : 'warning';
      case 'EXEC':
        return emergencyControls.systemStatus === 'ACTIVE' ? 'healthy' : 'warning';
      case 'INTL':
        return 'healthy'; // Could be based on model status
      default:
        return 'healthy';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-400';
      case 'warning': return 'bg-yellow-400';
      case 'error': return 'bg-red-400';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className={`${sidebarCollapsed ? 'w-16' : 'w-64'} bg-black border-r border-gray-600 flex flex-col transition-all duration-300 ease-in-out`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-600 flex items-center justify-between">
        {!sidebarCollapsed && (
          <div>
            <h2 className="text-orange-400 font-mono text-sm font-bold">
              NAVIGATION
            </h2>
            <p className="text-gray-500 text-xs mt-1">
              Research-Grade Trading Platform
            </p>
          </div>
        )}
        
        {/* Collapse Toggle */}
        <button
          onClick={toggleSidebar}
          className="p-1 text-gray-400 hover:text-orange-400 transition-colors"
          title={sidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-2 space-y-1">
          {sidebarItems.map((item, index) => {
            const isActive = currentDomain === item.key;
            const status = getItemStatus(item.key);
            
            return (
              <button
                key={item.key}
                onClick={() => setCurrentDomain(item.key)}
                className={`w-full text-left p-3 rounded-none border transition-all duration-200 group ${
                  isActive
                    ? 'bg-orange-900 border-orange-600 text-orange-400'
                    : 'border-gray-700 text-gray-300 hover:bg-gray-900 hover:border-gray-600 hover:text-orange-400'
                }`}
                title={sidebarCollapsed ? `${item.label} - ${item.description}` : undefined}
              >
                <div className="flex items-center justify-between">
                  <div className={`flex items-center ${sidebarCollapsed ? 'justify-center w-full' : 'space-x-3'}`}>
                    {/* Function Key */}
                    {!sidebarCollapsed && (
                      <div className={`w-8 h-6 flex items-center justify-center text-xs font-mono border ${
                        isActive 
                          ? 'border-orange-600 text-orange-400' 
                          : 'border-gray-600 text-gray-500 group-hover:border-gray-500 group-hover:text-orange-400'
                      }`}>
                        F{index + 1}
                      </div>
                    )}

                    {/* Icon */}
                    <div className={`${isActive ? 'text-orange-400' : 'text-gray-400 group-hover:text-orange-400'}`}>
                      {item.icon}
                    </div>

                    {/* Label and Description */}
                    {!sidebarCollapsed && (
                      <div>
                        <div className={`font-mono text-sm font-medium ${
                          isActive ? 'text-orange-400' : 'text-gray-300 group-hover:text-orange-400'
                        }`}>
                          {item.label}
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5 leading-tight">
                          {item.description}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Status Indicator */}
                  {!sidebarCollapsed && (
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(status)}`}></div>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      {!sidebarCollapsed && (
        <div className="p-4 border-t border-gray-600">
          <div className="text-xs text-gray-500 font-mono">
            <div className="flex justify-between items-center mb-2">
              <span>System Status</span>
              <span className={`px-2 py-1 rounded text-xs ${
                systemStatus === 'OPERATIONAL' 
                  ? 'bg-green-900 text-green-400' 
                  : systemStatus === 'DEGRADED'
                  ? 'bg-yellow-900 text-yellow-400'
                  : 'bg-red-900 text-red-400'
              }`}>
                {systemStatus}
              </span>
            </div>
            
            <div className="flex justify-between items-center mb-2">
              <span>Trading Status</span>
              <span className={`px-2 py-1 rounded text-xs ${
                emergencyControls.systemStatus === 'ACTIVE' 
                  ? 'bg-green-900 text-green-400' 
                  : emergencyControls.systemStatus === 'PAUSED'
                  ? 'bg-yellow-900 text-yellow-400'
                  : 'bg-red-900 text-red-400'
              }`}>
                {emergencyControls.systemStatus}
              </span>
            </div>

            <div className="text-center pt-2 border-t border-gray-700">
              <div className="text-gray-600">v1.0.0</div>
              <div className="text-gray-600">Academic Research</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
import { useEffect } from 'react';
import { useTradingStore } from './store/tradingStore';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import QuickActionModals from './components/QuickActionModals';
import { Dashboard } from './components/Dashboard';
import { Intelligence } from './components/Intelligence';
import { IntelligenceNew } from './components/IntelligenceNew';
import { Markets } from './components/Markets';
import { Strategies } from './components/Strategies';
import { Portfolio } from './components/Portfolio';
import { Execution } from './components/Execution';
import { Simulation } from './components/Simulation';
import { MLOps } from './components/MLOps';
import { System } from './components/System';
import { DataWorkspace } from './components/DataWorkspace';

export default function App() {
  const { currentDomain, setCurrentDomain, checkHealth } = useTradingStore();

  // Health check on mount and periodically
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Updated keyboard shortcuts for new navigation order (Option 1C)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key >= 'F1' && e.key <= 'F10') {
        e.preventDefault();
        // New Option 1C sequence: DASH, WORK, MLOPS, MKTS, INTL, STRT, SIMU, PORT, EXEC, SYST
        const domains = ['DASH', 'WORK', 'MLOPS', 'MKTS', 'INTL', 'STRT', 'SIMU', 'PORT', 'EXEC', 'SYST'];
        const index = parseInt(e.key.slice(1)) - 1;
        if (domains[index]) {
          setCurrentDomain(domains[index] as any);
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [setCurrentDomain]);

  const renderDomain = () => {
    switch (currentDomain) {
      case 'DASH':
        return <Dashboard />;
      case 'WORK':
        return <DataWorkspace />;
      case 'MLOPS':
        return <MLOps />;
      case 'MKTS':
        return <Markets />;
      case 'INTL':
        return <IntelligenceNew />;
      case 'STRT':
        return <Strategies />;
      case 'SIMU':
        return <Simulation />;
      case 'PORT':
        return <Portfolio />;
      case 'EXEC':
        return <Execution />;
      case 'SYST':
        return <System />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-black text-white overflow-hidden font-mono">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          {renderDomain()}
        </main>
      </div>

      {/* Quick Action Modals */}
      <QuickActionModals />
    </div>
  );
}

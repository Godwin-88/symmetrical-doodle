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
import { NautilusTrading } from './components/NautilusTrading';

export default function App() {
  const { currentDomain, setCurrentDomain, checkHealth } = useTradingStore();

  // Health check on mount and periodically
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Keyboard shortcuts matching FunctionKeyBar layout
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key >= 'F1' && e.key <= 'F10') {
        e.preventDefault();
        // Matches FunctionKeyBar: DASH, MKTS, INTL, STRT, PORT, EXEC, SIMU, WORK, NAUT, SYST
        const domains = ['DASH', 'MKTS', 'INTL', 'STRT', 'PORT', 'EXEC', 'SIMU', 'WORK', 'NAUT', 'SYST'];
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
      case 'NAUT':
        return <NautilusTrading />;
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

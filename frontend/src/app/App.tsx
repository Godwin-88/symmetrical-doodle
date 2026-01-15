import { useEffect } from 'react';
import { useTradingStore } from '@/app/store/tradingStore';
import { FunctionKeyBar } from '@/app/components/FunctionKeyBar';
import { StatusBar } from '@/app/components/StatusBar';
import { Dashboard } from '@/app/components/Dashboard';
import { Intelligence } from '@/app/components/Intelligence';
import { Markets } from '@/app/components/Markets';
import { Strategies } from '@/app/components/Strategies';
import { Portfolio } from '@/app/components/Portfolio';
import { Execution } from '@/app/components/Execution';
import { Simulation } from '@/app/components/Simulation';
import { DataModels } from '@/app/components/DataModels';
import { System } from '@/app/components/System';
import { DataWorkspace } from '@/app/components/DataWorkspace';

export default function App() {
  const { currentDomain, setCurrentDomain, checkHealth } = useTradingStore();

  // Health check on mount and periodically
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Keyboard shortcuts for function keys
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key >= 'F1' && e.key <= 'F9') {
        e.preventDefault();
        const domains = ['DASH', 'MKTS', 'INTL', 'STRT', 'PORT', 'EXEC', 'SIMU', 'DATA', 'SYST'];
        const index = parseInt(e.key[1]) - 1;
        setCurrentDomain(domains[index] as any);
      } else if (e.key === 'F10') {
        e.preventDefault();
        setCurrentDomain('WORK' as any);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [setCurrentDomain]);

  const renderDomain = () => {
    switch (currentDomain) {
      case 'DASH':
        return <Dashboard />;
      case 'MKTS':
        return <Markets />;
      case 'INTL':
        return <Intelligence />;
      case 'STRT':
        return <Strategies />;
      case 'PORT':
        return <Portfolio />;
      case 'EXEC':
        return <Execution />;
      case 'SIMU':
        return <Simulation />;
      case 'DATA':
        return <DataModels />;
      case 'SYST':
        return <System />;
      case 'WORK':
        return <DataWorkspace />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-black text-white overflow-hidden">
      <FunctionKeyBar />
      <div className="flex-1 overflow-hidden">
        {renderDomain()}
      </div>
      <StatusBar />
    </div>
  );
}

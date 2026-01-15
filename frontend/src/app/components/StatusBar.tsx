import { useTradingStore } from '@/app/store/tradingStore';
import { AlertTriangle, Power } from 'lucide-react';

export function StatusBar() {
  const {
    systemStatus,
    connectionStatus,
    latency,
    executionMode,
    currentRegime,
  } = useTradingStore();

  const now = new Date();
  const timestamp = now.toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  return (
    <div className="flex items-center justify-between bg-black border-t border-[#ff8c00] h-7 px-4 font-mono text-[11px] text-[#00ff00]">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <span className="text-[#ff8c00]">SYS:</span>
          <span className={systemStatus === 'OPERATIONAL' ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
            {systemStatus}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-[#ff8c00]">CONN:</span>
          <span className={connectionStatus === 'LIVE' ? 'text-[#00ff00]' : 'text-[#ffff00]'}>
            {connectionStatus}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-[#ff8c00]">LAT:</span>
          <span>{latency}ms</span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-[#ff8c00]">EXEC:</span>
          <span className="text-[#ffff00]">{executionMode}</span>
        </div>
        
        {currentRegime && (
          <div className="flex items-center gap-2">
            <span className="text-[#ff8c00]">REGIME:</span>
            <span>{currentRegime.name}</span>
            <span className="text-[#666]">[CONF: {currentRegime.probability}%]</span>
          </div>
        )}
      </div>
      
      <div className="flex items-center gap-4">
        <button className="flex items-center gap-2 px-3 py-1 bg-[#ff0000] text-black hover:bg-[#ff3333] transition-colors">
          <Power className="w-3 h-3" />
          <span className="font-bold">EMERGENCY HALT</span>
        </button>
        
        <span className="text-[#00ff00]">{timestamp} UTC</span>
      </div>
    </div>
  );
}

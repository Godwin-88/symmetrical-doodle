import { useTradingStore, type Domain } from '@/app/store/tradingStore';

const DOMAINS = [
  { key: 'F1', code: 'DASH', label: 'DASHBOARD' },
  { key: 'F2', code: 'MKTS', label: 'MARKETS' },
  { key: 'F3', code: 'INTL', label: 'INTELLIGENCE' },
  { key: 'F4', code: 'STRT', label: 'STRATEGIES' },
  { key: 'F5', code: 'PORT', label: 'PORTFOLIO' },
  { key: 'F6', code: 'EXEC', label: 'EXECUTION' },
  { key: 'F7', code: 'SIMU', label: 'SIMULATION' },
  { key: 'F8', code: 'DATA', label: 'DATA & MODELS' },
  { key: 'F9', code: 'SYST', label: 'SYSTEM' },
  { key: 'F10', code: 'WORK', label: 'WORKSPACE' },
];

export function FunctionKeyBar() {
  const { currentDomain, setCurrentDomain } = useTradingStore();

  return (
    <div className="flex items-center bg-black border-b border-[#ff8c00] h-8 font-mono text-xs">
      {DOMAINS.map((domain, idx) => (
        <button
          key={domain.code}
          onClick={() => setCurrentDomain(domain.code as Domain)}
          className={`
            flex-1 h-full px-2 border-r border-[#333]
            transition-colors
            ${currentDomain === domain.code
              ? 'bg-[#ff8c00] text-black font-bold'
              : 'bg-black text-[#ff8c00] hover:bg-[#1a1a1a]'
            }
          `}
        >
          <span className="text-[10px]">{domain.key}:</span>
          <span className="ml-1">{domain.code}</span>
        </button>
      ))}
    </div>
  );
}

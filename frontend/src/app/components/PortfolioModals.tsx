/**
 * Portfolio Modals - CRUD Forms for Portfolio Management
 */

import { useState } from 'react';
import type { PortfolioDefinition, RiskLimit, StressScenario } from './Portfolio';

interface CreatePortfolioModalProps {
  show: boolean;
  onClose: () => void;
  onCreate: (portfolio: any) => void;
  availableStrategies: any[];
}

export function CreatePortfolioModal({ show, onClose, onCreate, availableStrategies }: CreatePortfolioModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    baseCurrency: 'USD',
    initialCapital: 100000,
    mode: 'PAPER' as 'LIVE' | 'PAPER' | 'SIMULATED',
    allocationModel: 'VOL_TARGET' as any,
    rebalanceFrequency: 'WEEKLY' as any,
    turnoverConstraint: 20,
    strategyAllocations: [] as Array<{ strategyId: string; weight: number }>,
  });

  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [strategyWeight, setStrategyWeight] = useState(0);

  const addStrategy = () => {
    if (!selectedStrategy || strategyWeight <= 0) return;
    
    const totalWeight = formData.strategyAllocations.reduce((sum, s) => sum + s.weight, 0);
    if (totalWeight + strategyWeight > 1) {
      alert('Total weight cannot exceed 100%');
      return;
    }

    setFormData({
      ...formData,
      strategyAllocations: [
        ...formData.strategyAllocations,
        { strategyId: selectedStrategy, weight: strategyWeight }
      ]
    });
    setSelectedStrategy('');
    setStrategyWeight(0);
  };

  const removeStrategy = (index: number) => {
    setFormData({
      ...formData,
      strategyAllocations: formData.strategyAllocations.filter((_, i) => i !== index)
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (formData.strategyAllocations.length === 0) {
      alert('Please add at least one strategy allocation');
      return;
    }

    const totalWeight = formData.strategyAllocations.reduce((sum, s) => sum + s.weight, 0);
    if (Math.abs(totalWeight - 1) > 0.01) {
      alert('Total strategy weights must equal 100%');
      return;
    }

    onCreate({
      ...formData,
      status: 'ACTIVE',
      strategyAllocations: formData.strategyAllocations.map(s => ({
        ...s,
        capitalAllocated: formData.initialCapital * s.weight
      }))
    });
    onClose();
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">CREATE NEW PORTFOLIO</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            {/* Basic Info */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">PORTFOLIO NAME</label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., MAIN TRADING PORTFOLIO"
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">BASE CURRENCY</label>
                <select
                  value={formData.baseCurrency}
                  onChange={(e) => setFormData({ ...formData, baseCurrency: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="USD">USD</option>
                  <option value="EUR">EUR</option>
                  <option value="GBP">GBP</option>
                  <option value="JPY">JPY</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">INITIAL CAPITAL</label>
                <input
                  type="number"
                  required
                  value={formData.initialCapital}
                  onChange={(e) => setFormData({ ...formData, initialCapital: parseFloat(e.target.value) })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">MODE</label>
                <select
                  value={formData.mode}
                  onChange={(e) => setFormData({ ...formData, mode: e.target.value as any })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="SIMULATED">SIMULATED</option>
                  <option value="PAPER">PAPER</option>
                  <option value="LIVE">LIVE</option>
                </select>
              </div>
            </div>

            {/* Allocation Model */}
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">ALLOCATION MODEL</label>
                <select
                  value={formData.allocationModel}
                  onChange={(e) => setFormData({ ...formData, allocationModel: e.target.value as any })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="EQUAL_WEIGHT">EQUAL WEIGHT</option>
                  <option value="VOL_TARGET">VOL TARGET</option>
                  <option value="RISK_PARITY">RISK PARITY</option>
                  <option value="MAX_DIVERSIFICATION">MAX DIVERSIFICATION</option>
                  <option value="KELLY">KELLY</option>
                  <option value="CUSTOM">CUSTOM</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">REBALANCE FREQUENCY</label>
                <select
                  value={formData.rebalanceFrequency}
                  onChange={(e) => setFormData({ ...formData, rebalanceFrequency: e.target.value as any })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="DAILY">DAILY</option>
                  <option value="WEEKLY">WEEKLY</option>
                  <option value="MONTHLY">MONTHLY</option>
                  <option value="QUARTERLY">QUARTERLY</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">TURNOVER CONSTRAINT (%)</label>
                <input
                  type="number"
                  value={formData.turnoverConstraint}
                  onChange={(e) => setFormData({ ...formData, turnoverConstraint: parseFloat(e.target.value) })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
            </div>

            {/* Strategy Allocations */}
            <div>
              <label className="text-[#ff8c00] block mb-2 text-[10px]">STRATEGY ALLOCATIONS</label>
              
              {/* Add Strategy */}
              <div className="flex gap-2 mb-3">
                <select
                  value={selectedStrategy}
                  onChange={(e) => setSelectedStrategy(e.target.value)}
                  className="flex-1 bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">Select Strategy...</option>
                  {availableStrategies.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={strategyWeight}
                  onChange={(e) => setStrategyWeight(parseFloat(e.target.value))}
                  placeholder="Weight (0-1)"
                  className="w-32 bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
                <button
                  type="button"
                  onClick={addStrategy}
                  className="px-3 py-1 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
                >
                  ADD
                </button>
              </div>

              {/* Strategy List */}
              <div className="border border-[#444] bg-[#0a0a0a] max-h-40 overflow-y-auto">
                {formData.strategyAllocations.length === 0 ? (
                  <div className="p-3 text-[#666] text-[10px] text-center">No strategies added</div>
                ) : (
                  formData.strategyAllocations.map((alloc, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 border-b border-[#222]">
                      <span className="text-[#00ff00] text-[10px]">{alloc.strategyId}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[#fff] text-[10px]">{(alloc.weight * 100).toFixed(1)}%</span>
                        <button
                          type="button"
                          onClick={() => removeStrategy(idx)}
                          className="text-[#ff0000] text-[10px] hover:text-[#fff]"
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              <div className="mt-2 text-[10px]">
                <span className="text-[#666]">TOTAL WEIGHT: </span>
                <span className={`${Math.abs(formData.strategyAllocations.reduce((sum, s) => sum + s.weight, 0) - 1) < 0.01 ? 'text-[#00ff00]' : 'text-[#ffff00]'}`}>
                  {(formData.strategyAllocations.reduce((sum, s) => sum + s.weight, 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-4">
              <button
                type="submit"
                className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px]"
              >
                CREATE PORTFOLIO
              </button>
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CANCEL
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

// Export other modals as placeholders for now
export function EditPortfolioModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6">
        <h2 className="text-[#ff8c00] mb-4">EDIT PORTFOLIO (Coming Soon)</h2>
        <button onClick={onClose} className="px-4 py-2 border border-[#666] text-[#666]">CLOSE</button>
      </div>
    </div>
  );
}

export function RiskLimitModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6">
        <h2 className="text-[#ff8c00] mb-4">RISK LIMIT CONFIGURATION (Coming Soon)</h2>
        <button onClick={onClose} className="px-4 py-2 border border-[#666] text-[#666]">CLOSE</button>
      </div>
    </div>
  );
}

export function StressTestModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6">
        <h2 className="text-[#ff8c00] mb-4">STRESS TEST CONFIGURATION (Coming Soon)</h2>
        <button onClick={onClose} className="px-4 py-2 border border-[#666] text-[#666]">CLOSE</button>
      </div>
    </div>
  );
}

export function AllocationModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6">
        <h2 className="text-[#ff8c00] mb-4">REBALANCE ALLOCATION (Coming Soon)</h2>
        <button onClick={onClose} className="px-4 py-2 border border-[#666] text-[#666]">CLOSE</button>
      </div>
    </div>
  );
}

export function AttributionModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6">
        <h2 className="text-[#ff8c00] mb-4">P&L ATTRIBUTION ANALYSIS (Coming Soon)</h2>
        <button onClick={onClose} className="px-4 py-2 border border-[#666] text-[#666]">CLOSE</button>
      </div>
    </div>
  );
}

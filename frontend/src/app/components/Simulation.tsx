import { useState, useEffect } from 'react';
import { listStrategies, type StrategySpec } from '../../services/strategiesService';
import {
  listExperiments,
  getExperiment,
  createExperiment,
  updateExperiment,
  deleteExperiment,
  cloneExperiment,
  runExperiment,
  stopExperiment,
  listScenarios,
  runScenario,
  runParameterSweep,
  compareExperiments,
  type Experiment,
  type ExperimentResults,
  type ScenarioTest,
} from '../../services/simulationService';

// Types imported from simulationService

export function Simulation() {
  // State
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [availableStrategies, setAvailableStrategies] = useState<StrategySpec[]>([]);
  const [scenarioTests, setScenarioTests] = useState<ScenarioTest[]>([]);
  
  // Modal states
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showResultsModal, setShowResultsModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [showScenarioModal, setShowScenarioModal] = useState(false);
  const [showParameterSweepModal, setShowParameterSweepModal] = useState(false);
  
  // Filter states
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [tagFilter, setTagFilter] = useState<string>('ALL');
  
  // Comparison state
  const [compareExperiments, setCompareExperiments] = useState<string[]>([]);
  
  // Initialize data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [strategies, exps, scenarios] = await Promise.all([
          listStrategies(),
          listExperiments(),
          listScenarios(),
        ]);
        setAvailableStrategies(strategies);
        setExperiments(exps);
        setScenarioTests(scenarios);
      } catch (err) {
        console.warn('Failed to fetch data:', err);
      }
    };
    
    fetchData();
  }, []);
  
  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  // CRUD Operations
  const handleCreateExperiment = async (experiment: Omit<Experiment, 'id' | 'createdAt' | 'status'>) => {
    try {
      const newExperiment = await createExperiment(experiment);
      setExperiments([...experiments, newExperiment]);
      return newExperiment;
    } catch (err) {
      console.error('Failed to create experiment:', err);
    }
  };
  
  const handleUpdateExperiment = async (id: string, updates: Partial<Experiment>) => {
    try {
      const updated = await updateExperiment(id, updates);
      setExperiments(experiments.map(exp => exp.id === id ? updated : exp));
    } catch (err) {
      console.error('Failed to update experiment:', err);
    }
  };
  
  const handleDeleteExperiment = async (id: string) => {
    try {
      await deleteExperiment(id);
      setExperiments(experiments.filter(exp => exp.id !== id));
      if (selectedExperiment?.id === id) {
        setSelectedExperiment(null);
      }
    } catch (err) {
      console.error('Failed to delete experiment:', err);
    }
  };
  
  const handleCloneExperiment = async (id: string) => {
    try {
      const original = experiments.find(exp => exp.id === id);
      if (!original) return;
      
      const cloned = await cloneExperiment(id, `${original.name} (CLONE)`);
      setExperiments([...experiments, cloned]);
    } catch (err) {
      console.error('Failed to clone experiment:', err);
    }
  };
  
  const handleRunExperiment = async (id: string) => {
    try {
      const running = await runExperiment(id);
      setExperiments(experiments.map(exp => exp.id === id ? running : exp));
    } catch (err) {
      console.error('Failed to run experiment:', err);
    }
  };
  
  const handleStopExperiment = async (id: string) => {
    try {
      const stopped = await stopExperiment(id);
      setExperiments(experiments.map(exp => exp.id === id ? stopped : exp));
    } catch (err) {
      console.error('Failed to stop experiment:', err);
    }
  };
  
  const toggleCompare = (id: string) => {
    if (compareExperiments.includes(id)) {
      setCompareExperiments(compareExperiments.filter(expId => expId !== id));
    } else if (compareExperiments.length < 4) {
      setCompareExperiments([...compareExperiments, id]);
    }
  };
  
  // Filter experiments
  const filteredExperiments = experiments.filter(exp => {
    if (statusFilter !== 'ALL' && exp.status !== statusFilter) return false;
    if (tagFilter !== 'ALL' && !exp.tags.includes(tagFilter)) return false;
    return true;
  });
  
  // Get unique tags
  const allTags = Array.from(new Set(experiments.flatMap(exp => exp.tags)));
  
  // Calculate summary stats
  const summaryStats = {
    total: experiments.length,
    running: experiments.filter(e => e.status === 'RUNNING').length,
    queued: experiments.filter(e => e.status === 'QUEUED').length,
    completed: experiments.filter(e => e.status === 'COMPLETED').length,
    failed: experiments.filter(e => e.status === 'FAILED').length,
    avgSharpe: experiments
      .filter(e => e.results)
      .reduce((sum, e) => sum + (e.results?.sharpe || 0), 0) / 
      experiments.filter(e => e.results).length || 0,
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Experiment List */}
      <div className="w-96 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Header & Filters */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">EXPERIMENT REGISTRY</div>
            
            {/* Summary Stats */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL</div>
                <div className="text-[#fff] text-[14px]">{summaryStats.total}</div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">AVG SHARPE</div>
                <div className="text-[#00ff00] text-[14px]">{formatNumber(summaryStats.avgSharpe)}</div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">RUNNING</div>
                <div className="text-[#ffff00] text-[14px]">{summaryStats.running}</div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">COMPLETED</div>
                <div className="text-[#00ff00] text-[14px]">{summaryStats.completed}</div>
              </div>
            </div>
            
            {/* Filters */}
            <div className="space-y-2">
              <div>
                <label className="text-[#666] text-[9px] block mb-1">STATUS</label>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="ALL">ALL</option>
                  <option value="DRAFT">DRAFT</option>
                  <option value="QUEUED">QUEUED</option>
                  <option value="RUNNING">RUNNING</option>
                  <option value="COMPLETED">COMPLETED</option>
                  <option value="FAILED">FAILED</option>
                  <option value="CANCELLED">CANCELLED</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] text-[9px] block mb-1">TAG</label>
                <select
                  value={tagFilter}
                  onChange={(e) => setTagFilter(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="ALL">ALL</option>
                  {allTags.map(tag => (
                    <option key={tag} value={tag}>{tag}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="space-y-2 mb-4">
            <button
              onClick={() => setShowCreateModal(true)}
              className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
            >
              + NEW EXPERIMENT
            </button>
            <button
              onClick={() => setShowParameterSweepModal(true)}
              className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              PARAMETER SWEEP
            </button>
            <button
              onClick={() => setShowScenarioModal(true)}
              className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors"
            >
              SCENARIO TESTS
            </button>
            {compareExperiments.length >= 2 && (
              <button
                onClick={() => setShowCompareModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                COMPARE ({compareExperiments.length})
              </button>
            )}
          </div>
          
          {/* Experiment List */}
          <div className="space-y-2">
            {filteredExperiments.map((exp) => (
              <div
                key={exp.id}
                onClick={() => setSelectedExperiment(exp)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedExperiment?.id === exp.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="text-[#00ff00] text-[10px] font-bold">{exp.name}</div>
                    <div className="text-[#666] text-[9px]">{exp.id}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={compareExperiments.includes(exp.id)}
                      onChange={(e) => {
                        e.stopPropagation();
                        toggleCompare(exp.id);
                      }}
                      className="form-checkbox h-3 w-3"
                      title="Compare"
                    />
                    <span className={`
                      text-[8px] px-1
                      ${exp.status === 'COMPLETED' ? 'text-[#00ff00]' : ''}
                      ${exp.status === 'RUNNING' ? 'text-[#ffff00]' : ''}
                      ${exp.status === 'QUEUED' ? 'text-[#666]' : ''}
                      ${exp.status === 'FAILED' ? 'text-[#ff0000]' : ''}
                      ${exp.status === 'CANCELLED' ? 'text-[#ff8c00]' : ''}
                      ${exp.status === 'DRAFT' ? 'text-[#666]' : ''}
                    `}>
                      {exp.status}
                    </span>
                  </div>
                </div>
                
                <div className="space-y-1 text-[9px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">STRATEGY:</span>
                    <span className="text-[#fff]">{exp.strategyId}</span>
                  </div>
                  {exp.results && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-[#666]">SHARPE:</span>
                        <span className={exp.results.sharpe >= 1 ? 'text-[#00ff00]' : exp.results.sharpe >= 0 ? 'text-[#ffff00]' : 'text-[#ff0000]'}>
                          {formatNumber(exp.results.sharpe)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">MAX DD:</span>
                        <span className="text-[#ff0000]">{formatNumber(exp.results.maxDrawdown, 1)}%</span>
                      </div>
                    </>
                  )}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {exp.tags.map(tag => (
                      <span key={tag} className="px-1 bg-[#1a1a1a] border border-[#444] text-[#ff8c00] text-[8px]">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Center Panel - Experiment Details */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            {selectedExperiment ? `EXPERIMENT: ${selectedExperiment.name}` : 'SIMULATION & BACKTESTING - EXPERIMENT MANAGEMENT'}
          </div>
        </div>

        {selectedExperiment ? (
          <div className="space-y-6">
            {/* Experiment Metadata */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">METADATA</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-[#666]">EXPERIMENT ID</div>
                    <div className="text-[#00ff00]">{selectedExperiment.id}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">STATUS</div>
                    <div className={`
                      ${selectedExperiment.status === 'COMPLETED' ? 'text-[#00ff00]' : ''}
                      ${selectedExperiment.status === 'RUNNING' ? 'text-[#ffff00]' : ''}
                      ${selectedExperiment.status === 'FAILED' ? 'text-[#ff0000]' : ''}
                      ${selectedExperiment.status === 'QUEUED' ? 'text-[#666]' : ''}
                    `}>
                      {selectedExperiment.status}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#666]">STRATEGY</div>
                    <div className="text-[#fff]">{selectedExperiment.strategyId} v{selectedExperiment.strategyVersion}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">RESEARCHER</div>
                    <div className="text-[#fff]">{selectedExperiment.researcher}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">GIT COMMIT</div>
                    <div className="text-[#00ff00]">{selectedExperiment.gitCommit || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">CREATED</div>
                    <div className="text-[#fff]">{new Date(selectedExperiment.createdAt).toLocaleString()}</div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="text-[#666]">HYPOTHESIS</div>
                  <div className="text-[#fff]">{selectedExperiment.hypothesis}</div>
                </div>
              </div>
            </div>

            {/* Market & Data Configuration */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">MARKET & DATA CONFIGURATION</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-[#666]">ASSET UNIVERSE</div>
                    <div className="text-[#00ff00]">{selectedExperiment.assetUniverse.join(', ')}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">DATA SOURCE</div>
                    <div className="text-[#fff]">{selectedExperiment.dataSource}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">DATE RANGE</div>
                    <div className="text-[#fff]">{selectedExperiment.startDate} to {selectedExperiment.endDate}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">FREQUENCY</div>
                    <div className="text-[#fff]">{selectedExperiment.frequency}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">SURVIVORSHIP BIAS</div>
                    <div className={selectedExperiment.survivorshipBias ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                      {selectedExperiment.survivorshipBias ? 'ENABLED ⚠' : 'DISABLED ✓'}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#666]">LOOK-AHEAD BIAS</div>
                    <div className={selectedExperiment.lookAheadBias ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                      {selectedExperiment.lookAheadBias ? 'ENABLED ⚠' : 'DISABLED ✓'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Execution Model */}
            <div>
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">EXECUTION MODEL</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-[#666]">ORDER TYPES</div>
                    <div className="text-[#fff]">{selectedExperiment.executionModel.orderTypes.join(', ')}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">SLIPPAGE MODEL</div>
                    <div className="text-[#fff]">{selectedExperiment.executionModel.slippageModel}</div>
                  </div>
                  <div>
                    <div className="text-[#666]">SLIPPAGE</div>
                    <div className="text-[#ffff00]">{selectedExperiment.executionModel.slippageBps} bps</div>
                  </div>
                  <div>
                    <div className="text-[#666]">TRANSACTION COST</div>
                    <div className="text-[#ffff00]">{selectedExperiment.executionModel.transactionCostBps} bps</div>
                  </div>
                  <div>
                    <div className="text-[#666]">LATENCY</div>
                    <div className="text-[#fff]">{selectedExperiment.executionModel.latencyMs}ms</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Portfolio & Risk Configuration */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">PORTFOLIO CONSTRUCTION</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="space-y-2 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">POSITION SIZING</span>
                      <span className="text-[#fff]">{selectedExperiment.portfolioConfig.positionSizing}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">MAX LEVERAGE</span>
                      <span className="text-[#ffff00]">{selectedExperiment.portfolioConfig.maxLeverage}x</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">MAX POSITION</span>
                      <span className="text-[#fff]">{selectedExperiment.portfolioConfig.maxPositionPct}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">LONG/SHORT RATIO</span>
                      <span className="text-[#fff]">{selectedExperiment.portfolioConfig.longShortRatio}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK MANAGEMENT</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="space-y-2 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">STOP LOSS</span>
                      <span className="text-[#ff0000]">{selectedExperiment.riskConfig.stopLossPct}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">TAKE PROFIT</span>
                      <span className="text-[#00ff00]">{selectedExperiment.riskConfig.takeProfitPct}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">MAX DRAWDOWN</span>
                      <span className="text-[#ff0000]">{selectedExperiment.riskConfig.maxDrawdownPct}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">DAILY LOSS LIMIT</span>
                      <span className="text-[#ff0000]">{selectedExperiment.riskConfig.dailyLossLimitPct}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Results (if completed) */}
            {selectedExperiment.results && (
              <>
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">PERFORMANCE METRICS</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    <div className="grid grid-cols-4 gap-4 p-3">
                      <div>
                        <div className="text-[#666] text-[9px]">TOTAL RETURN</div>
                        <div className={`text-[14px] ${selectedExperiment.results.totalReturn >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                          {formatNumber(selectedExperiment.results.totalReturn, 1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">CAGR</div>
                        <div className={`text-[14px] ${selectedExperiment.results.cagr >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                          {formatNumber(selectedExperiment.results.cagr, 1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">SHARPE RATIO</div>
                        <div className={`text-[14px] ${selectedExperiment.results.sharpe >= 1 ? 'text-[#00ff00]' : selectedExperiment.results.sharpe >= 0 ? 'text-[#ffff00]' : 'text-[#ff0000]'}`}>
                          {formatNumber(selectedExperiment.results.sharpe)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">SORTINO RATIO</div>
                        <div className="text-[#00ff00] text-[14px]">{formatNumber(selectedExperiment.results.sortino)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">MAX DRAWDOWN</div>
                        <div className="text-[#ff0000] text-[14px]">{formatNumber(selectedExperiment.results.maxDrawdown, 1)}%</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">PROFIT FACTOR</div>
                        <div className="text-[#00ff00] text-[14px]">{formatNumber(selectedExperiment.results.profitFactor)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">WIN RATE</div>
                        <div className="text-[#00ff00] text-[14px]">{formatNumber(selectedExperiment.results.winRate, 0)}%</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">TURNOVER</div>
                        <div className="text-[#ffff00] text-[14px]">{formatNumber(selectedExperiment.results.turnover, 1)}x</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">TRADE STATISTICS</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    <div className="grid grid-cols-3 gap-4 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-[#666]">TOTAL TRADES</span>
                        <span className="text-[#fff]">{selectedExperiment.results.totalTrades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">WINNING</span>
                        <span className="text-[#00ff00]">{selectedExperiment.results.winningTrades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">LOSING</span>
                        <span className="text-[#ff0000]">{selectedExperiment.results.losingTrades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">AVG WIN</span>
                        <span className="text-[#00ff00]">${formatNumber(selectedExperiment.results.avgWin)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">AVG LOSS</span>
                        <span className="text-[#ff0000]">${formatNumber(selectedExperiment.results.avgLoss)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">AVG HOLD</span>
                        <span className="text-[#fff]">{formatNumber(selectedExperiment.results.avgHoldingPeriod, 1)}h</span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="text-center text-[#666] text-[10px] mt-20">
            SELECT AN EXPERIMENT TO VIEW DETAILS
          </div>
        )}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">EXPERIMENT ACTIONS</div>
          
          {selectedExperiment ? (
            <div className="space-y-2">
              {/* Run Controls */}
              {selectedExperiment.status === 'DRAFT' && (
                <button
                  onClick={() => handleRunExperiment(selectedExperiment.id)}
                  className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
                >
                  ▶ RUN EXPERIMENT
                </button>
              )}
              
              {selectedExperiment.status === 'RUNNING' && (
                <button
                  onClick={() => handleStopExperiment(selectedExperiment.id)}
                  className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                >
                  ■ STOP EXPERIMENT
                </button>
              )}
              
              {selectedExperiment.results && (
                <button
                  onClick={() => setShowResultsModal(true)}
                  className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                >
                  📊 VIEW FULL RESULTS
                </button>
              )}
              
              <button
                onClick={() => setShowEditModal(true)}
                className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#00ff00] transition-colors"
              >
                ✎ EDIT CONFIGURATION
              </button>
              
              <button
                onClick={() => handleCloneExperiment(selectedExperiment.id)}
                className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#00ff00] transition-colors"
              >
                ⎘ CLONE EXPERIMENT
              </button>
              
              <button
                onClick={() => {
                  if (confirm(`Delete experiment ${selectedExperiment.name}?`)) {
                    handleDeleteExperiment(selectedExperiment.id);
                  }
                }}
                className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
              >
                🗑 DELETE EXPERIMENT
              </button>
            </div>
          ) : (
            <div className="text-[#666] text-[10px]">
              Select an experiment to view actions
            </div>
          )}
          
          {/* Scenario Tests */}
          <div className="mt-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">SCENARIO TESTS</div>
            <div className="space-y-2">
              {scenarioTests.filter(s => s.enabled).map(scenario => (
                <div key={scenario.id} className="border border-[#444] p-2">
                  <div className="text-[#00ff00] text-[9px]">{scenario.name}</div>
                  <div className="text-[#666] text-[8px] mt-1">{scenario.type}</div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="mt-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK STATS</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="space-y-2 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">TOTAL EXPERIMENTS</span>
                  <span className="text-[#fff]">{experiments.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">AVG SHARPE</span>
                  <span className="text-[#00ff00]">{formatNumber(summaryStats.avgSharpe)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">RUNNING</span>
                  <span className="text-[#ffff00]">{summaryStats.running}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">COMPLETED</span>
                  <span className="text-[#00ff00]">{summaryStats.completed}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

pub mod clock;
pub mod backtesting;

pub use clock::{SimulationClock, TimeManager};
pub use backtesting::{BacktestEngine, MarketDataProvider, SimulationConfig};

use chrono::{DateTime, Utc};
use execution_core::{ExecutionCore, ExecutionCoreImpl, Config};
use anyhow::Result;

/// Main simulation engine for deterministic backtesting
pub struct SimulationEngine {
    pub execution_core: ExecutionCoreImpl,
    pub clock: SimulationClock,
    pub backtest_engine: BacktestEngine,
}

impl SimulationEngine {
    pub fn new(config: &Config, simulation_config: SimulationConfig) -> Result<Self> {
        Ok(Self {
            execution_core: ExecutionCoreImpl::new(config)?,
            clock: SimulationClock::new(simulation_config.start_time, simulation_config.end_time),
            backtest_engine: BacktestEngine::new(simulation_config),
        })
    }
    
    pub fn run_simulation(&mut self) -> Result<()> {
        tracing::info!("Starting simulation from {} to {}", 
                      self.clock.start_time(), self.clock.end_time());
        
        while !self.clock.is_finished() {
            let current_time = self.clock.current_time();
            
            // Process market data and events for current time
            self.backtest_engine.process_time_step(current_time, &mut self.execution_core)?;
            
            // Advance simulation clock
            self.clock.advance()?;
        }
        
        tracing::info!("Simulation completed");
        Ok(())
    }
}
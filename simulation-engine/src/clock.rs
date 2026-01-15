use chrono::{DateTime, Utc, Duration, Timelike};
use anyhow::Result;

/// Deterministic clock abstraction for simulation
pub struct SimulationClock {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    current_time: DateTime<Utc>,
    step_size: Duration,
}

impl SimulationClock {
    pub fn new(start_time: DateTime<Utc>, end_time: DateTime<Utc>) -> Self {
        Self {
            start_time,
            end_time,
            current_time: start_time,
            step_size: Duration::minutes(1), // Default 1-minute steps
        }
    }
    
    pub fn with_step_size(mut self, step_size: Duration) -> Self {
        self.step_size = step_size;
        self
    }
    
    pub fn current_time(&self) -> DateTime<Utc> {
        self.current_time
    }
    
    pub fn start_time(&self) -> DateTime<Utc> {
        self.start_time
    }
    
    pub fn end_time(&self) -> DateTime<Utc> {
        self.end_time
    }
    
    pub fn advance(&mut self) -> Result<()> {
        if self.current_time >= self.end_time {
            anyhow::bail!("Cannot advance past end time");
        }
        
        self.current_time += self.step_size;
        Ok(())
    }
    
    pub fn is_finished(&self) -> bool {
        self.current_time >= self.end_time
    }
    
    pub fn reset(&mut self) {
        self.current_time = self.start_time;
    }
    
    pub fn jump_to(&mut self, time: DateTime<Utc>) -> Result<()> {
        if time < self.start_time || time > self.end_time {
            anyhow::bail!("Time {} is outside simulation bounds", time);
        }
        
        self.current_time = time;
        Ok(())
    }
}

/// Time management utilities for simulation
pub struct TimeManager;

impl TimeManager {
    pub fn is_market_open(time: DateTime<Utc>) -> bool {
        // Simplified market hours check (9:30 AM - 4:00 PM EST)
        let hour = time.hour();
        hour >= 14 && hour < 21 // UTC hours for EST market
    }
    
    pub fn next_market_open(time: DateTime<Utc>) -> DateTime<Utc> {
        // Simplified - just advance to next 9:30 AM EST
        let mut next_open = time.date_naive().and_hms_opt(14, 30, 0).unwrap().and_utc();
        if next_open <= time {
            next_open += Duration::days(1);
        }
        next_open
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Property test for deterministic clock consistency
    // Feature: algorithmic-trading-system, Property 14: Deterministic Clock Consistency
    proptest! {
        #[test]
        fn prop_deterministic_clock_consistency(
            start_timestamp in 0i64..1_000_000_000i64,
            duration_hours in 1u32..168u32, // 1 hour to 1 week
            step_minutes in 1u32..1440u32,  // 1 minute to 1 day
        ) {
            // Create deterministic start and end times
            let start_time = DateTime::from_timestamp(start_timestamp, 0).unwrap_or_else(|| {
                DateTime::from_timestamp(0, 0).unwrap()
            });
            let end_time = start_time + Duration::hours(duration_hours as i64);
            let step_size = Duration::minutes(step_minutes as i64);

            // Create two identical clocks
            let mut clock1 = SimulationClock::new(start_time, end_time).with_step_size(step_size);
            let mut clock2 = SimulationClock::new(start_time, end_time).with_step_size(step_size);

            // Both clocks should start at the same time
            prop_assert_eq!(clock1.current_time(), clock2.current_time());
            prop_assert_eq!(clock1.start_time(), clock2.start_time());
            prop_assert_eq!(clock1.end_time(), clock2.end_time());

            // Advance both clocks in lockstep and verify they remain synchronized
            let mut step_count = 0;
            while !clock1.is_finished() && !clock2.is_finished() && step_count < 1000 {
                let time1_before = clock1.current_time();
                let time2_before = clock2.current_time();
                
                // Times should be identical before advancing
                prop_assert_eq!(time1_before, time2_before);
                
                // Advance both clocks
                let result1 = clock1.advance();
                let result2 = clock2.advance();
                
                // Results should be identical
                match (result1, result2) {
                    (Ok(()), Ok(())) => {
                        // Both succeeded - times should still be identical
                        prop_assert_eq!(clock1.current_time(), clock2.current_time());
                    },
                    (Err(_), Err(_)) => {
                        // Both failed - this is expected at end of simulation
                        break;
                    },
                    _ => {
                        // One succeeded, one failed - this should never happen
                        prop_assert!(false, "Clocks diverged in advance() results");
                    }
                }
                
                step_count += 1;
            }

            // Final states should be identical
            prop_assert_eq!(clock1.is_finished(), clock2.is_finished());
            prop_assert_eq!(clock1.current_time(), clock2.current_time());
        }
    }

    proptest! {
        #[test]
        fn prop_clock_reset_determinism(
            start_timestamp in 0i64..1_000_000_000i64,
            duration_hours in 1u32..24u32,
            step_minutes in 1u32..60u32,
            advance_steps in 1usize..100usize,
        ) {
            let start_time = DateTime::from_timestamp(start_timestamp, 0).unwrap_or_else(|| {
                DateTime::from_timestamp(0, 0).unwrap()
            });
            let end_time = start_time + Duration::hours(duration_hours as i64);
            let step_size = Duration::minutes(step_minutes as i64);

            let mut clock = SimulationClock::new(start_time, end_time).with_step_size(step_size);
            let original_time = clock.current_time();

            // Advance the clock some number of steps
            for _ in 0..advance_steps {
                if clock.is_finished() {
                    break;
                }
                let _ = clock.advance();
            }

            // Reset the clock
            clock.reset();

            // Clock should be back to original state
            prop_assert_eq!(clock.current_time(), original_time);
            prop_assert_eq!(clock.current_time(), start_time);
            prop_assert!(!clock.is_finished() || start_time >= end_time);
        }
    }

    proptest! {
        #[test]
        fn prop_clock_jump_determinism(
            start_timestamp in 0i64..1_000_000_000i64,
            duration_hours in 2u32..24u32,
            jump_offset_minutes in 1i64..1440i64,
        ) {
            let start_time = DateTime::from_timestamp(start_timestamp, 0).unwrap_or_else(|| {
                DateTime::from_timestamp(0, 0).unwrap()
            });
            let end_time = start_time + Duration::hours(duration_hours as i64);
            let jump_time = start_time + Duration::minutes(jump_offset_minutes);

            let mut clock1 = SimulationClock::new(start_time, end_time);
            let mut clock2 = SimulationClock::new(start_time, end_time);

            // If jump time is within bounds, both clocks should jump successfully
            if jump_time >= start_time && jump_time <= end_time {
                let result1 = clock1.jump_to(jump_time);
                let result2 = clock2.jump_to(jump_time);

                prop_assert!(result1.is_ok());
                prop_assert!(result2.is_ok());
                prop_assert_eq!(clock1.current_time(), clock2.current_time());
                prop_assert_eq!(clock1.current_time(), jump_time);
            } else {
                // If jump time is out of bounds, both should fail
                let result1 = clock1.jump_to(jump_time);
                let result2 = clock2.jump_to(jump_time);

                prop_assert!(result1.is_err());
                prop_assert!(result2.is_err());
                // Clocks should remain unchanged
                prop_assert_eq!(clock1.current_time(), start_time);
                prop_assert_eq!(clock2.current_time(), start_time);
            }
        }
    }

    #[test]
    fn test_clock_basic_functionality() {
        let start = DateTime::from_timestamp(1000000, 0).unwrap();
        let end = start + Duration::hours(1);
        let mut clock = SimulationClock::new(start, end);

        assert_eq!(clock.current_time(), start);
        assert_eq!(clock.start_time(), start);
        assert_eq!(clock.end_time(), end);
        assert!(!clock.is_finished());

        // Advance clock
        clock.advance().unwrap();
        assert_eq!(clock.current_time(), start + Duration::minutes(1));
        assert!(!clock.is_finished());

        // Reset clock
        clock.reset();
        assert_eq!(clock.current_time(), start);
    }

    #[test]
    fn test_time_manager_market_hours() {
        // Test market open during trading hours (2:30 PM UTC = 9:30 AM EST)
        let market_open = DateTime::from_timestamp(1000000, 0).unwrap()
            .date_naive()
            .and_hms_opt(15, 0, 0)
            .unwrap()
            .and_utc();
        assert!(TimeManager::is_market_open(market_open));

        // Test market closed outside trading hours
        let market_closed = DateTime::from_timestamp(1000000, 0).unwrap()
            .date_naive()
            .and_hms_opt(10, 0, 0)
            .unwrap()
            .and_utc();
        assert!(!TimeManager::is_market_open(market_closed));
    }
}
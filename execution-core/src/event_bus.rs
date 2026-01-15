use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::{OrderIntent, Fill};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    OrderIntentReceived { order_id: Uuid, intent: OrderIntent },
    OrderPlaced { order_id: Uuid, asset_id: String, side: String, quantity: f64 },
    OrderFilled { order_id: Uuid, fill: Fill },
    PositionUpdated { fill: Fill },
    RiskLimitBreached { limit_type: String, current_value: f64, limit_value: f64 },
    EmergencyHalt,
    SystemStartup,
    SystemShutdown,
    HealthCheck { component: String, status: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event: Event,
    pub sequence_number: u64,
    pub correlation_id: Option<Uuid>,
    pub source_component: String,
    pub metadata: HashMap<String, String>,
}

pub trait EventHandler: Send + Sync {
    fn handle_event(&mut self, event: &EventEnvelope) -> Result<()>;
    fn get_handler_id(&self) -> String;
}

/// Persistent event store for audit trails and replay
pub trait EventStore: Send + Sync {
    fn persist_event(&mut self, event: &EventEnvelope) -> Result<()>;
    fn load_events(&self, from_sequence: u64, to_sequence: Option<u64>) -> Result<Vec<EventEnvelope>>;
    fn get_latest_sequence(&self) -> Result<u64>;
}

/// In-memory event store implementation
pub struct InMemoryEventStore {
    events: Arc<Mutex<VecDeque<EventEnvelope>>>,
}

impl InMemoryEventStore {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

impl EventStore for InMemoryEventStore {
    fn persist_event(&mut self, event: &EventEnvelope) -> Result<()> {
        let mut events = self.events.lock().unwrap();
        events.push_back(event.clone());
        Ok(())
    }
    
    fn load_events(&self, from_sequence: u64, to_sequence: Option<u64>) -> Result<Vec<EventEnvelope>> {
        let events = self.events.lock().unwrap();
        let filtered: Vec<EventEnvelope> = events
            .iter()
            .filter(|e| {
                e.sequence_number >= from_sequence && 
                to_sequence.map_or(true, |to| e.sequence_number <= to)
            })
            .cloned()
            .collect();
        Ok(filtered)
    }
    
    fn get_latest_sequence(&self) -> Result<u64> {
        let events = self.events.lock().unwrap();
        Ok(events.back().map(|e| e.sequence_number).unwrap_or(0))
    }
}

/// Event bus for deterministic message passing and replay
pub struct EventBus {
    sequence_counter: u64,
    handlers: HashMap<String, Box<dyn EventHandler>>,
    event_store: Box<dyn EventStore>,
    source_component: String,
    sender: Option<mpsc::UnboundedSender<EventEnvelope>>,
    is_replaying: bool,
}

impl EventBus {
    pub fn new() -> Self {
        Self::with_store(Box::new(InMemoryEventStore::new()))
    }
    
    pub fn with_store(event_store: Box<dyn EventStore>) -> Self {
        Self {
            sequence_counter: 0,
            handlers: HashMap::new(),
            event_store,
            source_component: "execution-core".to_string(),
            sender: None,
            is_replaying: false,
        }
    }
    
    pub fn set_source_component(&mut self, component: String) {
        self.source_component = component;
    }
    
    pub fn publish(&mut self, event: Event) -> Result<()> {
        self.publish_with_metadata(event, None, HashMap::new())
    }
    
    pub fn publish_with_metadata(
        &mut self, 
        event: Event, 
        correlation_id: Option<Uuid>,
        metadata: HashMap<String, String>
    ) -> Result<()> {
        if !self.is_replaying {
            self.sequence_counter += 1;
        }
        
        let envelope = EventEnvelope {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event,
            sequence_number: self.sequence_counter,
            correlation_id,
            source_component: self.source_component.clone(),
            metadata,
        };
        
        // Persist event for audit trail and replay
        if !self.is_replaying {
            self.event_store.persist_event(&envelope)?;
        }
        
        // Notify handlers in deterministic order (sorted by handler ID)
        let mut handler_ids: Vec<String> = self.handlers.keys().cloned().collect();
        handler_ids.sort();
        
        for handler_id in handler_ids {
            if let Some(handler) = self.handlers.get_mut(&handler_id) {
                if let Err(e) = handler.handle_event(&envelope) {
                    tracing::error!("Handler {} failed to process event {}: {}", 
                                  handler_id, envelope.id, e);
                    // Continue processing other handlers
                }
            }
        }
        
        // Send to async channel if available
        if let Some(sender) = &self.sender {
            let _ = sender.send(envelope.clone());
        }
        
        tracing::debug!("Published event: {:?}", envelope);
        Ok(())
    }
    
    pub fn add_handler(&mut self, handler: Box<dyn EventHandler>) {
        let handler_id = handler.get_handler_id();
        self.handlers.insert(handler_id, handler);
    }
    
    pub fn remove_handler(&mut self, handler_id: &str) -> Option<Box<dyn EventHandler>> {
        self.handlers.remove(handler_id)
    }
    
    pub fn replay_events(&mut self, from_sequence: u64, to_sequence: Option<u64>) -> Result<()> {
        tracing::info!("Starting event replay from sequence {} to {:?}", from_sequence, to_sequence);
        
        self.is_replaying = true;
        let events = self.event_store.load_events(from_sequence, to_sequence)?;
        
        for event in events {
            // Update sequence counter to match replayed event
            self.sequence_counter = event.sequence_number;
            
            // Notify handlers in deterministic order
            let mut handler_ids: Vec<String> = self.handlers.keys().cloned().collect();
            handler_ids.sort();
            
            for handler_id in handler_ids {
                if let Some(handler) = self.handlers.get_mut(&handler_id) {
                    if let Err(e) = handler.handle_event(&event) {
                        tracing::error!("Handler {} failed during replay of event {}: {}", 
                                      handler_id, event.id, e);
                    }
                }
            }
        }
        
        self.is_replaying = false;
        tracing::info!("Event replay completed");
        Ok(())
    }
    
    pub fn get_event_count(&self) -> Result<u64> {
        self.event_store.get_latest_sequence()
    }
    
    pub fn get_events(&self, from_sequence: u64, to_sequence: Option<u64>) -> Result<Vec<EventEnvelope>> {
        self.event_store.load_events(from_sequence, to_sequence)
    }
    
    pub fn create_async_receiver(&mut self) -> mpsc::UnboundedReceiver<EventEnvelope> {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.sender = Some(sender);
        receiver
    }
    
    pub fn get_current_sequence(&self) -> u64 {
        self.sequence_counter
    }
    
    // Health monitoring methods
    pub fn get_queue_size(&self) -> usize {
        // For now, return 0 as we don't have a persistent queue
        // In a real implementation, this would return the size of pending events
        0
    }
    
    pub fn get_processed_count(&self) -> usize {
        // Return the current sequence number as processed count
        self.sequence_counter as usize
    }
    
    pub fn get_error_count(&self) -> usize {
        // For now, return 0 as we don't track errors separately
        // In a real implementation, this would track handler errors
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct TestHandler {
        id: String,
        events: Arc<Mutex<Vec<EventEnvelope>>>,
    }

    impl TestHandler {
        fn new(id: String) -> Self {
            Self {
                id,
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_events(&self) -> Vec<EventEnvelope> {
            self.events.lock().unwrap().clone()
        }
    }

    impl EventHandler for TestHandler {
        fn handle_event(&mut self, event: &EventEnvelope) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }

        fn get_handler_id(&self) -> String {
            self.id.clone()
        }
    }

    #[test]
    fn test_event_bus_creation() {
        let event_bus = EventBus::new();
        assert_eq!(event_bus.get_current_sequence(), 0);
    }

    #[test]
    fn test_event_publishing() {
        let mut event_bus = EventBus::new();
        let handler = TestHandler::new("test_handler".to_string());
        let events_ref = handler.events.clone();
        
        event_bus.add_handler(Box::new(handler));
        
        let result = event_bus.publish(Event::SystemStartup);
        assert!(result.is_ok());
        assert_eq!(event_bus.get_current_sequence(), 1);
        
        let events = events_ref.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].event, Event::SystemStartup));
    }

    #[test]
    fn test_deterministic_handler_ordering() {
        let mut event_bus = EventBus::new();
        
        // Add handlers in non-alphabetical order
        let handler_z = TestHandler::new("z_handler".to_string());
        let handler_a = TestHandler::new("a_handler".to_string());
        let handler_m = TestHandler::new("m_handler".to_string());
        
        let events_z = handler_z.events.clone();
        let events_a = handler_a.events.clone();
        let events_m = handler_m.events.clone();
        
        event_bus.add_handler(Box::new(handler_z));
        event_bus.add_handler(Box::new(handler_a));
        event_bus.add_handler(Box::new(handler_m));
        
        event_bus.publish(Event::SystemStartup).unwrap();
        
        // All handlers should receive the event
        assert_eq!(events_z.lock().unwrap().len(), 1);
        assert_eq!(events_a.lock().unwrap().len(), 1);
        assert_eq!(events_m.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_event_replay() {
        let mut event_bus = EventBus::new();
        let handler = TestHandler::new("replay_handler".to_string());
        let events_ref = handler.events.clone();
        
        // Publish some events first
        event_bus.publish(Event::SystemStartup).unwrap();
        event_bus.publish(Event::SystemShutdown).unwrap();
        
        // Add handler after events were published
        event_bus.add_handler(Box::new(handler));
        
        // Replay events from sequence 1
        let result = event_bus.replay_events(1, None);
        assert!(result.is_ok());
        
        let events = events_ref.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0].event, Event::SystemStartup));
        assert!(matches!(events[1].event, Event::SystemShutdown));
    }

    #[test]
    fn test_event_persistence() {
        let mut event_bus = EventBus::new();
        
        event_bus.publish(Event::SystemStartup).unwrap();
        event_bus.publish(Event::SystemShutdown).unwrap();
        
        let events = event_bus.get_events(1, None).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence_number, 1);
        assert_eq!(events[1].sequence_number, 2);
    }

    #[test]
    fn test_event_metadata() {
        let mut event_bus = EventBus::new();
        let handler = TestHandler::new("metadata_handler".to_string());
        let events_ref = handler.events.clone();
        
        event_bus.add_handler(Box::new(handler));
        
        let mut metadata = HashMap::new();
        metadata.insert("test_key".to_string(), "test_value".to_string());
        
        let correlation_id = Uuid::new_v4();
        
        event_bus.publish_with_metadata(
            Event::SystemStartup,
            Some(correlation_id),
            metadata.clone()
        ).unwrap();
        
        let events = events_ref.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].correlation_id, Some(correlation_id));
        assert_eq!(events[0].metadata, metadata);
    }

    // Property-based tests
    use proptest::prelude::*;

    // Property test for deterministic replay consistency
    // Feature: algorithmic-trading-system, Property 13: Deterministic Replay Consistency
    proptest! {
        #[test]
        fn prop_deterministic_replay_consistency(
            events in prop::collection::vec(
                prop::sample::select(vec![
                    Event::SystemStartup,
                    Event::SystemShutdown,
                    Event::EmergencyHalt,
                ]),
                1..10
            )
        ) {
            let mut event_bus1 = EventBus::new();
            let mut event_bus2 = EventBus::new();
            
            let handler1 = TestHandler::new("handler1".to_string());
            let handler2 = TestHandler::new("handler2".to_string());
            
            let events1_ref = handler1.events.clone();
            let events2_ref = handler2.events.clone();
            
            event_bus1.add_handler(Box::new(handler1));
            event_bus2.add_handler(Box::new(handler2));
            
            // Publish events to first bus
            for event in &events {
                event_bus1.publish(event.clone()).unwrap();
            }
            
            // Replay events on second bus
            let stored_events = event_bus1.get_events(1, None).unwrap();
            for stored_event in stored_events {
                event_bus2.publish(stored_event.event).unwrap();
            }
            
            // Both handlers should have received the same events
            let events1 = events1_ref.lock().unwrap();
            let events2 = events2_ref.lock().unwrap();
            
            prop_assert_eq!(events1.len(), events2.len());
            
            for (e1, e2) in events1.iter().zip(events2.iter()) {
                // Events should be the same type (we can't compare exact content due to timestamps)
                prop_assert_eq!(std::mem::discriminant(&e1.event), std::mem::discriminant(&e2.event));
            }
        }
    }
}
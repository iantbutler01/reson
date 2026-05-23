//! # reson-durable
//!
//! Durable execution primitives for Reson agent runtimes.
//!
//! The durable vocabulary is intentionally small:
//!
//! - [`Run`]
//! - [`Step`]
//! - [`State`]
//! - [`Effect`]
//! - [`Wait`]
//! - [`Event`]
//!
//! Storage, scheduling, and artifact transport are host responsibilities. This crate defines the
//! stable execution shape they implement against.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type Json = Value;

pub type RunId = Uuid;
pub type EventId = Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StepKey(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateRef(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EffectKey(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WaitKey(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Runnable,
    Running,
    Waiting,
    Finished,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaitStatus {
    Pending,
    Resolved,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub id: RunId,
    pub kind: String,
    pub version: u32,
    pub status: RunStatus,
    pub current_step: StepKey,
    pub current_state: StateRef,
    pub state_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub run_kind: String,
    pub run_version: u32,
    pub key: StepKey,
    pub input_schema_version: u32,
    pub output_schema_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub run_id: RunId,
    pub sequence: u64,
    pub step: StepKey,
    pub state_ref: StateRef,
    pub content_hash: ContentHash,
    pub schema_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Effect {
    pub run_id: RunId,
    pub key: EffectKey,
    pub kind: String,
    pub input_hash: ContentHash,
    pub status: EffectStatus,
    pub output_ref: Option<StateRef>,
    pub output_hash: Option<ContentHash>,
    pub error_ref: Option<StateRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wait {
    pub run_id: RunId,
    pub key: WaitKey,
    pub kind: String,
    pub status: WaitStatus,
    pub state_ref: StateRef,
    pub resolved_event_id: Option<EventId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub run_id: RunId,
    pub kind: String,
    pub payload_ref: StateRef,
    pub payload_hash: ContentHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepOutcome {
    Continue { next_step: StepKey, state: State },
    Wait { wait: Wait, state: State },
    Finished { state: State },
    Failed { state: State, error_ref: StateRef },
}

#[derive(Debug, thiserror::Error)]
pub enum DurableError {
    #[error("durable run {run_id} is at version {actual}, expected {expected}")]
    StaleRunVersion {
        run_id: RunId,
        expected: u64,
        actual: u64,
    },
    #[error("durable effect key {key:?} was reused with different input")]
    EffectInputMismatch { key: EffectKey },
    #[error("durable wait key {key:?} is not pending")]
    WaitNotPending { key: WaitKey },
    #[error("step {step:?} is not registered for {run_kind}@{run_version}")]
    StepNotRegistered {
        run_kind: String,
        run_version: u32,
        step: StepKey,
    },
    #[error("{0}")]
    Host(String),
}

pub type Result<T> = std::result::Result<T, DurableError>;

#[async_trait]
pub trait StepHandler: Send + Sync {
    fn step(&self) -> &Step;

    async fn run(&self, ctx: &mut dyn DurableContext, state: State) -> Result<StepOutcome>;
}

#[async_trait]
pub trait DurableContext: Send {
    async fn record_state(&mut self, state: State) -> Result<State>;

    async fn effect(&mut self, effect: Effect) -> Result<Effect>;

    async fn wait(&mut self, wait: Wait) -> Result<Wait>;

    async fn emit(&mut self, event: Event) -> Result<Event>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_run() {
        let run = Run {
            id: Uuid::new_v4(),
            kind: "conscious".to_string(),
            version: 1,
            status: RunStatus::Runnable,
            current_step: StepKey("start".to_string()),
            current_state: StateRef("nymfs://state/1".to_string()),
            state_version: 0,
        };

        let encoded = serde_json::to_string(&run).expect("serialize run");
        let decoded: Run = serde_json::from_str(&encoded).expect("deserialize run");
        assert_eq!(decoded, run);
    }
}

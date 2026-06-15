// @dive-file: Tracks spawned child process PIDs and dispatches centrally reaped exit status.
// @dive-rel: Used by portproxy/src/main.rs server lifecycle and services that spawn subprocesses.
// @dive-rel: Provides shared async PID registry for daemon/exec safety invariants.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::oneshot;

const MAX_UNCLAIMED_EXITS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildExit {
    Exited(i32),
    Signaled(i32),
}

impl ChildExit {
    pub fn protocol_code(self) -> i32 {
        match self {
            Self::Exited(code) => code,
            Self::Signaled(signal) => signal,
        }
    }

    pub fn interactive_code(self) -> i32 {
        match self {
            Self::Exited(code) => code,
            Self::Signaled(_) => -1,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("child pid {pid} exited before status could be delivered")]
pub struct ChildWaitError {
    pid: i32,
}

pub struct ChildWaiter {
    pid: i32,
    exit_rx: oneshot::Receiver<ChildExit>,
}

impl ChildWaiter {
    pub async fn wait(&mut self) -> Result<ChildExit, ChildWaitError> {
        (&mut self.exit_rx)
            .await
            .map_err(|_| ChildWaitError { pid: self.pid })
    }
}

#[derive(Clone, Default)]
pub struct ChildTracker {
    inner: Arc<std::sync::Mutex<ChildTrackerState>>,
}

#[derive(Default)]
struct ChildTrackerState {
    waiters: HashMap<i32, oneshot::Sender<ChildExit>>,
    unclaimed_exits: HashMap<i32, ChildExit>,
    unclaimed_order: VecDeque<i32>,
}

impl ChildTrackerState {
    fn remember_unclaimed_exit(&mut self, pid: i32, exit: ChildExit) {
        if self.unclaimed_exits.insert(pid, exit).is_some() {
            self.unclaimed_order.retain(|queued_pid| *queued_pid != pid);
        }
        self.unclaimed_order.push_back(pid);

        while self.unclaimed_order.len() > MAX_UNCLAIMED_EXITS {
            let Some(old_pid) = self.unclaimed_order.pop_front() else {
                break;
            };
            self.unclaimed_exits.remove(&old_pid);
        }
    }

    fn claim_unclaimed_exit(&mut self, pid: i32) -> Option<ChildExit> {
        let exit = self.unclaimed_exits.remove(&pid)?;
        self.unclaimed_order.retain(|queued_pid| *queued_pid != pid);
        Some(exit)
    }
}

impl ChildTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(ChildTrackerState::default())),
        }
    }

    pub fn register(&self, pid: i32) -> ChildWaiter {
        let (exit_tx, exit_rx) = oneshot::channel();
        let mut guard = self.inner.lock().expect("child tracker mutex poisoned");
        if let Some(exit) = guard.claim_unclaimed_exit(pid) {
            drop(guard);
            let _ = exit_tx.send(exit);
        } else {
            guard.waiters.insert(pid, exit_tx);
        }
        ChildWaiter { pid, exit_rx }
    }

    pub fn record_exit(&self, pid: i32, exit: ChildExit) -> bool {
        let exit_tx = {
            let mut guard = self.inner.lock().expect("child tracker mutex poisoned");
            match guard.waiters.remove(&pid) {
                Some(exit_tx) => Some(exit_tx),
                None => {
                    guard.remember_unclaimed_exit(pid, exit);
                    None
                }
            }
        };
        let Some(exit_tx) = exit_tx else {
            return false;
        };
        let _ = exit_tx.send(exit);
        true
    }

    pub fn snapshot(&self) -> Vec<i32> {
        let guard = self.inner.lock().expect("child tracker mutex poisoned");
        guard.waiters.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn waiter_receives_exit_recorded_after_registration() {
        let tracker = ChildTracker::new();
        let mut waiter = tracker.register(123);

        assert!(tracker.record_exit(123, ChildExit::Exited(7)));

        assert_eq!(waiter.wait().await.unwrap(), ChildExit::Exited(7));
    }

    #[tokio::test]
    async fn waiter_receives_exit_recorded_before_registration() {
        let tracker = ChildTracker::new();

        assert!(!tracker.record_exit(456, ChildExit::Exited(0)));
        let mut waiter = tracker.register(456);

        assert_eq!(waiter.wait().await.unwrap(), ChildExit::Exited(0));
    }

    #[test]
    fn snapshot_only_contains_waiting_children() {
        let tracker = ChildTracker::new();

        let _waiter = tracker.register(1);
        assert!(!tracker.record_exit(2, ChildExit::Exited(0)));

        assert_eq!(tracker.snapshot(), vec![1]);
    }
}

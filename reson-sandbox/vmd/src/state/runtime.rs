// @dive-file: In-memory runtime state for active VM processes and monitor handles.
// @dive-rel: Owned by vmd state manager lifecycle operations for running VM bookkeeping.
// @dive-rel: Carries process/socket paths and transient runtime flags not persisted to metadata.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Error;
use chrono::{DateTime, Utc};

use crate::fuse::FuseHandle;
use crate::state::types::VmState;
use crate::virt::{MonitorHandle, VirtiofsdHandle};

#[derive(Clone, Debug)]
pub struct VmRuntime {
    pub qmp_path: PathBuf,
    pub pid_path: PathBuf,
    pub state: VmState,
    pub started_at: Option<DateTime<Utc>>,
    pub monitor: Option<MonitorHandle>,
    pub command_pid: Option<u32>,
    pub exit_status: Arc<Mutex<Option<Error>>>,
    pub suspending: bool,
    /// Per-shared-mount virtiofsd children spawned alongside the qemu process. These
    /// are reaped on stop_vm / force_stop_vm; they also self-exit when qemu closes
    /// its vhost-user socket during a normal shutdown.
    pub virtiofsd_handles: Vec<VirtiofsdHandle>,
    pub fuse_handles: Vec<FuseHandle>,
}

impl VmRuntime {
    pub fn new(vm_dir: &PathBuf) -> Self {
        Self {
            qmp_path: vm_dir.join("qmp.sock"),
            pid_path: vm_dir.join("qemu.pid"),
            state: VmState::Stopped,
            started_at: None,
            monitor: None,
            command_pid: None,
            exit_status: Arc::new(Mutex::new(None)),
            suspending: false,
            virtiofsd_handles: Vec::new(),
            fuse_handles: Vec::new(),
        }
    }
}

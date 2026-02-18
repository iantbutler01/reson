use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use rand::RngCore;
use tokio::process::{Child, Command};
use tokio::sync::{OwnedMutexGuard, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::bootstrap;
use crate::config::{self, Config};
use crate::image::{self, BASE_IMAGE_EXT, BASE_IMAGE_SIZE_GB, PrebuiltImageStatus};
use crate::state::metadata::{load_metadata, save_metadata};
use crate::state::runtime::VmRuntime;
use crate::state::types::{
    CreateVmParams, ForkVmParams, NetworkSpec, SnapshotMetadata, SnapshotRecord, UpdateVmParams,
    Vm, VmInner, VmMetadata, VmSource, VmSourceType, VmState, new_snapshot_metadata,
    sanitize_name,
};
use crate::virt;

const ARCH_AMD64: &str = "amd64";
const ARCH_ARM64: &str = "arm64";
const FORK_BASES_DIR_NAME: &str = "_fork_bases";
const META_FORK_ID: &str = "reson.fork_id";
const META_FORK_BASE_PATH: &str = "reson.fork_base_path";
const META_PARENT_VM_ID: &str = "reson.parent_vm_id";
const META_FORK_SNAPSHOT: &str = "reson.fork_snapshot";

#[derive(Clone, Copy, Debug)]
pub enum CreateVmStage {
    DownloadImage,
    ConvertImage,
    StartVm,
}

#[derive(Clone, Debug)]
pub enum CreateVmProgressEvent {
    DownloadBytes {
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    StageProgress {
        stage: CreateVmStage,
        percent: u32,
        message: Option<String>,
    },
}

pub type CreateVmProgressCallback = Arc<dyn Fn(CreateVmProgressEvent) + Send + Sync>;

#[derive(thiserror::Error, Debug)]
pub enum ManagerError {
    #[error("vm not found")]
    VmNotFound,
    #[error("snapshot not found")]
    SnapshotNotFound,
    #[error("invalid VM state for operation")]
    InvalidState,
    #[error("operation cancelled")]
    Cancelled,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type ManagerResult<T> = Result<T, ManagerError>;

#[derive(Clone)]
pub struct SnapshotParams {
    pub label: String,
    pub description: String,
}

pub struct Manager {
    cfg: Config,
    host_arch: String,
    vms: RwLock<HashMap<String, Arc<Vm>>>,
    snapshots: RwLock<HashMap<String, SnapshotRecord>>,
}

impl Manager {
    #[instrument(skip(cfg))]
    pub async fn new(mut cfg: Config) -> ManagerResult<Self> {
        cfg.normalize().map_err(ManagerError::Other)?;
        let host_arch = normalize_arch(std::env::consts::ARCH)?;

        let manager = Self {
            cfg,
            host_arch,
            vms: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(HashMap::new()),
        };

        manager.discover().await?;
        Ok(manager)
    }

    pub fn host_architecture(&self) -> &str {
        &self.host_arch
    }

    pub fn normalize_architecture(&self, arch: &str) -> ManagerResult<String> {
        normalize_arch(arch)
    }

    async fn discover(&self) -> ManagerResult<()> {
        let entries = fs::read_dir(&self.cfg.data_dir)?;
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if entry.file_name() == config::BASE_IMAGES_DIR_NAME {
                continue;
            }
            if entry.file_name() == FORK_BASES_DIR_NAME {
                continue;
            }

            let vm_dir = entry.path();
            match load_metadata(&vm_dir) {
                Ok(mut meta) => {
                    if meta.id.is_empty() {
                        if let Some(name) = vm_dir.file_name().and_then(|n| n.to_str()) {
                            meta.id = name.to_string();
                        } else {
                            warn!(
                                dir = %vm_dir.display(),
                                "unable to determine VM ID from directory name"
                            );
                            continue;
                        }
                    }
                    if matches!(
                        meta.state,
                        VmState::Running | VmState::Paused | VmState::Creating
                    ) {
                        meta.state = VmState::Stopped;
                    }
                    if meta.architecture.is_empty() {
                        meta.architecture = self.host_arch.clone();
                    }

                    let vm = Arc::new(Vm::new(
                        meta.clone(),
                        VmRuntime::new(&vm_dir),
                        vm_dir.clone(),
                    ));
                    {
                        let mut inner = vm.lock().await;
                        inner.runtime.state = meta.state;
                    }

                    for snap in &meta.snapshots {
                        let record = SnapshotRecord {
                            vm_id: meta.id.clone(),
                            snapshot: snap.clone(),
                        };
                        self.snapshots
                            .write()
                            .await
                            .insert(record.snapshot.id.clone(), record);
                    }

                    let mut persisted = meta.clone();
                    if let Err(err) = save_metadata(&vm_dir, &mut persisted) {
                        warn!(
                            dir = %vm_dir.display(),
                            error = %err,
                            "failed to persist metadata during discovery"
                        );
                    }

                    self.vms.write().await.insert(meta.id.clone(), vm);
                }
                Err(err) => {
                    warn!(
                        dir = %vm_dir.display(),
                        error = %err,
                        "failed to load VM metadata during discovery"
                    );
                }
            }
        }
        Ok(())
    }

    pub async fn list(&self) -> Vec<VmMetadata> {
        let guard = self.vms.read().await;
        let mut vms = Vec::with_capacity(guard.len());
        for vm in guard.values() {
            let inner = vm.lock().await;
            vms.push(inner.metadata.clone());
        }
        vms
    }

    pub async fn get(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        let guard = vm.lock().await;
        Ok(guard.metadata.clone())
    }

    pub async fn get_with_runtime(&self, id: &str) -> ManagerResult<(VmMetadata, VmRuntime)> {
        let vm = self.vm_by_id(id).await?;
        let guard = vm.lock().await;
        Ok((guard.metadata.clone(), guard.runtime.clone()))
    }

    pub async fn create_vm(
        &self,
        params: CreateVmParams,
        progress: Option<CreateVmProgressCallback>,
    ) -> ManagerResult<VmMetadata> {
        let mut params = params;
        if params.resources.vcpu <= 0 {
            params.resources.vcpu = 1;
        }
        if params.resources.memory_mb <= 0 {
            params.resources.memory_mb = 1024;
        }
        if params.resources.disk_gb <= 0 {
            params.resources.disk_gb = 10;
        }
        let name = sanitize_name(&params.name);

        let mac = random_mac().map_err(ManagerError::Other)?;
        let requested_arch_str = normalize_arch(&params.architecture)?;
        let requested_arch = if requested_arch_str.is_empty() {
            None
        } else {
            Some(requested_arch_str)
        };

        let mut source_vm = None;
        let mut snapshot_record = None;
        let mut platform = None::<String>;

        let resolved_arch = match params.source.source_type {
            VmSourceType::Docker => {
                let (arch, plat) = self
                    .resolve_docker_platform(&params.source.reference, requested_arch.clone())
                    .await?;
                platform = Some(plat);
                Some(arch)
            }
            VmSourceType::Snapshot => {
                let record = self.snapshot_by_id(&params.source.reference).await?;
                let vm = self.vm_by_id(&record.vm_id).await?;
                {
                    let inner = vm.lock().await;
                    if let Some(req_arch) = &requested_arch {
                        if req_arch != &inner.metadata.architecture {
                            return Err(ManagerError::Other(anyhow!(
                                "snapshot architecture mismatch: requested {}, snapshot {}",
                                req_arch,
                                inner.metadata.architecture
                            )));
                        }
                    }
                    source_vm = Some(vm.clone());
                    snapshot_record = Some(record);
                    Some(inner.metadata.architecture.clone())
                }
            }
        };

        let arch = resolved_arch
            .or(requested_arch.clone())
            .unwrap_or_else(|| self.host_arch.clone());
        if arch != ARCH_AMD64 && arch != ARCH_ARM64 {
            return Err(ManagerError::Other(anyhow!(
                "unsupported architecture: {}",
                arch
            )));
        }

        let id = Uuid::new_v4().to_string();
        let vm_dir = PathBuf::from(&self.cfg.data_dir).join(&id);
        fs::create_dir_all(&vm_dir)?;

        let mut meta = VmMetadata {
            id: id.clone(),
            name,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            state: VmState::Creating,
            architecture: arch.clone(),
            source: VmSource {
                source_type: params.source.source_type.clone(),
                reference: params.source.reference.clone(),
            },
            resources: params.resources.clone(),
            network: NetworkSpec {
                mac,
                proxy_port: 0,
                rpc_port: 0,
            },
            metadata: params.metadata.clone(),
            snapshots: Vec::new(),
            suspended_snapshot: String::new(),
            suspended_boot_snapshot: String::new(),
            boot_snapshot: String::new(),
            started_at: None,
        };

        save_metadata(&vm_dir, &mut meta).map_err(ManagerError::Other)?;

        let runtime = VmRuntime::new(&vm_dir);
        let vm = Arc::new(Vm::new(meta.clone(), runtime, vm_dir.clone()));
        let mut vm_guard = vm.lock_owned().await;

        match params.source.source_type {
            VmSourceType::Docker => {
                vm_guard = self
                    .create_from_docker(
                        &vm,
                        vm_guard,
                        &params.source.reference,
                        &arch,
                        platform.as_deref(),
                        progress.clone(),
                    )
                    .await?;
            }
            VmSourceType::Snapshot => {
                if let (Some(source), Some(record)) = (source_vm, snapshot_record) {
                    vm_guard = self
                        .create_from_snapshot(&vm, vm_guard, &source, &record)
                        .await?;
                } else {
                    return Err(ManagerError::Other(anyhow!(
                        "snapshot source missing required metadata"
                    )));
                }
            }
        }

        vm_guard.metadata.state = VmState::Stopped;
        save_metadata(&vm.dir, &mut vm_guard.metadata).map_err(ManagerError::Other)?;

        let created_metadata = vm_guard.metadata.clone();
        let vm_name = created_metadata.name.clone();
        drop(vm_guard);

        self.vms.write().await.insert(id.clone(), vm.clone());

        if params.auto_start {
            emit_stage_progress(
                &progress,
                CreateVmStage::StartVm,
                0,
                format!("starting VM {}", vm_name.clone()),
            );
            let _ = self.start_vm(&id).await?;
            emit_stage_progress(
                &progress,
                CreateVmStage::StartVm,
                100,
                format!("VM {} started", vm_name),
            );
        } else {
            emit_stage_progress(
                &progress,
                CreateVmStage::StartVm,
                100,
                "auto-start disabled".to_string(),
            );
        }

        Ok(created_metadata)
    }

    #[instrument(skip(self))]
    pub async fn update_vm(&self, id: &str, params: UpdateVmParams) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        {
            let mut inner = vm.lock().await;
            if let Some(name) = params.name {
                inner.metadata.name = sanitize_name(&name);
            }
            if let Some(meta) = params.metadata {
                inner.metadata.metadata = meta;
            }
            save_metadata(&vm.dir, &mut inner.metadata).map_err(ManagerError::Other)?;
        }
        let updated = vm.lock().await.metadata.clone();
        Ok(updated)
    }

    pub async fn delete_vm(&self, id: &str, _purge_snapshots: bool) -> ManagerResult<()> {
        let vm = self.vm_by_id(id).await?;
        let fork_base_path = {
            let inner = vm.lock().await;
            inner.metadata.metadata.get(META_FORK_BASE_PATH).cloned()
        };
        {
            let inner = vm.lock().await;
            if matches!(inner.runtime.state, VmState::Running | VmState::Paused) {
                return Err(ManagerError::InvalidState);
            }
        }

        fs::remove_dir_all(&vm.dir)?;
        self.vms.write().await.remove(id);
        self.snapshots
            .write()
            .await
            .retain(|_, record| record.vm_id != id);
        self.cleanup_fork_base_if_unreferenced(fork_base_path).await;
        Ok(())
    }

    pub async fn fork_vm(
        &self,
        parent_id: &str,
        params: ForkVmParams,
    ) -> ManagerResult<(VmMetadata, VmMetadata, String)> {
        let parent_vm = self.vm_by_id(parent_id).await?;

        let parent_state = {
            let inner = parent_vm.lock().await;
            inner.runtime.state
        };
        let parent_was_running = matches!(parent_state, VmState::Running | VmState::Paused);

        if matches!(parent_state, VmState::Creating | VmState::Error) {
            return Err(ManagerError::InvalidState);
        }

        if matches!(parent_state, VmState::Running) {
            self.pause_vm(parent_id).await?;
        }

        let mut fork_snapshot_name = None::<String>;
        if parent_was_running {
            let snapshot = self
                .create_snapshot(
                    parent_id,
                    SnapshotParams {
                        label: format!("fork-{parent_id}"),
                        description: "reson fork point".to_string(),
                    },
                )
                .await?;
            fork_snapshot_name = Some(snapshot.name);
            self.force_stop_vm(parent_id).await?;
        }

        let (parent_name, parent_arch, parent_resources, parent_source, parent_metadata) = {
            let inner = parent_vm.lock().await;
            if !matches!(inner.runtime.state, VmState::Stopped) {
                return Err(ManagerError::InvalidState);
            }
            (
                inner.metadata.name.clone(),
                inner.metadata.architecture.clone(),
                inner.metadata.resources.clone(),
                inner.metadata.source.clone(),
                inner.metadata.metadata.clone(),
            )
        };

        let child_id = Uuid::new_v4().to_string();
        let fork_id = Uuid::new_v4().to_string();
        let child_name = sanitize_name(
            params
                .child_name
                .as_deref()
                .unwrap_or(&format!("{parent_name}-fork")),
        );
        let mut child_metadata = parent_metadata;
        child_metadata.extend(params.child_metadata);
        child_metadata.insert(META_FORK_ID.to_string(), fork_id.clone());
        child_metadata.insert(META_PARENT_VM_ID.to_string(), parent_id.to_string());
        child_metadata.remove(META_FORK_BASE_PATH);
        if let Some(snapshot_name) = fork_snapshot_name.as_ref() {
            child_metadata.insert(META_FORK_SNAPSHOT.to_string(), snapshot_name.clone());
        }

        let child_dir = PathBuf::from(&self.cfg.data_dir).join(&child_id);
        let parent_disk = parent_vm.disk_path();

        let mut child_meta = VmMetadata {
            id: child_id.clone(),
            name: child_name,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            state: VmState::Stopped,
            architecture: parent_arch.clone(),
            source: VmSource {
                source_type: parent_source.source_type,
                reference: parent_source.reference,
            },
            resources: parent_resources.clone(),
            network: NetworkSpec {
                mac: random_mac().map_err(ManagerError::Other)?,
                proxy_port: 0,
                rpc_port: 0,
            },
            metadata: child_metadata,
            snapshots: Vec::new(),
            suspended_snapshot: String::new(),
            suspended_boot_snapshot: String::new(),
            boot_snapshot: fork_snapshot_name.clone().unwrap_or_default(),
            started_at: None,
        };

        let parent_before_metadata = {
            let inner = parent_vm.lock().await;
            inner.metadata.clone()
        };

        let parent_after;
        if let Some(snapshot_name) = fork_snapshot_name.clone() {
            if let Err(err) = fs::create_dir_all(&child_dir) {
                let _ = self.start_vm(parent_id).await;
                return Err(ManagerError::Io(err));
            }

            let child_disk = child_dir.join("disk.qcow2");
            if let Err(err) = virt::clone_file_cow(parent_disk.as_path(), child_disk.as_path()).await {
                let _ = fs::remove_dir_all(&child_dir);
                let _ = self.start_vm(parent_id).await;
                return Err(ManagerError::Other(err));
            }

            let bootstrap_path = child_dir.join("bootstrap.iso");
            if let Err(err) = bootstrap::create_iso(
                &bootstrap_path,
                bootstrap::Config {
                    instance_id: child_id.clone(),
                    hostname: child_meta.name.clone(),
                    arch: parent_arch,
                },
            ) {
                let _ = fs::remove_dir_all(&child_dir);
                let _ = self.start_vm(parent_id).await;
                return Err(ManagerError::Other(err));
            }
            if let Err(err) = save_metadata(&child_dir, &mut child_meta) {
                let _ = fs::remove_dir_all(&child_dir);
                let _ = self.start_vm(parent_id).await;
                return Err(ManagerError::Other(err));
            }

            {
                let mut inner = parent_vm.lock().await;
                inner
                    .metadata
                    .metadata
                    .insert(META_FORK_ID.to_string(), fork_id.clone());
                inner
                    .metadata
                    .metadata
                    .insert(META_FORK_SNAPSHOT.to_string(), snapshot_name.clone());
                inner.metadata.metadata.remove(META_FORK_BASE_PATH);
                inner.metadata.boot_snapshot = snapshot_name;
                if let Err(err) = save_metadata(&parent_vm.dir, &mut inner.metadata) {
                    let _ = fs::remove_dir_all(&child_dir);
                    let _ = self.start_vm(parent_id).await;
                    return Err(ManagerError::Other(err));
                }
            }

            let started_parent = self.start_vm(parent_id).await;
            if let Err(err) = started_parent {
                let _ = fs::remove_dir_all(&child_dir);
                let mut inner = parent_vm.lock().await;
                inner.metadata = parent_before_metadata.clone();
                let _ = save_metadata(&parent_vm.dir, &mut inner.metadata);
                return Err(err);
            }

            {
                let mut inner = parent_vm.lock().await;
                inner.metadata.boot_snapshot.clear();
                if let Err(err) = save_metadata(&parent_vm.dir, &mut inner.metadata) {
                    warn!(
                        vm_id = %parent_id,
                        error = %err,
                        "failed to clear one-shot parent fork snapshot pointer"
                    );
                }
            }
            parent_after = self.get(parent_id).await?;
        } else {
            let fork_root = self.fork_base_root().join(&fork_id);
            fs::create_dir_all(&fork_root)?;
            let fork_base = fork_root.join("base.qcow2");
            fs::rename(&parent_disk, &fork_base)?;

            let parent_size_gb = parent_resources.disk_gb.max(BASE_IMAGE_SIZE_GB);
            if let Err(err) = virt::create_overlay(
                &self.cfg.qemu_img_bin,
                fork_base.as_path(),
                parent_disk.as_path(),
                parent_size_gb,
            )
            .await
            {
                Self::rollback_stopped_fork_artifacts(
                    parent_disk.as_path(),
                    fork_base.as_path(),
                    fork_root.as_path(),
                    child_dir.as_path(),
                );
                return Err(ManagerError::Other(err));
            }

            if let Err(err) = fs::create_dir_all(&child_dir) {
                Self::rollback_stopped_fork_artifacts(
                    parent_disk.as_path(),
                    fork_base.as_path(),
                    fork_root.as_path(),
                    child_dir.as_path(),
                );
                return Err(ManagerError::Io(err));
            }

            child_meta.metadata.insert(
                META_FORK_BASE_PATH.to_string(),
                fork_base.to_string_lossy().to_string(),
            );

            let child_disk = child_dir.join("disk.qcow2");
            let child_size_gb = child_meta.resources.disk_gb.max(BASE_IMAGE_SIZE_GB);
            if let Err(err) = virt::create_overlay(
                &self.cfg.qemu_img_bin,
                fork_base.as_path(),
                child_disk.as_path(),
                child_size_gb,
            )
            .await
            {
                Self::rollback_stopped_fork_artifacts(
                    parent_disk.as_path(),
                    fork_base.as_path(),
                    fork_root.as_path(),
                    child_dir.as_path(),
                );
                return Err(ManagerError::Other(err));
            }

            let bootstrap_path = child_dir.join("bootstrap.iso");
            if let Err(err) = bootstrap::create_iso(
                &bootstrap_path,
                bootstrap::Config {
                    instance_id: child_id.clone(),
                    hostname: child_meta.name.clone(),
                    arch: parent_arch,
                },
            ) {
                Self::rollback_stopped_fork_artifacts(
                    parent_disk.as_path(),
                    fork_base.as_path(),
                    fork_root.as_path(),
                    child_dir.as_path(),
                );
                return Err(ManagerError::Other(err));
            }
            if let Err(err) = save_metadata(&child_dir, &mut child_meta) {
                Self::rollback_stopped_fork_artifacts(
                    parent_disk.as_path(),
                    fork_base.as_path(),
                    fork_root.as_path(),
                    child_dir.as_path(),
                );
                return Err(ManagerError::Other(err));
            }

            {
                let mut inner = parent_vm.lock().await;
                inner
                    .metadata
                    .metadata
                    .insert(META_FORK_ID.to_string(), fork_id.clone());
                inner
                    .metadata
                    .metadata
                    .insert(META_FORK_BASE_PATH.to_string(), fork_base.to_string_lossy().to_string());
                inner.metadata.metadata.remove(META_FORK_SNAPSHOT);
                if let Err(err) = save_metadata(&parent_vm.dir, &mut inner.metadata) {
                    Self::rollback_stopped_fork_artifacts(
                        parent_disk.as_path(),
                        fork_base.as_path(),
                        fork_root.as_path(),
                        child_dir.as_path(),
                    );
                    return Err(ManagerError::Other(err));
                }
                parent_after = inner.metadata.clone();
            }
        }

        let child_vm = Arc::new(Vm::new(
            child_meta.clone(),
            VmRuntime::new(&child_dir),
            child_dir,
        ));
        self.vms.write().await.insert(child_id.clone(), child_vm);

        let child_after = if params.auto_start_child {
            self.start_vm(&child_id).await?
        } else {
            child_meta
        };

        Ok((parent_after, child_after, fork_id))
    }

    pub async fn list_snapshots(&self, vm_id: &str) -> ManagerResult<Vec<SnapshotMetadata>> {
        let vm = self.vm_by_id(vm_id).await?;
        let inner = vm.lock().await;
        Ok(inner.metadata.snapshots.clone())
    }

    pub async fn snapshot(
        &self,
        vm_id: &str,
        snapshot_id: &str,
    ) -> ManagerResult<SnapshotMetadata> {
        let vm = self.vm_by_id(vm_id).await?;
        let inner = vm.lock().await;
        inner
            .metadata
            .snapshots
            .iter()
            .find(|snap| snap.id == snapshot_id)
            .cloned()
            .ok_or(ManagerError::SnapshotNotFound)
    }

    pub async fn create_snapshot(
        &self,
        vm_id: &str,
        params: SnapshotParams,
    ) -> ManagerResult<SnapshotMetadata> {
        let vm = self.vm_by_id(vm_id).await?;
        let (state, monitor, disk_path) = {
            let inner = vm.lock().await;
            (
                inner.runtime.state,
                inner.runtime.monitor.clone(),
                vm.disk_path(),
            )
        };

        let mut meta = new_snapshot_metadata(params.label, params.description);

        match state {
            VmState::Running | VmState::Paused => {
                let monitor = monitor
                    .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?;
                virt::save_vm(&monitor, &meta.name)
                    .await
                    .map_err(ManagerError::Other)?;
            }
            VmState::Stopped => {
                virt::create_snapshot_offline(
                    &self.cfg.qemu_img_bin,
                    disk_path.as_path(),
                    &meta.name,
                )
                .await
                .map_err(ManagerError::Other)?;
                meta.disk_only = true;
            }
            _ => return Err(ManagerError::InvalidState),
        }

        {
            let mut inner = vm.lock().await;
            inner.metadata.snapshots.push(meta.clone());
            save_metadata(&vm.dir, &mut inner.metadata).map_err(ManagerError::Other)?;
        }
        self.snapshots.write().await.insert(
            meta.id.clone(),
            SnapshotRecord {
                vm_id: vm_id.to_string(),
                snapshot: meta.clone(),
            },
        );
        Ok(meta)
    }

    pub async fn delete_snapshot(&self, vm_id: &str, snapshot_id: &str) -> ManagerResult<()> {
        let vm = self.vm_by_id(vm_id).await?;
        let (state, monitor, disk_path, snapshot, boot_snapshot_match) = {
            let inner = vm.lock().await;
            let position = inner
                .metadata
                .snapshots
                .iter()
                .position(|snap| snap.id == snapshot_id)
                .ok_or(ManagerError::SnapshotNotFound)?;
            let snapshot = inner.metadata.snapshots[position].clone();
            let boot_match = inner.metadata.boot_snapshot == snapshot.name;
            (
                inner.runtime.state,
                inner.runtime.monitor.clone(),
                vm.disk_path(),
                snapshot,
                boot_match,
            )
        };

        match state {
            VmState::Running | VmState::Paused => {
                if let Some(monitor) = monitor {
                    if let Err(err) = virt::delete_snapshot(&monitor, &snapshot.name).await {
                        if !is_snapshot_not_found(&err) {
                            return Err(ManagerError::Other(err));
                        }
                    }
                }
            }
            VmState::Stopped => {
                virt::delete_snapshot_offline(
                    &self.cfg.qemu_img_bin,
                    disk_path.as_path(),
                    &snapshot.name,
                )
                .await
                .map_err(ManagerError::Other)?;
            }
            _ => return Err(ManagerError::InvalidState),
        }

        {
            let mut inner = vm.lock().await;
            inner
                .metadata
                .snapshots
                .retain(|snap| snap.id != snapshot_id);
            if boot_snapshot_match {
                inner.metadata.boot_snapshot.clear();
            }
            save_metadata(&vm.dir, &mut inner.metadata).map_err(ManagerError::Other)?;
        }

        self.snapshots.write().await.remove(snapshot_id);
        Ok(())
    }

    pub async fn start_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;

        let (binary, meta_snapshot, vm_dir, qmp_path, pid_path, log_path) = {
            let mut inner = vm.lock().await;
            if matches!(inner.runtime.state, VmState::Running) {
                return Ok(inner.metadata.clone());
            }

            let binary = self.qemu_binary_for_arch(&inner.metadata.architecture)?;

            let vm_dir = vm.dir.clone();
            let qmp_path = inner.runtime.qmp_path.clone();
            let pid_path = inner.runtime.pid_path.clone();
            let log_path = vm_dir.join("qemu.log");

            let _ = fs::remove_file(&qmp_path);
            let _ = fs::remove_file(&pid_path);

            let mut reserved = HashSet::new();
            inner.metadata.network.proxy_port =
                allocate_host_port(inner.metadata.network.proxy_port, &mut reserved)?;
            inner.metadata.network.rpc_port =
                allocate_host_port(inner.metadata.network.rpc_port, &mut reserved)?;

            inner.runtime.monitor = None;
            inner.runtime.command_pid = None;
            inner.runtime.started_at = None;
            if let Ok(mut exit) = inner.runtime.exit_status.lock() {
                *exit = None;
            }
            inner.runtime.state = VmState::Creating;
            inner.metadata.state = VmState::Creating;

            let snapshot = inner.metadata.clone();
            save_metadata(&vm.dir, &mut inner.metadata).map_err(ManagerError::Other)?;

            (binary, snapshot, vm_dir, qmp_path, pid_path, log_path)
        };

        let mut port_map = HashMap::new();
        port_map.insert(meta_snapshot.network.proxy_port, 13337);
        port_map.insert(meta_snapshot.network.rpc_port, 13338);

        let args = build_qemu_args(
            &meta_snapshot,
            vm_dir.as_path(),
            qmp_path.as_path(),
            pid_path.as_path(),
            &self.host_arch,
            &port_map,
        )
        .map_err(ManagerError::Other)?;
        debug!(
            vm_id = %id,
            binary = %binary,
            args = ?args,
            log_path = %log_path.display(),
            "starting qemu process"
        );

        let mut cmd = Command::new(&binary);
        cmd.args(&args);
        cmd.current_dir(&vm_dir);
        cmd.stdin(std::process::Stdio::null());

        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("open qemu log {}", log_path.display()))
            .map_err(ManagerError::Other)?;
        let stderr_file = log_file
            .try_clone()
            .with_context(|| format!("clone qemu log {}", log_path.display()))
            .map_err(ManagerError::Other)?;

        cmd.stdout(std::process::Stdio::from(log_file));
        cmd.stderr(std::process::Stdio::from(stderr_file));

        let child = match cmd
            .spawn()
            .with_context(|| format!("spawn qemu {}", binary))
        {
            Ok(child) => child,
            Err(err) => {
                mark_vm_state(&vm, VmState::Error).await;
                return Err(ManagerError::Other(err));
            }
        };

        let monitor =
            match virt::wait_for_monitor(qmp_path.as_path(), Duration::from_secs(20)).await {
                Ok(handle) => handle,
                Err(err) => return Err(abort_launch(vm.clone(), child, &log_path, err).await),
            };
        if let Err(err) = virt::wait_for_running(&monitor, Duration::from_secs(20)).await {
            return Err(abort_launch(vm.clone(), child, &log_path, err).await);
        }

        {
            let mut inner = vm.lock().await;
            inner.runtime.monitor = Some(monitor.clone());
            inner.runtime.state = VmState::Running;
            inner.runtime.started_at = Some(Utc::now());
            inner.runtime.command_pid = child.id();
            if let Ok(mut exit) = inner.runtime.exit_status.lock() {
                *exit = None;
            }
            inner.metadata.state = VmState::Running;
            inner.metadata.started_at = inner.runtime.started_at;
            if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
                warn!(vm_id = %id, error = %err, "failed to persist metadata after start");
            }
        }

        spawn_exit_task(vm.clone(), child, log_path);
        info!(vm_id = %id, "VM started");
        let meta = vm.lock().await.metadata.clone();
        Ok(meta)
    }

    pub async fn stop_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        let monitor = {
            let inner = vm.lock().await;
            match inner.runtime.state {
                VmState::Running => inner
                    .runtime
                    .monitor
                    .clone()
                    .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?,
                VmState::Paused => {
                    return Ok(inner.metadata.clone());
                }
                _ => return Ok(inner.metadata.clone()),
            }
        };

        virt::system_powerdown(&monitor)
            .await
            .map_err(ManagerError::Other)?;
        wait_for_exit(&vm, Duration::from_secs(60)).await?;

        let meta = vm.lock().await.metadata.clone();
        Ok(meta)
    }

    pub async fn restart_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        self.stop_vm(id).await?;
        self.start_vm(id).await
    }

    pub async fn pause_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        let monitor = {
            let inner = vm.lock().await;
            if !matches!(inner.runtime.state, VmState::Running) {
                return Ok(inner.metadata.clone());
            }
            inner
                .runtime
                .monitor
                .clone()
                .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?
        };

        virt::stop(&monitor).await.map_err(ManagerError::Other)?;

        {
            let mut inner = vm.lock().await;
            inner.runtime.state = VmState::Paused;
            inner.metadata.state = VmState::Paused;
            if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
                warn!(vm_id = %id, error = %err, "failed to persist metadata after pause");
            }
        }

        Ok(vm.lock().await.metadata.clone())
    }

    pub async fn resume_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        let monitor = {
            let inner = vm.lock().await;
            if !matches!(inner.runtime.state, VmState::Paused) {
                return Ok(inner.metadata.clone());
            }
            inner
                .runtime
                .monitor
                .clone()
                .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?
        };

        virt::cont(&monitor).await.map_err(ManagerError::Other)?;

        {
            let mut inner = vm.lock().await;
            inner.runtime.state = VmState::Running;
            inner.runtime.started_at = Some(Utc::now());
            inner.metadata.state = VmState::Running;
            inner.metadata.started_at = inner.runtime.started_at;
            if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
                warn!(vm_id = %id, error = %err, "failed to persist metadata after resume");
            }
        }

        Ok(vm.lock().await.metadata.clone())
    }

    pub async fn force_stop_vm(&self, id: &str) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(id).await?;
        let (monitor, pid) = {
            let inner = vm.lock().await;
            if !matches!(inner.runtime.state, VmState::Running | VmState::Paused) {
                return Ok(inner.metadata.clone());
            }
            let monitor = inner
                .runtime
                .monitor
                .clone()
                .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?;
            let pid = inner.runtime.command_pid.unwrap_or_default();
            (monitor, pid)
        };

        let quit_result = virt::quit(&monitor).await;
        if quit_result.is_err() {
            if pid > 0 {
                kill_process(pid).map_err(ManagerError::Other)?;
            }
        }

        wait_for_exit(&vm, Duration::from_secs(30)).await?;
        let meta = vm.lock().await.metadata.clone();
        Ok(meta)
    }

    pub async fn restore_snapshot(
        &self,
        vm_id: &str,
        snapshot_id: &str,
    ) -> ManagerResult<VmMetadata> {
        let vm = self.vm_by_id(vm_id).await?;
        let (state, snapshot, disk_path, monitor) = {
            let inner = vm.lock().await;
            let state = inner.runtime.state;
            let snapshot = inner
                .metadata
                .snapshots
                .iter()
                .find(|snap| snap.id == snapshot_id)
                .cloned()
                .ok_or(ManagerError::SnapshotNotFound)?;
            let disk_path = vm.disk_path();
            let monitor = inner.runtime.monitor.clone();
            (state, snapshot, disk_path, monitor)
        };

        match state {
            VmState::Running | VmState::Paused => {
                if snapshot.disk_only {
                    return Err(ManagerError::InvalidState);
                }
                let monitor = monitor
                    .ok_or_else(|| ManagerError::Other(anyhow!("vm monitor unavailable")))?;
                virt::stop(&monitor).await.map_err(ManagerError::Other)?;
                virt::load_vm(&monitor, &snapshot.name)
                    .await
                    .map_err(ManagerError::Other)?;
                virt::cont(&monitor).await.map_err(ManagerError::Other)?;

                let mut inner = vm.lock().await;
                inner.runtime.state = VmState::Running;
                inner.runtime.started_at = Some(Utc::now());
                inner.metadata.state = VmState::Running;
                inner.metadata.started_at = inner.runtime.started_at;
                if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
                    warn!(vm_id = %vm_id, error = %err, "persist metadata after restore failed");
                }
            }
            VmState::Stopped => {
                virt::revert_snapshot_offline(
                    &self.cfg.qemu_img_bin,
                    disk_path.as_path(),
                    &snapshot.name,
                )
                .await
                .map_err(ManagerError::Other)?;
                let mut inner = vm.lock().await;
                inner.runtime.state = VmState::Stopped;
                inner.metadata.state = VmState::Stopped;
                inner.metadata.started_at = None;
                if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
                    warn!(vm_id = %vm_id, error = %err, "persist metadata after restore failed");
                }
            }
            _ => return Err(ManagerError::InvalidState),
        }

        Ok(vm.lock().await.metadata.clone())
    }

    async fn create_from_docker(
        &self,
        vm: &Arc<Vm>,
        mut guard: OwnedMutexGuard<VmInner>,
        reference: &str,
        arch: &str,
        platform: Option<&str>,
        progress: Option<CreateVmProgressCallback>,
    ) -> ManagerResult<OwnedMutexGuard<VmInner>> {
        let vm_id = guard.metadata.id.clone();
        debug!(
            vm_id = %vm_id,
            reference = %reference,
            arch = %arch,
            platform = ?platform,
            "preparing VM disk from docker source"
        );

        let base_dir = PathBuf::from(&self.cfg.data_dir).join(config::BASE_IMAGES_DIR_NAME);
        fs::create_dir_all(&base_dir)?;

        let base_name = image::base_image_file_name(reference, arch);
        let base_path = base_dir.join(&base_name);
        let abs_base_path = base_path
            .canonicalize()
            .unwrap_or_else(|_| base_path.clone());

        emit_stage_progress(
            &progress,
            CreateVmStage::DownloadImage,
            0,
            format!("preparing base image for {reference}"),
        );

        let mut need_convert = true;
        if let Ok(metadata) = fs::metadata(&abs_base_path) {
            if metadata.len() > 0 {
                need_convert = false;
                debug!(
                    vm_id = %vm_id,
                    base = %abs_base_path.display(),
                    "using cached base image"
                );
                emit_stage_progress(
                    &progress,
                    CreateVmStage::DownloadImage,
                    100,
                    format!("using cached base image {}", abs_base_path.display()),
                );
            } else {
                let _ = fs::remove_file(&abs_base_path);
            }
        }

        if need_convert {
            if self.cfg.force_local_build {
                emit_stage_progress(
                    &progress,
                    CreateVmStage::DownloadImage,
                    100,
                    format!("skipping prebuilt image download for {reference}"),
                );
            } else {
                debug!(
                    vm_id = %vm_id,
                    reference = %reference,
                    arch = %arch,
                    "no cached base image, attempting to download or build"
                );
                emit_stage_progress(
                    &progress,
                    CreateVmStage::DownloadImage,
                    0,
                    format!("downloading {reference}"),
                );
                let dl_progress = progress.clone();
                let download_result = image::download_prebuilt_image_with_progress(
                    reference,
                    arch,
                    abs_base_path.as_path(),
                    move |update| {
                        emit_download_progress(
                            &dl_progress,
                            update.downloaded_bytes,
                            update.total_bytes,
                        );
                    },
                )
                .await;
                match download_result {
                    Ok(PrebuiltImageStatus::Downloaded { .. }) => {
                        need_convert = false;
                        debug!(
                            vm_id = %vm_id,
                            base = %abs_base_path.display(),
                            "downloaded prebuilt VM disk"
                        );
                        emit_stage_progress(
                            &progress,
                            CreateVmStage::DownloadImage,
                            100,
                            format!("downloaded VM disk to {}", abs_base_path.display()),
                        );
                    }
                    Ok(PrebuiltImageStatus::NotFound) => {
                        emit_stage_progress(
                            &progress,
                            CreateVmStage::DownloadImage,
                            100,
                            format!("no prebuilt VM disk available for {reference}"),
                        );
                    }
                    Err(err) => {
                        let err_msg = err.to_string();
                        warn!(
                            vm_id = %vm_id,
                            reference = %reference,
                            error = %err_msg,
                            "failed to download prebuilt VM image"
                        );
                        emit_stage_progress(
                            &progress,
                            CreateVmStage::DownloadImage,
                            100,
                            format!("failed to download prebuilt image: {err_msg}"),
                        );
                        return Err(ManagerError::Other(err));
                    }
                }
            }
        }

        if !need_convert {
            emit_stage_progress(
                &progress,
                CreateVmStage::ConvertImage,
                100,
                "base image ready".to_string(),
            );
        }

        if need_convert {
            emit_stage_progress(
                &progress,
                CreateVmStage::ConvertImage,
                0,
                format!("converting docker image {}", reference),
            );
            let tmp_name = format!(
                "{}-{}.{}",
                base_name.trim_end_matches(BASE_IMAGE_EXT),
                Uuid::new_v4(),
                BASE_IMAGE_EXT.trim_start_matches('.')
            );
            let tmp_path = base_dir.join(&tmp_name);
            let options = virt::D2VmOptions {
                image: reference.to_string(),
                output: tmp_name.clone(),
                disk_gb: BASE_IMAGE_SIZE_GB,
                pull: true,
                platform: platform.map(ToString::to_string),
                include_bootstrap: true,
            };
            debug!(
                vm_id = %vm_id,
                reference = %reference,
                output = %tmp_path.display(),
                arch = %arch,
                platform = ?options.platform,
                "running d2vm to convert VM disk"
            );
            virt::run_d2vm(&self.cfg.docker_bin, base_dir.as_path(), options)
                .await
                .map_err(ManagerError::Other)?;
            fs::rename(&tmp_path, &abs_base_path)?;
            debug!(
                vm_id = %vm_id,
                base = %abs_base_path.display(),
                "stored converted VM disk"
            );
            emit_stage_progress(
                &progress,
                CreateVmStage::ConvertImage,
                100,
                "finished converting VM disk".to_string(),
            );
        }

        if let Err(err) = fs::remove_file(vm.disk_path()) {
            if err.kind() != std::io::ErrorKind::NotFound {
                return Err(ManagerError::Other(err.into()));
            }
        }

        let size_gb = guard.metadata.resources.disk_gb.max(BASE_IMAGE_SIZE_GB);
        let overlay_path = vm.disk_path();
        debug!(
            vm_id = %vm_id,
            base = %abs_base_path.display(),
            overlay = %overlay_path.display(),
            size_gb,
            "creating overlay disk"
        );
        virt::create_overlay(
            &self.cfg.qemu_img_bin,
            abs_base_path.as_path(),
            overlay_path.as_path(),
            size_gb,
        )
        .await
        .map_err(ManagerError::Other)?;

        let bootstrap_path = vm.dir.join("bootstrap.iso");
        debug!(vm_id = %vm_id, path = %bootstrap_path.display(), "writing bootstrap seed image");
        let cfg = bootstrap::Config {
            instance_id: guard.metadata.id.clone(),
            hostname: guard.metadata.name.clone(),
            arch: arch.to_string(),
        };
        bootstrap::create_iso(&bootstrap_path, cfg).map_err(ManagerError::Other)?;

        guard.metadata.architecture = arch.to_string();
        save_metadata(&vm.dir, &mut guard.metadata).map_err(ManagerError::Other)?;

        Ok(guard)
    }

    async fn create_from_snapshot(
        &self,
        vm: &Arc<Vm>,
        mut guard: OwnedMutexGuard<VmInner>,
        source_vm: &Arc<Vm>,
        record: &SnapshotRecord,
    ) -> ManagerResult<OwnedMutexGuard<VmInner>> {
        let vm_id = guard.metadata.id.clone();
        debug!(
            vm_id = %vm_id,
            source_vm_id = %record.vm_id,
            snapshot_id = %record.snapshot.id,
            "restoring VM disk from snapshot"
        );
        let (snapshot, src_disk, dst_disk) = {
            let source = source_vm.lock().await;
            let src_snap = source
                .metadata
                .snapshots
                .iter()
                .find(|snap| snap.id == record.snapshot.id)
                .cloned()
                .ok_or(ManagerError::SnapshotNotFound)?;
            let src_disk = source_vm.disk_path();
            let dst_disk = vm.disk_path();
            (src_snap, src_disk, dst_disk)
        };

        virt::copy_file(&src_disk, &dst_disk)
            .await
            .map_err(ManagerError::Other)?;
        debug!(
            vm_id = %vm_id,
            path = %dst_disk.display(),
            "copied base disk from snapshot"
        );

        if !snapshot.disk_only {
            guard.metadata.boot_snapshot = snapshot.name;
            save_metadata(&vm.dir, &mut guard.metadata).map_err(ManagerError::Other)?;
        }

        Ok(guard)
    }

    fn qemu_binary_for_arch(&self, arch: &str) -> ManagerResult<String> {
        let normalized = if arch.trim().is_empty() {
            self.host_arch.clone()
        } else {
            normalize_arch(arch)?
        };
        match normalized.as_str() {
            ARCH_AMD64 => Ok(self.cfg.qemu_bin.clone()),
            ARCH_ARM64 => Ok(self.cfg.qemu_arm64_bin.clone()),
            other => Err(ManagerError::Other(anyhow!(
                "unsupported architecture: {other}"
            ))),
        }
    }

    async fn resolve_docker_platform(
        &self,
        reference: &str,
        requested: Option<String>,
    ) -> ManagerResult<(String, String)> {
        let platforms = virt::inspect_image_platforms(&self.cfg.docker_bin, reference)
            .await
            .map_err(ManagerError::Other)?;

        if let Some(req_arch) = requested {
            if !platform_supports_architecture(&platforms, &req_arch) {
                return Err(ManagerError::Other(anyhow!(
                    "image {reference} does not provide linux/{req_arch}"
                )));
            }
            Ok((req_arch.clone(), format!("linux/{req_arch}")))
        } else {
            let arch =
                choose_preferred_architecture(&platforms, &self.host_arch).ok_or_else(|| {
                    anyhow!("image {reference} does not provide a supported architecture")
                })?;
            Ok((arch.clone(), format!("linux/{arch}")))
        }
    }

    async fn vm_by_id(&self, id: &str) -> ManagerResult<Arc<Vm>> {
        let guard = self.vms.read().await;
        guard.get(id).cloned().ok_or(ManagerError::VmNotFound)
    }

    fn fork_base_root(&self) -> PathBuf {
        PathBuf::from(&self.cfg.data_dir).join(FORK_BASES_DIR_NAME)
    }

    fn rollback_stopped_fork_artifacts(
        parent_disk: &Path,
        fork_base: &Path,
        fork_root: &Path,
        child_dir: &Path,
    ) {
        let _ = fs::remove_dir_all(child_dir);
        let _ = fs::remove_file(parent_disk);
        if fork_base.exists() {
            let _ = fs::rename(fork_base, parent_disk);
        }
        let _ = fs::remove_dir(fork_root);
    }

    async fn cleanup_fork_base_if_unreferenced(&self, fork_base_path: Option<String>) {
        let Some(path) = fork_base_path else {
            return;
        };

        let vm_refs: Vec<_> = {
            let guard = self.vms.read().await;
            guard.values().cloned().collect()
        };
        for vm in vm_refs {
            let inner = vm.lock().await;
            if inner
                .metadata
                .metadata
                .get(META_FORK_BASE_PATH)
                .map(|entry| entry == &path)
                .unwrap_or(false)
            {
                return;
            }
        }

        let base_path = PathBuf::from(path);
        let _ = fs::remove_file(&base_path);
        if let Some(parent) = base_path.parent() {
            let _ = fs::remove_dir(parent);
        }
    }

    async fn snapshot_by_id(&self, id: &str) -> ManagerResult<SnapshotRecord> {
        let guard = self.snapshots.read().await;
        guard.get(id).cloned().ok_or(ManagerError::SnapshotNotFound)
    }
}

fn build_qemu_args(
    meta: &VmMetadata,
    vm_dir: &Path,
    qmp_path: &Path,
    pid_path: &Path,
    host_arch: &str,
    port_map: &HashMap<i32, i32>,
) -> Result<Vec<String>> {
    let guest_arch = if meta.architecture.trim().is_empty() {
        ARCH_AMD64
    } else {
        meta.architecture.as_str()
    };

    let (machine, cpu, bios) = match guest_arch {
        ARCH_AMD64 => {
            let machine = "q35,accel=kvm:tcg";
            let cpu = if host_arch == ARCH_AMD64 && cfg!(target_os = "linux") {
                "host"
            } else {
                "qemu64"
            };
            (machine.to_string(), cpu.to_string(), None)
        }
        ARCH_ARM64 => {
            let mut machine = "virt".to_string();
            let mut cpu = "cortex-a72".to_string();
            if host_arch == ARCH_ARM64 && cfg!(target_os = "linux") {
                machine = "virt,accel=kvm:tcg".to_string();
                cpu = "host".to_string();
            }
            (machine, cpu, Some("edk2-aarch64-code.fd".to_string()))
        }
        other => bail!("unsupported architecture: {other}"),
    };

    let disk_path = vm_dir.join("disk.qcow2");
    let mut netdev = String::from("user,id=net0");
    for (host_port, guest_port) in port_map {
        netdev.push_str(&format!(
            ",hostfwd=tcp:127.0.0.1:{}-:{}",
            host_port, guest_port
        ));
    }

    let mut args = Vec::new();
    args.push("-machine".to_string());
    args.push(machine);
    args.push("-cpu".to_string());
    args.push(cpu);
    args.push("-smp".to_string());
    args.push(meta.resources.vcpu.to_string());
    args.push("-m".to_string());
    args.push(meta.resources.memory_mb.to_string());
    args.push("-drive".to_string());
    args.push(format!(
        "file={},if=virtio,index=0,cache=none,aio=threads,format=qcow2",
        disk_path.display()
    ));
    args.push("-netdev".to_string());
    args.push(netdev);
    args.push("-device".to_string());
    args.push(format!(
        "virtio-net-pci,netdev=net0,mac={}",
        meta.network.mac
    ));
    args.push("-display".to_string());
    args.push("none".to_string());
    args.push("-serial".to_string());
    args.push("mon:stdio".to_string());
    args.push("-qmp".to_string());
    args.push(format!("unix:{},server=on,wait=off", qmp_path.display()));
    args.push("-pidfile".to_string());
    args.push(pid_path.display().to_string());

    if guest_arch == ARCH_ARM64 && host_arch != ARCH_ARM64 {
        args.push("-accel".to_string());
        args.push("tcg".to_string());
    }

    if let Some(bios) = bios {
        args.push("-bios".to_string());
        args.push(bios);
    }

    if !meta.boot_snapshot.is_empty() {
        args.push("-loadvm".to_string());
        args.push(meta.boot_snapshot.clone());
    }

    let bootstrap_iso = vm_dir.join("bootstrap.iso");
    if bootstrap_iso.exists() {
        args.push("-drive".to_string());
        args.push(format!(
            "file={},if=virtio,index=1,id=bootstrap,media=cdrom,readonly=on",
            bootstrap_iso.display()
        ));
    }

    Ok(args)
}

async fn mark_vm_state(vm: &Arc<Vm>, state: VmState) {
    let mut inner = vm.lock().await;
    inner.runtime.state = state;
    if state != VmState::Running {
        inner.runtime.started_at = None;
    }
    inner.metadata.state = state;
    inner.metadata.started_at = inner.runtime.started_at;
    if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
        warn!(vm_id = %inner.metadata.id, error = %err, "persist metadata failed");
    }
}

async fn abort_launch(
    vm: Arc<Vm>,
    mut child: Child,
    log_path: &Path,
    err: anyhow::Error,
) -> ManagerError {
    let _ = child.kill().await;
    let _ = child.wait().await;
    mark_vm_state(&vm, VmState::Error).await;
    let tail = read_log_tail(log_path, 8192);
    let message = if let Some(tail) = tail {
        format!("{err}: {tail}")
    } else {
        err.to_string()
    };
    ManagerError::Other(anyhow!(message))
}

async fn wait_for_exit(vm: &Arc<Vm>, timeout: Duration) -> ManagerResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        {
            let inner = vm.lock().await;
            match inner.runtime.state {
                VmState::Running | VmState::Paused => {}
                _ => return Ok(()),
            }
        }

        if Instant::now() >= deadline {
            return Err(ManagerError::Other(anyhow!(
                "timeout waiting for VM to stop"
            )));
        }
        sleep(Duration::from_millis(200)).await;
    }
}

fn allocate_host_port(preferred: i32, reserved: &mut HashSet<i32>) -> Result<i32> {
    if preferred > 0 && !reserved.contains(&preferred) {
        if port_available(preferred) {
            reserved.insert(preferred);
            return Ok(preferred);
        }
    }

    loop {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port() as i32;
        drop(listener);
        if reserved.contains(&port) {
            continue;
        }
        reserved.insert(port);
        return Ok(port);
    }
}

fn port_available(port: i32) -> bool {
    if port <= 0 {
        return false;
    }
    match TcpListener::bind(("127.0.0.1", port as u16)) {
        Ok(listener) => {
            drop(listener);
            true
        }
        Err(_) => false,
    }
}

fn read_log_tail(path: &Path, limit: usize) -> Option<String> {
    let mut file = OpenOptions::new().read(true).open(path).ok()?;
    let len = file.seek(SeekFrom::End(0)).ok()?;
    let start = if len as usize > limit {
        len - limit as u64
    } else {
        0
    };
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    Some(buf.trim().to_string())
}

fn spawn_exit_task(vm: Arc<Vm>, mut child: Child, log_path: PathBuf) {
    tokio::spawn(async move {
        let wait_result = child.wait().await;
        let tail = read_log_tail(&log_path, 8192);

        let (new_state, log_message, mut exit_error) = match wait_result {
            Ok(status) if status.success() => (VmState::Stopped, None, None),
            Ok(status) => {
                let err = anyhow!("qemu exited with status {status}");
                let msg = err.to_string();
                (VmState::Error, Some(msg), Some(err))
            }
            Err(err) => {
                let err = anyhow!(err);
                let msg = err.to_string();
                (VmState::Error, Some(msg), Some(err))
            }
        };

        let mut inner = vm.lock().await;
        let vm_id = inner.metadata.id.clone();
        inner.runtime.monitor = None;
        inner.runtime.command_pid = None;
        inner.runtime.started_at = None;

        {
            let mut exit_guard = inner
                .runtime
                .exit_status
                .lock()
                .expect("exit status lock poisoned");
            *exit_guard = exit_error.take();
        }

        inner.runtime.state = new_state;
        inner.metadata.state = new_state;
        inner.metadata.started_at = inner.runtime.started_at;

        if let Some(msg) = log_message {
            if let Some(tail) = &tail {
                error!(vm_id = %vm_id, error = %msg, tail = %tail, "VM exited with failure");
            } else {
                error!(vm_id = %vm_id, error = %msg, "VM exited with failure");
            }
        }

        if let Err(err) = save_metadata(&vm.dir, &mut inner.metadata) {
            warn!(vm_id = %vm_id, error = %err, "persist metadata after exit failed");
        }
    });
}

fn kill_process(pid: u32) -> Result<()> {
    if pid == 0 {
        return Ok(());
    }
    let result = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

fn normalize_arch(arch: &str) -> ManagerResult<String> {
    match arch.trim().to_lowercase().as_str() {
        "" => Ok(String::new()),
        "amd64" | "x86_64" => Ok(ARCH_AMD64.to_string()),
        "arm64" | "aarch64" => Ok(ARCH_ARM64.to_string()),
        other => Err(ManagerError::Other(anyhow!(
            "unsupported architecture: {other}"
        ))),
    }
}

fn random_mac() -> Result<String> {
    let mut bytes = [0u8; 6];
    let mut rng = rand::rng();
    rng.fill_bytes(&mut bytes);
    bytes[0] = (bytes[0] | 0x02) & 0xfe;
    Ok(format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]
    ))
}

fn platform_supports_architecture(platforms: &[virt::Platform], arch: &str) -> bool {
    platforms.iter().any(|p| {
        p.os.to_lowercase() == "linux"
            && normalize_arch(&p.arch).map(|a| a == arch).unwrap_or(false)
    })
}

fn choose_preferred_architecture(platforms: &[virt::Platform], host: &str) -> Option<String> {
    let mut preferred = Vec::new();
    if !host.is_empty() {
        preferred.push(host.to_string());
    }
    preferred.push(ARCH_AMD64.to_string());
    preferred.push(ARCH_ARM64.to_string());

    for candidate in preferred {
        if platform_supports_architecture(platforms, &candidate) {
            return Some(candidate);
        }
    }
    None
}

fn emit_download_progress(
    progress: &Option<CreateVmProgressCallback>,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
) {
    if let Some(callback) = progress {
        callback(CreateVmProgressEvent::DownloadBytes {
            downloaded_bytes,
            total_bytes,
        });
    }
}

fn emit_stage_progress(
    progress: &Option<CreateVmProgressCallback>,
    stage: CreateVmStage,
    percent: u32,
    message: impl Into<String>,
) {
    if let Some(callback) = progress {
        callback(CreateVmProgressEvent::StageProgress {
            stage,
            percent,
            message: Some(message.into()),
        });
    }
}

fn is_snapshot_not_found(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("snapshot") && msg.contains("not found")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qemu_img_available() -> bool {
        std::process::Command::new("qemu-img")
            .arg("--version")
            .output()
            .is_ok()
    }

    fn qemu_backing_file(path: &Path) -> Option<String> {
        let output = std::process::Command::new("qemu-img")
            .arg("info")
            .arg(path)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Some(rest) = line.trim().strip_prefix("backing file:") {
                let trimmed = rest.trim();
                let backing = trimmed
                    .split(" (actual")
                    .next()
                    .map(str::trim)
                    .unwrap_or_default();
                if !backing.is_empty() {
                    return Some(backing.to_string());
                }
            }
        }
        None
    }

    #[tokio::test]
    async fn fork_vm_stopped_parent_uses_shared_cow_backing() {
        if !qemu_img_available() {
            eprintln!("qemu-img unavailable; skipping fork CoW runtime test");
            return;
        }

        let test_arch = if cfg!(target_arch = "aarch64") {
            ARCH_ARM64
        } else {
            ARCH_AMD64
        };

        let tmp = tempfile::tempdir().expect("create tempdir");
        let data_dir = tmp.path().to_path_buf();
        fs::create_dir_all(data_dir.join(config::BASE_IMAGES_DIR_NAME)).expect("create base images");

        let cfg = Config {
            listen_address: "127.0.0.1:0".to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            qemu_bin: "qemu-system-x86_64".to_string(),
            qemu_arm64_bin: "qemu-system-aarch64".to_string(),
            qemu_img_bin: "qemu-img".to_string(),
            docker_bin: "docker".to_string(),
            log_level: "info".to_string(),
            force_local_build: false,
        };

        let manager = Manager {
            cfg,
            host_arch: test_arch.to_string(),
            vms: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(HashMap::new()),
        };

        let proxy_bin_dir = data_dir.join("proxy-bin");
        fs::create_dir_all(&proxy_bin_dir).expect("create proxy bin dir");
        fs::write(proxy_bin_dir.join("portproxy-linux-amd64"), b"stub-portproxy-amd64")
            .expect("write amd64 proxy stub");
        fs::write(proxy_bin_dir.join("portproxy-linux-arm64"), b"stub-portproxy-arm64")
            .expect("write arm64 proxy stub");
        unsafe {
            std::env::set_var("PROXY_BIN", &proxy_bin_dir);
        }

        let parent_id = "parent-vm".to_string();
        let parent_dir = data_dir.join(&parent_id);
        fs::create_dir_all(&parent_dir).expect("create parent dir");
        let parent_disk = parent_dir.join("disk.qcow2");
        let create_output = std::process::Command::new("qemu-img")
            .args(["create", "-f", "qcow2"])
            .arg(&parent_disk)
            .arg("1G")
            .output()
            .expect("run qemu-img create");
        assert!(
            create_output.status.success(),
            "qemu-img create failed: {}",
            String::from_utf8_lossy(&create_output.stderr)
        );

        let mut parent_meta = VmMetadata {
            id: parent_id.clone(),
            name: "parent".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            state: VmState::Stopped,
            architecture: test_arch.to_string(),
            source: VmSource {
                source_type: VmSourceType::Docker,
                reference: "mock/image:latest".to_string(),
            },
            resources: crate::state::ResourceSpec {
                vcpu: 1,
                memory_mb: 1024,
                disk_gb: 1,
            },
            network: NetworkSpec {
                mac: "02:00:00:00:00:01".to_string(),
                proxy_port: 0,
                rpc_port: 0,
            },
            metadata: HashMap::new(),
            snapshots: Vec::new(),
            suspended_snapshot: String::new(),
            suspended_boot_snapshot: String::new(),
            boot_snapshot: String::new(),
            started_at: None,
        };
        save_metadata(&parent_dir, &mut parent_meta).expect("save parent metadata");

        let parent_vm = Arc::new(Vm::new(
            parent_meta.clone(),
            VmRuntime::new(&parent_dir),
            parent_dir.clone(),
        ));
        manager
            .vms
            .write()
            .await
            .insert(parent_id.clone(), parent_vm);

        let mut child_meta_extra = HashMap::new();
        child_meta_extra.insert("reson.session_id".to_string(), "child-session".to_string());
        let (parent_after, child_after, fork_id) = manager
            .fork_vm(
                &parent_id,
                ForkVmParams {
                    child_name: Some("child".to_string()),
                    child_metadata: child_meta_extra,
                    auto_start_child: false,
                },
            )
            .await
            .expect("fork stopped parent");

        assert_eq!(parent_after.id, parent_id);
        assert_eq!(parent_after.state, VmState::Stopped);
        assert_eq!(child_after.state, VmState::Stopped);
        assert_ne!(child_after.id, parent_after.id);
        assert_eq!(
            child_after.metadata.get(META_PARENT_VM_ID).map(String::as_str),
            Some(parent_id.as_str())
        );
        assert_eq!(
            child_after.metadata.get(META_FORK_ID).map(String::as_str),
            Some(fork_id.as_str())
        );

        let fork_base_path = parent_after
            .metadata
            .get(META_FORK_BASE_PATH)
            .expect("parent metadata missing fork base path")
            .clone();
        assert!(
            Path::new(&fork_base_path).exists(),
            "expected fork base file to exist: {fork_base_path}"
        );

        let child_disk = data_dir.join(&child_after.id).join("disk.qcow2");
        assert!(parent_disk.exists(), "parent disk missing");
        assert!(child_disk.exists(), "child disk missing");

        let parent_backing =
            qemu_backing_file(&parent_disk).expect("parent overlay missing backing file");
        let child_backing =
            qemu_backing_file(&child_disk).expect("child overlay missing backing file");
        assert_eq!(
            parent_backing, child_backing,
            "parent and child must share same backing file"
        );
        assert_eq!(
            PathBuf::from(parent_backing),
            PathBuf::from(fork_base_path),
            "overlay backing should point to persisted fork base"
        );

        unsafe {
            std::env::remove_var("PROXY_BIN");
        }
    }

    #[tokio::test]
    async fn discover_normalizes_running_state_to_stopped() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let data_dir = tmp.path().to_path_buf();
        fs::create_dir_all(data_dir.join(config::BASE_IMAGES_DIR_NAME)).expect("create base images");

        let vm_id = "discover-vm".to_string();
        let vm_dir = data_dir.join(&vm_id);
        fs::create_dir_all(&vm_dir).expect("create vm dir");

        let mut meta = VmMetadata {
            id: vm_id.clone(),
            name: "discover".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            state: VmState::Running,
            architecture: ARCH_AMD64.to_string(),
            source: VmSource {
                source_type: VmSourceType::Docker,
                reference: "mock/image:latest".to_string(),
            },
            resources: crate::state::ResourceSpec {
                vcpu: 1,
                memory_mb: 1024,
                disk_gb: 10,
            },
            network: NetworkSpec {
                mac: "02:00:00:00:00:01".to_string(),
                proxy_port: 0,
                rpc_port: 0,
            },
            metadata: HashMap::new(),
            snapshots: Vec::new(),
            suspended_snapshot: String::new(),
            suspended_boot_snapshot: String::new(),
            boot_snapshot: String::new(),
            started_at: None,
        };
        save_metadata(&vm_dir, &mut meta).expect("save vm metadata");

        let cfg = Config {
            listen_address: "127.0.0.1:0".to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            qemu_bin: "qemu-system-x86_64".to_string(),
            qemu_arm64_bin: "qemu-system-aarch64".to_string(),
            qemu_img_bin: "qemu-img".to_string(),
            docker_bin: "docker".to_string(),
            log_level: "info".to_string(),
            force_local_build: false,
        };

        let manager = Manager {
            cfg,
            host_arch: ARCH_AMD64.to_string(),
            vms: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(HashMap::new()),
        };

        manager.discover().await.expect("discover VMs");
        let listed = manager.list().await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, vm_id);
        assert_eq!(listed[0].state, VmState::Stopped);

        let persisted = load_metadata(&vm_dir).expect("reload persisted metadata");
        assert_eq!(persisted.state, VmState::Stopped);
    }
}

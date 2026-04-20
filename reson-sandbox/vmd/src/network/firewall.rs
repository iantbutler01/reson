// @dive-file: Pod-local iptables enforcement for guest egress hard barriers keyed to the non-root qemu UID.
// @dive-rel: Installed once at daemon startup so slirp-originated guest traffic is forced through proxy/DNS carveouts and denied elsewhere.
use std::net::SocketAddr;
use std::process::Command;
use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use tracing::warn;

use crate::config::Config;

const FILTER_TABLE: &str = "filter";
const CHAIN_NAME: &str = "RESON_QEMU_EGRESS";
const ACCOUNT_CHAIN_NAME: &str = "RESON_QEMU_ACCOUNT";
const NAT_TABLE: &str = "nat";
const DNS_REDIRECT_CHAIN_NAME: &str = "RESON_QEMU_DNS_REDIRECT";
const PRIVATE_RANGES: [&str; 4] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
];
const WEBRTC_ICE_PORTS: &str = "19302,3478";
const WEBRTC_EPHEMERAL_PORT_RANGE: &str = "32768:65535";
const WEBRTC_NEW_FLOW_LIMIT: &str = "240/minute";
const WEBRTC_NEW_FLOW_BURST: &str = "48";

pub struct FirewallHandle {
    installed: bool,
    qemu_uid: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VmIptablesCounterSnapshot {
    pub new_connections: u64,
    pub total_bytes: u64,
}

impl FirewallHandle {
    pub fn shutdown(self) {
        if !self.installed {
            return;
        }
        if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 {
            return;
        }
        if let Err(err) = remove_output_jump(self.qemu_uid) {
            warn!(error = %err, "failed removing qemu firewall OUTPUT jump");
        }
        if let Err(err) = remove_account_output_jump(self.qemu_uid) {
            warn!(error = %err, "failed removing qemu accounting OUTPUT jump");
        }
        if let Err(err) = remove_dns_redirect_output_jump(self.qemu_uid) {
            warn!(error = %err, "failed removing qemu DNS redirect OUTPUT jump");
        }
        if let Err(err) = flush_account_chain() {
            warn!(error = %err, "failed flushing qemu accounting chain");
        }
        if let Err(err) = flush_chain() {
            warn!(error = %err, "failed flushing qemu firewall chain");
        }
        if let Err(err) = flush_dns_redirect_chain() {
            warn!(error = %err, "failed flushing qemu DNS redirect chain");
        }
        if let Err(err) = delete_chain() {
            warn!(error = %err, "failed deleting qemu firewall chain");
        }
        if let Err(err) = delete_account_chain() {
            warn!(error = %err, "failed deleting qemu accounting chain");
        }
        if let Err(err) = delete_dns_redirect_chain() {
            warn!(error = %err, "failed deleting qemu DNS redirect chain");
        }
    }
}

pub fn install(config: &Config) -> Result<FirewallHandle> {
    if !cfg!(target_os = "linux") {
        return Ok(FirewallHandle {
            installed: false,
            qemu_uid: config.qemu_process.run_as_uid,
        });
    }
    if unsafe { libc::geteuid() } != 0 {
        warn!("skipping guest firewall install because vmd is not running as root");
        return Ok(FirewallHandle {
            installed: false,
            qemu_uid: config.qemu_process.run_as_uid,
        });
    }

    let dns_addr = guest_dns_redirect_addr(config)
        .map(|addr| addr.parse::<SocketAddr>())
        .transpose()
        .with_context(|| {
            format!(
                "parse firewall dns resolver addr {}",
                config.network_services.coredns_bind_addr
            )
        })?;
    let proxy_addr = config
        .guest_network
        .http_proxy_upstream_addr
        .as_deref()
        .unwrap_or("127.0.0.1:3128")
        .parse::<SocketAddr>()
        .with_context(|| "parse firewall proxy upstream addr".to_string())?;

    if dns_addr.is_some() {
        create_dns_redirect_chain_if_missing()?;
    }
    create_account_chain_if_missing()?;
    create_chain_if_missing()?;
    if dns_addr.is_some() {
        flush_dns_redirect_chain()?;
    }
    flush_account_chain()?;
    flush_chain()?;
    if let Some(dns_addr) = dns_addr {
        append_dns_redirect_rule(dns_addr, "udp")?;
        append_dns_redirect_rule(dns_addr, "tcp")?;
    }
    for port in ["25,465,587", "6667,6697"] {
        append_rule(&[
            "-p",
            "tcp",
            "-m",
            "multiport",
            "--dports",
            port,
            "-j",
            "DROP",
        ])?;
        append_rule(&[
            "-p",
            "udp",
            "-m",
            "multiport",
            "--dports",
            port,
            "-j",
            "DROP",
        ])?;
    }

    // Allow replies on already-established hostfwd/control-plane connections before the
    // private-range deny rules. Without this, the guest can accept an inbound probe via the
    // vmd pod's forwarded port, but the reply packets back to the calling pod IP are dropped
    // as RFC1918 egress.
    append_rule(&[
        "-m",
        "conntrack",
        "--ctstate",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    ])?;

    for cidr in PRIVATE_RANGES {
        // Local/browser watch sessions often advertise host ICE candidates on RFC1918 or
        // carrier-grade NAT space. Allow only high-port UDP so WebRTC media can flow without
        // reopening private-range TCP access or low-port service scanning inside the cluster.
        append_rule(&[
            "-p",
            "udp",
            "-d",
            cidr,
            "--dport",
            WEBRTC_EPHEMERAL_PORT_RANGE,
            "-j",
            "ACCEPT",
        ])?;
        append_rule(&["-d", cidr, "-j", "DROP"])?;
    }

    if let Some(dns_addr) = dns_addr {
        append_rule(&[
            "-p",
            "udp",
            "-d",
            &dns_addr.ip().to_string(),
            "--dport",
            &dns_addr.port().to_string(),
            "-j",
            "ACCEPT",
        ])?;
        append_rule(&[
            "-p",
            "tcp",
            "-d",
            &dns_addr.ip().to_string(),
            "--dport",
            &dns_addr.port().to_string(),
            "-j",
            "ACCEPT",
        ])?;
    } else {
        append_rule(&["-p", "udp", "--dport", "53", "-j", "ACCEPT"])?;
        append_rule(&["-p", "tcp", "--dport", "53", "-j", "ACCEPT"])?;
    }
    append_rule(&[
        "-p",
        "tcp",
        "-d",
        &proxy_addr.ip().to_string(),
        "--dport",
        &proxy_addr.port().to_string(),
        "-j",
        "ACCEPT",
    ])?;

    append_rule(&[
        "-p",
        "tcp",
        "-m",
        "multiport",
        "--dports",
        "80,443",
        "-j",
        "DROP",
    ])?;
    append_rule(&[
        "-p",
        "udp",
        "-m",
        "multiport",
        "--dports",
        "80,443",
        "-j",
        "DROP",
    ])?;
    if dns_addr.is_some() {
        append_rule(&["-p", "udp", "--dport", "53", "-j", "DROP"])?;
        append_rule(&["-p", "tcp", "--dport", "53", "-j", "DROP"])?;
    }
    // The current watch surface uses Google's public STUN service in GKE and coturn on
    // 3478 in local dev. Keep those explicit and avoid reopening arbitrary low-port UDP.
    append_rule(&[
        "-p",
        "udp",
        "-m",
        "multiport",
        "--dports",
        WEBRTC_ICE_PORTS,
        "-j",
        "ACCEPT",
    ])?;
    // WebRTC media rides public high UDP ports. Limit only *new* flow fanout here; once
    // a peer path is established, the earlier ESTABLISHED,RELATED rule carries the media
    // stream without tripping this abuse guardrail.
    append_rule(&[
        "-p",
        "udp",
        "--dport",
        WEBRTC_EPHEMERAL_PORT_RANGE,
        "-m",
        "conntrack",
        "--ctstate",
        "NEW",
        "-m",
        "hashlimit",
        "--hashlimit-above",
        WEBRTC_NEW_FLOW_LIMIT,
        "--hashlimit-burst",
        WEBRTC_NEW_FLOW_BURST,
        "--hashlimit-mode",
        "dstip,dstport",
        "--hashlimit-name",
        "reson-qemu-webrtc-new",
        "-j",
        "DROP",
    ])?;
    append_rule(&[
        "-p",
        "udp",
        "--dport",
        WEBRTC_EPHEMERAL_PORT_RANGE,
        "-j",
        "ACCEPT",
    ])?;
    append_rule(&["-j", "DROP"])?;

    ensure_account_output_jump(config.qemu_process.run_as_uid)?;
    if dns_addr.is_some() {
        ensure_dns_redirect_output_jump(config.qemu_process.run_as_uid)?;
    } else {
        remove_dns_redirect_output_jump(config.qemu_process.run_as_uid)?;
    }
    ensure_output_jump(config.qemu_process.run_as_uid)?;

    Ok(FirewallHandle {
        installed: true,
        qemu_uid: config.qemu_process.run_as_uid,
    })
}

fn guest_dns_redirect_addr(config: &Config) -> Option<&str> {
    if config.guest_network.dns_server.is_some()
        || config.guest_network.http_proxy_upstream_addr.is_some()
    {
        Some(config.network_services.coredns_bind_addr.as_str())
    } else {
        None
    }
}

pub fn allow_proxy_listener(listen_addr: SocketAddr) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 {
        return Ok(());
    }
    let ip = listen_addr.ip().to_string();
    let port = listen_addr.port().to_string();
    let check_args = [
        "-t",
        FILTER_TABLE,
        "-C",
        CHAIN_NAME,
        "-p",
        "tcp",
        "-d",
        ip.as_str(),
        "--dport",
        port.as_str(),
        "-j",
        "ACCEPT",
    ];
    if Command::new("iptables")
        .args(check_args)
        .status()
        .context("check proxy listener allow rule")?
        .success()
    {
        return Ok(());
    }
    run_iptables([
        "-t",
        FILTER_TABLE,
        "-I",
        CHAIN_NAME,
        "1",
        "-p",
        "tcp",
        "-d",
        ip.as_str(),
        "--dport",
        port.as_str(),
        "-j",
        "ACCEPT",
    ])
    .context("install proxy listener allow rule")
}

pub fn revoke_proxy_listener(listen_addr: SocketAddr) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 {
        return Ok(());
    }
    let ip = listen_addr.ip().to_string();
    let port = listen_addr.port().to_string();
    let delete_args = [
        "-t",
        FILTER_TABLE,
        "-D",
        CHAIN_NAME,
        "-p",
        "tcp",
        "-d",
        ip.as_str(),
        "--dport",
        port.as_str(),
        "-j",
        "ACCEPT",
    ];
    let _ = Command::new("iptables").args(delete_args).status();
    Ok(())
}

pub fn block_vm_process(vm_id: &str, pid: u32, cgroup_path: Option<&str>) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 || pid == 0 {
        return Ok(());
    }
    let comment = vm_block_comment(vm_id);
    if let Some(cgroup_path) = cgroup_path.filter(|value| !value.trim().is_empty()) {
        let check_args = [
            "-t",
            FILTER_TABLE,
            "-C",
            CHAIN_NAME,
            "-m",
            "comment",
            "--comment",
            comment.as_str(),
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
            "-j",
            "DROP",
        ];
        if Command::new("iptables")
            .args(check_args)
            .status()
            .context("check vm-specific qemu DROP cgroup rule")?
            .success()
        {
            return Ok(());
        }

        let cgroup_rule = [
            "-t",
            FILTER_TABLE,
            "-I",
            CHAIN_NAME,
            "1",
            "-m",
            "comment",
            "--comment",
            comment.as_str(),
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
            "-j",
            "DROP",
        ];
        return run_iptables(cgroup_rule)
            .with_context(|| format!("install vm-specific qemu DROP rule for {vm_id}"));
    }

    let pid = pid.to_string();
    let check_args = [
        "-t",
        FILTER_TABLE,
        "-C",
        CHAIN_NAME,
        "-m",
        "comment",
        "--comment",
        comment.as_str(),
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
        "-j",
        "DROP",
    ];
    if Command::new("iptables")
        .args(check_args)
        .status()
        .context("check vm-specific qemu DROP rule")?
        .success()
    {
        return Ok(());
    }

    let pid_rule = [
        "-t",
        FILTER_TABLE,
        "-I",
        CHAIN_NAME,
        "1",
        "-m",
        "comment",
        "--comment",
        comment.as_str(),
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
        "-j",
        "DROP",
    ];
    run_iptables(pid_rule)
        .with_context(|| format!("install vm-specific qemu DROP rule for {vm_id}"))
}

pub fn unblock_vm_process(vm_id: &str, pid: u32, cgroup_path: Option<&str>) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 || pid == 0 {
        return Ok(());
    }
    let comment = vm_block_comment(vm_id);
    if let Some(cgroup_path) = cgroup_path.filter(|value| !value.trim().is_empty()) {
        let cgroup_args = [
            "-t",
            FILTER_TABLE,
            "-D",
            CHAIN_NAME,
            "-m",
            "comment",
            "--comment",
            comment.as_str(),
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
            "-j",
            "DROP",
        ];
        let _ = Command::new("iptables")
            .args(cgroup_args)
            .status()
            .context("remove vm-specific qemu DROP cgroup rule")?;
        return Ok(());
    }

    let pid = pid.to_string();
    let pid_args = [
        "-t",
        FILTER_TABLE,
        "-D",
        CHAIN_NAME,
        "-m",
        "comment",
        "--comment",
        comment.as_str(),
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
        "-j",
        "DROP",
    ];
    let _ = Command::new("iptables")
        .args(pid_args)
        .status()
        .context("remove vm-specific qemu DROP rule")?;
    Ok(())
}

pub fn install_vm_counter_rules(vm_id: &str, pid: u32, cgroup_path: Option<&str>) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 || pid == 0 {
        return Ok(());
    }
    install_counter_rule(
        vm_connection_comment(vm_id).as_str(),
        pid,
        cgroup_path,
        true,
    )?;
    install_counter_rule(vm_bytes_comment(vm_id).as_str(), pid, cgroup_path, false)
}

pub fn remove_vm_counter_rules(vm_id: &str, pid: u32, cgroup_path: Option<&str>) -> Result<()> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 || pid == 0 {
        return Ok(());
    }
    remove_counter_rule(
        vm_connection_comment(vm_id).as_str(),
        pid,
        cgroup_path,
        true,
    )?;
    remove_counter_rule(vm_bytes_comment(vm_id).as_str(), pid, cgroup_path, false)
}

pub fn vm_counter_snapshot(vm_id: &str) -> Result<Option<VmIptablesCounterSnapshot>> {
    if !cfg!(target_os = "linux") || unsafe { libc::geteuid() } != 0 {
        return Ok(None);
    }
    let output = Command::new("iptables-save")
        .args(["-c", "-t", FILTER_TABLE])
        .output()
        .context("read iptables accounting counters")?;
    if !output.status.success() {
        bail!("iptables-save failed with status {}", output.status);
    }
    let contents = String::from_utf8_lossy(&output.stdout);
    let new_connections = parse_counter_for_comment(
        contents.as_ref(),
        ACCOUNT_CHAIN_NAME,
        &vm_connection_comment(vm_id),
    )
    .unwrap_or_default()
    .0;
    let total_bytes = parse_counter_for_comment(
        contents.as_ref(),
        ACCOUNT_CHAIN_NAME,
        &vm_bytes_comment(vm_id),
    )
    .unwrap_or_default()
    .1;
    if new_connections == 0 && total_bytes == 0 {
        return Ok(None);
    }
    Ok(Some(VmIptablesCounterSnapshot {
        new_connections,
        total_bytes,
    }))
}

pub fn cgroup_path_for_pid(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    let path = format!("/proc/{pid}/cgroup");
    parse_cgroup_path(&fs::read_to_string(path).ok()?)
}

fn parse_cgroup_path(contents: &str) -> Option<String> {
    contents
        .lines()
        .filter_map(|line| line.rsplit_once(':').map(|(_, path)| path.trim()))
        .find(|path| !path.is_empty() && *path != "/")
        .map(|path| {
            if Path::new(path).is_absolute() {
                path.to_string()
            } else {
                format!("/{path}")
            }
        })
}

fn create_chain_if_missing() -> Result<()> {
    let status = Command::new("iptables")
        .args(["-t", FILTER_TABLE, "-N", CHAIN_NAME])
        .status()
        .context("create qemu firewall chain")?;
    if status.success() || chain_exists()? {
        return Ok(());
    }
    bail!("failed creating iptables chain {CHAIN_NAME}");
}

fn create_account_chain_if_missing() -> Result<()> {
    let status = Command::new("iptables")
        .args(["-t", FILTER_TABLE, "-N", ACCOUNT_CHAIN_NAME])
        .status()
        .context("create qemu accounting chain")?;
    if status.success() || account_chain_exists()? {
        return Ok(());
    }
    bail!("failed creating iptables chain {ACCOUNT_CHAIN_NAME}");
}

fn create_dns_redirect_chain_if_missing() -> Result<()> {
    let status = Command::new("iptables")
        .args(["-t", NAT_TABLE, "-N", DNS_REDIRECT_CHAIN_NAME])
        .status()
        .context("create qemu dns redirect chain")?;
    if status.success() || dns_redirect_chain_exists()? {
        return Ok(());
    }
    bail!("failed creating iptables chain {DNS_REDIRECT_CHAIN_NAME}");
}

fn flush_chain() -> Result<()> {
    run_iptables(["-t", FILTER_TABLE, "-F", CHAIN_NAME]).context("flush qemu firewall chain")
}

fn flush_account_chain() -> Result<()> {
    run_iptables(["-t", FILTER_TABLE, "-F", ACCOUNT_CHAIN_NAME])
        .context("flush qemu accounting chain")
}

fn flush_dns_redirect_chain() -> Result<()> {
    run_iptables(["-t", NAT_TABLE, "-F", DNS_REDIRECT_CHAIN_NAME])
        .context("flush qemu dns redirect chain")
}

fn delete_chain() -> Result<()> {
    run_iptables(["-t", FILTER_TABLE, "-X", CHAIN_NAME]).context("delete qemu firewall chain")
}

fn delete_account_chain() -> Result<()> {
    run_iptables(["-t", FILTER_TABLE, "-X", ACCOUNT_CHAIN_NAME])
        .context("delete qemu accounting chain")
}

fn delete_dns_redirect_chain() -> Result<()> {
    run_iptables(["-t", NAT_TABLE, "-X", DNS_REDIRECT_CHAIN_NAME])
        .context("delete qemu dns redirect chain")
}

fn chain_exists() -> Result<bool> {
    let status = Command::new("iptables")
        .args(["-t", FILTER_TABLE, "-S", CHAIN_NAME])
        .status()
        .context("inspect qemu firewall chain")?;
    Ok(status.success())
}

fn account_chain_exists() -> Result<bool> {
    let status = Command::new("iptables")
        .args(["-t", FILTER_TABLE, "-S", ACCOUNT_CHAIN_NAME])
        .status()
        .context("inspect qemu accounting chain")?;
    Ok(status.success())
}

fn dns_redirect_chain_exists() -> Result<bool> {
    let status = Command::new("iptables")
        .args(["-t", NAT_TABLE, "-S", DNS_REDIRECT_CHAIN_NAME])
        .status()
        .context("inspect qemu dns redirect chain")?;
    Ok(status.success())
}

fn append_rule(args: &[&str]) -> Result<()> {
    let mut command = vec!["-t", FILTER_TABLE, "-A", CHAIN_NAME];
    command.extend_from_slice(args);
    run_iptables(command).with_context(|| format!("append iptables rule to {CHAIN_NAME}"))
}

fn append_dns_redirect_rule(dns_addr: SocketAddr, protocol: &str) -> Result<()> {
    let dns_target = format!("{}:{}", dns_addr.ip(), dns_addr.port());
    run_iptables([
        "-t",
        NAT_TABLE,
        "-A",
        DNS_REDIRECT_CHAIN_NAME,
        "-p",
        protocol,
        "--dport",
        "53",
        "-j",
        "DNAT",
        "--to-destination",
        dns_target.as_str(),
    ])
    .with_context(|| format!("append {protocol} dns redirect rule to {DNS_REDIRECT_CHAIN_NAME}"))
}

fn ensure_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    let check_args = [
        "-t",
        FILTER_TABLE,
        "-C",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        CHAIN_NAME,
    ];
    if Command::new("iptables")
        .args(check_args)
        .status()
        .context("check qemu firewall OUTPUT jump")?
        .success()
    {
        return Ok(());
    }

    run_iptables([
        "-t",
        FILTER_TABLE,
        "-A",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        CHAIN_NAME,
    ])
    .context("install qemu firewall OUTPUT jump")
}

fn ensure_account_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    let check_args = [
        "-t",
        FILTER_TABLE,
        "-C",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        ACCOUNT_CHAIN_NAME,
    ];
    if Command::new("iptables")
        .args(check_args)
        .status()
        .context("check qemu accounting OUTPUT jump")?
        .success()
    {
        return Ok(());
    }

    run_iptables([
        "-t",
        FILTER_TABLE,
        "-I",
        "OUTPUT",
        "1",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        ACCOUNT_CHAIN_NAME,
    ])
    .context("install qemu accounting OUTPUT jump")
}

fn ensure_dns_redirect_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    let check_args = [
        "-t",
        NAT_TABLE,
        "-C",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-p",
        "udp",
        "--dport",
        "53",
        "-j",
        DNS_REDIRECT_CHAIN_NAME,
    ];
    if Command::new("iptables")
        .args(check_args)
        .status()
        .context("check qemu dns redirect OUTPUT jump")?
        .success()
    {
        return Ok(());
    }

    for protocol in ["udp", "tcp"] {
        run_iptables([
            "-t",
            NAT_TABLE,
            "-A",
            "OUTPUT",
            "-m",
            "owner",
            "--uid-owner",
            uid.as_str(),
            "-p",
            protocol,
            "--dport",
            "53",
            "-j",
            DNS_REDIRECT_CHAIN_NAME,
        ])
        .with_context(|| format!("install qemu dns redirect OUTPUT jump for {protocol}"))?;
    }
    Ok(())
}

fn remove_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    let args = [
        "-t",
        FILTER_TABLE,
        "-D",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        CHAIN_NAME,
    ];
    if !Command::new("iptables")
        .args(args)
        .status()
        .context("remove qemu firewall OUTPUT jump")?
        .success()
    {
        warn!("qemu firewall OUTPUT jump was already absent");
    }
    Ok(())
}

fn remove_account_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    let args = [
        "-t",
        FILTER_TABLE,
        "-D",
        "OUTPUT",
        "-m",
        "owner",
        "--uid-owner",
        uid.as_str(),
        "-j",
        ACCOUNT_CHAIN_NAME,
    ];
    let _ = Command::new("iptables")
        .args(args)
        .status()
        .context("remove qemu accounting OUTPUT jump")?;
    Ok(())
}

fn remove_dns_redirect_output_jump(qemu_uid: u32) -> Result<()> {
    let uid = qemu_uid.to_string();
    for protocol in ["udp", "tcp"] {
        let args = [
            "-t",
            NAT_TABLE,
            "-D",
            "OUTPUT",
            "-m",
            "owner",
            "--uid-owner",
            uid.as_str(),
            "-p",
            protocol,
            "--dport",
            "53",
            "-j",
            DNS_REDIRECT_CHAIN_NAME,
        ];
        let _ = Command::new("iptables")
            .args(args)
            .status()
            .context("remove qemu dns redirect OUTPUT jump")?;
    }
    Ok(())
}

fn run_iptables<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let status = Command::new("iptables")
        .args(args)
        .status()
        .context("run iptables command")?;
    if !status.success() {
        bail!("iptables command failed with status {status}");
    }
    Ok(())
}

fn vm_block_comment(vm_id: &str) -> String {
    format!("reson-vm-block:{vm_id}")
}

fn vm_connection_comment(vm_id: &str) -> String {
    format!("reson-vm-conn:{vm_id}")
}

fn vm_bytes_comment(vm_id: &str) -> String {
    format!("reson-vm-bytes:{vm_id}")
}

fn install_counter_rule(
    comment: &str,
    pid: u32,
    cgroup_path: Option<&str>,
    conntrack_new_only: bool,
) -> Result<()> {
    if let Some(cgroup_path) = cgroup_path.filter(|value| !value.trim().is_empty()) {
        let mut cgroup_check = vec![
            "-t",
            FILTER_TABLE,
            "-C",
            ACCOUNT_CHAIN_NAME,
            "-m",
            "comment",
            "--comment",
            comment,
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
        ];
        if conntrack_new_only {
            cgroup_check.extend(["-m", "conntrack", "--ctstate", "NEW"]);
        }
        cgroup_check.extend(["-j", "RETURN"]);
        if Command::new("iptables")
            .args(&cgroup_check)
            .status()
            .context("check vm accounting cgroup rule")?
            .success()
        {
            return Ok(());
        }

        let mut cgroup_install = vec![
            "-t",
            FILTER_TABLE,
            "-I",
            ACCOUNT_CHAIN_NAME,
            "1",
            "-m",
            "comment",
            "--comment",
            comment,
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
        ];
        if conntrack_new_only {
            cgroup_install.extend(["-m", "conntrack", "--ctstate", "NEW"]);
        }
        cgroup_install.extend(["-j", "RETURN"]);
        return run_iptables(cgroup_install);
    }

    let pid = pid.to_string();
    let mut check_rule = vec![
        "-t",
        FILTER_TABLE,
        "-C",
        ACCOUNT_CHAIN_NAME,
        "-m",
        "comment",
        "--comment",
        comment,
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
    ];
    if conntrack_new_only {
        check_rule.extend(["-m", "conntrack", "--ctstate", "NEW"]);
    }
    check_rule.extend(["-j", "RETURN"]);
    if Command::new("iptables")
        .args(&check_rule)
        .status()
        .context("check vm accounting rule")?
        .success()
    {
        return Ok(());
    }

    let mut install_rule = vec![
        "-t",
        FILTER_TABLE,
        "-I",
        ACCOUNT_CHAIN_NAME,
        "1",
        "-m",
        "comment",
        "--comment",
        comment,
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
    ];
    if conntrack_new_only {
        install_rule.extend(["-m", "conntrack", "--ctstate", "NEW"]);
    }
    install_rule.extend(["-j", "RETURN"]);
    run_iptables(install_rule)
}

fn remove_counter_rule(
    comment: &str,
    pid: u32,
    cgroup_path: Option<&str>,
    conntrack_new_only: bool,
) -> Result<()> {
    if let Some(cgroup_path) = cgroup_path.filter(|value| !value.trim().is_empty()) {
        let mut cgroup_rule = vec![
            "-t",
            FILTER_TABLE,
            "-D",
            ACCOUNT_CHAIN_NAME,
            "-m",
            "comment",
            "--comment",
            comment,
            "-m",
            "cgroup",
            "--path",
            cgroup_path,
        ];
        if conntrack_new_only {
            cgroup_rule.extend(["-m", "conntrack", "--ctstate", "NEW"]);
        }
        cgroup_rule.extend(["-j", "RETURN"]);
        let _ = Command::new("iptables")
            .args(&cgroup_rule)
            .status()
            .context("remove vm accounting cgroup rule")?;
        return Ok(());
    }

    let pid = pid.to_string();
    let mut pid_rule = vec![
        "-t",
        FILTER_TABLE,
        "-D",
        ACCOUNT_CHAIN_NAME,
        "-m",
        "comment",
        "--comment",
        comment,
        "-m",
        "owner",
        "--pid-owner",
        pid.as_str(),
    ];
    if conntrack_new_only {
        pid_rule.extend(["-m", "conntrack", "--ctstate", "NEW"]);
    }
    pid_rule.extend(["-j", "RETURN"]);
    let _ = Command::new("iptables")
        .args(&pid_rule)
        .status()
        .context("remove vm accounting rule")?;

    Ok(())
}

fn parse_counter_for_comment(
    contents: &str,
    chain_name: &str,
    comment: &str,
) -> Option<(u64, u64)> {
    for line in contents.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('[') || !trimmed.contains(chain_name) || !trimmed.contains(comment)
        {
            continue;
        }
        let end = trimmed.find(']')?;
        let counters = &trimmed[1..end];
        let (packets, bytes) = counters.split_once(':')?;
        return Some((packets.parse().ok()?, bytes.parse().ok()?));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GuestNetworkConfig, QemuProcessConfig};
    use std::net::IpAddr;

    #[test]
    fn install_uses_qemu_uid_and_local_proxy_dns_endpoints() {
        let mut cfg = Config::default();
        cfg.qemu_process = QemuProcessConfig {
            run_as_uid: 1000,
            run_as_gid: 1000,
        };
        cfg.network_services.coredns_bind_addr = "127.0.0.53:53".to_string();
        cfg.guest_network.http_proxy_upstream_addr = Some("127.0.0.1:3128".to_string());

        let dns_addr = cfg
            .network_services
            .coredns_bind_addr
            .parse::<SocketAddr>()
            .expect("dns addr");
        let proxy_addr = cfg
            .guest_network
            .http_proxy_upstream_addr
            .as_deref()
            .expect("proxy addr")
            .parse::<SocketAddr>()
            .expect("proxy addr parse");

        assert_eq!(dns_addr.ip(), IpAddr::from([127, 0, 0, 53]));
        assert_eq!(dns_addr.port(), 53);
        assert_eq!(proxy_addr.ip(), IpAddr::from([127, 0, 0, 1]));
        assert_eq!(proxy_addr.port(), 3128);
        assert_eq!(cfg.qemu_process.run_as_uid, 1000);
    }

    #[test]
    fn dns_redirect_uses_nat_chain_and_local_coredns_target() {
        let dns_addr: SocketAddr = "127.0.0.53:53".parse().expect("dns addr");
        let dns_target = format!("{}:{}", dns_addr.ip(), dns_addr.port());
        assert_eq!(NAT_TABLE, "nat");
        assert_eq!(DNS_REDIRECT_CHAIN_NAME, "RESON_QEMU_DNS_REDIRECT");
        assert_eq!(dns_target, "127.0.0.53:53");
    }

    #[test]
    fn guest_dns_redirect_addr_requires_guest_dns_or_proxy_configuration() {
        let mut cfg = Config::default();
        cfg.network_services.coredns_bind_addr = "127.0.0.53:53".to_string();
        cfg.guest_network = GuestNetworkConfig::default();
        assert_eq!(guest_dns_redirect_addr(&cfg), None);

        cfg.guest_network.dns_server = Some("10.0.2.3".to_string());
        assert_eq!(guest_dns_redirect_addr(&cfg), Some("127.0.0.53:53"));

        cfg.guest_network = GuestNetworkConfig::default();
        cfg.guest_network.http_proxy_upstream_addr = Some("127.0.0.1:3128".to_string());
        assert_eq!(guest_dns_redirect_addr(&cfg), Some("127.0.0.53:53"));
    }

    #[test]
    fn parse_cgroup_path_prefers_non_root_entry() {
        let parsed = parse_cgroup_path("0::/\n1:name=systemd:/kubepods.slice/pod123/vm.scope\n");
        assert_eq!(parsed.as_deref(), Some("/kubepods.slice/pod123/vm.scope"));
    }

    #[test]
    fn parse_counter_for_comment_extracts_packets_and_bytes() {
        let contents = r#"
[12:3456] -A RESON_QEMU_ACCOUNT -m comment --comment "reson-vm-conn:vm-1" -m owner --pid-owner 123 -m conntrack --ctstate NEW -j RETURN
[34:5678] -A RESON_QEMU_ACCOUNT -m comment --comment "reson-vm-bytes:vm-1" -m owner --pid-owner 123 -j RETURN
"#;
        assert_eq!(
            parse_counter_for_comment(contents, ACCOUNT_CHAIN_NAME, "reson-vm-conn:vm-1"),
            Some((12, 3456))
        );
        assert_eq!(
            parse_counter_for_comment(contents, ACCOUNT_CHAIN_NAME, "reson-vm-bytes:vm-1"),
            Some((34, 5678))
        );
    }
}

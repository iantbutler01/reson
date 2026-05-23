// @dive-file: Small Unix process-group helpers for guest command cleanup.
// @dive-rel: Used by services.rs, daemon.rs, and main.rs so timeouts and shutdowns reach child trees.

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;

#[cfg(unix)]
pub fn configure_child_process_group(command: &mut tokio::process::Command) {
    unsafe {
        command.pre_exec(|| {
            if nix::libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
pub fn configure_child_process_group(_command: &mut tokio::process::Command) {}

pub fn signal_process_group_or_pid(pid: i32, signal: Signal) -> Result<(), nix::errno::Errno> {
    if pid <= 0 {
        return Ok(());
    }
    match kill(Pid::from_raw(-pid), signal) {
        Ok(()) => Ok(()),
        Err(nix::errno::Errno::ESRCH) => kill(Pid::from_raw(pid), signal),
        Err(err) => Err(err),
    }
}

pub fn kill_process_group_or_child(pid: i32, child: &mut tokio::process::Child) {
    if pid > 0 && signal_process_group_or_pid(pid, Signal::SIGKILL).is_ok() {
        return;
    }
    let _ = child.start_kill();
}

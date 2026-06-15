use std::time::Duration;

use chevalier_sandbox::{
    ExecEvent, ExecOptions, OpenComputerBackendConfig, Sandbox, SandboxConfig,
    SandboxProviderConfig, SessionOptions, ShellEvent, ShellInput, ShellOptions,
};
use futures::StreamExt;
use tokio::time::timeout;

fn live_config() -> Option<SandboxConfig> {
    if std::env::var("OPENCOMPUTER_LIVE").ok().as_deref() != Some("1") {
        return None;
    }
    let opencomputer = OpenComputerBackendConfig::from_env().ok()?;
    Some(SandboxConfig {
        provider: SandboxProviderConfig::OpenComputer(opencomputer),
        ..SandboxConfig::default()
    })
}

async fn collect_exec_stdout(mut handle: chevalier_sandbox::ExecHandle) -> String {
    let mut stdout = Vec::new();
    while let Some(event) = handle.events.next().await {
        match event.expect("exec event") {
            ExecEvent::Stdout(bytes) => stdout.extend(bytes),
            ExecEvent::Stderr(_) => {}
            ExecEvent::Exit(code) => {
                assert_eq!(code, 0);
                break;
            }
            ExecEvent::Timeout => panic!("exec timed out"),
        }
    }
    String::from_utf8(stdout).expect("utf8 stdout")
}

async fn collect_shell_output(mut handle: chevalier_sandbox::ShellHandle) -> String {
    handle
        .input
        .send(ShellInput::Data(b"printf shell-ok\nexit\n".to_vec()))
        .await
        .expect("send shell command");
    let mut output = Vec::new();
    while let Some(event) = handle.events.next().await {
        match event.expect("shell event") {
            ShellEvent::Output(bytes) => output.extend(bytes),
            ShellEvent::Exit(code) => {
                assert_eq!(code, 0);
                break;
            }
        }
    }
    String::from_utf8(output).expect("utf8 shell output")
}

#[tokio::test(flavor = "multi_thread")]
async fn opencomputer_live_facade_smoke() {
    let Some(config) = live_config() else {
        eprintln!(
            "skipping OpenComputer live test; set OPENCOMPUTER_LIVE=1 and OPENCOMPUTER_API_KEY"
        );
        return;
    };

    let sandbox = Sandbox::new(config).await.expect("opencomputer sandbox");
    let session = sandbox
        .session(SessionOptions {
            name: Some("chevalier-opencomputer-live".to_string()),
            ..SessionOptions::default()
        })
        .await
        .expect("opencomputer session");

    let exec_output = timeout(
        Duration::from_secs(60),
        collect_exec_stdout(
            session
                .exec("printf exec-ok", ExecOptions::default())
                .await
                .expect("opencomputer exec"),
        ),
    )
    .await
    .expect("exec timeout");
    assert_eq!(exec_output, "exec-ok");

    session
        .write_file("/tmp/chevalier-opencomputer-live.txt", b"file-ok".to_vec())
        .await
        .expect("write file");
    let file = session
        .read_file("/tmp/chevalier-opencomputer-live.txt")
        .await
        .expect("read file");
    assert_eq!(file, b"file-ok");

    let shell_output = timeout(
        Duration::from_secs(60),
        collect_shell_output(
            session
                .shell(ShellOptions::default())
                .await
                .expect("opencomputer shell"),
        ),
    )
    .await
    .expect("shell timeout");
    assert!(shell_output.contains("shell-ok"));

    session.discard().await.expect("discard session");
}

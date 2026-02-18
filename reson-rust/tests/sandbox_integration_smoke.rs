use std::time::{SystemTime, UNIX_EPOCH};

use reson_sandbox::{ForkOptions, Sandbox, SandboxConfig, SessionOptions};

#[test]
fn sandbox_config_defaults_are_stable() {
    let cfg = SandboxConfig::default();
    assert!(cfg.connect_timeout.as_secs() > 0);
    assert!(!cfg.endpoint.is_empty());
}

#[tokio::test]
async fn sandbox_smoke_connect_when_enabled() {
    if std::env::var("RESON_SANDBOX_SMOKE").ok().as_deref() != Some("1") {
        return;
    }

    let endpoint = std::env::var("RESON_SANDBOX_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:8052".to_string());

    let cfg = SandboxConfig {
        auto_spawn: false,
        endpoint: endpoint.clone(),
        ..SandboxConfig::default()
    };

    let sandbox = Sandbox::connect(endpoint, cfg)
        .await
        .expect("failed to connect to sandbox daemon");

    let _ = sandbox
        .list_sessions()
        .await
        .expect("list_sessions should succeed");
}

#[tokio::test]
async fn sandbox_smoke_session_and_fork_when_enabled() {
    if std::env::var("RESON_SANDBOX_SMOKE").ok().as_deref() != Some("1") {
        return;
    }
    if std::env::var("RESON_SANDBOX_SMOKE_FORK").ok().as_deref() != Some("1") {
        return;
    }

    let endpoint = std::env::var("RESON_SANDBOX_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:8052".to_string());
    let image = std::env::var("RESON_SANDBOX_IMAGE")
        .unwrap_or_else(|_| "ghcr.io/bracketdevelopers/uv-builder:main".to_string());

    let cfg = SandboxConfig {
        auto_spawn: false,
        endpoint: endpoint.clone(),
        ..SandboxConfig::default()
    };

    let sandbox = Sandbox::connect(endpoint, cfg)
        .await
        .expect("failed to connect to sandbox daemon");

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time")
        .as_secs();
    let parent_session_id = format!("reson-rust-smoke-parent-{suffix}");

    let parent = sandbox
        .session(SessionOptions {
            session_id: Some(parent_session_id.clone()),
            name: Some(format!("reson-rust-parent-{suffix}")),
            image: Some(image),
            auto_start: true,
            ..SessionOptions::default()
        })
        .await
        .expect("create parent session");

    let fork = parent
        .fork(ForkOptions::default())
        .await
        .expect("fork parent session");

    let listed = sandbox
        .list_sessions()
        .await
        .expect("list sessions after fork");
    assert!(
        listed.iter().any(|s| s.session_id == parent_session_id),
        "parent session should be listed"
    );
    assert!(
        listed.iter().any(|s| s.session_id == fork.child_session_id),
        "child session should be listed"
    );

    parent.discard().await.expect("discard parent");
    fork.child.discard().await.expect("discard child");
}

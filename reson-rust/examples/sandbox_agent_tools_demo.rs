//! Agent-style sandbox demo with a true model-driven tool loop.
//!
//! This mirrors the production pattern used in cleoapp:
//! - call model with `RunParams { history, system, ... }`
//! - detect tool calls from model response
//! - execute tools
//! - inject `ToolCall` + `ToolResult` into history
//! - repeat until final assistant answer
//!
//! Required env for model backend:
//! - `RESON_AGENT_MODEL` (or `LOCAL_LLM`)
//! - plus provider auth as required by that model backend
//!
//! Run with:
//! `cargo run --example sandbox_agent_tools_demo`

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use reson_agentic::error::{Error as ResonError, Result as ResonResult};
use reson_agentic::runtime::{RunParams, Runtime, ToolFunction};
use reson_agentic::types::{ChatMessage, CreateResult, ToolCall, ToolResult};
use reson_agentic::utils::ConversationMessage;
use reson_sandbox::{
    ExecEvent, ExecHandle, ExecInput, ExecOptions, Sandbox, SandboxConfig, SandboxError, Session,
    SessionOptions,
};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use uuid::Uuid;

const MAX_TURNS: usize = 20;

#[derive(Clone, Debug, Default)]
struct ExecOutcome {
    stdout: String,
    stderr: String,
    exit_code: i32,
    timed_out: bool,
}

#[derive(Clone, Debug, Default)]
struct JobSnapshot {
    status: String,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    error: Option<String>,
}

type JobStore = Arc<RwLock<HashMap<String, Arc<RwLock<JobSnapshot>>>>>;

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs()
}

fn is_retryable_exec_error(err: &SandboxError) -> bool {
    match err {
        SandboxError::Grpc(status) => {
            let msg = status.message().to_lowercase();
            msg.contains("transport error")
                || msg.contains("connection reset")
                || msg.contains("connection refused")
        }
        _ => false,
    }
}

fn exec_ready_timeout_secs() -> u64 {
    std::env::var("RESON_EXEC_READY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(180)
}

fn resolve_agent_model() -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(model) = std::env::var("RESON_AGENT_MODEL") {
        if !model.trim().is_empty() {
            return Ok(model);
        }
    }

    if let Ok(model) = std::env::var("LOCAL_LLM") {
        if !model.trim().is_empty() {
            return Ok(model);
        }
    }

    Err("missing model config: set RESON_AGENT_MODEL or LOCAL_LLM".into())
}

async fn start_exec_with_retry(session: &Session, command: &str) -> Result<ExecHandle, SandboxError> {
    const EXEC_RETRY_DELAY_SECS: u64 = 2;
    let deadline = Instant::now() + Duration::from_secs(exec_ready_timeout_secs());

    loop {
        match session.exec(command, ExecOptions::default()).await {
            Ok(handle) => return Ok(handle),
            Err(err) if is_retryable_exec_error(&err) && Instant::now() < deadline => {
                eprintln!("exec stream not ready yet: {err}");
                tokio::time::sleep(Duration::from_secs(EXEC_RETRY_DELAY_SECS)).await;
            }
            Err(err) => return Err(err),
        }
    }
}

async fn exec_collect(session: &Session, command: &str) -> Result<ExecOutcome, SandboxError> {
    let handle = start_exec_with_retry(session, command).await?;
    let _ = handle.input.send(ExecInput::Eof).await;

    let mut outcome = ExecOutcome {
        exit_code: -1,
        ..ExecOutcome::default()
    };
    let mut events = handle.events;

    while let Some(event) = events.next().await {
        match event? {
            ExecEvent::Stdout(bytes) => outcome.stdout.push_str(&String::from_utf8_lossy(&bytes)),
            ExecEvent::Stderr(bytes) => outcome.stderr.push_str(&String::from_utf8_lossy(&bytes)),
            ExecEvent::Timeout => outcome.timed_out = true,
            ExecEvent::Exit(code) => {
                outcome.exit_code = code;
                break;
            }
        }
    }

    Ok(outcome)
}

fn tool_err(message: impl Into<String>) -> ResonError {
    ResonError::NonRetryable(message.into())
}

fn tool_output_json(tool_name: &str, raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => value,
        Err(_) => json!({
            "tool": tool_name,
            "raw": raw,
        }),
    }
}

fn response_to_assistant_text(response: &Value) -> String {
    response
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| serde_json::to_string_pretty(response).unwrap_or_else(|_| response.to_string()))
}

async fn register_sandbox_tools(runtime: &Runtime, session: Session, jobs: JobStore) -> ResonResult<()> {
    let session_for_exec = session.clone();
    runtime
        .register_tool_with_schema(
            "run_in_sandbox",
            "Run a shell command inside the sandbox VM and return stdout/stderr/exit status.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run inside the sandbox VM."
                    }
                },
                "required": ["command"]
            }),
            ToolFunction::Async(Box::new(move |args: Value| {
                let session = session_for_exec.clone();
                Box::pin(async move {
                    let command = args
                        .get("command")
                        .and_then(Value::as_str)
                        .ok_or_else(|| tool_err("run_in_sandbox requires string field `command`"))?;

                    let outcome = exec_collect(&session, command)
                        .await
                        .map_err(|err| tool_err(format!("sandbox exec failed: {err}")))?;

                    Ok(json!({
                        "command": command,
                        "exit_code": outcome.exit_code,
                        "timed_out": outcome.timed_out,
                        "stdout": outcome.stdout,
                        "stderr": outcome.stderr,
                    })
                    .to_string())
                })
            })),
        )
        .await?;

    let session_for_job_start = session.clone();
    let jobs_for_job_start = jobs.clone();
    runtime
        .register_tool_with_schema(
            "start_sandbox_job",
            "Start a long-running command in the sandbox and return a job_id for polling.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run asynchronously inside the sandbox VM."
                    }
                },
                "required": ["command"]
            }),
            ToolFunction::Async(Box::new(move |args: Value| {
                let session = session_for_job_start.clone();
                let jobs = jobs_for_job_start.clone();
                Box::pin(async move {
                    let command = args
                        .get("command")
                        .and_then(Value::as_str)
                        .ok_or_else(|| tool_err("start_sandbox_job requires string field `command`"))?
                        .to_string();
                    let job_id = Uuid::new_v4().to_string();

                    let snapshot = Arc::new(RwLock::new(JobSnapshot {
                        status: "starting".to_string(),
                        ..JobSnapshot::default()
                    }));
                    jobs.write().await.insert(job_id.clone(), snapshot.clone());

                    let handle = start_exec_with_retry(&session, &command)
                        .await
                        .map_err(|err| tool_err(format!("failed to start async sandbox job: {err}")))?;
                    let _ = handle.input.send(ExecInput::Eof).await;

                    {
                        let mut state = snapshot.write().await;
                        state.status = "running".to_string();
                    }

                    tokio::spawn(async move {
                        let mut events = handle.events;
                        while let Some(event) = events.next().await {
                            match event {
                                Ok(ExecEvent::Stdout(bytes)) => {
                                    let mut state = snapshot.write().await;
                                    state.stdout.push_str(&String::from_utf8_lossy(&bytes));
                                }
                                Ok(ExecEvent::Stderr(bytes)) => {
                                    let mut state = snapshot.write().await;
                                    state.stderr.push_str(&String::from_utf8_lossy(&bytes));
                                }
                                Ok(ExecEvent::Timeout) => {
                                    let mut state = snapshot.write().await;
                                    state.status = "failed".to_string();
                                    state.error = Some("job timed out".to_string());
                                }
                                Ok(ExecEvent::Exit(code)) => {
                                    let mut state = snapshot.write().await;
                                    state.exit_code = Some(code);
                                    state.status = if code == 0 {
                                        "completed".to_string()
                                    } else {
                                        "failed".to_string()
                                    };
                                    break;
                                }
                                Err(err) => {
                                    let mut state = snapshot.write().await;
                                    state.status = "failed".to_string();
                                    state.error = Some(err.to_string());
                                    break;
                                }
                            }
                        }
                    });

                    Ok(json!({
                        "job_id": job_id,
                        "status": "running",
                        "command": command,
                    })
                    .to_string())
                })
            })),
        )
        .await?;

    let jobs_for_poll = jobs.clone();
    runtime
        .register_tool_with_schema(
            "poll_sandbox_job",
            "Poll a previously started sandbox job by job_id and return status/output.",
            json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Identifier returned by start_sandbox_job."
                    }
                },
                "required": ["job_id"]
            }),
            ToolFunction::Async(Box::new(move |args: Value| {
                let jobs = jobs_for_poll.clone();
                Box::pin(async move {
                    let job_id = args
                        .get("job_id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| tool_err("poll_sandbox_job requires string field `job_id`"))?
                        .to_string();

                    let job = jobs
                        .read()
                        .await
                        .get(&job_id)
                        .cloned()
                        .ok_or_else(|| tool_err(format!("job_id not found: {job_id}")))?;
                    let snapshot = job.read().await.clone();

                    Ok(json!({
                        "job_id": job_id,
                        "status": snapshot.status,
                        "exit_code": snapshot.exit_code,
                        "stdout": snapshot.stdout,
                        "stderr": snapshot.stderr,
                        "error": snapshot.error,
                    })
                    .to_string())
                })
            })),
        )
        .await?;

    Ok(())
}

async fn discard_with_retry(
    session: Session,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    const DISCARD_RETRY_DELAY_SECS: u64 = 1;
    let discard_timeout_secs = std::env::var("RESON_DISCARD_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30);
    let deadline = Instant::now() + Duration::from_secs(discard_timeout_secs);

    loop {
        match session.clone().discard().await {
            Ok(()) => return Ok(()),
            Err(SandboxError::Grpc(status))
                if status.message().to_lowercase().contains("invalid vm state")
                    && Instant::now() < deadline =>
            {
                eprintln!("discard pending state transition for {label}: {status}");
                tokio::time::sleep(Duration::from_secs(DISCARD_RETRY_DELAY_SECS)).await;
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
}

fn collect_tool_call_values(runtime: &Runtime, response: &Value) -> Vec<Value> {
    let response_is_tool_array = response
        .as_array()
        .map(|arr| arr.iter().all(|value| runtime.is_tool_call(value)))
        .unwrap_or(false);

    let mut tool_call_values = Vec::new();
    if runtime.is_tool_call(response) {
        tool_call_values.push(response.clone());
    } else if response_is_tool_array {
        if let Some(arr) = response.as_array() {
            tool_call_values.extend(arr.iter().cloned());
        }
    }

    tool_call_values
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::var("RESON_SANDBOX_MODE").unwrap_or_else(|_| "local".to_string());
    let endpoint = std::env::var("RESON_SANDBOX_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:18072".to_string());
    let image = std::env::var("RESON_SANDBOX_IMAGE")
        .unwrap_or_else(|_| "ghcr.io/bracketdevelopers/uv-builder:main".to_string());
    let attach_session_id = std::env::var("RESON_ATTACH_SESSION_ID").ok();
    let model = resolve_agent_model()?;

    let mut cfg = SandboxConfig::default();
    cfg.connect_timeout = Duration::from_secs(20);
    cfg.daemon_start_timeout = Duration::from_secs(120);

    let sandbox = if mode == "remote" {
        println!("mode=remote endpoint={endpoint}");
        Sandbox::connect(endpoint.clone(), cfg).await?
    } else {
        let vmd_bin = std::env::var("RESON_VMD_BIN")
            .unwrap_or_else(|_| "../reson-sandbox/target/debug/vmd".to_string());
        let proxy_bin_dir = std::env::var("RESON_PROXY_BIN_DIR")
            .unwrap_or_else(|_| "../reson-sandbox/portproxy/bin".to_string());
        let data_dir = std::env::var("RESON_SANDBOX_DATA_DIR")
            .unwrap_or_else(|_| "/tmp/reson-agent-tools-vmd".to_string());

        unsafe {
            std::env::set_var("PROXY_BIN", &proxy_bin_dir);
        }

        cfg.endpoint = endpoint.clone();
        cfg.auto_spawn = true;
        cfg.daemon_listen = endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .to_string();
        cfg.daemon_bin = Some(PathBuf::from(vmd_bin));
        cfg.daemon_data_dir = Some(PathBuf::from(data_dir));

        println!("mode=local endpoint={} daemon_bin={:?}", cfg.endpoint, cfg.daemon_bin);
        Sandbox::new(cfg).await?
    };

    let suffix = unix_secs();
    let (session, attached_parent, session_id) = if let Some(existing) = attach_session_id {
        println!("attaching existing session {existing}");
        let attached = sandbox.attach_session(&existing).await?;
        (attached, true, existing)
    } else {
        let session_id = format!("agent-tools-parent-{suffix}");
        println!("creating parent session {session_id}");
        let created = sandbox
            .session(SessionOptions {
                session_id: Some(session_id.clone()),
                name: Some(format!("reson-agent-tools-{suffix}")),
                image: Some(image),
                auto_start: true,
                ..SessionOptions::default()
            })
            .await?;
        (created, false, session_id)
    };

    println!("session_id={} vm_id={}", session_id, session.vm_id());
    println!("agent_model={model}");

    let mut runtime = Runtime::new();
    let jobs: JobStore = Arc::new(RwLock::new(HashMap::new()));
    register_sandbox_tools(&runtime, session.clone(), jobs).await?;

    let system_prompt = r#"You are an execution agent operating a sandboxed VM.

You MUST use tools to complete the task.
Available tools:
- run_in_sandbox(command)
- start_sandbox_job(command)
- poll_sandbox_job(job_id)

Required plan:
1. Inspect the sandbox environment with a command that prints architecture and UTC time.
2. Start an async job that emits progress ticks and writes 'async-done' to /tmp/reson_async_job.txt.
3. Poll job status until it is completed.
4. Read /tmp/reson_async_job.txt.
5. Return a concise final report including command outputs.

Never claim completion without tool evidence."#;

    let user_task = "Execute the required plan now and provide a final report.";

    let mut history: Vec<ConversationMessage> = vec![ConversationMessage::Chat(ChatMessage::user(user_task))];
    let mut final_answer: Option<String> = None;

    for turn in 0..MAX_TURNS {
        println!("\n[agent] turn {}", turn + 1);

        let response = runtime
            .run(RunParams {
                system: Some(system_prompt.to_string()),
                history: Some(history.clone()),
                model: Some(model.clone()),
                timeout: Some(Duration::from_secs(180)),
                max_tokens: Some(4096),
                ..Default::default()
            })
            .await?;

        let tool_call_values = collect_tool_call_values(&runtime, &response);

        if tool_call_values.is_empty() {
            let assistant_text = response_to_assistant_text(&response);
            println!("[agent] final response:\n{}", assistant_text);
            history.push(ConversationMessage::Chat(ChatMessage::assistant(
                assistant_text.clone(),
            )));
            final_answer = Some(assistant_text);
            break;
        }

        for call_value in &tool_call_values {
            match ToolCall::create(call_value.clone()) {
                Ok(CreateResult::Single(tool_call)) => {
                    history.push(ConversationMessage::ToolCall(tool_call.clone()));

                    let execution_result = runtime.execute_tool(call_value).await;
                    let tool_result = match execution_result {
                        Ok(content) => {
                            println!("[tool:{}] {}", tool_call.tool_name, tool_output_json(&tool_call.tool_name, &content));
                            ToolResult::success_with_name(
                                tool_call.tool_use_id.clone(),
                                tool_call.tool_name.clone(),
                                content,
                            )
                            .with_tool_obj(tool_call.args.clone())
                        }
                        Err(err) => {
                            eprintln!("[tool:{}] execution failed: {}", tool_call.tool_name, err);
                            ToolResult::error(
                                tool_call.tool_use_id.clone(),
                                format!("Tool execution failed: {}", err),
                            )
                            .with_tool_name(tool_call.tool_name.clone())
                            .with_tool_obj(tool_call.args.clone())
                        }
                    };

                    history.push(ConversationMessage::ToolResult(tool_result));
                }
                Ok(CreateResult::Multiple(tool_calls)) => {
                    for tool_call in tool_calls {
                        history.push(ConversationMessage::ToolCall(tool_call.clone()));

                        let payload = json!({
                            "_tool_name": tool_call.tool_name,
                            "function": {
                                "arguments": serde_json::to_string(&tool_call.args)
                                    .unwrap_or_else(|_| "{}".to_string())
                            }
                        });

                        let execution_result = runtime.execute_tool(&payload).await;
                        let tool_result = match execution_result {
                            Ok(content) => ToolResult::success_with_name(
                                tool_call.tool_use_id.clone(),
                                tool_call.tool_name.clone(),
                                content,
                            )
                            .with_tool_obj(tool_call.args.clone()),
                            Err(err) => ToolResult::error(
                                tool_call.tool_use_id.clone(),
                                format!("Tool execution failed: {}", err),
                            )
                            .with_tool_name(tool_call.tool_name.clone())
                            .with_tool_obj(tool_call.args.clone()),
                        };

                        history.push(ConversationMessage::ToolResult(tool_result));
                    }
                }
                Ok(CreateResult::Empty) => {}
                Err(err) => {
                    eprintln!("[agent] failed to parse tool call payload: {err}");
                }
            }
        }
    }

    if final_answer.is_none() {
        return Err(format!("agent did not converge within {} turns", MAX_TURNS).into());
    }

    if attached_parent {
        println!("\nleaving attached session intact");
    } else {
        println!("\ncleaning up created session");
        discard_with_retry(session, "agent-tools-parent").await?;
    }

    println!("sandbox agent tool demo complete");
    Ok(())
}

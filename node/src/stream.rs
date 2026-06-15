//! Streaming: a `StreamHandle` that yields engine `ResponseStreamEvent`s as a
//! JS-friendly discriminated union. The Rust stream is driven by a spawned task
//! that owns the runtime lock guard + the stream together (avoiding a
//! self-referential struct), forwarding events over an mpsc channel.

use std::sync::Arc;

use chevalier_agentic::types::{ResponsePart, ResponseStreamEvent};
use napi_derive::napi;
use tokio::sync::mpsc::Receiver;
use tokio::sync::Mutex;

use crate::types::ToolCallJs;

/// A single streamed event. Discriminated by `type`:
/// `content` | `reasoning` | `signature` | `toolCall` | `toolPartial` | `usage` | `complete`.
#[napi(object)]
pub struct StreamEvent {
    #[napi(js_name = "type")]
    pub kind: String,
    /// Present for `content` / `reasoning` / `signature`.
    pub text: Option<String>,
    /// Present for `toolCall`.
    pub tool_call: Option<ToolCallJs>,
    /// Raw payload for `toolPartial` (partial args), `usage`, `complete` (full response).
    pub data: Option<serde_json::Value>,
}

impl From<ResponseStreamEvent> for StreamEvent {
    fn from(ev: ResponseStreamEvent) -> Self {
        let base = |kind: &str| StreamEvent {
            kind: kind.to_string(),
            text: None,
            tool_call: None,
            data: None,
        };
        match ev {
            ResponseStreamEvent::Output(part) => match part {
                ResponsePart::Text { text } => StreamEvent {
                    text: Some(text),
                    ..base("content")
                },
                ResponsePart::Reasoning { text } => StreamEvent {
                    text: Some(text),
                    ..base("reasoning")
                },
                ResponsePart::Signature { value } => StreamEvent {
                    text: Some(value),
                    ..base("signature")
                },
                ResponsePart::Tool { call } => StreamEvent {
                    tool_call: Some(ToolCallJs::from(&call)),
                    ..base("toolCall")
                },
            },
            ResponseStreamEvent::ToolPartial(v) => StreamEvent {
                data: Some(v),
                ..base("toolPartial")
            },
            ResponseStreamEvent::Usage(u) => StreamEvent {
                data: serde_json::to_value(u).ok(),
                ..base("usage")
            },
            ResponseStreamEvent::Complete(r) => StreamEvent {
                data: serde_json::to_value(r).ok(),
                ..base("complete")
            },
        }
    }
}

/// Async cursor over a streaming run. Call `next()` until it returns `null`.
/// The TS layer wraps this with `Symbol.asyncIterator` for `for await`.
#[napi]
pub struct StreamHandle {
    pub(crate) rx: Arc<Mutex<Receiver<Result<StreamEvent, String>>>>,
}

#[napi]
impl StreamHandle {
    /// The next event, or `null` when the stream is exhausted. Throws if the
    /// underlying run errored.
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<StreamEvent>> {
        let rx = self.rx.clone();
        let mut guard = rx.lock().await;
        match guard.recv().await {
            Some(Ok(ev)) => Ok(Some(ev)),
            Some(Err(msg)) => Err(napi::Error::new(napi::Status::GenericFailure, msg)),
            None => Ok(None),
        }
    }
}

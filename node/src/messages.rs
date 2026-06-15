//! Conversation message inputs (typed napi objects) → engine types.
//! Covers chat (user/assistant/system), tool results, and reasoning segments —
//! the message kinds a multi-turn agent loop needs. (Multimodal history is a
//! follow-up.)

use chevalier_agentic::types::{
    ChatMessage, ChatRole, MediaPart, MediaSource, MultimodalMessage, ReasoningSegment, ToolResult,
};
use chevalier_agentic::utils::ConversationMessage;
use napi_derive::napi;

/// A part of a multimodal message. `type` is `text` or `image`.
#[napi(object)]
pub struct MediaPartInput {
    #[napi(js_name = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub image_base64: Option<String>,
    pub mime_type: Option<String>,
    pub image_url: Option<String>,
}

/// A conversation message. `type` selects the variant:
/// - `chat` → `{ role: "user"|"assistant"|"system", content }`
/// - `toolResult` → `{ toolUseId, content, isError?, toolName? }`
/// - `reasoning` → `{ content }`
/// - `multimodal` → `{ parts: [{ type:"text", text } | { type:"image", imageBase64, mimeType }] }`
#[napi(object)]
pub struct Message {
    #[napi(js_name = "type")]
    pub kind: String,
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_use_id: Option<String>,
    pub tool_name: Option<String>,
    pub is_error: Option<bool>,
    pub parts: Option<Vec<MediaPartInput>>,
}

fn to_media_part(p: &MediaPartInput) -> MediaPart {
    match p.kind.as_str() {
        "image" => {
            if let Some(url) = &p.image_url {
                MediaPart::image(MediaSource::url(url.clone()))
            } else {
                MediaPart::image(MediaSource::base64(
                    p.image_base64.clone().unwrap_or_default(),
                    p.mime_type.clone().unwrap_or_else(|| "image/png".to_string()),
                ))
            }
        }
        _ => MediaPart::text(p.text.clone().unwrap_or_default()),
    }
}

fn role_from(s: &str) -> ChatRole {
    match s {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

/// Build a `ChatMessage` from a message (used for system-message prefixes).
pub fn to_chat_message(m: &Message) -> ChatMessage {
    let content = m.content.clone().unwrap_or_default();
    match role_from(m.role.as_deref().unwrap_or("user")) {
        ChatRole::System => ChatMessage::system(content),
        ChatRole::Assistant => ChatMessage::assistant(content),
        _ => ChatMessage::user(content),
    }
}

/// Build a `ConversationMessage` (history item) from a message.
pub fn to_conversation_message(m: &Message) -> ConversationMessage {
    match m.kind.as_str() {
        "toolResult" => {
            let id = m.tool_use_id.clone().unwrap_or_default();
            let content = m.content.clone().unwrap_or_default();
            let mut tr = if m.is_error.unwrap_or(false) {
                ToolResult::error(id, content)
            } else {
                ToolResult::success(id, content)
            };
            if let Some(name) = &m.tool_name {
                tr = tr.with_tool_name(name.clone());
            }
            ConversationMessage::ToolResult(tr)
        }
        "reasoning" => {
            ConversationMessage::Reasoning(ReasoningSegment::new(m.content.clone().unwrap_or_default()))
        }
        "multimodal" => {
            let parts = m
                .parts
                .as_ref()
                .map(|ps| ps.iter().map(to_media_part).collect())
                .unwrap_or_default();
            ConversationMessage::Multimodal(MultimodalMessage::user(parts))
        }
        _ => ConversationMessage::Chat(to_chat_message(m)),
    }
}

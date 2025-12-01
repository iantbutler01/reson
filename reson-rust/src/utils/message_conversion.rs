//! Message conversion utilities
//!
//! Converts mixed ChatMessage/ToolResult/ToolCall/ReasoningSegment lists to provider-specific formats.
//! Handles message coalescing for Anthropic/Google providers.

use crate::error::Result;
use crate::types::{ChatMessage, ChatRole, Provider, ReasoningSegment, ToolCall, ToolResult};
use serde_json::{json, Value};

/// Message type that can appear in conversation history
#[derive(Debug, Clone)]
pub enum ConversationMessage {
    /// Regular chat message (user, assistant, system)
    Chat(ChatMessage),
    /// Tool call from the assistant
    ToolCall(ToolCall),
    /// Tool result from execution
    ToolResult(ToolResult),
    /// Reasoning/thinking segment
    Reasoning(ReasoningSegment),
}

impl From<ChatMessage> for ConversationMessage {
    fn from(msg: ChatMessage) -> Self {
        ConversationMessage::Chat(msg)
    }
}

impl From<ToolCall> for ConversationMessage {
    fn from(tool_call: ToolCall) -> Self {
        ConversationMessage::ToolCall(tool_call)
    }
}

impl From<ToolResult> for ConversationMessage {
    fn from(result: ToolResult) -> Self {
        ConversationMessage::ToolResult(result)
    }
}

impl From<ReasoningSegment> for ConversationMessage {
    fn from(segment: ReasoningSegment) -> Self {
        ConversationMessage::Reasoning(segment)
    }
}

/// Convert messages to provider-specific format with coalescing
///
/// Provider-aware behavior:
/// - Anthropic/Bedrock/GoogleAnthropic: Coalesce consecutive ToolResults into a single user turn
///   with a content array of tool_result blocks. If the immediate next user ChatMessage exists,
///   append it as a trailing {"type":"text","text":"..."} block in the same message.
///
/// - Google GenAI: Coalesce consecutive ToolResults into a single user turn where 'content' is a list
///   of {"functionResponse": {...}} dicts. If the immediate next user ChatMessage exists, append
///   {"text": "..."} as a trailing entry.
///
/// - OpenAI/OpenRouter: Map each ToolResult to a proper tool-role message:
///   {"role":"tool","tool_call_id": tool_use_id, "content": result}
///   Keep subsequent user messages separate (no coalescing).
pub fn convert_messages_to_provider_format(
    messages: &[ConversationMessage],
    provider: Provider,
) -> Result<Vec<Value>> {
    let mut converted_messages: Vec<Value> = Vec::new();
    let mut pending_anthropic_blocks: Vec<Value> = Vec::new();
    let mut pending_google_parts: Vec<Value> = Vec::new();

    let flush_pending = |converted: &mut Vec<Value>,
                         anthropic_blocks: &mut Vec<Value>,
                         google_parts: &mut Vec<Value>| {
        match provider {
            Provider::Anthropic | Provider::Bedrock | Provider::GoogleAnthropic => {
                if !anthropic_blocks.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": anthropic_blocks.clone()
                    }));
                    anthropic_blocks.clear();
                }
            }
            Provider::GoogleGenAI => {
                if !google_parts.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": google_parts.clone()
                    }));
                    google_parts.clear();
                }
            }
            _ => {}
        }
    };

    for (_idx, msg) in messages.iter().enumerate() {
        match msg {
            ConversationMessage::Chat(chat_msg) => {
                // Check if this is a user message that should be merged with pending tool results
                if chat_msg.role == ChatRole::User
                    && matches!(
                        provider,
                        Provider::Anthropic
                            | Provider::Bedrock
                            | Provider::GoogleAnthropic
                            | Provider::GoogleGenAI
                    )
                {
                    // Merge with pending tool results
                    if matches!(
                        provider,
                        Provider::Anthropic | Provider::Bedrock | Provider::GoogleAnthropic
                    ) && !pending_anthropic_blocks.is_empty()
                    {
                        // Only add text block if content is non-empty
                        // (Anthropic rejects "text content blocks must be non-empty")
                        if !chat_msg.content.is_empty() {
                            pending_anthropic_blocks.push(json!({
                                "type": "text",
                                "text": chat_msg.content
                            }));
                        }
                        flush_pending(
                            &mut converted_messages,
                            &mut pending_anthropic_blocks,
                            &mut pending_google_parts,
                        );
                        continue;
                    } else if provider == Provider::GoogleGenAI && !pending_google_parts.is_empty()
                    {
                        // Only add text part if content is non-empty
                        if !chat_msg.content.is_empty() {
                            pending_google_parts.push(json!({
                                "text": chat_msg.content
                            }));
                        }
                        flush_pending(
                            &mut converted_messages,
                            &mut pending_anthropic_blocks,
                            &mut pending_google_parts,
                        );
                        continue;
                    }
                }

                // Flush pending before adding this message
                flush_pending(
                    &mut converted_messages,
                    &mut pending_anthropic_blocks,
                    &mut pending_google_parts,
                );

                // Add regular chat message
                let role = match chat_msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                };

                converted_messages.push(json!({
                    "role": role,
                    "content": chat_msg.content
                }));
            }

            ConversationMessage::ToolCall(tool_call) => {
                // Flush pending before adding tool call
                flush_pending(
                    &mut converted_messages,
                    &mut pending_anthropic_blocks,
                    &mut pending_google_parts,
                );

                // Convert tool call to provider-specific assistant message format
                converted_messages.push(tool_call.to_provider_assistant_message(provider));
            }

            ConversationMessage::ToolResult(tool_result) => {
                match provider {
                    Provider::OpenAI | Provider::OpenRouter => {
                        // OpenAI: Each ToolResult becomes a separate tool message
                        flush_pending(
                            &mut converted_messages,
                            &mut pending_anthropic_blocks,
                            &mut pending_google_parts,
                        );
                        converted_messages.push(tool_result.to_provider_format(provider));
                    }
                    Provider::Anthropic | Provider::Bedrock | Provider::GoogleAnthropic => {
                        // Anthropic: Accumulate tool results for coalescing
                        pending_anthropic_blocks.push(tool_result.to_provider_format(provider));
                    }
                    Provider::GoogleGenAI => {
                        // Google GenAI: Accumulate function responses
                        pending_google_parts.push(tool_result.to_provider_format(provider));
                    }
                }
            }

            ConversationMessage::Reasoning(reasoning_segment) => {
                // Flush pending before adding reasoning
                flush_pending(
                    &mut converted_messages,
                    &mut pending_anthropic_blocks,
                    &mut pending_google_parts,
                );

                // Add reasoning segment as-is (providers handle differently)
                converted_messages.push(reasoning_segment.to_provider_format(provider));
            }
        }
    }

    // Flush any trailing pending tool results
    flush_pending(
        &mut converted_messages,
        &mut pending_anthropic_blocks,
        &mut pending_google_parts,
    );

    Ok(converted_messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_no_coalescing() {
        // OpenAI should NOT coalesce tool results
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Use the weather tool")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_1", "Sunny, 72°F")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_2", "Rainy, 55°F")),
            ConversationMessage::Chat(ChatMessage::user("Thanks!")),
        ];

        let result = convert_messages_to_provider_format(&messages, Provider::OpenAI).unwrap();

        // Should be 4 separate messages
        assert_eq!(result.len(), 4);

        // First message: user
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "Use the weather tool");

        // Second message: tool result
        assert_eq!(result[1]["role"], "tool");
        assert_eq!(result[1]["tool_call_id"], "toolu_1");

        // Third message: tool result
        assert_eq!(result[2]["role"], "tool");
        assert_eq!(result[2]["tool_call_id"], "toolu_2");

        // Fourth message: user
        assert_eq!(result[3]["role"], "user");
        assert_eq!(result[3]["content"], "Thanks!");
    }

    #[test]
    fn test_anthropic_coalesces_tool_results() {
        // Anthropic SHOULD coalesce consecutive tool results
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Use the weather tool")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_1", "Sunny, 72°F")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_2", "Rainy, 55°F")),
        ];

        let result =
            convert_messages_to_provider_format(&messages, Provider::Anthropic).unwrap();

        // Should be 2 messages: original user message + coalesced tool results
        assert_eq!(result.len(), 2);

        // First message: user
        assert_eq!(result[0]["role"], "user");

        // Second message: coalesced tool results
        assert_eq!(result[1]["role"], "user");
        let content = result[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
        assert_eq!(content[1]["type"], "tool_result");
        assert_eq!(content[1]["tool_use_id"], "toolu_2");
    }

    #[test]
    fn test_anthropic_merges_trailing_user_message() {
        // Anthropic SHOULD merge trailing user message with tool results
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Use the weather tool")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_1", "Sunny, 72°F")),
            ConversationMessage::Chat(ChatMessage::user("Thanks!")),
        ];

        let result =
            convert_messages_to_provider_format(&messages, Provider::Anthropic).unwrap();

        // Should be 2 messages: original user + coalesced (tool result + trailing user text)
        assert_eq!(result.len(), 2);

        // Second message should have both tool result AND text
        let content = result[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "Thanks!");
    }

    #[test]
    fn test_google_genai_coalescing() {
        // Google GenAI should coalesce tool results as functionResponse parts
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Use the weather tool")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_1", "Sunny, 72°F")),
            ConversationMessage::ToolResult(ToolResult::success("toolu_2", "Rainy, 55°F")),
        ];

        let result =
            convert_messages_to_provider_format(&messages, Provider::GoogleGenAI).unwrap();

        // Should be 2 messages
        assert_eq!(result.len(), 2);

        // Second message: coalesced function responses
        assert_eq!(result[1]["role"], "user");
        let content = result[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert!(content[0].get("functionResponse").is_some());
        assert!(content[1].get("functionResponse").is_some());
    }

    #[test]
    fn test_google_genai_merges_trailing_user_message() {
        // Google GenAI should merge trailing user message as text part
        let messages = vec![
            ConversationMessage::ToolResult(ToolResult::success("toolu_1", "Sunny, 72°F")),
            ConversationMessage::Chat(ChatMessage::user("Thanks!")),
        ];

        let result =
            convert_messages_to_provider_format(&messages, Provider::GoogleGenAI).unwrap();

        // Should be 1 message with both functionResponse and text
        assert_eq!(result.len(), 1);

        let content = result[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert!(content[0].get("functionResponse").is_some());
        assert_eq!(content[1]["text"], "Thanks!");
    }

    #[test]
    fn test_system_message_preserved() {
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::system("You are helpful")),
            ConversationMessage::Chat(ChatMessage::user("Hello")),
        ];

        let result = convert_messages_to_provider_format(&messages, Provider::OpenAI).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[0]["content"], "You are helpful");
    }

    #[test]
    fn test_reasoning_segments_not_coalesced() {
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Think about this")),
            ConversationMessage::Reasoning(ReasoningSegment::with_index("Hmm...", 0)),
            ConversationMessage::Chat(ChatMessage::assistant("I understand")),
        ];

        let result =
            convert_messages_to_provider_format(&messages, Provider::Anthropic).unwrap();

        // Reasoning segments should be separate messages
        assert_eq!(result.len(), 3);
        assert_eq!(result[1]["type"], "thinking");
    }
}

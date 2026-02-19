//! OpenAI/OpenRouter Responses streaming parser
//!
//! Parses Responses API SSE events into StreamChunk values.

use crate::providers::StreamChunk;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct ResponsesToolAccumulator {
    current_tool_calls: HashMap<usize, PartialToolCall>,
}

#[derive(Debug, Clone)]
struct PartialToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

impl ResponsesToolAccumulator {
    pub fn new() -> Self {
        Self {
            current_tool_calls: HashMap::new(),
        }
    }

    pub fn start_tool(&mut self, index: usize, call_id: Option<&str>, name: Option<&str>) {
        let entry = self
            .current_tool_calls
            .entry(index)
            .or_insert_with(|| PartialToolCall {
                call_id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });

        if let Some(id) = call_id {
            entry.call_id = id.to_string();
        }
        if let Some(tool_name) = name {
            entry.name = tool_name.to_string();
        }
    }

    pub fn append_args(&mut self, index: usize, delta: &str) {
        let entry = self
            .current_tool_calls
            .entry(index)
            .or_insert_with(|| PartialToolCall {
                call_id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        entry.arguments.push_str(delta);
    }

    pub fn set_args(&mut self, index: usize, args: &str) {
        let entry = self
            .current_tool_calls
            .entry(index)
            .or_insert_with(|| PartialToolCall {
                call_id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        entry.arguments = args.to_string();
    }

    pub fn tool_partial(&self, index: usize) -> Option<Value> {
        self.current_tool_calls.get(&index).and_then(|tool| {
            if tool.name.is_empty() {
                return None;
            }
            let call_id = if tool.call_id.is_empty() {
                format!("call_{}", index)
            } else {
                tool.call_id.clone()
            };

            Some(serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": tool.arguments
                }
            }))
        })
    }

    pub fn has_name(&self, index: usize) -> bool {
        self.current_tool_calls
            .get(&index)
            .map(|tool| !tool.name.is_empty())
            .unwrap_or(false)
    }

    pub fn complete_tool(&mut self, index: usize) -> Option<Value> {
        self.current_tool_calls.remove(&index).map(|tool| {
            let call_id = if tool.call_id.is_empty() {
                format!("call_{}", index)
            } else {
                tool.call_id
            };

            serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": tool.arguments
                }
            })
        })
    }
}

/// Parse a Responses API SSE event into StreamChunks.
pub fn parse_openai_responses_event(
    event_json: &Value,
    accumulator: &mut ResponsesToolAccumulator,
    has_tools: bool,
) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    let event_type = event_json
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match event_type {
        "response.content_part.delta" => {
            if let Some(delta) = event_json.get("delta").and_then(|v| v.as_str()) {
                if !delta.is_empty() {
                    chunks.push(StreamChunk::Content(delta.to_string()));
                }
            }
        }
        "response.output_text.delta" => {
            if let Some(delta) = event_json.get("delta").and_then(|v| v.as_str()) {
                if !delta.is_empty() {
                    chunks.push(StreamChunk::Content(delta.to_string()));
                }
            }
        }
        "response.reasoning.delta" => {
            if let Some(delta) = event_json.get("delta").and_then(|v| v.as_str()) {
                if !delta.is_empty() {
                    chunks.push(StreamChunk::Reasoning(delta.to_string()));
                }
            }
        }
        "response.output_item.added" => {
            if has_tools {
                let output_index = event_json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if let Some(item) = event_json.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()));
                        let name = item.get("name").and_then(|v| v.as_str());
                        accumulator.start_tool(output_index, call_id, name);
                        if let Some(partial) = accumulator.tool_partial(output_index) {
                            chunks.push(StreamChunk::ToolCallPartial(partial));
                        }
                    }
                }
            }
        }
        "response.function_call_arguments.delta" => {
            if has_tools {
                let output_index = event_json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if let Some(delta) = event_json.get("delta").and_then(|v| v.as_str()) {
                    accumulator.append_args(output_index, delta);
                    if let Some(partial) = accumulator.tool_partial(output_index) {
                        chunks.push(StreamChunk::ToolCallPartial(partial));
                    }
                }
            }
        }
        "response.function_call_arguments.done" => {
            if has_tools {
                let output_index = event_json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if let Some(args) = event_json.get("arguments").and_then(|v| v.as_str()) {
                    accumulator.set_args(output_index, args);
                }
                if accumulator.has_name(output_index) {
                    if let Some(completed) = accumulator.complete_tool(output_index) {
                        chunks.push(StreamChunk::ToolCallComplete(completed));
                    }
                }
            }
        }
        "response.output_item.done" => {
            if has_tools {
                let output_index = event_json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if let Some(item) = event_json.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()));
                        let name = item.get("name").and_then(|v| v.as_str());
                        accumulator.start_tool(output_index, call_id, name);
                        if let Some(args) = item.get("arguments").and_then(|v| v.as_str()) {
                            accumulator.set_args(output_index, args);
                        }
                        if let Some(completed) = accumulator.complete_tool(output_index) {
                            chunks.push(StreamChunk::ToolCallComplete(completed));
                        }
                    }
                }
            }
        }
        "response.done" => {
            if let Some(response) = event_json.get("response") {
                if let Some(usage) = response.get("usage") {
                    chunks.push(StreamChunk::Usage {
                        input_tokens: usage
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        output_tokens: usage
                            .get("output_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        cached_tokens: usage
                            .get("input_tokens_details")
                            .and_then(|d| d.get("cached_tokens"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                    });
                }
            }
        }
        _ => {}
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_content_delta() {
        let event = serde_json::json!({
            "type": "response.content_part.delta",
            "delta": "Hello"
        });
        let mut acc = ResponsesToolAccumulator::new();
        let chunks = parse_openai_responses_event(&event, &mut acc, false);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Content(s) => assert_eq!(s, "Hello"),
            _ => panic!("Expected Content chunk"),
        }
    }

    #[test]
    fn test_parse_tool_call_flow() {
        let added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": ""
            }
        });
        let delta = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"city\":\"SF\"}"
        });
        let done = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "arguments": "{\"city\":\"SF\"}"
        });

        let mut acc = ResponsesToolAccumulator::new();
        let added_chunks = parse_openai_responses_event(&added, &mut acc, true);
        assert!(matches!(
            added_chunks.first(),
            Some(StreamChunk::ToolCallPartial(_))
        ));

        let delta_chunks = parse_openai_responses_event(&delta, &mut acc, true);
        assert!(matches!(
            delta_chunks.first(),
            Some(StreamChunk::ToolCallPartial(_))
        ));

        let chunks = parse_openai_responses_event(&done, &mut acc, true);

        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::ToolCallComplete(tc) => {
                assert_eq!(tc["id"], "call_123");
                assert_eq!(tc["function"]["name"], "get_weather");
            }
            _ => panic!("Expected ToolCallComplete chunk"),
        }
    }

    #[test]
    fn test_parse_usage_done() {
        let event = serde_json::json!({
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            }
        });
        let mut acc = ResponsesToolAccumulator::new();
        let chunks = parse_openai_responses_event(&event, &mut acc, false);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Usage {
                input_tokens,
                output_tokens,
                ..
            } => {
                assert_eq!(*input_tokens, 10);
                assert_eq!(*output_tokens, 5);
            }
            _ => panic!("Expected Usage chunk"),
        }
    }
}

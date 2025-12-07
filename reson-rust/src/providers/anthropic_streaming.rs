//! Anthropic streaming implementation
//!
//! Handles SSE parsing and progressive tool call accumulation for Anthropic API.

use std::collections::HashMap;
use serde_json::{json, Value};

use crate::providers::StreamChunk;

/// Accumulates tool calls during streaming (similar to OpenAI)
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    /// Track tool blocks by index
    current_tool_blocks: HashMap<usize, PartialToolCall>,
}

#[derive(Debug, Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    input: String, // Accumulated JSON
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self {
            current_tool_blocks: HashMap::new(),
        }
    }

    /// Start tracking a new tool call
    pub fn start_tool(&mut self, index: usize, id: String, name: String) {
        self.current_tool_blocks.insert(
            index,
            PartialToolCall {
                id,
                name,
                input: String::new(),
            },
        );
    }

    /// Accumulate partial JSON input
    pub fn accumulate_input(&mut self, index: usize, partial_json: &str) -> Option<Value> {
        if let Some(tool) = self.current_tool_blocks.get_mut(&index) {
            tool.input.push_str(partial_json);

            // Return OpenAI-format partial tool call
            Some(json!({
                "id": tool.id,
                "function": {
                    "name": tool.name,
                    "arguments": tool.input
                }
            }))
        } else {
            None
        }
    }

    /// Complete and remove a tool call
    pub fn complete_tool(&mut self, index: usize) -> Option<Value> {
        if let Some(tool) = self.current_tool_blocks.remove(&index) {
            // Parse and re-serialize JSON to ensure validity
            let arguments = if tool.input.is_empty() {
                "{}".to_string()
            } else {
                match serde_json::from_str::<Value>(&tool.input) {
                    Ok(parsed) => serde_json::to_string(&parsed).unwrap_or_else(|_| "{}".to_string()),
                    Err(_) => "{}".to_string(),
                }
            };

            // Return OpenAI-format complete tool call
            Some(json!({
                "id": tool.id,
                "function": {
                    "name": tool.name,
                    "arguments": arguments
                }
            }))
        } else {
            None
        }
    }
}

/// Parse Anthropic streaming chunk into StreamChunk
pub fn parse_anthropic_chunk(
    chunk_json: &Value,
    accumulator: &mut ToolCallAccumulator,
    has_tools: bool,
) -> Vec<StreamChunk> {
    let mut results = Vec::new();

    let chunk_type = match chunk_json["type"].as_str() {
        Some(t) => t,
        None => return results,
    };

    match chunk_type {
        "content_block_delta" => {
            let delta = &chunk_json["delta"];
            let content_type = delta["type"].as_str().unwrap_or("");

            match content_type {
                "text_delta" => {
                    if let Some(text) = delta["text"].as_str() {
                        results.push(StreamChunk::Content(text.to_string()));
                    }
                }
                "input_json_delta" if has_tools => {
                    if let Some(partial_json) = delta["partial_json"].as_str() {
                        let index = chunk_json["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(partial_tool) = accumulator.accumulate_input(index, partial_json) {
                            results.push(StreamChunk::ToolCallPartial(partial_tool));
                        }
                    }
                }
                "signature_delta" => {
                    if let Some(signature) = delta["signature"].as_str() {
                        results.push(StreamChunk::Signature(signature.to_string()));
                    }
                }
                _ => {}
            }
        }

        "content_block_start" => {
            let content_block = &chunk_json["content_block"];
            if content_block["type"] == "tool_use" {
                let index = chunk_json["index"].as_u64().unwrap_or(0) as usize;
                let id = content_block["id"].as_str().unwrap_or("").to_string();
                let name = content_block["name"].as_str().unwrap_or("").to_string();
                accumulator.start_tool(index, id, name);
            }
        }

        "content_block_stop" => {
            if has_tools {
                let index = chunk_json["index"].as_u64().unwrap_or(0) as usize;
                if let Some(complete_tool) = accumulator.complete_tool(index) {
                    results.push(StreamChunk::ToolCallComplete(complete_tool));
                }
            }
        }

        // message_start and message_delta are handled for usage tracking but don't emit chunks
        _ => {}
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_accumulator_start() {
        let mut acc = ToolCallAccumulator::new();
        acc.start_tool(0, "toolu_123".to_string(), "get_weather".to_string());

        assert_eq!(acc.current_tool_blocks.len(), 1);
        assert_eq!(acc.current_tool_blocks[&0].id, "toolu_123");
        assert_eq!(acc.current_tool_blocks[&0].name, "get_weather");
    }

    #[test]
    fn test_tool_accumulator_accumulate() {
        let mut acc = ToolCallAccumulator::new();
        acc.start_tool(0, "toolu_123".to_string(), "get_weather".to_string());

        let partial1 = acc.accumulate_input(0, "{\"city\":").unwrap();
        assert_eq!(partial1["function"]["arguments"], "{\"city\":");

        let partial2 = acc.accumulate_input(0, "\"SF\"}").unwrap();
        assert_eq!(partial2["function"]["arguments"], "{\"city\":\"SF\"}");
    }

    #[test]
    fn test_tool_accumulator_complete() {
        let mut acc = ToolCallAccumulator::new();
        acc.start_tool(0, "toolu_123".to_string(), "get_weather".to_string());
        acc.accumulate_input(0, "{\"city\":\"SF\"}");

        let complete = acc.complete_tool(0).unwrap();
        assert_eq!(complete["id"], "toolu_123");
        assert_eq!(complete["function"]["name"], "get_weather");
        assert_eq!(complete["function"]["arguments"], "{\"city\":\"SF\"}");
        assert_eq!(acc.current_tool_blocks.len(), 0); // Removed
    }

    #[test]
    fn test_parse_text_delta() {
        let mut acc = ToolCallAccumulator::new();
        let chunk = json!({
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        });

        let chunks = parse_anthropic_chunk(&chunk, &mut acc, false);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Content(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Content chunk"),
        }
    }

    #[test]
    fn test_parse_tool_use_flow() {
        let mut acc = ToolCallAccumulator::new();

        // 1. content_block_start
        let start_chunk = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather"
            }
        });
        let chunks = parse_anthropic_chunk(&start_chunk, &mut acc, true);
        assert_eq!(chunks.len(), 0); // Just initializes

        // 2. input_json_delta
        let delta_chunk = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\"city\":\"SF\"}"
            }
        });
        let chunks = parse_anthropic_chunk(&delta_chunk, &mut acc, true);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::ToolCallPartial(tool) => {
                assert_eq!(tool["id"], "toolu_123");
                assert_eq!(tool["function"]["arguments"], "{\"city\":\"SF\"}");
            }
            _ => panic!("Expected ToolCallPartial"),
        }

        // 3. content_block_stop
        let stop_chunk = json!({
            "type": "content_block_stop",
            "index": 0
        });
        let chunks = parse_anthropic_chunk(&stop_chunk, &mut acc, true);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::ToolCallComplete(tool) => {
                assert_eq!(tool["id"], "toolu_123");
                assert_eq!(tool["function"]["name"], "get_weather");
            }
            _ => panic!("Expected ToolCallComplete"),
        }
    }

    #[test]
    fn test_parse_signature_delta() {
        let mut acc = ToolCallAccumulator::new();
        let chunk = json!({
            "type": "content_block_delta",
            "delta": {
                "type": "signature_delta",
                "signature": "sig_abc123"
            }
        });

        let chunks = parse_anthropic_chunk(&chunk, &mut acc, false);
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Signature(sig) => assert_eq!(sig, "sig_abc123"),
            _ => panic!("Expected Signature chunk"),
        }
    }
}

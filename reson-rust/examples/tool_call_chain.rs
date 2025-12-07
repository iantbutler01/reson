//! Example: Tool Call Chain with History Reinjection
//!
//! Demonstrates how an assistant-issued tool call can be executed, converted into
//! a [`ToolResult`], and injected back into the conversation so downstream calls
//! can consume the new context.
//!
//! Run with: `cargo run --example tool_call_chain`

use reson_agentic::types::{ChatMessage, ToolCall, ToolResult};
use reson_agentic::utils::ConversationMessage;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool Call Chain Example ===\n");

    // Initial conversation history before any tool calls fire.
    let mut history = vec![
        ConversationMessage::Chat(ChatMessage::system(
            "You are a weather-savvy assistant that uses tools when needed.",
        )),
        ConversationMessage::Chat(ChatMessage::user(
            "Check the weather for Saturday in San Francisco and plan a picnic time.",
        )),
    ];

    print_history("Initial history", &history);

    // -----------------------------------------------------------------------------
    // Step 1: Assistant emits a get_weather tool call.
    // -----------------------------------------------------------------------------
    let mut weather_call = ToolCall::new(
        "get_weather",
        json!({
            "city": "San Francisco",
            "day": "Saturday",
            "units": "imperial"
        }),
    );
    weather_call.tool_use_id = "toolu_weather_1".to_string();
    history.push(ConversationMessage::ToolCall(weather_call.clone()));

    println!("Executing {}", describe_message(history.last().unwrap()));
    let weather_result = execute_fake_tool(&weather_call)?;

    // Convert tool output into ToolResult and reinject into history.
    history.push(ConversationMessage::ToolResult(weather_result.clone()));
    print_history("After weather tool", &history);

    // Parse the tool content so the next tool in the chain can reuse it.
    let weather_payload: Value = serde_json::from_str(&weather_result.content)?;
    let city = weather_payload
        .get("city")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown city")
        .to_string();
    let forecast = weather_payload
        .get("forecast")
        .and_then(|v| v.as_str())
        .unwrap_or("pleasant conditions")
        .to_string();
    let weather_window = weather_payload
        .get("best_window")
        .cloned()
        .unwrap_or_else(|| json!({ "start": "10:00", "end": "14:00" }));

    // -----------------------------------------------------------------------------
    // Step 2: Assistant uses the weather context to call plan_picnic.
    // -----------------------------------------------------------------------------
    let plan_args = json!({
        "city": city,
        "attendees": ["Taylor", "Jordan"],
        "weather_window": weather_window,
        "notes": forecast,
    });
    let mut plan_call = ToolCall::new("plan_picnic", plan_args);
    plan_call.tool_use_id = "toolu_plan_1".to_string();
    history.push(ConversationMessage::ToolCall(plan_call.clone()));

    println!("Executing {}", describe_message(history.last().unwrap()));
    let plan_result = execute_fake_tool(&plan_call)?;

    history.push(ConversationMessage::ToolResult(plan_result.clone()));
    print_history("Final history after chaining", &history);

    println!(
        "✅ Tool call chain completed. The assistant can now respond with the aggregated context."
    );
    Ok(())
}

/// Very small fake tool executor so the example stays self-contained.
fn execute_fake_tool(tool_call: &ToolCall) -> Result<ToolResult, Box<dyn std::error::Error>> {
    match tool_call.tool_name.as_str() {
        "get_weather" => {
            let city = tool_call
                .args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown city");
            let payload = json!({
                "city": city,
                "forecast": format!("{} will be sunny with highs around 68F.", city),
                "best_window": { "start": "11:00", "end": "14:00" },
                "uv_index": 5
            });
            let content = serde_json::to_string_pretty(&payload)?;
            Ok(ToolResult::success_with_name(
                tool_call.tool_use_id.clone(),
                tool_call.tool_name.clone(),
                content,
            )
            .with_tool_obj(tool_call.args.clone()))
        }
        "plan_picnic" => {
            let city = tool_call
                .args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown city")
                .to_string();
            let attendees: Vec<String> = tool_call
                .args
                .get("attendees")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let notes = tool_call
                .args
                .get("notes")
                .and_then(|v| v.as_str())
                .unwrap_or("clear skies and mild temps");
            let window_value = tool_call
                .args
                .get("weather_window")
                .cloned()
                .unwrap_or_else(|| json!({ "start": "10:00", "end": "14:00" }));
            let start = window_value
                .get("start")
                .and_then(|v| v.as_str())
                .unwrap_or("10:00");
            let end = window_value
                .get("end")
                .and_then(|v| v.as_str())
                .unwrap_or("14:00");
            let summary = format!(
                "Schedule picnic between {} and {} in {} while it's {}.",
                start, end, city, notes
            );

            let payload = json!({
                "city": city,
                "attendees": attendees,
                "window": window_value,
                "plan": summary,
                "checklist": ["blanket", "sunscreen", "sparkling water"]
            });
            let content = serde_json::to_string_pretty(&payload)?;
            Ok(ToolResult::success_with_name(
                tool_call.tool_use_id.clone(),
                tool_call.tool_name.clone(),
                content,
            )
            .with_tool_obj(tool_call.args.clone()))
        }
        _ => Ok(ToolResult::error(
            tool_call.tool_use_id.clone(),
            format!("No mock handler registered for {}", tool_call.tool_name),
        )
        .with_tool_name(tool_call.tool_name.clone())),
    }
}

fn print_history(label: &str, history: &[ConversationMessage]) {
    println!("--- {} ({} messages) ---", label, history.len());
    for (idx, message) in history.iter().enumerate() {
        println!("{:>2}. {}", idx + 1, describe_message(message));
    }
    println!();
}

fn describe_message(message: &ConversationMessage) -> String {
    match message {
        ConversationMessage::Chat(chat) => format!("{:?}: {}", chat.role, chat.content),
        ConversationMessage::ToolCall(tc) => {
            format!("ToolCall -> {} {}", tc.tool_name, pretty_json(&tc.args))
        }
        ConversationMessage::ToolResult(result) => format!(
            "ToolResult -> {} ({}): {}",
            result.tool_use_id,
            result.tool_name.as_deref().unwrap_or("<unknown tool>"),
            result.content.trim()
        ),
        ConversationMessage::Reasoning(_) => "Reasoning segment (not used here)".to_string(),
        ConversationMessage::Multimodal(_) => "Multimodal message (not used here)".to_string(),
    }
}

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

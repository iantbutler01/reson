//! Simple example demonstrating tool registration and schema generation
//!
//! Run with: cargo run --example simple_tools

use reson::runtime::{Runtime, ToolFunction};
use reson::storage::MemoryStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime (native tools are always enabled)
    let runtime = Runtime::with_config(
        Some("anthropic:claude-3-5-sonnet-20241022".to_string()),
        None, // Will use ANTHROPIC_API_KEY from env
        Arc::new(MemoryStore::new()),
    );

    // Register a simple sync tool
    runtime
        .register_tool(
            "get_weather",
            ToolFunction::Sync(Box::new(|args: serde_json::Value| {
                let location = args["location"].as_str().unwrap_or("unknown");
                Ok(format!("The weather in {} is sunny", location))
            })),
            None,
        )
        .await;

    // Register an async tool
    use futures::future::BoxFuture;
    runtime
        .register_tool(
            "calculate",
            ToolFunction::Async(Box::new(|args: serde_json::Value| {
                Box::pin(async move {
                    let a = args["a"].as_f64().unwrap_or(0.0);
                    let b = args["b"].as_f64().unwrap_or(0.0);
                    let op = args["operation"].as_str().unwrap_or("add");

                    let result = match op {
                        "add" => a + b,
                        "subtract" => a - b,
                        "multiply" => a * b,
                        "divide" => a / b,
                        _ => 0.0,
                    };

                    Ok(result.to_string())
                })
            })),
            None,
        )
        .await;

    println!("✅ Registered 2 tools successfully");
    println!("   - get_weather (sync)");
    println!("   - calculate (async)");
    println!();
    println!("Tools are ready for use with native tool calling!");
    println!();
    println!("Note: Actual LLM calls would require a valid API key.");
    println!("This example demonstrates the tool registration API.");

    Ok(())
}

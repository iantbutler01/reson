//! Integration tests for native tool calling with #[agentic] macro
//!
//! Run with: ANTHROPIC_API_KEY=xxx cargo test --test test_native_tools -- --nocapture --ignored
//! Or: OPENROUTER_API_KEY=xxx cargo test --test test_native_tools -- --nocapture --ignored

use futures::future::BoxFuture;
use reson_agentic::agentic;
use reson_agentic::error::Result;
use reson_agentic::parsers::Deserializable;
use reson_agentic::runtime::Runtime;
use reson_agentic::types::ToolResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct SearchQuery {
    #[serde(default)]
    text: String,
    #[serde(default = "default_category")]
    category: String,
    #[serde(default = "default_max_results")]
    max_results: i32,
}

fn default_category() -> String {
    "general".to_string()
}

fn default_max_results() -> i32 {
    5
}

impl Deserializable for SearchQuery {
    fn from_partial(value: serde_json::Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| {
            reson_agentic::error::Error::NonRetryable(format!("Failed to parse SearchQuery: {}", e))
        })
    }

    fn validate_complete(&self) -> Result<()> {
        Ok(())
    }

    fn field_descriptions() -> Vec<reson_agentic::parsers::FieldDescription> {
        vec![
            reson_agentic::parsers::FieldDescription {
                name: "text".to_string(),
                field_type: "string".to_string(),
                description: "The search text".to_string(),
                required: true,
            },
            reson_agentic::parsers::FieldDescription {
                name: "category".to_string(),
                field_type: "string".to_string(),
                description: "Category to search in".to_string(),
                required: false,
            },
            reson_agentic::parsers::FieldDescription {
                name: "max_results".to_string(),
                field_type: "number".to_string(),
                description: "Maximum number of results".to_string(),
                required: false,
            },
        ]
    }
}

/// Tool type for math operations
#[derive(Debug, Serialize, Deserialize)]
struct MathOperation {
    #[serde(default)]
    a: i64,
    #[serde(default)]
    b: i64,
}

impl Deserializable for MathOperation {
    fn from_partial(value: serde_json::Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| {
            reson_agentic::error::Error::NonRetryable(format!(
                "Failed to parse MathOperation: {}",
                e
            ))
        })
    }

    fn validate_complete(&self) -> Result<()> {
        Ok(())
    }

    fn field_descriptions() -> Vec<reson_agentic::parsers::FieldDescription> {
        vec![
            reson_agentic::parsers::FieldDescription {
                name: "a".to_string(),
                field_type: "number".to_string(),
                description: "First number".to_string(),
                required: true,
            },
            reson_agentic::parsers::FieldDescription {
                name: "b".to_string(),
                field_type: "number".to_string(),
                description: "Second number".to_string(),
                required: true,
            },
        ]
    }
}

/// Tool type for factorial
#[derive(Debug, Serialize, Deserialize)]
struct FactorialInput {
    #[serde(default)]
    n: i32,
}

impl Deserializable for FactorialInput {
    fn from_partial(value: serde_json::Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| {
            reson_agentic::error::Error::NonRetryable(format!(
                "Failed to parse FactorialInput: {}",
                e
            ))
        })
    }

    fn validate_complete(&self) -> Result<()> {
        Ok(())
    }

    fn field_descriptions() -> Vec<reson_agentic::parsers::FieldDescription> {
        vec![reson_agentic::parsers::FieldDescription {
            name: "n".to_string(),
            field_type: "number".to_string(),
            description: "Number to calculate factorial of".to_string(),
            required: true,
        }]
    }
}

// Handlers now receive typed structs directly, not ParsedTool
fn search_database(query: SearchQuery) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move {
        Ok(format!(
            "Found {} results for '{}' in category '{}'",
            query.max_results, query.text, query.category
        ))
    })
}

fn add_numbers(op: MathOperation) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move { Ok((op.a + op.b).to_string()) })
}

fn multiply_numbers(op: MathOperation) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move { Ok((op.a * op.b).to_string()) })
}

fn factorial(input: FactorialInput) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move {
        fn calc_factorial(n: i32) -> i32 {
            if n <= 1 {
                1
            } else {
                n * calc_factorial(n - 1)
            }
        }

        Ok(calc_factorial(input.n).to_string())
    })
}

#[agentic(model = "openrouter:anthropic/claude-sonnet-4")]
async fn native_multi_agent(query: String, runtime: Runtime) -> Result<String> {
    // Register tools - each handler receives its typed struct directly
    runtime
        .tool::<SearchQuery, _>(search_database, Some("search_database"))
        .await?;
    runtime
        .tool::<MathOperation, _>(add_numbers, Some("add_numbers"))
        .await?;
    runtime
        .tool::<MathOperation, _>(multiply_numbers, Some("multiply_numbers"))
        .await?;
    runtime
        .tool::<FactorialInput, _>(factorial, Some("factorial"))
        .await?;

    let mut history = Vec::new();
    let mut result = runtime
        .run(
            Some(&query),
            None,
            Some(history.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await?;

    println!("📞 Native initial call: {:?}", result);

    let mut turn_count = 0;
    let max_turns = 5;

    while runtime.is_tool_call(&result) && turn_count < max_turns {
        turn_count += 1;
        let tool_name = runtime
            .get_tool_name(&result)
            .ok_or_else(|| reson_agentic::error::Error::NonRetryable("No tool name".to_string()))?;

        println!("🔧 Native turn {}: {}", turn_count, tool_name);

        // Execute tool
        let tool_result_str = runtime.execute_tool(&result).await?;
        println!("🔧 Native tool result {}: {}", turn_count, tool_result_str);

        // Create ToolResult message
        let tool_result = ToolResult {
            tool_use_id: runtime.get_tool_name(&result).unwrap_or_default(),
            tool_name: None,
            content: tool_result_str,
            is_error: false,
            signature: None,
            tool_obj: None,
        };

        // Add to history
        history.push(reson_agentic::utils::ConversationMessage::ToolResult(
            tool_result,
        ));

        // Continue conversation
        result = runtime
            .run(
                Some("Continue with the next step"),
                None,
                Some(history.clone()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await?;

        println!("📞 Native turn {} result: {:?}", turn_count + 1, result);
    }

    println!(
        "✅ Native conversation completed with {} tool calls",
        turn_count
    );
    Ok(format!("Completed with {} tool calls", turn_count))
}

#[agentic(model = "anthropic:claude-haiku-4-5-20251001")]
async fn single_tool_agent(query: String, runtime: Runtime) -> Result<String> {
    runtime
        .tool::<MathOperation, _>(add_numbers, Some("add_numbers"))
        .await?;

    let result = runtime
        .run(
            Some(&query),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await?;
    println!("📊 Result: {:?}", result);

    if runtime.is_tool_call(&result) {
        println!("✅ Tool call detected");
        let tool_result = runtime.execute_tool(&result).await?;
        println!("✅ Tool executed: {}", tool_result);
        return Ok(tool_result);
    } else {
        println!("📝 Got text response");
        Ok(format!("{:?}", result))
    }
}

#[tokio::test]
#[ignore] // Run with --ignored flag
async fn test_native_5_turn_conversation() {
    println!("\n🧪 Testing Native Tools - 5-turn conversation");

    let query = "I need you to: 1) Calculate 8 + 7, 2) Then multiply that result by 4, 3) Then calculate the factorial of 5, 4) Search for 'python tutorials' with max 2 results, 5) Finally give me a summary of all the results".to_string();

    match native_multi_agent(query).await {
        Ok(result) => {
            println!("✅ Test completed: {}", result);
        }
        Err(e) => {
            panic!("❌ Test failed: {:?}", e);
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_single_tool_call() {
    println!("\n🧪 Testing Single Tool Call");

    match single_tool_agent("Add 12 and 8 together".to_string()).await {
        Ok(result) => {
            println!("✅ Test completed: {}", result);
            assert!(result.contains("20") || result.contains("tool"));
        }
        Err(e) => {
            panic!("❌ Test failed: {:?}", e);
        }
    }
}

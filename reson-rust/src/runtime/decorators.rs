//! Decorator implementations for agentic functions
//!
//! Note: These are Rust function equivalents. The actual decorators will be
//! implemented as proc macros in the reson-macros crate.

use crate::error::Result;

/// Helper to create empty value for a type (for streaming initialization)
pub fn create_empty_value_for_type(type_name: &str) -> serde_json::Value {
    match type_name {
        "String" | "str" => serde_json::json!(""),
        "i32" | "i64" | "u32" | "u64" | "isize" | "usize" => serde_json::json!(0),
        "f32" | "f64" => serde_json::json!(0.0),
        "bool" => serde_json::json!(false),
        "Vec" | "Array" => serde_json::json!([]),
        "HashMap" | "Map" | "Object" => serde_json::json!({}),
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_value() {
        assert_eq!(create_empty_value_for_type("String"), serde_json::json!(""));
        assert_eq!(create_empty_value_for_type("i32"), serde_json::json!(0));
        assert_eq!(create_empty_value_for_type("f64"), serde_json::json!(0.0));
        assert_eq!(create_empty_value_for_type("bool"), serde_json::json!(false));
        assert_eq!(create_empty_value_for_type("Vec"), serde_json::json!([]));
        assert_eq!(create_empty_value_for_type("HashMap"), serde_json::json!({}));
        assert_eq!(create_empty_value_for_type("Unknown"), serde_json::Value::Null);
    }
}

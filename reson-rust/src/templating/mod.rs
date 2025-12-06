//! Templating system using minijinja
//!
//! Provides Jinja2-like templating with support for type interpolation and context variables.

#[cfg(feature = "templating")]
use minijinja::{Environment, Error as TemplateError};

/// Template renderer with support for type interpolation
#[cfg(feature = "templating")]
pub struct TemplateEngine {
    env: Environment<'static>,
}

#[cfg(feature = "templating")]
impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        let mut env = Environment::new();

        // Add custom filters
        env.add_filter("json", json_filter);

        Self { env }
    }

    /// Render a template string with the given context
    ///
    /// # Example
    /// ```no_run
    /// use reson_agentic::templating::TemplateEngine;
    ///
    /// let engine = TemplateEngine::new();
    /// let result = engine.render_str("Hello {{ name }}!", &[("name", "World")]).unwrap();
    /// assert_eq!(result, "Hello World!");
    /// ```
    pub fn render_str<K, V>(&self, template: &str, context: &[(K, V)]) -> Result<String, TemplateError>
    where
        K: AsRef<str>,
        V: serde::Serialize,
    {
        let tmpl = self.env.template_from_str(template)?;

        // Convert context to HashMap
        let mut ctx_map = HashMap::new();
        for (key, value) in context {
            let json_value = serde_json::to_value(value)
                .map_err(|e| TemplateError::new(minijinja::ErrorKind::InvalidOperation, format!("Serialization failed: {}", e)))?;
            ctx_map.insert(key.as_ref().to_string(), json_value);
        }

        tmpl.render(ctx_map)
    }

    /// Render a template string with a hashmap context
    pub fn render_with_context(
        &self,
        template: &str,
        context: HashMap<String, serde_json::Value>,
    ) -> Result<String, TemplateError> {
        let tmpl = self.env.template_from_str(template)?;
        tmpl.render(context)
    }
}

#[cfg(feature = "templating")]
impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON filter for pretty-printing values
#[cfg(feature = "templating")]
fn json_filter(_state: &minijinja::State, value: minijinja::Value) -> Result<minijinja::Value, TemplateError> {
    let json_str = serde_json::to_string_pretty(&value)
        .map_err(|e| TemplateError::new(minijinja::ErrorKind::InvalidOperation, format!("JSON serialization failed: {}", e)))?;
    Ok(minijinja::Value::from(json_str))
}

/// Render a simple template with context variables
///
/// This is a convenience function for quick template rendering.
#[cfg(feature = "templating")]
pub fn render_template<K, V>(template: &str, context: &[(K, V)]) -> Result<String, TemplateError>
where
    K: AsRef<str>,
    V: serde::Serialize,
{
    let engine = TemplateEngine::new();
    engine.render_str(template, context)
}

#[cfg(all(test, feature = "templating"))]
mod tests {
    use super::*;

    #[test]
    fn test_template_engine_new() {
        let engine = TemplateEngine::new();
        let result = engine.render_str("Hello!", &[("name", "World")]).unwrap();
        assert_eq!(result, "Hello!");
    }

    #[test]
    fn test_template_engine_simple_interpolation() {
        let engine = TemplateEngine::new();
        let result = engine.render_str("Hello {{ name }}!", &[("name", "World")]).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_engine_multiple_variables() {
        let engine = TemplateEngine::new();
        let context = [
            ("first", "John"),
            ("last", "Doe"),
        ];
        let result = engine.render_str("{{ first }} {{ last }}", &context).unwrap();
        assert_eq!(result, "John Doe");
    }

    #[test]
    fn test_template_engine_json_filter() {
        let engine = TemplateEngine::new();
        let data = serde_json::json!({"name": "Alice", "age": 30});
        let result = engine.render_str("Data: {{ data | json }}", &[("data", &data)]).unwrap();
        assert!(result.contains("Alice"));
        assert!(result.contains("30"));
    }

    #[test]
    fn test_template_engine_with_context() {
        let engine = TemplateEngine::new();
        let mut context = HashMap::new();
        context.insert("name".to_string(), serde_json::json!("Bob"));
        context.insert("age".to_string(), serde_json::json!(25));

        let result = engine.render_with_context("{{ name }} is {{ age }}", context).unwrap();
        assert_eq!(result, "Bob is 25");
    }

    #[test]
    fn test_template_engine_conditional() {
        let engine = TemplateEngine::new();
        let result = engine.render_str(
            "{% if show %}Hello!{% endif %}",
            &[("show", true)]
        ).unwrap();
        assert_eq!(result, "Hello!");
    }

    #[test]
    fn test_template_engine_loop() {
        let engine = TemplateEngine::new();
        let items = vec!["a", "b", "c"];
        let result = engine.render_str(
            "{% for item in items %}{{ item }}{% endfor %}",
            &[("items", items)]
        ).unwrap();
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_render_template_convenience() {
        let result = render_template("Hi {{ name }}", &[("name", "World")]).unwrap();
        assert_eq!(result, "Hi World");
    }
}

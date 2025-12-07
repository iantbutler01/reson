//! Example demonstrating template rendering with minijinja
//!
//! Run with: cargo run --example templating_example --features templating

#[cfg(feature = "templating")]
use reson_agentic::templating::TemplateEngine;

#[cfg(feature = "templating")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Reson Templating Example ===\n");

    let engine = TemplateEngine::new();

    // 1. Simple variable interpolation
    println!("1. Simple interpolation:");
    let result = engine.render_str("Hello {{ name }}!", &[("name", "World")])?;
    println!("   {}\n", result);

    // 2. Multiple variables
    println!("2. Multiple variables:");
    let result = engine.render_str(
        "User: {{ first }} {{ last }}, Age: {{ age }}",
        &[
            ("first", "John"),
            ("last", "Doe"),
            ("age", "30"),
        ],
    )?;
    println!("   {}\n", result);

    // 3. Conditionals
    println!("3. Conditionals:");
    let result = engine.render_str(
        "{% if is_admin %}Admin access granted{% else %}Regular user{% endif %}",
        &[("is_admin", true)],
    )?;
    println!("   {}\n", result);

    // 4. Loops
    println!("4. Loops:");
    let items = vec!["apple", "banana", "cherry"];
    let result = engine.render_str(
        "Fruits: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}",
        &[("items", items)],
    )?;
    println!("   {}\n", result);

    // 5. JSON filter
    println!("5. JSON filter:");
    let data = serde_json::json!({
        "name": "Alice",
        "age": 30,
        "active": true
    });
    let result = engine.render_str("User data: {{ user | json }}", &[("user", &data)])?;
    println!("   {}\n", result);

    println!("✅ Templating example completed successfully!");

    Ok(())
}

#[cfg(not(feature = "templating"))]
fn main() {
    println!("This example requires the 'templating' feature to be enabled.");
    println!("Run with: cargo run --example templating_example --features templating");
}

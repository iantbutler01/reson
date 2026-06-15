//! Python bindings for the chevalier LLM agent framework
//!
//! This crate provides PyO3 bindings to expose the Rust chevalier library to Python,
//! allowing existing Python integration tests to run unchanged.

use pyo3::prelude::*;

mod decorators;
mod errors;
mod runtime;
mod services;
mod stores;
mod types;
mod utils;

/// Create the `types` submodule
fn create_types_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let types_module = PyModule::new(m.py(), "types")?;
    types_module.add_class::<types::ChatRole>()?;
    types_module.add_class::<types::ChatMessage>()?;
    types_module.add_class::<types::ToolCall>()?;
    types_module.add_class::<types::ToolResult>()?;
    types_module.add_class::<types::ReasoningSegment>()?;
    types_module.add_class::<types::Deserializable>()?;
    m.add_submodule(&types_module)?;

    // Make it importable as `from chevalier.types import ...`
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("chevalier.types", types_module)?;

    Ok(())
}

/// Create the `stores` submodule
fn create_stores_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let stores_module = PyModule::new(m.py(), "stores")?;
    stores_module.add_class::<stores::MemoryStore>()?;
    stores_module.add_class::<stores::MemoryStoreConfig>()?;
    m.add_submodule(&stores_module)?;

    // Make it importable as `from chevalier.stores import ...`
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("chevalier.stores", stores_module)?;

    Ok(())
}

/// Create the `services` submodule with nested `inference_clients`
fn create_services_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let services_module = PyModule::new(m.py(), "services")?;

    // Create inference_clients submodule
    let inference_clients_module = PyModule::new(m.py(), "inference_clients")?;
    inference_clients_module.add_class::<services::InferenceProvider>()?;
    inference_clients_module.add_class::<services::InferenceClient>()?;
    // Also add types that are commonly imported from inference_clients
    inference_clients_module.add_class::<types::ChatMessage>()?;
    inference_clients_module.add_class::<types::ChatRole>()?;
    inference_clients_module.add_class::<types::ToolCall>()?;
    inference_clients_module.add_class::<types::ToolResult>()?;
    services_module.add_submodule(&inference_clients_module)?;

    m.add_submodule(&services_module)?;

    // Make them importable
    let sys_modules = m.py().import("sys")?.getattr("modules")?;
    sys_modules.set_item("chevalier.services", services_module)?;
    sys_modules.set_item("chevalier.services.inference_clients", inference_clients_module)?;

    Ok(())
}

/// Create the `_utils` submodule with `schema_generators` (internal, wrapped by Python)
fn create_utils_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Export Rust utils under chevalier.chevalier (the Rust module namespace)
    // The Python wrapper in chevalier/utils/ will import from here and wrap them
    m.add_function(wrap_pyfunction!(utils::supports_native_tools, m)?)?;
    m.add_function(wrap_pyfunction!(utils::get_schema_generator, m)?)?;
    m.add_class::<utils::SchemaGenerator>()?;

    Ok(())
}

/// Create the `chevalier.chevalier` submodule (for `from chevalier.chevalier import ...`)
fn create_chevalier_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let chevalier_module = PyModule::new(m.py(), "chevalier")?;

    // Add main exports that tests import from chevalier.chevalier
    chevalier_module.add_class::<runtime::Runtime>()?;
    chevalier_module.add_function(wrap_pyfunction!(decorators::agentic, &chevalier_module)?)?;
    chevalier_module.add_function(wrap_pyfunction!(
        decorators::agentic_generator,
        &chevalier_module
    )?)?;

    // Add _generate_native_tool_schemas function (used by some tests)
    chevalier_module.add_function(wrap_pyfunction!(
        utils::_generate_native_tool_schemas,
        &chevalier_module
    )?)?;

    m.add_submodule(&chevalier_module)?;

    // Make it importable
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("chevalier.chevalier", chevalier_module)?;

    Ok(())
}

/// The chevalier Python module
#[pymodule]
fn chevalier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception types at top level
    m.add(
        "NonRetryableException",
        m.py().get_type::<errors::NonRetryableException>(),
    )?;
    m.add(
        "InferenceException",
        m.py().get_type::<errors::InferenceException>(),
    )?;
    m.add(
        "ContextLengthExceeded",
        m.py().get_type::<errors::ContextLengthExceeded>(),
    )?;
    m.add(
        "RetriesExceeded",
        m.py().get_type::<errors::RetriesExceeded>(),
    )?;

    // Register type classes at top level (for backwards compatibility)
    m.add_class::<types::ChatRole>()?;
    m.add_class::<types::ChatMessage>()?;
    m.add_class::<types::ToolCall>()?;
    m.add_class::<types::ToolResult>()?;
    m.add_class::<types::ReasoningSegment>()?;
    m.add_class::<types::Deserializable>()?;

    // Register store classes at top level (for backwards compatibility)
    m.add_class::<stores::MemoryStore>()?;
    m.add_class::<stores::MemoryStoreConfig>()?;

    // Register runtime class
    m.add_class::<runtime::Runtime>()?;

    // Register decorator functions
    m.add_function(wrap_pyfunction!(decorators::agentic, m)?)?;
    m.add_function(wrap_pyfunction!(decorators::agentic_generator, m)?)?;

    // Register utility functions at top level (also available via chevalier.chevalier for tests)
    m.add_function(wrap_pyfunction!(utils::_generate_native_tool_schemas, m)?)?;

    // Create submodules for `from chevalier.X import Y` style imports
    create_types_submodule(m)?;
    create_stores_submodule(m)?;
    create_services_submodule(m)?;
    create_utils_submodule(m)?;
    create_chevalier_submodule(m)?;

    Ok(())
}

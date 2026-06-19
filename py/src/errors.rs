//! Python exception types wrapping chevalier_core::error::Error

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Define Python exception types
pyo3::create_exception!(chevalier, NonRetryableException, PyException);
pyo3::create_exception!(chevalier, InferenceException, PyException);
pyo3::create_exception!(chevalier, ContextLengthExceeded, InferenceException);
pyo3::create_exception!(chevalier, RetriesExceeded, InferenceException);

/// Convert chevalier_core::error::Error to Python exception
pub fn to_py_err(err: chevalier_core::error::Error) -> PyErr {
    match err {
        chevalier_core::error::Error::Inference(msg) => InferenceException::new_err(msg),
        chevalier_core::error::Error::ContextLengthExceeded(msg) => {
            ContextLengthExceeded::new_err(msg)
        }
        chevalier_core::error::Error::RetriesExceeded(_) => {
            RetriesExceeded::new_err("Maximum retries exceeded")
        }
        chevalier_core::error::Error::NonRetryable(msg) => NonRetryableException::new_err(msg),
        chevalier_core::error::Error::Parse(msg) => {
            NonRetryableException::new_err(format!("Parse error: {}", msg))
        }
        chevalier_core::error::Error::Validation(msg) => {
            NonRetryableException::new_err(format!("Validation error: {}", msg))
        }
        chevalier_core::error::Error::ToolNotFound(name) => {
            NonRetryableException::new_err(format!("Tool not found: {}", name))
        }
        chevalier_core::error::Error::Network(e) => {
            InferenceException::new_err(format!("Network error: {}", e))
        }
        chevalier_core::error::Error::Json(e) => {
            NonRetryableException::new_err(format!("JSON error: {}", e))
        }
        chevalier_core::error::Error::Utf8(e) => {
            NonRetryableException::new_err(format!("UTF-8 error: {}", e))
        }
        chevalier_core::error::Error::FromUtf8(e) => {
            NonRetryableException::new_err(format!("UTF-8 error: {}", e))
        }
        chevalier_core::error::Error::Io(e) => {
            InferenceException::new_err(format!("IO error: {}", e))
        }
        chevalier_core::error::Error::InvalidProvider(msg) => {
            NonRetryableException::new_err(format!("Invalid provider: {}", msg))
        }
        chevalier_core::error::Error::MissingApiKey(msg) => {
            NonRetryableException::new_err(format!("Missing API key: {}", msg))
        }
        chevalier_core::error::Error::RuntimeNotUsed => {
            NonRetryableException::new_err("Runtime was not used in agentic function")
        }
    }
}

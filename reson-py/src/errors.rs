//! Python exception types wrapping reson_agentic::error::Error

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Define Python exception types
pyo3::create_exception!(reson, NonRetryableException, PyException);
pyo3::create_exception!(reson, InferenceException, PyException);
pyo3::create_exception!(reson, ContextLengthExceeded, InferenceException);
pyo3::create_exception!(reson, RetriesExceeded, InferenceException);

/// Convert reson_agentic::error::Error to Python exception
pub fn to_py_err(err: reson_agentic::error::Error) -> PyErr {
    match err {
        reson_agentic::error::Error::Inference(msg) => InferenceException::new_err(msg),
        reson_agentic::error::Error::ContextLengthExceeded(msg) => {
            ContextLengthExceeded::new_err(msg)
        }
        reson_agentic::error::Error::RetriesExceeded(_) => {
            RetriesExceeded::new_err("Maximum retries exceeded")
        }
        reson_agentic::error::Error::NonRetryable(msg) => NonRetryableException::new_err(msg),
        reson_agentic::error::Error::Parse(msg) => {
            NonRetryableException::new_err(format!("Parse error: {}", msg))
        }
        reson_agentic::error::Error::Validation(msg) => {
            NonRetryableException::new_err(format!("Validation error: {}", msg))
        }
        reson_agentic::error::Error::ToolNotFound(name) => {
            NonRetryableException::new_err(format!("Tool not found: {}", name))
        }
        reson_agentic::error::Error::Network(e) => {
            InferenceException::new_err(format!("Network error: {}", e))
        }
        reson_agentic::error::Error::Json(e) => {
            NonRetryableException::new_err(format!("JSON error: {}", e))
        }
        reson_agentic::error::Error::Utf8(e) => {
            NonRetryableException::new_err(format!("UTF-8 error: {}", e))
        }
        reson_agentic::error::Error::FromUtf8(e) => {
            NonRetryableException::new_err(format!("UTF-8 error: {}", e))
        }
        reson_agentic::error::Error::Io(e) => {
            InferenceException::new_err(format!("IO error: {}", e))
        }
        reson_agentic::error::Error::InvalidProvider(msg) => {
            NonRetryableException::new_err(format!("Invalid provider: {}", msg))
        }
        reson_agentic::error::Error::MissingApiKey(msg) => {
            NonRetryableException::new_err(format!("Missing API key: {}", msg))
        }
        reson_agentic::error::Error::RuntimeNotUsed => {
            NonRetryableException::new_err("Runtime was not used in agentic function")
        }
    }
}

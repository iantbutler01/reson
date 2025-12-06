//! Python exception types wrapping reson::error::Error

use pyo3::prelude::*;
use pyo3::exceptions::PyException;

// Define Python exception types
pyo3::create_exception!(reson, NonRetryableException, PyException);
pyo3::create_exception!(reson, InferenceException, PyException);
pyo3::create_exception!(reson, ContextLengthExceeded, InferenceException);
pyo3::create_exception!(reson, RetriesExceeded, InferenceException);

/// Convert reson::error::Error to Python exception
pub fn to_py_err(err: reson::error::Error) -> PyErr {
    match err {
        reson::error::Error::Inference(msg) => InferenceException::new_err(msg),
        reson::error::Error::ContextLengthExceeded(msg) => ContextLengthExceeded::new_err(msg),
        reson::error::Error::RetriesExceeded => RetriesExceeded::new_err("Maximum retries exceeded"),
        reson::error::Error::NonRetryable(msg) => NonRetryableException::new_err(msg),
        reson::error::Error::Parse(msg) => NonRetryableException::new_err(format!("Parse error: {}", msg)),
        reson::error::Error::Validation(msg) => NonRetryableException::new_err(format!("Validation error: {}", msg)),
        reson::error::Error::ToolNotFound(name) => NonRetryableException::new_err(format!("Tool not found: {}", name)),
        reson::error::Error::Network(e) => InferenceException::new_err(format!("Network error: {}", e)),
        reson::error::Error::Json(e) => NonRetryableException::new_err(format!("JSON error: {}", e)),
        reson::error::Error::Utf8(e) => NonRetryableException::new_err(format!("UTF-8 error: {}", e)),
        reson::error::Error::FromUtf8(e) => NonRetryableException::new_err(format!("UTF-8 error: {}", e)),
        reson::error::Error::Io(e) => InferenceException::new_err(format!("IO error: {}", e)),
        reson::error::Error::InvalidProvider(msg) => NonRetryableException::new_err(format!("Invalid provider: {}", msg)),
        reson::error::Error::MissingApiKey(msg) => NonRetryableException::new_err(format!("Missing API key: {}", msg)),
        reson::error::Error::RuntimeNotUsed => NonRetryableException::new_err("Runtime was not used in agentic function"),
    }
}

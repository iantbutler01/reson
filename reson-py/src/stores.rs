//! Python store wrappers

use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::to_py_err;

/// In-memory store exposed to Python
#[pyclass]
pub struct MemoryStore {
    inner: Arc<RwLock<reson_agentic::storage::MemoryKVStore>>,
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(reson_agentic::storage::MemoryKVStore::new())),
        }
    }

    fn get<'py>(&self, py: Python<'py>, key: String, default: Option<PyObject>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            let result: Option<serde_json::Value> = reson_agentic::storage::Store::get(&*store, &key)
                .await
                .map_err(to_py_err)?;

            Python::with_gil(|py| -> PyResult<PyObject> {
                match result {
                    Some(value) => pythonize::pythonize(py, &value)
                        .map(|b| b.unbind())
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
                    None => Ok(default.unwrap_or_else(|| py.None())),
                }
            })
        })
    }

    fn set<'py>(&self, py: Python<'py>, key: String, value: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let json_value: serde_json::Value = Python::with_gil(|py| {
            pythonize::depythonize(value.bind(py))
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.write().await;
            reson_agentic::storage::Store::set(&*store, &key, &json_value)
                .await
                .map_err(to_py_err)?;
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn delete<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.write().await;
            reson_agentic::storage::Store::delete(&*store, &key)
                .await
                .map_err(to_py_err)?;
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.write().await;
            reson_agentic::storage::Store::clear(&*store)
                .await
                .map_err(to_py_err)?;
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn get_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            let result = reson_agentic::storage::Store::get_all(&*store)
                .await
                .map_err(to_py_err)?;

            Python::with_gil(|py| -> PyResult<PyObject> {
                pythonize::pythonize(py, &result)
                    .map(|b| b.unbind())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            })
        })
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            let result = reson_agentic::storage::Store::keys(&*store)
                .await
                .map_err(to_py_err)?;

            Python::with_gil(|py| -> PyResult<PyObject> {
                let set = pyo3::types::PySet::new(py, result.iter())?;
                Ok(set.into_any().unbind())
            })
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            reson_agentic::storage::Store::close(&*store)
                .await
                .map_err(to_py_err)?;
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn publish_to_mailbox<'py>(&self, py: Python<'py>, mailbox_id: String, message: PyObject) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let json_value: serde_json::Value = Python::with_gil(|py| {
            pythonize::depythonize(message.bind(py))
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.write().await;
            reson_agentic::storage::Store::publish_to_mailbox(&*store, &mailbox_id, &json_value)
                .await
                .map_err(to_py_err)?;
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    #[pyo3(signature = (mailbox_id, timeout=None))]
    fn get_message<'py>(&self, py: Python<'py>, mailbox_id: String, timeout: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.write().await;
            let result = reson_agentic::storage::Store::get_message(&*store, &mailbox_id, timeout)
                .await
                .map_err(to_py_err)?;

            Python::with_gil(|py| -> PyResult<PyObject> {
                match result {
                    Some(value) => pythonize::pythonize(py, &value)
                        .map(|b| b.unbind())
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
                    None => Ok(py.None()),
                }
            })
        })
    }
}

/// Memory store config for decorator
#[pyclass]
#[derive(Clone)]
pub struct MemoryStoreConfig {
    #[pyo3(get)]
    pub kind: String,
}

#[pymethods]
impl MemoryStoreConfig {
    #[new]
    fn new() -> Self {
        Self { kind: "memory".to_string() }
    }
}

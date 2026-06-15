//! Python store wrappers.

use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Default)]
struct MemoryStoreInner {
    kv: HashMap<String, serde_json::Value>,
    mailboxes: HashMap<String, VecDeque<serde_json::Value>>,
}

/// In-memory store exposed to Python.
#[pyclass]
pub struct MemoryStore {
    inner: Arc<RwLock<MemoryStoreInner>>,
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MemoryStoreInner::default())),
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: String,
        default: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            let result = store.kv.get(&key).cloned();

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

    fn set<'py>(
        &self,
        py: Python<'py>,
        key: String,
        value: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let json_value: serde_json::Value =
            Python::with_gil(|py| pythonize::depythonize(value.bind(py)))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut store = inner.write().await;
            store.kv.insert(key, json_value);
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn delete<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut store = inner.write().await;
            store.kv.remove(&key);
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut store = inner.write().await;
            store.kv.clear();
            store.mailboxes.clear();
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn get_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = inner.read().await;
            let result = store.kv.clone();

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
            let keys: Vec<String> = store.kv.keys().cloned().collect();
            Python::with_gil(|py| -> PyResult<PyObject> {
                let set = pyo3::types::PySet::new(py, keys.iter())?;
                Ok(set.into_any().unbind())
            })
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn publish_to_mailbox<'py>(
        &self,
        py: Python<'py>,
        mailbox_id: String,
        message: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let json_value: serde_json::Value =
            Python::with_gil(|py| pythonize::depythonize(message.bind(py)))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut store = inner.write().await;
            store
                .mailboxes
                .entry(mailbox_id)
                .or_insert_with(VecDeque::new)
                .push_back(json_value);
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    #[pyo3(signature = (mailbox_id, timeout=None))]
    fn get_message<'py>(
        &self,
        py: Python<'py>,
        mailbox_id: String,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let deadline = timeout.map(|secs| {
                tokio::time::Instant::now() + tokio::time::Duration::from_secs_f64(secs.max(0.0))
            });

            loop {
                {
                    let mut store = inner.write().await;
                    if let Some(queue) = store.mailboxes.get_mut(&mailbox_id) {
                        if let Some(value) = queue.pop_front() {
                            return Python::with_gil(|py| -> PyResult<PyObject> {
                                pythonize::pythonize(py, &value)
                                    .map(|b| b.unbind())
                                    .map_err(|e| {
                                        pyo3::exceptions::PyValueError::new_err(e.to_string())
                                    })
                            });
                        }
                    }
                }

                match deadline {
                    Some(end) if tokio::time::Instant::now() >= end => {
                        return Ok(Python::with_gil(|py| py.None()));
                    }
                    Some(_) => tokio::time::sleep(tokio::time::Duration::from_millis(10)).await,
                    None => return Ok(Python::with_gil(|py| py.None())),
                }
            }
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
        Self {
            kind: "memory".to_string(),
        }
    }
}

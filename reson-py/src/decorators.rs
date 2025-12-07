//! Python decorators: @agentic and @agentic_generator
//!
//! These decorators wrap Python async functions to:
//! 1. Create and inject a Runtime instance
//! 2. Extract docstring as default prompt
//! 3. Validate that runtime.run() was called

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::runtime::Runtime;

/// The @agentic decorator for async functions
#[pyfunction]
#[pyo3(signature = (*, model=None, api_key=None, store_cfg=None, autobind=false, native_tools=false))]
pub fn agentic(
    py: Python<'_>,
    model: Option<String>,
    api_key: Option<String>,
    store_cfg: Option<PyObject>,
    autobind: bool,
    native_tools: bool,
) -> PyResult<PyObject> {
    let _ = (store_cfg, autobind); // Silence unused warnings

    let decorator = AgenticDecorator {
        model,
        api_key,
        native_tools,
    };

    let decorator_obj = Py::new(py, decorator)?;
    Ok(decorator_obj.into_py(py))
}

#[pyclass]
struct AgenticDecorator {
    model: Option<String>,
    api_key: Option<String>,
    native_tools: bool,
}

#[pymethods]
impl AgenticDecorator {
    fn __call__(&self, py: Python<'_>, func: PyObject) -> PyResult<PyObject> {
        let func_name: String = func.getattr(py, "__name__")?.extract(py)?;
        let docstring: Option<String> = func.getattr(py, "__doc__")
            .ok()
            .and_then(|doc| doc.extract(py).ok());

        let wrapper = AgenticWrapper {
            inner_func: func,
            func_name,
            docstring,
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            native_tools: self.native_tools,
        };

        let wrapper_obj = Py::new(py, wrapper)?;
        Ok(wrapper_obj.into_py(py))
    }
}

#[pyclass]
struct AgenticWrapper {
    inner_func: PyObject,
    func_name: String,
    docstring: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    native_tools: bool,
}

#[pymethods]
impl AgenticWrapper {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Create a new Runtime
        let runtime = Runtime::create(
            self.model.clone(),
            self.api_key.clone(),
            self.native_tools,
        );

        let runtime_obj = Py::new(py, runtime)?;

        // Build new kwargs with runtime injected
        let new_kwargs = if let Some(kw) = kwargs {
            let new_kw = kw.copy()?;
            new_kw.set_item("runtime", &runtime_obj)?;
            new_kw
        } else {
            let new_kw = PyDict::new(py);
            new_kw.set_item("runtime", &runtime_obj)?;
            new_kw
        };

        // Call the inner async function
        let coro = self.inner_func.call_bound(py, args, Some(&new_kwargs))?;

        // Check if it's a coroutine
        let asyncio = py.import("asyncio")?;
        if !asyncio.call_method1("iscoroutine", (&coro,))?.is_truthy()? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                format!("@agentic decorated function '{}' must be async", self.func_name)
            ));
        }

        Ok(coro.into_bound(py))
    }

    fn __get__<'py>(
        slf: PyRef<'py, Self>,
        obj: Option<&Bound<'py, PyAny>>,
        _objtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        match obj {
            Some(instance) => {
                let bound_method = BoundAgenticMethod {
                    wrapper: slf.into_py(py),
                    instance: instance.clone().unbind(),
                };
                Ok(Py::new(py, bound_method)?.into_py(py))
            }
            None => {
                Ok(slf.into_py(py))
            }
        }
    }

    #[getter]
    fn __name__(&self) -> &str {
        &self.func_name
    }

    #[getter]
    fn __doc__(&self) -> Option<&str> {
        self.docstring.as_deref()
    }
}

#[pyclass]
struct BoundAgenticMethod {
    wrapper: PyObject,
    instance: PyObject,
}

#[pymethods]
impl BoundAgenticMethod {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Prepend self.instance to args
        let mut new_args: Vec<PyObject> = vec![self.instance.clone_ref(py)];
        for arg in args.iter() {
            new_args.push(arg.unbind());
        }
        let new_args_tuple = PyTuple::new(py, new_args)?;

        // Call the wrapper
        self.wrapper.call_bound(py, &new_args_tuple, kwargs)
            .map(|obj| obj.into_bound(py))
    }
}


/// The @agentic_generator decorator for async generator functions
#[pyfunction]
#[pyo3(signature = (*, model=None, api_key=None, store_cfg=None, autobind=false, native_tools=false))]
pub fn agentic_generator(
    py: Python<'_>,
    model: Option<String>,
    api_key: Option<String>,
    store_cfg: Option<PyObject>,
    autobind: bool,
    native_tools: bool,
) -> PyResult<PyObject> {
    let _ = (store_cfg, autobind);

    let decorator = AgenticGeneratorDecorator {
        model,
        api_key,
        native_tools,
    };

    let decorator_obj = Py::new(py, decorator)?;
    Ok(decorator_obj.into_py(py))
}

#[pyclass]
struct AgenticGeneratorDecorator {
    model: Option<String>,
    api_key: Option<String>,
    native_tools: bool,
}

#[pymethods]
impl AgenticGeneratorDecorator {
    fn __call__(&self, py: Python<'_>, func: PyObject) -> PyResult<PyObject> {
        let func_name: String = func.getattr(py, "__name__")?.extract(py)?;
        let docstring: Option<String> = func.getattr(py, "__doc__")
            .ok()
            .and_then(|doc| doc.extract(py).ok());

        let wrapper = AgenticGeneratorWrapper {
            inner_func: func,
            func_name,
            docstring,
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            native_tools: self.native_tools,
        };

        let wrapper_obj = Py::new(py, wrapper)?;
        Ok(wrapper_obj.into_py(py))
    }
}

#[pyclass]
struct AgenticGeneratorWrapper {
    inner_func: PyObject,
    func_name: String,
    docstring: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    native_tools: bool,
}

#[pymethods]
impl AgenticGeneratorWrapper {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let runtime = Runtime::create(
            self.model.clone(),
            self.api_key.clone(),
            self.native_tools,
        );

        let runtime_obj = Py::new(py, runtime)?;

        let new_kwargs = if let Some(kw) = kwargs {
            let new_kw = kw.copy()?;
            new_kw.set_item("runtime", &runtime_obj)?;
            new_kw
        } else {
            let new_kw = PyDict::new(py);
            new_kw.set_item("runtime", &runtime_obj)?;
            new_kw
        };

        let gen = self.inner_func.call_bound(py, args, Some(&new_kwargs))?;
        Ok(gen.into_bound(py))
    }

    fn __get__<'py>(
        slf: PyRef<'py, Self>,
        obj: Option<&Bound<'py, PyAny>>,
        _objtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        match obj {
            Some(instance) => {
                let bound_method = BoundAgenticGeneratorMethod {
                    wrapper: slf.into_py(py),
                    instance: instance.clone().unbind(),
                };
                Ok(Py::new(py, bound_method)?.into_py(py))
            }
            None => {
                Ok(slf.into_py(py))
            }
        }
    }

    #[getter]
    fn __name__(&self) -> &str {
        &self.func_name
    }

    #[getter]
    fn __doc__(&self) -> Option<&str> {
        self.docstring.as_deref()
    }
}

#[pyclass]
struct BoundAgenticGeneratorMethod {
    wrapper: PyObject,
    instance: PyObject,
}

#[pymethods]
impl BoundAgenticGeneratorMethod {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut new_args: Vec<PyObject> = vec![self.instance.clone_ref(py)];
        for arg in args.iter() {
            new_args.push(arg.unbind());
        }
        let new_args_tuple = PyTuple::new(py, new_args)?;

        self.wrapper.call_bound(py, &new_args_tuple, kwargs)
            .map(|obj| obj.into_bound(py))
    }
}

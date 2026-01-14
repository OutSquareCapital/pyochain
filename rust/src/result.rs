use crate::option::{PyNone, PySome, get_none_singleton};
use crate::types::{PyClassInit, call_func};
use pyderive::*;
use pyo3::exceptions::PyValueError;
use pyo3::{
    prelude::*,
    types::{PyDict, PyString, PyTuple},
};
/// Exception raised when unwrapping fails on Result types
#[pyclass(extends = PyValueError)]
pub struct ResultUnwrapError;

#[pymethods]
impl ResultUnwrapError {
    #[new]
    fn new(_exc_arg: &Bound<'_, PyAny>) -> Self {
        ResultUnwrapError
    }
}
#[pyclass(frozen, name = "Result", generic)]
pub struct PyochainResult;

#[derive(PyMatchArgs)]
#[pyclass(frozen, name = "Ok", generic)]
pub struct PyOk {
    #[pyo3(get)]
    pub value: Py<PyAny>,
}

#[pymethods]
impl PyOk {
    #[new]
    fn new(value: Py<PyAny>) -> Self {
        PyOk { value }
    }

    fn is_ok(&self) -> bool {
        true
    }

    fn is_err(&self) -> bool {
        false
    }

    fn ok(&self, py: Python<'_>) -> PyResult<Py<PySome>> {
        Ok(PySome::new(self.value.clone_ref(py)).init(py)?)
    }

    fn err(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        get_none_singleton(py)
    }

    fn unwrap(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    fn expect(&self, msg: &Bound<'_, PyString>) -> Py<PyAny> {
        self.value.clone_ref(msg.py())
    }

    fn unwrap_or(&self, default: &Bound<'_, PyAny>) -> Py<PyAny> {
        self.value.clone_ref(default.py())
    }

    fn unwrap_or_else(&self, f: &Bound<'_, PyAny>) -> Py<PyAny> {
        self.value.clone_ref(f.py())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn map(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(PyOk {
            value: call_func(func, &self.value.bind(func.py()), args, kwargs)?.unbind(),
        })
    }

    fn and_(&self, resb: &Bound<'_, PyAny>) -> Py<PyAny> {
        resb.to_owned().unbind()
    }

    fn or_(&self, rese: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyOk {
            value: self.value.clone_ref(rese.py()),
        })
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &self.value.bind(func.py()), args, kwargs)?.unbind())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> Self {
        PyOk {
            value: self.value.clone_ref(f.py()),
        }
    }

    fn unwrap_err(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(
            "called `unwrap_err` on Ok",
        ))
    }

    fn expect_err(&self, msg: &Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
        let ok_repr = self.value.bind(msg.py()).repr()?.to_string();
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "{}: expected Err, got Ok({})",
            msg, ok_repr
        )))
    }

    fn flatten(&self, py: Python<'_>) -> Py<PyAny> {
        // For Ok[Result[T, E], E], the self.value IS the inner Result
        self.value.clone_ref(py)
    }

    fn iter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.ok(py)?.bind(py).call_method0("iter")?.unbind())
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyOk {
            value: func
                .call(self.value.bind(func.py()).cast::<PyTuple>()?, None)?
                .unbind(),
        })
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value_tuple = self.value.bind(func.py()).cast::<PyTuple>()?;
        Ok(func.call(value_tuple, None)?.unbind())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &slf, args, kwargs)?.unbind())
    }

    #[pyo3(signature = (pred, *args, **kwargs))]
    fn is_ok_and(
        &self,
        pred: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        call_func(pred, &self.value.bind(pred.py()), args, kwargs)?.is_truthy()
    }

    #[pyo3(signature = (_pred, *_args, **_kwargs))]
    fn is_err_and(
        &self,
        _pred: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn map_err(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyOk {
            value: self.value.clone_ref(func.py()),
        }
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn inspect_err(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyOk {
            value: self.value.clone_ref(func.py()),
        }
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.value.bind(py);
        if let Ok(some_ref) = inner.extract::<PyRef<PySome>>() {
            let ok_value = Py::new(
                py,
                PyOk {
                    value: some_ref.value.clone_ref(py),
                },
            )?
            .into_any();
            Ok(PySome::new(ok_value).init(py)?.into_any())
        } else if inner.is_instance_of::<PyNone>() {
            get_none_singleton(py)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Some or NONE result",
            ))
        }
    }

    #[pyo3(signature = (default, func, *args, **kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &self.value.bind(default.py()), args, kwargs)?.unbind())
    }

    fn map_or_else(&self, ok: &Bound<'_, PyAny>, _err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(ok.call1((&self.value,))?.unbind())
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = f.py();
        call_func(f, &self.value.bind(py), args, kwargs)?;
        Ok(PyOk {
            value: self.value.clone_ref(py),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let value_repr = self.value.bind(py).repr()?;
        Ok(format!("Ok({})", value_repr))
    }
}

/// Err(error) - Result variant containing an error value
#[derive(PyMatchArgs)]
#[pyclass(frozen, name = "Err", generic)]
pub struct PyErr {
    #[pyo3(get)]
    pub error: Py<PyAny>,
}

#[pymethods]
impl PyErr {
    #[new]
    fn new(error: Py<PyAny>) -> Self {
        PyErr { error }
    }

    fn is_ok(&self) -> bool {
        false
    }

    fn is_err(&self) -> bool {
        true
    }

    fn ok(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        get_none_singleton(py)
    }

    fn err(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PySome::new(self.error.clone_ref(py)).init(py)?.into_any())
    }

    fn unwrap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let err_repr = self.error.bind(py).repr()?.to_string();
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "called `unwrap` on an `Err`: {}",
            err_repr
        )))
    }

    fn expect(&self, msg: &Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
        let err_repr = self.error.bind(msg.py()).repr()?.to_string();
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "{}: {}",
            msg, err_repr
        )))
    }

    fn expect_err(&self, _msg: String, py: Python<'_>) -> Py<PyAny> {
        self.error.clone_ref(py)
    }

    fn iter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyModule::import(py, "pyochain")?
            .getattr("Iter")?
            .call1((PyTuple::empty(py),))?
            .unbind())
    }

    fn unwrap_or(&self, default: Py<PyAny>) -> Py<PyAny> {
        default
    }

    fn unwrap_or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let result = f.call1((&self.error,))?;
        Ok(result.unbind())
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn map(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(func.py()),
        }
    }

    fn and_(&self, resb: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(resb.py()),
        }
    }

    fn or_(&self, rese: &Bound<'_, PyAny>) -> Py<PyAny> {
        rese.to_owned().unbind()
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(func.py()),
        }
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.error,))?.unbind())
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(func.py()),
        }
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(func.py()),
        }
    }

    fn unwrap_err(&self, py: Python<'_>) -> Py<PyAny> {
        self.error.clone_ref(py)
    }

    #[pyo3(signature = (_pred, *_args, **_kwargs))]
    fn is_ok_and(
        &self,
        _pred: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (pred, *args, **kwargs))]
    fn is_err_and(
        &self,
        pred: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        call_func(pred, &self.error.bind(pred.py()), args, kwargs)?.is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn map_err(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(PyErr {
            error: call_func(func, &self.error.bind(func.py()), args, kwargs)?.unbind(),
        })
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn inspect_err(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = func.py();
        call_func(func, &self.error.bind(py), args, kwargs)?;
        Ok(PyErr {
            error: self.error.clone_ref(py),
        })
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let err_value = Py::new(
            py,
            PyErr {
                error: self.error.clone_ref(py),
            },
        )?
        .into_any();
        Ok(PySome::new(err_value).init(py)?.into_any())
    }

    #[pyo3(signature = (default, _func, *_args, **_kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Py<PyAny> {
        default.to_owned().unbind()
    }

    fn map_or_else(&self, _ok: &Bound<'_, PyAny>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(err.call1((&self.error,))?.unbind())
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn filter(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(func.py()),
        }
    }

    #[pyo3(signature = (f, *_args, **_kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(f.py()),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let error_repr = self.error.bind(py).repr()?;
        Ok(format!("Err({})", error_repr))
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &slf, args, kwargs)?.unbind())
    }
}

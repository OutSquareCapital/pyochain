use crate::option::{PyNone, PySome, PyochainOption, get_none_singleton};
use crate::types::{ResultUnwrapError, concatenate, concatenate_self};
use pyderive::*;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
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

    fn ok(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: self.value.clone_ref(py),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn err(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        get_none_singleton(py)
    }

    fn unwrap(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    fn expect(&self, py: Python<'_>, _msg: String) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    fn unwrap_or(&self, py: Python<'_>, _default: Py<PyAny>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    fn unwrap_or_else(&self, py: Python<'_>, _f: &Bound<'_, PyAny>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn map(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let all_args = unsafe { concatenate(py, &self.value, args) };
        Ok(PyOk {
            value: func.call(&all_args, kwargs)?.unbind(),
        })
    }

    fn and_(&self, _py: Python<'_>, resb: &Bound<'_, PyAny>) -> Py<PyAny> {
        resb.to_owned().unbind()
    }

    fn or_(&self, py: Python<'_>, _rese: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyOk {
            value: self.value.clone_ref(py),
        })
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn and_then(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = unsafe { concatenate(py, &self.value, args) };
        Ok(func.call(&all_args, kwargs)?.unbind())
    }

    fn or_else(&self, py: Python<'_>, _f: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyOk {
            value: self.value.clone_ref(py),
        })
    }

    fn unwrap_err(&self) -> PyResult<Py<PyAny>> {
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(
            "called `unwrap_err` on Ok",
        ))
    }

    fn expect_err(&self, py: Python<'_>, msg: String) -> PyResult<Py<PyAny>> {
        let ok_repr = self.value.bind(py).repr()?.to_string();
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "{}: expected Err, got Ok({})",
            msg, ok_repr
        )))
    }

    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // For Ok[Result[T, E], E], the self.value IS the inner Result
        Ok(self.value.clone_ref(py))
    }

    fn iter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.ok(py)?.bind(py).call_method0("iter")?.unbind())
    }

    fn map_star(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyOk {
            value: func
                .call(self.value.bind(py).cast::<PyTuple>()?, None)?
                .unbind(),
        })
    }

    fn and_then_star(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value_tuple = self.value.bind(py).cast::<PyTuple>()?;
        Ok(func.call(value_tuple, None)?.unbind())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        concatenate_self(slf.py(), func, slf.as_ptr(), args, kwargs)
    }

    #[pyo3(signature = (pred, *args, **kwargs))]
    fn is_ok_and(
        &self,
        py: Python<'_>,
        pred: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let all_args = unsafe { concatenate(py, &self.value, args) };
        pred.call(&all_args, kwargs)?.is_truthy()
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

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn map_err(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyOk {
            value: self.value.clone_ref(py),
        }
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn inspect_err(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyOk {
            value: self.value.clone_ref(py),
        }
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.value.bind(py);
        if let Ok(some_ref) = inner.extract::<PyRef<PySome>>() {
            let unwrapped = some_ref.value.clone_ref(py);
            let ok_value = Py::new(py, PyOk { value: unwrapped })?.into_any();
            let some_init = PyClassInitializer::from(crate::option::PyochainOption)
                .add_subclass(PySome { value: ok_value });
            Ok(Py::new(py, some_init)?.into_any())
        } else if inner.is_instance_of::<PyNone>() {
            get_none_singleton(py)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Some or NONE result",
            ))
        }
    }

    #[pyo3(signature = (_default, func, *args, **kwargs))]
    fn map_or(
        &self,
        py: Python<'_>,
        _default: &Bound<'_, PyAny>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = unsafe { concatenate(py, &self.value, args) };
        Ok(func.call(&all_args, kwargs)?.unbind())
    }

    fn map_or_else(
        &self,
        _py: Python<'_>,
        ok: &Bound<'_, PyAny>,
        _err: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(ok.call1((&self.value,))?.unbind())
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        py: Python<'_>,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let all_args = unsafe { concatenate(py, &self.value, args) };
        f.call(&all_args, kwargs)?;
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
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: self.error.clone_ref(py),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn unwrap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let err_repr = self.error.bind(py).repr()?.to_string();
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "called `unwrap` on an `Err`: {}",
            err_repr
        )))
    }

    fn expect(&self, py: Python<'_>, msg: String) -> PyResult<Py<PyAny>> {
        let err_repr = self.error.bind(py).repr()?.to_string();
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

    fn unwrap_or(&self, _py: Python<'_>, default: Py<PyAny>) -> Py<PyAny> {
        default
    }

    fn unwrap_or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let result = f.call1((&self.error,))?;
        Ok(result.unbind())
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn map(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
        }
    }

    fn and_(&self, py: Python<'_>, _resb: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
        }
    }

    fn or_(&self, _py: Python<'_>, rese: &Bound<'_, PyAny>) -> Py<PyAny> {
        rese.to_owned().unbind()
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn and_then(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
        }
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.error,))?.unbind())
    }

    fn map_star(&self, py: Python<'_>, _func: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
        }
    }

    fn and_then_star(&self, py: Python<'_>, _func: &Bound<'_, PyAny>) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
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
        py: Python<'_>,
        pred: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let all_args = unsafe { concatenate(py, &self.error, args) };
        pred.call(&all_args, kwargs)?.is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn map_err(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let all_args = unsafe { concatenate(py, &self.error, args) };
        let result = func.call(&all_args, kwargs)?;
        Ok(PyErr {
            error: result.unbind(),
        })
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn inspect_err(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let all_args = unsafe { concatenate(py, &self.error, args) };
        func.call(&all_args, kwargs)?;
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
        let some_init = PyClassInitializer::from(crate::option::PyochainOption)
            .add_subclass(PySome { value: err_value });
        Ok(Py::new(py, some_init)?.into_any())
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

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn filter(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
        }
    }

    #[pyo3(signature = (_f, *_args, **_kwargs))]
    fn inspect(
        &self,
        py: Python<'_>,
        _f: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyErr {
            error: self.error.clone_ref(py),
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
        concatenate_self(slf.py(), func, slf.as_ptr(), args, kwargs)
    }
}

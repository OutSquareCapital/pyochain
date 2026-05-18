use crate::args::{Args, Concatenate, Kwargs};
use crate::errors::ResultUnwrapError;
use crate::hasher::hash_fn;
use crate::option::{PyNull, PySome, get_null};
use pyderive::*;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyBaseException;
use pyo3::{
    prelude::*,
    types::{PyString, PyTuple},
};
fn format_err_value(error: &Bound<'_, PyAny>) -> PyResult<String> {
    match error.is_instance_of::<PyBaseException>() {
        true => {
            let error_type = error.get_type();
            let module_name = error_type.getattr("__module__")?.extract::<String>()?;
            let type_name = error_type.getattr("__qualname__")?.extract::<String>()?;
            let display_name = match module_name.as_str() {
                "builtins" => type_name,
                _ => format!("{}.{}", module_name, type_name),
            };
            let error_message = error.str()?.extract::<String>()?;
            match error_message.is_empty() {
                true => Ok(display_name),
                false => Ok(format!("{}: {}", display_name, error_message)),
            }
        }
        false => error.repr()?.extract::<String>(),
    }
}

/// Result[T, E] - Generic Result type with Ok and Err variants for Python typing
#[pyclass(frozen, name = "Result", generic)]
pub struct PyochainResult;

#[derive(PyMatchArgs)]
#[pyclass(frozen, name = "Ok", generic)]
pub struct PyoOk {
    #[pyo3(get)]
    pub value: Py<PyAny>,
}

#[pymethods]
impl PyoOk {
    #[new]
    pub fn new(value: Py<PyAny>) -> Self {
        PyoOk { value }
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let value_repr = self.value.bind(py).repr()?;
        Ok(format!("Ok({})", value_repr))
    }
    fn is_ok(&self) -> bool {
        true
    }

    fn is_err(&self) -> bool {
        false
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(hash_fn(0_u8, self.value.bind(py).hash()?))
    }

    fn ok(&self, py: Python<'_>) -> PySome {
        PySome::new(self.value.clone_ref(py))
    }

    fn err(&self, py: Python<'_>) -> Py<PyNull> {
        get_null(py)
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
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Self> {
        Ok(PyoOk::new(
            func.concat(&self.value.bind(func.py()), args, kwargs)?
                .unbind(),
        ))
    }

    fn and_(&self, resb: &Bound<'_, PyAny>) -> Py<PyAny> {
        resb.to_owned().unbind()
    }

    fn or_(&self, rese: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyoOk::new(self.value.clone_ref(rese.py())))
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(func
            .concat(&self.value.bind(func.py()), args, kwargs)?
            .unbind())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> Self {
        PyoOk::new(self.value.clone_ref(f.py()))
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
        Ok(self.ok(py).into_py_any(py)?.call_method0(py, "iter")?)
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyoOk::new(
            func.call(self.value.bind(func.py()).cast::<PyTuple>()?, None)?
                .unbind(),
        ))
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value_tuple = self.value.bind(func.py()).cast::<PyTuple>()?;
        Ok(func.call(value_tuple, None)?.unbind())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(func.concat(&slf, args, kwargs)?.unbind())
    }

    #[pyo3(signature = (pred, *args, **kwargs))]
    fn is_ok_and(
        &self,
        pred: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<bool> {
        pred.concat(&self.value.bind(pred.py()), args, kwargs)?
            .is_truthy()
    }

    #[pyo3(signature = (_pred, *_args, **_kwargs))]
    fn is_err_and(
        &self,
        _pred: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn map_err(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Self {
        PyoOk::new(self.value.clone_ref(func.py()))
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn inspect_err(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Self {
        PyoOk::new(self.value.clone_ref(func.py()))
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.value.bind(py).cast_exact::<PySome>() {
            Ok(some_ref) => {
                let ok_value = PyoOk::new(some_ref.get().value.clone_ref(py)).into_py_any(py)?;
                PySome::new(ok_value).into_py_any(py)
            }
            Err(_) => get_null(py).into_py_any(py),
        }
    }

    #[pyo3(signature = (default, func, *args, **kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(func
            .concat(&self.value.bind(default.py()), args, kwargs)?
            .unbind())
    }

    fn map_or_else(&self, ok: &Bound<'_, PyAny>, _err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(ok.call1((&self.value,))?.unbind())
    }
    fn swap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyoErr::new(self.value.clone_ref(py)).into_py_any(py)?)
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Self> {
        let py = f.py();
        f.concat(&self.value.bind(py), args, kwargs)?;
        Ok(PyoOk::new(self.value.clone_ref(py)))
    }
}

/// Err(error) - Result variant containing an error value
#[derive(PyMatchArgs)]
#[pyclass(frozen, name = "Err", generic)]
pub struct PyoErr {
    #[pyo3(get)]
    pub error: Py<PyAny>,
}

#[pymethods]
impl PyoErr {
    #[new]
    pub fn new(error: Py<PyAny>) -> Self {
        PyoErr { error }
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let error_repr = self.error.bind(py).repr()?;
        Ok(format!("Err({})", error_repr))
    }

    fn is_ok(&self) -> bool {
        false
    }

    fn is_err(&self) -> bool {
        true
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(hash_fn(1, self.error.bind(py).hash()?))
    }

    fn ok(&self, py: Python<'_>) -> Py<PyNull> {
        get_null(py)
    }

    fn err(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        PySome::new(self.error.clone_ref(py)).into_py_any(py)
    }

    fn unwrap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let err_repr = format_err_value(&self.error.bind(py))?;
        Err(pyo3::PyErr::new::<ResultUnwrapError, _>(format!(
            "called `unwrap` on an `Err`: {}",
            err_repr
        )))
    }

    fn expect(&self, msg: &Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
        let err_repr = format_err_value(&self.error.bind(msg.py()))?;
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
    fn map(&self, func: &Bound<'_, PyAny>, _args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        PyoErr::new(self.error.clone_ref(func.py()))
    }

    fn and_(&self, resb: &Bound<'_, PyAny>) -> Self {
        PyoErr::new(self.error.clone_ref(resb.py()))
    }

    fn or_(&self, rese: &Bound<'_, PyAny>) -> Py<PyAny> {
        rese.to_owned().unbind()
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Self {
        PyoErr::new(self.error.clone_ref(func.py()))
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.error,))?.unbind())
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> Self {
        PyoErr::new(self.error.clone_ref(func.py()))
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> Self {
        PyoErr::new(self.error.clone_ref(func.py()))
    }

    fn unwrap_err(&self, py: Python<'_>) -> Py<PyAny> {
        self.error.clone_ref(py)
    }

    #[pyo3(signature = (_pred, *_args, **_kwargs))]
    fn is_ok_and(
        &self,
        _pred: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (pred, *args, **kwargs))]
    fn is_err_and(
        &self,
        pred: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<bool> {
        pred.concat(&self.error.bind(pred.py()), args, kwargs)?
            .is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn map_err(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Self> {
        Ok(PyoErr::new(
            func.concat(&self.error.bind(func.py()), args, kwargs)?
                .unbind(),
        ))
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn inspect_err(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Self> {
        let py = func.py();
        func.concat(&self.error.bind(py), args, kwargs)?;
        Ok(PyoErr::new(self.error.clone_ref(py)))
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let err_value = PyoErr::new(self.error.clone_ref(py)).into_py_any(py)?;
        PySome::new(err_value).into_py_any(py)
    }

    #[pyo3(signature = (default, _func, *_args, **_kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        _func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
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
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Self {
        PyoErr::new(self.error.clone_ref(func.py()))
    }

    #[pyo3(signature = (f, *_args, **_kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Self {
        PyoErr::new(self.error.clone_ref(f.py()))
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(func.concat(&slf, args, kwargs)?.unbind())
    }
    fn swap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyoOk::new(self.error.clone_ref(py)).into_py_any(py)?)
    }
}

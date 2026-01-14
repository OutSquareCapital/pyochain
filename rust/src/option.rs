use crate::result;
use crate::types::{PyClassInit, call_func};
use pyo3::exceptions::PyValueError;
use pyo3::{
    ffi,
    prelude::*,
    sync::PyOnceLock,
    types::{PyDict, PyString, PyTuple},
};
use std::sync::atomic::{AtomicPtr, Ordering};
/// Exception raised when unwrapping fails on Option types
#[pyclass(extends = PyValueError)]
pub struct OptionUnwrapError;

#[pymethods]
impl OptionUnwrapError {
    #[new]
    fn new(_exc_arg: &Bound<'_, PyAny>) -> Self {
        OptionUnwrapError
    }
}

// Singleton for NONE - initialized once per Python interpreter
pub static NONE_SINGLETON: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

// Raw pointer for fast identity comparison (avoids clone_ref + bind overhead)
static NONE_PTR: AtomicPtr<ffi::PyObject> = AtomicPtr::new(std::ptr::null_mut());
#[inline]
pub fn get_none_singleton(py: Python<'_>) -> PyResult<Py<PyAny>> {
    NONE_SINGLETON
        .get_or_try_init(py, || {
            let singleton = PyNone::new().init(py)?.into_any();
            NONE_PTR.store(singleton.as_ptr(), Ordering::Release);
            Ok(singleton)
        })
        .map(|singleton| singleton.clone_ref(py))
}

#[pyclass(frozen, name = "Option", generic, subclass)]
pub struct PyochainOption;

#[pymethods]
impl PyochainOption {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Py<PyochainOption>> {
        // SAFETY: This unsafe block is justified because:
        // 1. `into_ptr()` transfers ownership of Py<PyAny> to a raw pointer
        // 2. We immediately reconstruct Py<PyAny> via `from_owned_ptr` with the same Python context
        // 3. No other code touches the pointer between these operations
        // 4. PyO3's lifetime guarantees ensure `py` remains valid throughout
        unsafe {
            let py = value.py();
            let result_ptr = if value.is_none() {
                get_none_singleton(py)?.into_ptr()
            } else {
                PySome::new(value.to_owned().unbind())
                    .init(py)?
                    .into_any()
                    .into_ptr()
            };
            let py_any: Py<PyAny> = Py::from_owned_ptr(py, result_ptr);
            py_any
                .extract::<Py<PyochainOption>>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (value, *, predicate))]
    fn if_true(value: &Bound<'_, PyAny>, predicate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = value.py();
        if predicate.call1((value,))?.is_truthy()? {
            Ok(PySome::new(value.to_owned().unbind()).init(py)?.into_any())
        } else {
            get_none_singleton(py)
        }
    }

    #[staticmethod]
    fn if_some(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = value.py();
        if value.is_truthy()? {
            Ok(PySome::new(value.to_owned().unbind()).init(py)?.into_any())
        } else {
            get_none_singleton(py)
        }
    }
}

#[pyclass(frozen, name = "Some", generic, extends = PyochainOption)]
pub struct PySome {
    #[pyo3(get)]
    pub value: Py<PyAny>,
}

#[pymethods]
impl PySome {
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("value",)
    }

    #[new]
    pub fn new(value: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyochainOption).add_subclass(PySome { value })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        if let Ok(other_some) = other.extract::<PyRef<PySome>>() {
            self.value.bind(py).eq(&other_some.value)
        } else if other.is_instance_of::<PyNone>() {
            Ok(false)
        } else {
            self.value.bind(py).eq(other)
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.",
        ))
    }

    fn is_some(&self) -> bool {
        true
    }

    fn is_none(&self) -> bool {
        false
    }

    #[pyo3(signature = (predicate, *args, **kwargs))]
    fn is_some_and(
        &self,
        predicate: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let py = predicate.py();
        call_func(predicate, &self.value.bind(py), args, kwargs)?.is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn is_none_or(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let py = func.py();
        call_func(func, &self.value.bind(py), args, kwargs)?.is_truthy()
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
    ) -> PyResult<Py<PyAny>> {
        let py = func.py();
        Ok(
            PySome::new(call_func(func, &self.value.bind(py), args, kwargs)?.unbind())
                .init(py)?
                .into_any(),
        )
    }

    fn and_(&self, optb: &Bound<'_, PyAny>) -> Py<PyAny> {
        optb.to_owned().unbind()
    }
    fn or_(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = optb.py();
        Ok(PySome::new(self.value.clone_ref(py)).init(py)?.into_any())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = func.py();
        Ok(call_func(func, &self.value.bind(py), args, kwargs)?.unbind())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = f.py();
        Ok(PySome::new(self.value.clone_ref(py)).init(py)?.into_any())
    }

    fn ok_or(&self, err: &Bound<'_, PyAny>) -> PyResult<result::PyOk> {
        Ok(result::PyOk {
            value: self.value.clone_ref(err.py()),
        })
    }

    fn ok_or_else(&self, err: &Bound<'_, PyAny>) -> PyResult<result::PyOk> {
        Ok(result::PyOk {
            value: self.value.clone_ref(err.py()),
        })
    }

    #[pyo3(signature = (default, f, *args, **kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = default.py();
        Ok(call_func(f, &self.value.bind(py), args, kwargs)?.unbind())
    }

    fn map_or_else(
        &self,
        _default: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.value,))?.unbind())
    }

    #[pyo3(signature = (predicate, *args, **kwargs))]
    fn filter(
        &self,
        predicate: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = predicate.py();
        if call_func(predicate, &self.value.bind(py), args, kwargs)?.is_truthy()? {
            Ok(PySome::new(self.value.clone_ref(py)).init(py)?.into_any())
        } else {
            get_none_singleton(py)
        }
    }

    fn flatten(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = f.py();
        call_func(f, &self.value.bind(py), args, kwargs)?;
        Ok(PySome::new(self.value.clone_ref(py)).init(py)?.into_any())
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (a, b) = self.value.bind(py).extract::<(Py<PyAny>, Py<PyAny>)>()?;
        Ok((
            PySome::new(a).init(py)?.into_any(),
            PySome::new(b).init(py)?.into_any(),
        ))
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = func.py();

        Ok(
            PySome::new(func.call1(self.value.bind(py).cast::<PyTuple>()?)?.unbind())
                .init(py)?
                .into_any(),
        )
    }
    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(func
            .call1(self.value.bind(func.py()).cast::<PyTuple>()?)?
            .unbind())
    }

    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_instance_of::<PyNone>() {
            return get_none_singleton(py);
        }
        let init = PySome::new(
            PyTuple::new(
                py,
                [
                    self.value.bind(py).clone(),
                    other.extract::<PyRef<PySome>>()?.value.bind(py).clone(),
                ],
            )?
            .unbind()
            .into_any(),
        );
        Ok(init.init(py)?.into_any())
    }

    fn zip_with(&self, other: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_instance_of::<PyNone>() {
            return get_none_singleton(py);
        }
        let value = f
            .call1((&self.value, &other.extract::<PyRef<PySome>>()?.value))?
            .unbind();
        Ok(PySome::new(value).init(py)?.into_any())
    }

    fn reduce(&self, other: &Bound<'_, PyAny>, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let value = if other.is_instance_of::<PyNone>() {
            self.value.clone_ref(py)
        } else {
            let other_some = other.extract::<PyRef<PySome>>()?;
            func.call1((&self.value, &other_some.value))?.unbind()
        };
        Ok(PySome::new(value).init(py)?.into_any())
    }

    fn xor(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = optb.py();
        if optb.is_instance_of::<PyNone>() {
            Ok(PySome::new(self.value.clone_ref(py)).init(py)?.into_any())
        } else {
            get_none_singleton(py)
        }
    }

    fn iter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyochain")?
            .getattr("Iter")?
            .call_method1("once", (&self.value,))?
            .unbind())
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.value.bind(py);
        if let Ok(ok_ref) = inner.extract::<PyRef<result::PyOk>>() {
            let unwrapped = ok_ref.value.clone_ref(py);
            let some_value = PySome::new(unwrapped).init(py)?.into_any();
            Ok(Py::new(py, result::PyOk { value: some_value })?.into_any())
        } else if let Ok(err_ref) = inner.extract::<PyRef<result::PyErr>>() {
            Ok(Py::new(
                py,
                result::PyErr {
                    error: err_ref.error.clone_ref(py),
                },
            )?
            .into_any())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Ok or Err result",
            ))
        }
    }

    fn eq(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_some) = other.extract::<PyRef<PySome>>() {
            self.value.bind(other.py()).eq(&other_some.value)
        } else {
            Ok(false)
        }
    }

    fn ne(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!self.eq(other)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let value_repr = self.value.bind(py).repr()?;
        Ok(format!("Some({})", value_repr))
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

#[pyclass(frozen, name = "NoneOption", extends = PyochainOption)]
#[derive(Clone, Copy)]
pub struct PyNone;

#[pymethods]
impl PyNone {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyochainOption).add_subclass(PyNone)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.",
        ))
    }

    fn is_some(&self) -> bool {
        false
    }

    fn is_none(&self) -> bool {
        true
    }

    #[pyo3(signature = (_predicate, *_args, **_kwargs))]
    fn is_some_and(
        &self,
        _predicate: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn is_none_or(
        &self,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> bool {
        true
    }

    fn unwrap(&self) -> PyResult<Py<PyAny>> {
        Err(PyErr::new::<OptionUnwrapError, _>(
            "called `unwrap` on a `None`",
        ))
    }

    fn expect(&self, msg: String) -> PyResult<Py<PyAny>> {
        Err(PyErr::new::<OptionUnwrapError, _>(format!(
            "{} (called `expect` on a `None`)",
            msg
        )))
    }

    fn unwrap_or(&self, default: Py<PyAny>) -> Py<PyAny> {
        default
    }

    fn unwrap_or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call0()?.unbind())
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn map(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        get_none_singleton(func.py())
    }
    fn and_(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        get_none_singleton(optb.py())
    }

    fn or_(&self, optb: Py<PyAny>) -> Py<PyAny> {
        optb
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        get_none_singleton(func.py())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call0()?.unbind())
    }

    fn ok_or(&self, err: &Bound<'_, PyAny>) -> result::PyErr {
        result::PyErr {
            error: err.to_owned().unbind(),
        }
    }

    fn ok_or_else(&self, err: &Bound<'_, PyAny>) -> PyResult<result::PyErr> {
        Ok(result::PyErr {
            error: err.call0()?.unbind(),
        })
    }

    #[pyo3(signature = (default, _f, *_args, **_kwargs))]
    fn map_or(
        &self,
        default: Py<PyAny>,
        _f: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Py<PyAny> {
        default
    }

    fn map_or_else(
        &self,
        default: &Bound<'_, PyAny>,
        _f: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(default.call0()?.unbind())
    }

    #[pyo3(signature = (predicate, *_args, **_kwargs))]
    fn filter(
        &self,
        predicate: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        get_none_singleton(predicate.py())
    }

    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        get_none_singleton(py)
    }

    #[pyo3(signature = (f, *_args, **_kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        get_none_singleton(f.py())
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let none = get_none_singleton(py)?;
        Ok((none.clone_ref(py), none))
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        get_none_singleton(func.py())
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        get_none_singleton(func.py())
    }

    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        get_none_singleton(other.py())
    }

    fn zip_with(&self, other: &Bound<'_, PyAny>, _f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        get_none_singleton(other.py())
    }

    fn reduce(&self, other: Py<PyAny>, _func: &Bound<'_, PyAny>) -> Py<PyAny> {
        other
    }

    fn xor(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if optb.is_instance_of::<PyNone>() {
            get_none_singleton(optb.py())
        } else {
            Ok(optb.clone().unbind())
        }
    }

    fn iter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyochain")?
            .getattr("Iter")?
            .call_method0("new")?
            .unbind())
    }

    fn transpose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(
            py,
            result::PyOk {
                value: get_none_singleton(py)?,
            },
        )?
        .into_any())
    }

    fn eq(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        slf.is(other)
    }

    fn ne(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        !slf.is(other)
    }

    fn __repr__(&self) -> &'static str {
        "NONE"
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.is_none() || other.is_instance_of::<PyNone>()
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

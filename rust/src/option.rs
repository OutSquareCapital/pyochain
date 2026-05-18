use crate::args::{Args, Concatenate, Kwargs};
use crate::errors::OptionUnwrapError;
use crate::hasher::hash_fn;
use crate::result::{PyoErr, PyoOk};
use pyo3::IntoPyObjectExt;
use pyo3::{
    prelude::*,
    sync::PyOnceLock,
    types::{PyString, PyTuple},
};

/// Singleton for NONE - initialized once per Python interpreter
static NONE: PyOnceLock<Py<PyNull>> = PyOnceLock::new();
/// Trait to check if a PyAny is the NONE singleton
pub trait IsNull<'py> {
    fn is_null(self) -> bool;
}
impl IsNull<'_> for &Bound<'_, PyAny> {
    #[inline]
    fn is_null(self) -> bool {
        self.as_ptr()
            == NONE
                .get(self.py())
                .expect("NONE singleton not initialized")
                .as_ptr()
    }
}
/// Called once on import to initialize the NONE singleton for the interpreter
pub fn init_null(py: Python<'_>) -> PyResult<()> {
    match NONE.get(py) {
        Some(_) => Ok(()),
        None => {
            NONE.set(py, Py::new(py, PyNull)?)
                .expect("NONE singleton should only be initialized once per interpreter");
            Ok(())
        }
    }
}

#[inline]
pub fn get_null(py: Python<'_>) -> Py<PyNull> {
    NONE.get(py)
        .expect("NONE singleton not initialized")
        .clone_ref(py)
}
/// Option[T] - Generic Option type with Some and None variants for Python typing
#[pyclass(frozen, name = "Option", generic)]
pub struct PyochainOption;
#[pyfunction]
pub fn option(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if value.is_none() {
        Ok(get_null(py).into_any())
    } else {
        PySome::new(value.to_owned().unbind()).into_py_any(py)
    }
}

#[pyfunction]
pub fn then_if_some(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if value.is_truthy()? {
        PySome::new(value.to_owned().unbind()).into_py_any(py)
    } else {
        Ok(get_null(py).into_any())
    }
}

#[pyfunction(signature = (value, *, predicate))]
pub fn then_if_true(value: &Bound<'_, PyAny>, predicate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if predicate.call1((value,))?.is_truthy()? {
        PySome::new(value.to_owned().unbind()).into_py_any(py)
    } else {
        Ok(get_null(py).into_any())
    }
}

#[pyclass(frozen, name = "Some", generic)]
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
    pub fn new(value: Py<PyAny>) -> Self {
        PySome { value }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        match other.cast_exact::<PySome>() {
            Ok(other_some) => self.value.bind(other.py()).eq(&other_some.get().value),
            Err(_) => Ok(false),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.",
        ))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(hash_fn(0_u8, self.value.bind(py).hash()?))
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
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<bool> {
        predicate
            .concat(&self.value.bind(predicate.py()), args, kwargs)?
            .is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn is_none_or(
        &self,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<bool> {
        func.concat(&self.value.bind(func.py()), args, kwargs)?
            .is_truthy()
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
    ) -> PyResult<Py<PyAny>> {
        let py = func.py();
        PySome::new(func.concat(&self.value.bind(py), args, kwargs)?.unbind()).into_py_any(py)
    }

    fn and_(&self, optb: &Bound<'_, PyAny>) -> Py<PyAny> {
        optb.to_owned().unbind()
    }
    fn or_(&self, optb: &Bound<'_, PyAny>) -> Self {
        let py = optb.py();
        PySome::new(self.value.clone_ref(py))
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
        let py = f.py();
        PySome::new(self.value.clone_ref(py))
    }

    fn ok_or(&self, err: &Bound<'_, PyAny>) -> PyoOk {
        PyoOk::new(self.value.clone_ref(err.py()))
    }

    fn ok_or_else(&self, err: &Bound<'_, PyAny>) -> PyoOk {
        PyoOk::new(self.value.clone_ref(err.py()))
    }

    #[pyo3(signature = (default, f, *args, **kwargs))]
    fn map_or(
        &self,
        default: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(f.concat(&self.value.bind(default.py()), args, kwargs)?
            .unbind())
    }

    #[allow(unused_variables)]
    fn map_or_else(&self, default: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.value,))?.unbind())
    }

    #[pyo3(signature = (predicate, *args, **kwargs))]
    fn filter(
        &self,
        predicate: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        let py = predicate.py();
        if predicate
            .concat(&self.value.bind(py), args, kwargs)?
            .is_truthy()?
        {
            PySome::new(self.value.clone_ref(py)).into_py_any(py)
        } else {
            Ok(get_null(py).into_any())
        }
    }

    fn flatten(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        let py = f.py();
        f.concat(&self.value.bind(py), args, kwargs)?;
        PySome::new(self.value.clone_ref(py)).into_py_any(py)
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (a, b) = self.value.bind(py).extract::<(Py<PyAny>, Py<PyAny>)>()?;
        Ok((
            PySome::new(a).into_py_any(py)?,
            PySome::new(b).into_py_any(py)?,
        ))
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = func.py();

        PySome::new(func.call1(self.value.bind(py).cast::<PyTuple>()?)?.unbind()).into_py_any(py)
    }
    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(func
            .call1(self.value.bind(func.py()).cast::<PyTuple>()?)?
            .unbind())
    }

    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_null() {
            return Ok(get_null(py).into_any());
        }
        let init = PySome::new(
            PyTuple::new(
                py,
                [
                    self.value.bind(py).clone(),
                    other.cast_exact::<PySome>()?.get().value.bind(py).clone(),
                ],
            )?
            .unbind()
            .into_any(),
        );
        init.into_py_any(py)
    }

    fn zip_with(&self, other: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_null() {
            return Ok(get_null(py).into_any());
        }
        let value = f
            .call1((&self.value, &other.cast_exact::<PySome>()?.get().value))?
            .unbind();
        PySome::new(value).into_py_any(py)
    }

    fn reduce(&self, other: &Bound<'_, PyAny>, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let value = if other.is_null() {
            self.value.clone_ref(py)
        } else {
            let other_some = other.cast_exact::<PySome>()?.get();
            func.call1((&self.value, &other_some.value))?.unbind()
        };
        PySome::new(value).into_py_any(py)
    }

    fn xor(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = optb.py();
        if optb.is_null() {
            PySome::new(self.value.clone_ref(py)).into_py_any(py)
        } else {
            Ok(get_null(py).into_any())
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
        match inner.cast_exact::<PyoOk>() {
            Ok(ok_ref) => {
                let some_value = PySome::new(ok_ref.get().value.clone_ref(py)).into_py_any(py)?;
                Ok(PyoOk::new(some_value).into_py_any(py)?)
            }
            Err(_) => {
                let err_ref = inner.cast_exact::<PyoErr>()?;
                Ok(PyoErr::new(err_ref.get().error.clone_ref(py)).into_py_any(py)?)
            }
        }
    }
    fn eq(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        match other.cast_exact::<PySome>() {
            Ok(other_some) => self.value.bind(other.py()).eq(&other_some.get().value),
            Err(_) => Ok(false),
        }
    }

    fn ne(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!self.eq(other)?)
    }
    fn unwrap_or_none(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let value_repr = self.value.bind(py).repr()?;
        Ok(format!("Some({})", value_repr))
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
}

#[pyclass(frozen, name = "Null")]
pub struct PyNull;

#[pymethods]
impl PyNull {
    #[new]
    fn new(py: Python<'_>) -> Py<PyNull> {
        get_null(py)
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.",
        ))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        py.None().bind(py).hash()
    }

    fn is_some(&self) -> bool {
        false
    }

    fn is_none(&self) -> bool {
        true
    }
    #[allow(unused_variables)]
    #[pyo3(signature = (predicate, *_args, **_kwargs))]
    fn is_some_and(
        &self,
        predicate: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> bool {
        false
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn is_none_or(
        &self,
        _func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
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
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Py<PyNull> {
        get_null(func.py())
    }
    fn and_(&self, optb: &Bound<'_, PyAny>) -> Py<Self> {
        get_null(optb.py())
    }

    fn or_(&self, optb: Py<PyAny>) -> Py<PyAny> {
        optb
    }

    #[pyo3(signature = (func, *_args, **_kwargs))]
    fn and_then(
        &self,
        func: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Py<Self> {
        get_null(func.py())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call0()?.unbind())
    }

    fn ok_or(&self, err: &Bound<'_, PyAny>) -> PyoErr {
        PyoErr::new(err.to_owned().unbind())
    }

    fn ok_or_else(&self, err: &Bound<'_, PyAny>) -> PyResult<PyoErr> {
        Ok(PyoErr {
            error: err.call0()?.unbind(),
        })
    }

    #[pyo3(signature = (default, _f, *_args, **_kwargs))]
    fn map_or(
        &self,
        default: Py<PyAny>,
        _f: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Py<PyAny> {
        default
    }
    #[allow(unused_variables)]
    fn map_or_else(&self, default: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(default.call0()?.unbind())
    }

    #[pyo3(signature = (predicate, *_args, **_kwargs))]
    fn filter(
        &self,
        predicate: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Py<Self> {
        get_null(predicate.py())
    }

    fn flatten(&self, py: Python<'_>) -> Py<Self> {
        get_null(py)
    }

    #[pyo3(signature = (f, *_args, **_kwargs))]
    fn inspect(
        &self,
        f: &Bound<'_, PyAny>,
        _args: &Args<'_>,
        _kwargs: Option<&Kwargs<'_>>,
    ) -> Py<Self> {
        get_null(f.py())
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<Self>, Py<Self>)> {
        let none = get_null(py);
        Ok((none.clone_ref(py), none))
    }

    fn map_star(&self, func: &Bound<'_, PyAny>) -> Py<Self> {
        get_null(func.py())
    }

    fn and_then_star(&self, func: &Bound<'_, PyAny>) -> Py<Self> {
        get_null(func.py())
    }

    fn zip(&self, other: &Bound<'_, PyAny>) -> Py<Self> {
        get_null(other.py())
    }

    fn zip_with(&self, other: &Bound<'_, PyAny>, _f: &Bound<'_, PyAny>) -> Py<Self> {
        get_null(other.py())
    }

    fn reduce(&self, other: Py<PyAny>, _func: &Bound<'_, PyAny>) -> Py<PyAny> {
        other
    }

    fn xor(&self, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if optb.is_null() {
            get_null(optb.py()).into_py_any(optb.py())
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
        PyoOk::new(get_null(py).into_any()).into_py_any(py)
    }

    fn eq(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        slf.is(other)
    }

    fn ne(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        !slf.is(other)
    }
    fn unwrap_or_none(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }
    fn __repr__(&self) -> &'static str {
        "NONE"
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.is_none() || other.is_null()
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
}

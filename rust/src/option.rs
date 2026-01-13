use crate::result;
use crate::types::{OptionUnwrapError, build_args};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple, PyType},
};
#[pyclass(frozen, name = "Option", generic, subclass)]
pub struct PyochainOption;

#[pymethods]
impl PyochainOption {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Py<PyochainOption>> {
        let result: Py<PyochainOption> = value
            .py()
            .get_type::<PyochainOption>()
            .call_method1("__internal_new__", (value,))?
            .extract()?;
        Ok(result)
    }
    // This is a hack to make the __new__ method generic over subclasses works
    // Otherwise, PyO3 does not allow calling __new__ of a base class with another return type
    // When this is fixed in PyO3, dropping the subclassing could be considered
    // This could also allow to derive PyMatchArgs for Some
    #[classmethod]
    fn __internal_new__(cls: &Bound<'_, PyType>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = cls.py();
        if value.is_none() {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
                value: value.to_owned().unbind(),
            });
            Ok(Py::new(py, init)?.into_any())
        }
    }

    #[staticmethod]
    #[pyo3(signature = (value, *, predicate))]
    fn if_true(
        py: Python<'_>,
        value: Py<PyAny>,
        predicate: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let result = predicate.call1((value.bind(py),))?;
        if result.is_truthy()? {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value });
            Ok(Py::new(py, init)?.into_any())
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
        }
    }

    #[staticmethod]
    fn if_some(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if value.bind(py).is_truthy()? {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value });
            Ok(Py::new(py, init)?.into_any())
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
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
    fn new(value: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyochainOption).add_subclass(PySome { value })
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
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
        py: Python<'_>,
        predicate: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let all_args = build_args(py, &self.value, args)?;
        predicate.call(&all_args, kwargs)?.is_truthy()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn is_none_or(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let all_args = build_args(py, &self.value, args)?;
        func.call(&all_args, kwargs)?.is_truthy()
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
    ) -> PyResult<Py<PyAny>> {
        let all_args = build_args(py, &self.value, args)?;
        let result = func.call(&all_args, kwargs)?;
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: result.unbind(),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn and_(&self, _py: Python<'_>, optb: &Bound<'_, PyAny>) -> Py<PyAny> {
        optb.to_owned().unbind()
    }

    fn or_(&self, py: Python<'_>, _optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: self.value.clone_ref(py),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn and_then(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = build_args(py, &self.value, args)?;
        Ok(func.call(&all_args, kwargs)?.unbind())
    }

    fn or_else(&self, py: Python<'_>, _f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: self.value.clone_ref(py),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn ok_or(&self, py: Python<'_>, _err: &Bound<'_, PyAny>) -> PyResult<result::PyOk> {
        Ok(result::PyOk {
            value: self.value.clone_ref(py),
        })
    }

    fn ok_or_else(&self, py: Python<'_>, _err: &Bound<'_, PyAny>) -> PyResult<result::PyOk> {
        Ok(result::PyOk {
            value: self.value.clone_ref(py),
        })
    }

    #[pyo3(signature = (_default, f, *args, **kwargs))]
    fn map_or(
        &self,
        py: Python<'_>,
        _default: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = build_args(py, &self.value, args)?;
        Ok(f.call(&all_args, kwargs)?.unbind())
    }

    fn map_or_else(
        &self,
        _py: Python<'_>,
        _default: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(f.call1((&self.value,))?.unbind())
    }

    #[pyo3(signature = (predicate, *args, **kwargs))]
    fn filter(
        &self,
        py: Python<'_>,
        predicate: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = build_args(py, &self.value, args)?;
        if predicate.call(&all_args, kwargs)?.is_truthy()? {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
                value: self.value.clone_ref(py),
            });
            Ok(Py::new(py, init)?.into_any())
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
        }
    }

    fn flatten(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        &self,
        py: Python<'_>,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let all_args = build_args(py, &self.value, args)?;
        f.call(&all_args, kwargs)?;
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: self.value.clone_ref(py),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let tuple = self.value.bind(py);
        let (a, b) = tuple.extract::<(Py<PyAny>, Py<PyAny>)>()?;
        let init_a = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value: a });
        let init_b = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value: b });
        Ok((
            Py::new(py, init_a)?.into_any(),
            Py::new(py, init_b)?.into_any(),
        ))
    }

    fn map_star(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value = func
            .call(self.value.bind(py).cast::<PyTuple>()?, None)?
            .unbind();
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value });
        Ok(Py::new(py, init)?.into_any())
    }

    fn and_then_star(&self, py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(func
            .call(self.value.bind(py).cast::<PyTuple>()?, None)?
            .unbind())
    }

    fn zip(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyNone>() {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            return Ok(Py::new(py, init)?.into_any());
        }
        let tuple = PyTuple::new(
            py,
            [
                self.value.bind(py).clone(),
                other.extract::<PyRef<PySome>>()?.value.bind(py).clone(),
            ],
        )?;
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
            value: tuple.unbind().into_any(),
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn zip_with(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyNone>() {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            return Ok(Py::new(py, init)?.into_any());
        }
        let value = f
            .call1((&self.value, &other.extract::<PyRef<PySome>>()?.value))?
            .unbind();
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value });
        Ok(Py::new(py, init)?.into_any())
    }

    fn reduce(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let value = if other.is_instance_of::<PyNone>() {
            self.value.clone_ref(py)
        } else {
            let other_some = other.extract::<PyRef<PySome>>()?;
            func.call1((&self.value, &other_some.value))?.unbind()
        };
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome { value });
        Ok(Py::new(py, init)?.into_any())
    }

    fn xor(&self, py: Python<'_>, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if optb.is_instance_of::<PyNone>() {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
                value: self.value.clone_ref(py),
            });
            Ok(Py::new(py, init)?.into_any())
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
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

        // Extract as a result to match on is_ok/is_err
        if let Ok(ok_ref) = inner.extract::<PyRef<result::PyOk>>() {
            let unwrapped = ok_ref.value.clone_ref(py);
            let some_init =
                PyClassInitializer::from(PyochainOption).add_subclass(PySome { value: unwrapped });
            let some_value = Py::new(py, some_init)?.into_any();
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

    fn eq(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_some) = other.extract::<PyRef<PySome>>() {
            self.value.bind(py).eq(&other_some.value)
        } else if other.is_instance_of::<PyNone>() {
            Ok(false)
        } else {
            Ok(false)
        }
    }

    fn ne(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!self.eq(py, other)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let value_repr = self.value.bind(py).repr()?;
        Ok(format!("Some({})", value_repr))
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let self_ptr = self as *const _ as *mut pyo3::ffi::PyObject;
        crate::types::call_with_self_prepended(py, func, self_ptr, args, kwargs)
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

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn map(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn and_(&self, py: Python<'_>, _optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn or_(&self, optb: Py<PyAny>) -> Py<PyAny> {
        optb
    }

    #[pyo3(signature = (_func, *_args, **_kwargs))]
    fn and_then(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn or_else(&self, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(f.call0()?.unbind())
    }

    fn ok_or(&self, _py: Python<'_>, err: &Bound<'_, PyAny>) -> PyResult<result::PyErr> {
        Ok(result::PyErr {
            error: err.to_owned().unbind(),
        })
    }

    fn ok_or_else(&self, _py: Python<'_>, err: &Bound<'_, PyAny>) -> PyResult<result::PyErr> {
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

    #[pyo3(signature = (_predicate, *_args, **_kwargs))]
    fn filter(
        &self,
        py: Python<'_>,
        _predicate: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    #[pyo3(signature = (_f, *_args, **_kwargs))]
    fn inspect(
        &self,
        py: Python<'_>,
        _f: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn unzip(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let init1 = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        let init2 = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok((
            Py::new(py, init1)?.into_any(),
            Py::new(py, init2)?.into_any(),
        ))
    }

    fn zip(&self, py: Python<'_>, _other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn zip_with(
        &self,
        py: Python<'_>,
        _other: &Bound<'_, PyAny>,
        _f: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        Ok(Py::new(py, init)?.into_any())
    }

    fn reduce(&self, other: Py<PyAny>, _func: &Bound<'_, PyAny>) -> Py<PyAny> {
        other
    }

    fn xor(&self, py: Python<'_>, optb: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if optb.is_instance_of::<PyNone>() {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
            Ok(Py::new(py, init)?.into_any())
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
        let none_init = PyClassInitializer::from(PyochainOption).add_subclass(PyNone);
        let none_value = Py::new(py, none_init)?.into_any();
        Ok(Py::new(py, result::PyOk { value: none_value })?.into_any())
    }

    fn eq(&self, other: &Bound<'_, PyAny>) -> bool {
        other.is_instance_of::<PyNone>()
    }

    fn ne(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.eq(other)
    }

    fn __repr__(&self) -> &'static str {
        "NONE"
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.is_none() || other.is_instance_of::<PyNone>()
    }

    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let self_ptr = self as *const _ as *mut pyo3::ffi::PyObject;
        crate::types::call_with_self_prepended(py, func, self_ptr, args, kwargs)
    }
}

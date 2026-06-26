use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tap::prelude::*;
/// Type alias representing the `*args` parameter in Python functions (or any argument that is expected to be a tuple)
pub type Args<'py> = Bound<'py, PyTuple>;
/// Type alias representing the `**kwargs` parameter in Python functions
pub type Kwargs<'py> = Bound<'py, PyDict>;

/// In python, you can make a very generic function signature like this:
/// ```python
/// from collections.abc import Callable
/// from typing import Concatenate
/// def foo[**P, T, R](
///     function: Callable[Concatenate[T, P], R],
///     value: T,
///     *args: P.args,
///     **kwargs: P.kwargs,
/// ) -> R:
///     return function(value, *args, **kwargs)
/// ```
/// This trait provides the `concat` method which allows you to implement this kind of behavior in Rust.\
/// It is implemented for `&Bound<'py, PyAny>`, so it can be used on any Python object.\
/// `self` is the function to call, `value` is the value to concatenate with `*args`, and `kwargs` are the keyword arguments to pass to the function.\
/// The provided methods handle various cases with presence or absence of args/kwargs, as well as the special case where `value` is itself a tuple that needs to be unpacked (similar to `itertools.starmap`).
pub trait Concatenate<'py> {
    /// Concatenate the provided value with the given `*args` and call the function with the resulting arguments and `**kwargs`
    fn concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// Same as concat star, but does not handle `**kwargs`. Use this whenever possible as it is faster.
    fn concat1(self, value: &Bound<'py, PyAny>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>>;
    /// Akin to `itertools::map_starmap`, where *value* is expected to be a tuple of arguments.\
    /// Unpack each item in *value* and concatenate it with the given `*args`, then call the function with the resulting arguments and `**kwargs`
    fn concat_star(
        self,
        value: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// same as `concat_star`, but does not handle `**kwargs`. Use this whenever possible as it is faster.
    fn concat_star1(self, value: &Args<'py>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>>;

    /// Prepend `acc` to `item` and concatenate with `args`, then call the function with `**kwargs`
    fn fold_concat_star(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// same as `fold_concat_star`, but does not handle `**kwargs`
    fn fold_concat_star1(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> Concatenate<'py> for &Bound<'py, PyAny> {
    #[inline]
    fn concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args_len = args.len();
        match args_len {
            0 => self.call((value,), kwargs),
            _ => self.call(
                unsafe { concat_val_with_args(&value, args, args_len) },
                kwargs,
            ),
        }
    }
    #[inline]
    fn concat1(self, value: &Bound<'py, PyAny>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call1(unsafe { concat_val_with_args(&value, args, args.len()) })
    }
    #[inline]
    fn concat_star(
        self,
        value: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args_len = args.len();
        match args_len {
            0 => self.call(value, kwargs),
            _ => self.call(
                unsafe { concat_tup_with_args(value, args, args_len) },
                kwargs,
            ),
        }
    }
    #[inline]
    fn concat_star1(self, value: &Args<'py>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call1(unsafe { concat_tup_with_args(value, args, args.len()) })
    }
    #[inline]
    fn fold_concat_star(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call(
            unsafe { concat_acc_tup_with_args(acc, item, args, args.len()) },
            kwargs,
        )
    }

    #[inline]
    fn fold_concat_star1(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call1(unsafe { concat_acc_tup_with_args(acc, item, args, args.len()) })
    }
}
pub trait ConcatWith<'py> {
    fn concat_with(self, others: &Args<'py>) -> PyResult<Bound<'py, PyTuple>>;
}
impl<'py> ConcatWith<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn concat_with(self, others: &Args<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let py = self.py();
        self.pipe(std::iter::once)
            .chain(others.iter())
            .collect::<Vec<Bound<'py, PyAny>>>()
            .pipe_ref(|x| PyTuple::new(py, x))
    }
}
#[inline]
unsafe fn concat_val_with_args<'py>(
    value: &Bound<'py, PyAny>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    unsafe {
        let ptr = value.as_ptr();
        let new_argc = args_len + 1;
        let new_args_ptr = ffi::PyTuple_New(new_argc as ffi::Py_ssize_t);
        ffi::Py_INCREF(ptr);
        ffi::PyTuple_SetItem(new_args_ptr, 0, ptr);

        let args_ptr = args.as_ptr();
        for i in 0..args_len {
            let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, (i + 1) as ffi::Py_ssize_t, item);
        }
        Bound::from_owned_ptr(value.py(), new_args_ptr).cast_into_unchecked::<PyTuple>()
    }
}
#[inline]
unsafe fn concat_tup_with_args<'py>(
    value: &Args<'py>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    unsafe {
        let tuple_len = value.len();
        let total_len = tuple_len + args_len;
        let new_args_ptr = ffi::PyTuple_New(total_len as ffi::Py_ssize_t);
        let tuple_ptr = value.as_ptr();
        for i in 0..tuple_len {
            let item = ffi::PyTuple_GET_ITEM(tuple_ptr, i as ffi::Py_ssize_t);
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, i as ffi::Py_ssize_t, item);
        }
        let args_ptr = args.as_ptr();
        for i in 0..args_len {
            let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, (tuple_len + i) as ffi::Py_ssize_t, item);
        }

        Bound::from_owned_ptr(value.py(), new_args_ptr).cast_into_unchecked::<PyTuple>()
    }
}
#[inline]
unsafe fn concat_acc_tup_with_args<'py>(
    acc: &Bound<'py, PyAny>,
    value: &Args<'py>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    unsafe {
        let tuple_len = value.len();
        let total_len = 1 + tuple_len + args_len;
        let new_args_ptr = ffi::PyTuple_New(total_len as ffi::Py_ssize_t);

        ffi::Py_INCREF(acc.as_ptr());
        ffi::PyTuple_SetItem(new_args_ptr, 0, acc.as_ptr());

        let tuple_ptr = value.as_ptr();
        for i in 0..tuple_len {
            let item = ffi::PyTuple_GET_ITEM(tuple_ptr, i as ffi::Py_ssize_t);
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, (1 + i) as ffi::Py_ssize_t, item);
        }
        let args_ptr = args.as_ptr();
        for i in 0..args_len {
            let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, (1 + tuple_len + i) as ffi::Py_ssize_t, item);
        }

        Bound::from_owned_ptr(acc.py(), new_args_ptr).cast_into_unchecked::<PyTuple>()
    }
}

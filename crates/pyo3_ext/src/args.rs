use pyo3::{
    ffi,
    prelude::*,
    types::{PyDict, PyTuple},
};
use tap::prelude::*;

use crate::iter::CollectBoundIterator;
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
pub trait CallConcat<'py> {
    /// Concatenate the provided value with the given `*args` and call the function with the resulting arguments and `**kwargs`
    fn call_concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// Same as concat star, but does not handle `**kwargs`. Use this whenever possible as it is faster.
    fn call_concat1(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// Akin to `itertools::map_starmap`, where *value* is expected to be a tuple of arguments.\
    /// Unpack each item in *value* and concatenate it with the given `*args`, then call the function with the resulting arguments and `**kwargs`
    fn call_concat_star(
        self,
        value: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// same as `call_concat_star`, but does not handle `**kwargs`. Use this whenever possible as it is faster.
    fn call_concat_star1(self, value: &Args<'py>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>>;

    /// Prepend `acc` to `item` and concatenate with `args`, then call the function with `**kwargs`
    fn call_fold_concat_star(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// same as `call_fold_concat_star`, but does not handle `**kwargs`
    fn call_fold_concat_star1(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> CallConcat<'py> for &Bound<'py, PyAny> {
    #[inline]
    fn call_concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args_len = args.len();
        match args_len {
            0 => self.call((value,), kwargs),
            _ => self.call(concat_val_with_args(value, args, args_len), kwargs),
        }
    }
    #[inline]
    fn call_concat1(
        self,
        value: &Bound<'py, PyAny>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call1(concat_val_with_args(value, args, args.len()))
    }
    #[inline]
    fn call_concat_star(
        self,
        value: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args_len = args.len();
        match args_len {
            0 => self.call(value, kwargs),
            _ => self.call(concat_tup_with_args(value, args, args_len), kwargs),
        }
    }
    #[inline]
    fn call_concat_star1(self, value: &Args<'py>, args: &Args<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call1(concat_tup_with_args(value, args, args.len()))
    }
    #[inline]
    fn call_fold_concat_star(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call(
            concat_acc_tup_with_args(acc, item, args, args.len()),
            kwargs,
        )
    }

    #[inline]
    fn call_fold_concat_star1(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call1(concat_acc_tup_with_args(acc, item, args, args.len()))
    }
}
pub trait CallWith<'py> {
    fn call_with(self, others: &Args<'py>) -> PyResult<Bound<'py, PyTuple>>;
    fn call_with_2(self, b: &Bound<'py, PyAny>, others: &Args<'py>) -> Bound<'py, PyTuple>;
}
impl<'py> CallWith<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn call_with(self, others: &Args<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let py = self.py();
        self.pipe(std::iter::once)
            .chain(others.iter())
            .collect::<Vec<Bound<'py, PyAny>>>()
            .into_iter()
            .collect_bound::<PyTuple>(py)
    }
    #[allow(clippy::cast_possible_wrap)]
    #[inline]
    fn call_with_2(
        self: Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        args: &Args<'py>,
    ) -> Bound<'py, PyTuple> {
        let mut builder = PyTupleBuilder::new(self.py(), args.len() + 2);
        builder.push(&self);
        builder.push(b);
        builder.extend(args);
        builder.finish()
    }
}

#[inline]
fn concat_val_with_args<'py>(
    value: &Bound<'py, PyAny>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    let mut builder = PyTupleBuilder::new(value.py(), args_len + 1);
    builder.push(value);
    builder.extend(args);
    builder.finish()
}

#[inline]
fn concat_tup_with_args<'py>(
    value: &Args<'py>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    let mut builder = PyTupleBuilder::new(value.py(), value.len() + args_len);
    builder.extend(value);
    builder.extend(args);
    builder.finish()
}

#[inline]
fn concat_acc_tup_with_args<'py>(
    acc: &Bound<'py, PyAny>,
    value: &Args<'py>,
    args: &Args<'py>,
    args_len: usize,
) -> Bound<'py, PyTuple> {
    let mut builder = PyTupleBuilder::new(acc.py(), 1 + value.len() + args_len);
    builder.push(acc);
    builder.extend(value);
    builder.extend(args);
    builder.finish()
}

struct PyTupleBuilder<'py> {
    py: Python<'py>,
    ptr: *mut ffi::PyObject,
    next_index: ffi::Py_ssize_t,
}

impl<'py> PyTupleBuilder<'py> {
    #[allow(clippy::cast_possible_wrap)]
    #[inline]
    fn new(py: Python<'py>, len: usize) -> Self {
        Self {
            py,
            ptr: unsafe { ffi::PyTuple_New(len as ffi::Py_ssize_t) },
            next_index: 0,
        }
    }

    #[inline]
    fn push(&mut self, value: &Bound<'py, PyAny>) {
        unsafe {
            let ptr = value.as_ptr();
            ffi::Py_INCREF(ptr);
            ffi::PyTuple_SetItem(self.ptr, self.next_index, ptr);
        }
        self.next_index += 1;
    }

    #[inline]
    fn extend<T: IntoIterator<Item = Bound<'py, PyAny>>>(&mut self, values: T) {
        values.into_iter().for_each(|value| self.push(&value));
    }

    #[inline]
    fn finish(self) -> Bound<'py, PyTuple> {
        unsafe { Bound::from_owned_ptr(self.py, self.ptr).cast_into_unchecked::<PyTuple>() }
    }
}

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use smallvec::SmallVec;
type ArgsBuf = SmallVec<[*mut ffi::PyObject; 8]>;

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

    fn concat1<S: ArgSource<'py>>(self, sources: S) -> PyResult<Bound<'py, PyAny>>;
    /// Akin to `itertools::map_starmap`, where *value* is expected to be a tuple of arguments.\
    /// Unpack each item in *value* and concatenate it with the given `*args`, then call the function with the resulting arguments and `**kwargs`
    fn concat_star(
        self,
        value: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    /// Prepend `acc` to `item` and concatenate with `args`, then call the function with `**kwargs`
    fn fold_concat_star(
        self,
        acc: &Bound<'py, PyAny>,
        item: &Args<'py>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    fn call_sources_kw<S: ArgSource<'py>>(
        self,
        args: S,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> Concatenate<'py> for &Bound<'py, PyAny> {
    #[inline(always)]
    fn call_sources_kw<S: ArgSource<'py>>(
        self,
        args: S,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        let (args, total) = build_args_buf(&args);
        unsafe {
            let ret = ffi::PyObject_Vectorcall(
                self.as_ptr(),
                args.as_ptr().add(1),
                (total as usize) | ffi::PY_VECTORCALL_ARGUMENTS_OFFSET,
                kwargs
                    .map(|k| k.as_ptr())
                    .unwrap_or_else(core::ptr::null_mut),
            );
            Bound::from_owned_ptr_or_err(py, ret)
        }
    }
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
    fn concat1<S: ArgSource<'py>>(self, args: S) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        let (args, total) = build_args_buf(&args);
        unsafe {
            let ret = ffi::PyObject_Vectorcall(
                self.as_ptr(),
                args.as_ptr().add(1),
                (total as usize) | ffi::PY_VECTORCALL_ARGUMENTS_OFFSET,
                core::ptr::null_mut(),
            );
            Bound::from_owned_ptr_or_err(py, ret)
        }
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
}
pub trait ArgSource<'py> {
    fn arg_len(&self) -> usize;
    fn write_borrowed_ptrs(&self, dst: &mut [*mut ffi::PyObject]);
}

impl<'py, 'a, T: ArgSource<'py> + ?Sized> ArgSource<'py> for &'a T {
    #[inline(always)]
    fn arg_len(&self) -> usize {
        (**self).arg_len()
    }
    #[inline(always)]
    fn write_borrowed_ptrs(&self, dst: &mut [*mut ffi::PyObject]) {
        (**self).write_borrowed_ptrs(dst)
    }
}
impl<'py> ArgSource<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn arg_len(&self) -> usize {
        1
    }
    #[inline(always)]
    fn write_borrowed_ptrs(&self, dst: &mut [*mut ffi::PyObject]) {
        dst[0] = self.as_ptr();
    }
}
impl<'py> ArgSource<'py> for Bound<'py, PyTuple> {
    #[inline(always)]
    fn arg_len(&self) -> usize {
        self.len()
    }
    #[inline(always)]
    fn write_borrowed_ptrs(&self, dst: &mut [*mut ffi::PyObject]) {
        let ptr = self.as_ptr();
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = unsafe { ffi::PyTuple_GET_ITEM(ptr, i as ffi::Py_ssize_t) };
        }
    }
}

macro_rules! impl_concat_sources {
        ($( ($($T:ident : $idx:tt),+) ),+ $(,)?) => {
            $(
                impl<'py, $($T: ArgSource<'py>),+> ArgSource<'py> for ($($T,)+) {
                    #[inline(always)]
                    fn arg_len(&self) -> usize {
                        0 $(+ self.$idx.arg_len())+
                    }
                    #[inline(always)]
                    fn write_borrowed_ptrs(&self, dst: &mut [*mut ffi::PyObject]) {
                        #[allow(unused_mut, unused_variables)]
                        let mut offset = 0;
                        $(
                            let len = self.$idx.arg_len();
                            self.$idx.write_borrowed_ptrs(&mut dst[offset..offset + len]);
                            #[allow(unused_assignments)]
                            {
                                offset += len;
                            }
                        )+
                    }
                }
            )+
        };
    }

impl_concat_sources!(
    (A:0, B:1),
    (A:0, B:1, C:2),
    (A:0, B:1, C:2, D:3),
);

#[inline(always)]
fn build_args_buf<'py, S: ArgSource<'py>>(sources: &S) -> (ArgsBuf, usize) {
    let total = sources.arg_len();
    let mut buf: ArgsBuf = SmallVec::from_elem(core::ptr::null_mut(), 1 + total);
    sources.write_borrowed_ptrs(&mut buf[1..]);
    (buf, total)
}

use pyo3::{
    ffi,
    prelude::*,
    types::{PyDict, PyTuple},
};
use smallvec::{Array, SmallVec};
use std::mem::MaybeUninit;
use std::ptr;
type Kwargs<'a, 'py> = Option<&'a Bound<'py, PyDict>>;
type CallResult<'py> = PyResult<Bound<'py, PyAny>>;
pub trait CallConcat<'py> {
    fn call_concat<A: IntoArgs<'py>>(self, args: A, kwargs: Kwargs<'_, 'py>) -> CallResult<'py>;
    fn call_concat1<A: IntoArgs<'py>>(self, args: A) -> CallResult<'py>;
}

macro_rules! dispatch {
    ($func:expr, $args:expr, $kwargs:expr) => {
        seq_macro::seq!(i in 1..=8 {
            match $args.len_unpacked() {
                #(
                i => FixedBuf::<i>::from_args($args).call($func,$kwargs),
                )*
                _ => VecOfPtr::from_args($args).call($func, $kwargs),
            }
        })
    };
}

macro_rules! dispatch1 {
    ( $func:expr, $args:expr) => {
        seq_macro::seq!(i in 1..=8 {
            match $args.len_unpacked() {
                #(
                i => FixedBuf::<i>::from_args($args).call1($func),
                )*
                _ => VecOfPtr::from_args($args).call1($func),
            }
        })
    };
}
impl<'py> CallConcat<'py> for &Bound<'py, PyAny> {
    #[inline(always)]
    fn call_concat1<A: IntoArgs<'py>>(self, args: A) -> CallResult<'py> {
        dispatch1!(self, args)
    }
    #[inline(always)]
    fn call_concat<A: IntoArgs<'py>>(self, args: A, kwargs: Kwargs<'_, 'py>) -> CallResult<'py> {
        dispatch!(self, args, kwargs)
    }
}

/// # Safety
/// Implementors must call `buf.push_ptr` from `extend_buf` exactly `len_unpacked()` times.
pub unsafe trait IntoArgs<'py> {
    fn len_unpacked(&self) -> usize;
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B);
}

unsafe impl<'py> IntoArgs<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn len_unpacked(&self) -> usize {
        1
    }
    #[inline(always)]
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B) {
        buf.push_ptr(self.as_ptr());
    }
}

unsafe impl<'py> IntoArgs<'py> for Bound<'py, PyTuple> {
    #[inline(always)]
    fn len_unpacked(&self) -> usize {
        self.len()
    }
    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B) {
        let ptr = self.as_ptr();
        for i in 0..self.len() {
            buf.push_ptr(unsafe { ffi::PyTuple_GET_ITEM(ptr, i as ffi::Py_ssize_t) });
        }
    }
}
unsafe impl<A> IntoArgs<'_> for SmallVec<A>
where
    A: Array<Item = Py<PyAny>>,
{
    #[inline(always)]
    fn len_unpacked(&self) -> usize {
        self.len()
    }
    #[inline(always)]
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B) {
        self.iter().for_each(|item| buf.push_ptr(item.as_ptr()));
    }
}
unsafe impl<'py, T: IntoArgs<'py> + ?Sized> IntoArgs<'py> for &T {
    #[inline(always)]
    fn len_unpacked(&self) -> usize {
        (**self).len_unpacked()
    }
    #[inline(always)]
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B) {
        (**self).extend_buf(buf);
    }
}

macro_rules! impl_arg_concat_tuple {
    ($($T:ident : $idx:tt),+) => {
        unsafe impl<'py, $($T: IntoArgs<'py>),+> IntoArgs<'py> for ($($T,)+) {
            #[inline(always)]
            fn len_unpacked(&self) -> usize {
                0 $(+ self.$idx.len_unpacked())+
            }
            #[inline(always)]
            fn extend_buf<T: ArgBuffer>(&self, buf: &mut T) {
                $( self.$idx.extend_buf(buf); )+
            }
        }
    };
}
impl_arg_concat_tuple!(A:0, B:1);
impl_arg_concat_tuple!(A:0, B:1, C:2);
impl_arg_concat_tuple!(A:0, B:1, C:2, D:3);
mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for VecOfPtr {}
impl<const N: usize> sealed::Sealed for FixedBuf<N> {}
pub trait ArgBuffer: Sized + sealed::Sealed {
    fn from_args<'py, A: IntoArgs<'py>>(args: A) -> Self;
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject);
    fn as_ptr(&self) -> *const *mut ffi::PyObject;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline(always)]
    fn call1<'py>(self, func: &Bound<'py, PyAny>) -> CallResult<'py> {
        unsafe {
            let res =
                ffi::PyObject_Vectorcall(func.as_ptr(), self.as_ptr(), self.len(), ptr::null_mut());
            Bound::from_owned_ptr_or_err(func.py(), res)
        }
    }
    #[inline(always)]
    fn call<'py>(self, func: &Bound<'py, PyAny>, kwargs: Kwargs<'_, 'py>) -> CallResult<'py> {
        let kw = kwargs.map_or(ptr::null_mut(), Bound::as_ptr);
        unsafe {
            let res = ffi::PyObject_VectorcallDict(func.as_ptr(), self.as_ptr(), self.len(), kw);
            Bound::from_owned_ptr_or_err(func.py(), res)
        }
    }
}
struct VecOfPtr(SmallVec<[*mut ffi::PyObject; 16]>);
impl ArgBuffer for VecOfPtr {
    fn from_args<'py, A: IntoArgs<'py>>(args: A) -> Self {
        let mut buf = Self(SmallVec::with_capacity(args.len_unpacked()));
        args.extend_buf(&mut buf);
        buf
    }
    #[inline(always)]
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject) {
        self.0.push(ptr);
    }
    fn as_ptr(&self) -> *const *mut ffi::PyObject {
        self.0.as_ptr()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
struct FixedBuf<const N: usize> {
    slice: [MaybeUninit<*mut ffi::PyObject>; N],
    len: usize,
}

impl<const N: usize> ArgBuffer for FixedBuf<N> {
    #[inline(always)]
    fn from_args<'py, A: IntoArgs<'py>>(args: A) -> Self {
        let mut buf = Self {
            slice: [const { MaybeUninit::uninit() }; N],
            len: 0,
        };
        args.extend_buf(&mut buf);
        buf
    }
    #[inline(always)]
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject) {
        debug_assert!(self.len < N);
        unsafe { self.slice.get_unchecked_mut(self.len).write(ptr) };
        self.len += 1;
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const *mut ffi::PyObject {
        self.slice.as_ptr().cast()
    }
    fn len(&self) -> usize {
        N
    }
}

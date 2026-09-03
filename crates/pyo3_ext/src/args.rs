use pyo3::{
    ffi,
    prelude::*,
    types::{PyDict, PyTuple},
};
use smallvec::SmallVec;
use std::mem::MaybeUninit;
type VecOfPtr = SmallVec<[*mut ffi::PyObject; 8]>;
pub trait CallConcat<'py> {
    fn call_concat<A: ArgsConcat<'py>>(
        self,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    fn call_concat1<A: ArgsConcat<'py>>(self, args: A) -> PyResult<Bound<'py, PyAny>>;
}

macro_rules! dispatch {
    ($n:expr, $func:expr, $args:expr, $kwargs:expr, [$($i:literal),+]) => {
        match $n {
            $($i => call_fixed::<$i, _>($func, $args, $kwargs),)+
            n => call_dyn($func, $args, $kwargs, n)
        }
    };
}

macro_rules! dispatch1 {
    ($n:expr, $func:expr, $args:expr, [$($i:literal),+]) => {
        match $n {
            $($i => call_fixed1::<$i, _>($func, $args),)+
            n => call_dyn1($func, $args, n)
        }
    };
}
impl<'py> CallConcat<'py> for &Bound<'py, PyAny> {
    #[inline(always)]
    fn call_concat1<A: ArgsConcat<'py>>(self, args: A) -> PyResult<Bound<'py, PyAny>> {
        dispatch1!(args.len_unpacked(), self, args, [1, 2, 3, 4])
    }

    #[inline(always)]
    fn call_concat<A: ArgsConcat<'py>>(
        self,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dispatch!(args.len_unpacked(), self, args, kwargs, [1, 2, 3, 4])
    }
}

pub trait ArgBuffer {
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject);
}

impl ArgBuffer for VecOfPtr {
    #[inline(always)]
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject) {
        self.push(ptr);
    }
}

pub trait ArgsConcat<'py> {
    fn len_unpacked(&self) -> usize;
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B);
}

impl<'py> ArgsConcat<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn len_unpacked(&self) -> usize {
        1
    }
    #[inline(always)]
    fn extend_buf<B: ArgBuffer>(&self, buf: &mut B) {
        buf.push_ptr(self.as_ptr());
    }
}

impl<'py> ArgsConcat<'py> for Bound<'py, PyTuple> {
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
impl<'py, T: ArgsConcat<'py> + ?Sized> ArgsConcat<'py> for &T {
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
        impl<'py, $($T: ArgsConcat<'py>),+> ArgsConcat<'py> for ($($T,)+) {
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

struct FixedBuf<const N: usize> {
    buf: [MaybeUninit<*mut ffi::PyObject>; N],
    len: usize,
}

impl<const N: usize> FixedBuf<N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            buf: [const { MaybeUninit::uninit() }; N],
            len: 0,
        }
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const *mut ffi::PyObject {
        self.buf.as_ptr().cast()
    }
}

impl<const N: usize> ArgBuffer for FixedBuf<N> {
    #[inline(always)]
    fn push_ptr(&mut self, ptr: *mut ffi::PyObject) {
        debug_assert!(self.len < N);
        unsafe { self.buf.get_unchecked_mut(self.len).write(ptr) };
        self.len += 1;
    }
}

fn call_dyn1<'py, A: ArgsConcat<'py>>(
    func: &Bound<'py, PyAny>,
    args: A,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mut buf = VecOfPtr::with_capacity(n);
    args.extend_buf(&mut buf);
    vectorcall1(func, buf.as_ptr(), n)
}
fn call_dyn<'py, A: ArgsConcat<'py>>(
    func: &Bound<'py, PyAny>,
    args: A,
    kwargs: Option<&Bound<'py, PyDict>>,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mut buf = VecOfPtr::with_capacity(n);
    args.extend_buf(&mut buf);
    vectorcall(func, buf.as_ptr(), n, kwargs)
}
#[inline(always)]
fn call_fixed<'py, const N: usize, A: ArgsConcat<'py>>(
    func: &Bound<'py, PyAny>,
    args: A,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut buf = FixedBuf::<N>::new();
    args.extend_buf(&mut buf);
    vectorcall(func, buf.as_ptr(), N, kwargs)
}
#[inline(always)]
fn call_fixed1<'py, const N: usize, A: ArgsConcat<'py>>(
    func: &Bound<'py, PyAny>,
    args: A,
) -> PyResult<Bound<'py, PyAny>> {
    let mut buf = FixedBuf::<N>::new();
    args.extend_buf(&mut buf);
    vectorcall1(func, buf.as_ptr(), N)
}
#[inline(always)]
fn vectorcall1<'py>(
    func: &Bound<'py, PyAny>,
    ptr: *const *mut ffi::PyObject,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    unsafe {
        let result = ffi::PyObject_Vectorcall(func.as_ptr(), ptr, n, std::ptr::null_mut());
        Bound::from_owned_ptr_or_err(func.py(), result)
    }
}

#[inline(always)]
fn vectorcall<'py>(
    func: &Bound<'py, PyAny>,
    ptr: *const *mut ffi::PyObject,
    n: usize,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let kw = kwargs.map_or(std::ptr::null_mut(), pyo3::Bound::as_ptr);
    unsafe {
        let result = ffi::PyObject_VectorcallDict(func.as_ptr(), ptr, n, kw);
        Bound::from_owned_ptr_or_err(func.py(), result)
    }
}

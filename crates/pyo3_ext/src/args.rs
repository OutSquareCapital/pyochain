use pyo3::{
    ffi,
    prelude::*,
    types::{PyDict, PyTuple},
};
use smallvec::SmallVec;

type VecOfPtr = SmallVec<[*mut ffi::PyObject; 8]>;
pub trait CallConcat<'py> {
    fn call_concat<A: ArgsConcat<'py>>(
        self,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    fn call_concat1<A: ArgsConcat<'py>>(self, args: A) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> CallConcat<'py> for &Bound<'py, PyAny> {
    #[inline(always)]
    fn call_concat1<A: ArgsConcat<'py>>(self, args: A) -> PyResult<Bound<'py, PyAny>> {
        let buf = get_buffer(args);
        let result = unsafe {
            ffi::PyObject_Vectorcall(self.as_ptr(), buf.as_ptr(), buf.len(), std::ptr::null_mut())
        };
        unsafe { Bound::from_owned_ptr_or_err(self.py(), result) }
    }

    #[inline(always)]
    fn call_concat<A: ArgsConcat<'py>>(
        self,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let buf = get_buffer(args);
        let result = unsafe {
            ffi::PyObject_VectorcallDict(
                self.as_ptr(),
                buf.as_ptr(),
                buf.len(),
                kwargs.map_or(std::ptr::null_mut(), Bound::as_ptr),
            )
        };
        unsafe { Bound::from_owned_ptr_or_err(self.py(), result) }
    }
}
fn get_buffer<'py, A: ArgsConcat<'py>>(args: A) -> VecOfPtr {
    let mut buf = VecOfPtr::with_capacity(args.len_as_args());
    args.extend_buf(&mut buf);
    buf
}

pub trait ArgsConcat<'py> {
    fn len_as_args(&self) -> usize;
    fn extend_buf(&self, buf: &mut VecOfPtr);
}

impl<'py> ArgsConcat<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn len_as_args(&self) -> usize {
        1
    }
    #[inline(always)]
    fn extend_buf(&self, buf: &mut VecOfPtr) {
        buf.push(self.as_ptr());
    }
}

impl<'py> ArgsConcat<'py> for Bound<'py, PyTuple> {
    #[inline(always)]
    fn len_as_args(&self) -> usize {
        self.len()
    }
    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    fn extend_buf(&self, buf: &mut VecOfPtr) {
        let ptr = self.as_ptr();
        for i in 0..self.len() {
            buf.push(unsafe { ffi::PyTuple_GET_ITEM(ptr, i as ffi::Py_ssize_t) });
        }
    }
}
impl<'py, T: ArgsConcat<'py> + ?Sized> ArgsConcat<'py> for &T {
    #[inline(always)]
    fn len_as_args(&self) -> usize {
        (**self).len_as_args()
    }
    #[inline(always)]
    fn extend_buf(&self, buf: &mut VecOfPtr) {
        (**self).extend_buf(buf);
    }
}

macro_rules! impl_arg_source_tuple {
    ($($T:ident : $idx:tt),+) => {
        impl<'py, $($T: ArgsConcat<'py>),+> ArgsConcat<'py> for ($($T,)+) {
            #[inline(always)]
            fn len_as_args(&self) -> usize {
                0 $(+ self.$idx.len_as_args())+
            }
            #[inline(always)]
            fn extend_buf(&self, buf: &mut VecOfPtr) {
                $( self.$idx.extend_buf(buf); )+
            }
        }
    };
}
impl_arg_source_tuple!(A:0, B:1);
impl_arg_source_tuple!(A:0, B:1, C:2);
impl_arg_source_tuple!(A:0, B:1, C:2, D:3);

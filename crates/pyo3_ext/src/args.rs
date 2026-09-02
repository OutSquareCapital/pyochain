use pyo3::{
    ffi,
    prelude::*,
    types::{PyDict, PyTuple},
};
use smallvec::SmallVec;
trait FuncCaller<'py> {
    fn call_concatenate1(&self, args: ArgsBuilder) -> PyResult<Bound<'py, PyAny>>;
    fn call_concatenate(
        &self,
        args: ArgsBuilder,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> FuncCaller<'py> for Bound<'py, PyAny> {
    #[inline]
    fn call_concatenate1(&self, args: ArgsBuilder) -> PyResult<Bound<'py, PyAny>> {
        let result = unsafe {
            ffi::PyObject_Vectorcall(
                self.as_ptr(),
                args.0.as_ptr(),
                args.0.len(),
                std::ptr::null_mut(),
            )
        };
        unsafe { Bound::from_owned_ptr_or_err(self.py(), result) }
    }
    #[inline]
    fn call_concatenate(
        &self,
        args: ArgsBuilder,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let result = unsafe {
            ffi::PyObject_VectorcallDict(
                self.as_ptr(),
                args.0.as_ptr(),
                args.0.len(),
                kwargs.map_or(std::ptr::null_mut(), Bound::as_ptr),
            )
        };
        unsafe { Bound::from_owned_ptr_or_err(self.py(), result) }
    }
}

pub trait CallConcat<'py> {
    fn call_concat<S: ArgSource<'py>>(
        self,
        args: S,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    fn call_concat1<S: ArgSource<'py>>(self, args: S) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> CallConcat<'py> for &Bound<'py, PyAny> {
    #[inline(always)]
    fn call_concat1<S: ArgSource<'py>>(self, args: S) -> PyResult<Bound<'py, PyAny>> {
        self.call_concatenate1(ArgsBuilder::from_source(&args))
    }

    #[inline(always)]
    fn call_concat<S: ArgSource<'py>>(
        self,
        args: S,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call_concatenate(ArgsBuilder::from_source(&args), kwargs)
    }
}

struct ArgsBuilder(SmallVec<[*mut ffi::PyObject; 8]>);
impl ArgsBuilder {
    #[inline(always)]
    fn from_source<'py, S: ArgSource<'py>>(sources: &S) -> Self {
        let mut buf = SmallVec::with_capacity(sources.source_len());
        sources.for_each_ptr(|ptr| buf.push(ptr));
        Self(buf)
    }
}

pub trait ArgSource<'py> {
    fn source_len(&self) -> usize;
    fn for_each_ptr<F: FnMut(*mut ffi::PyObject)>(&self, f: F);
}

impl<'py> ArgSource<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn source_len(&self) -> usize {
        1
    }
    #[inline(always)]
    fn for_each_ptr<F: FnMut(*mut ffi::PyObject)>(&self, mut f: F) {
        f(self.as_ptr());
    }
}

impl<'py> ArgSource<'py> for Bound<'py, PyTuple> {
    #[inline(always)]
    fn source_len(&self) -> usize {
        self.len()
    }
    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    fn for_each_ptr<F: FnMut(*mut ffi::PyObject)>(&self, mut f: F) {
        let ptr = self.as_ptr();
        for i in 0..self.len() {
            f(unsafe { ffi::PyTuple_GET_ITEM(ptr, i as ffi::Py_ssize_t) });
        }
    }
}
impl<'py, T: ArgSource<'py> + ?Sized> ArgSource<'py> for &T {
    #[inline(always)]
    fn source_len(&self) -> usize {
        (**self).source_len()
    }
    #[inline(always)]
    fn for_each_ptr<F: FnMut(*mut ffi::PyObject)>(&self, f: F) {
        (**self).for_each_ptr(f);
    }
}

macro_rules! impl_arg_source_tuple {
    ($($T:ident : $idx:tt),+) => {
        impl<'py, $($T: ArgSource<'py>),+> ArgSource<'py> for ($($T,)+) {
            #[inline(always)]
            fn source_len(&self) -> usize {
                0 $(+ self.$idx.source_len())+
            }
            #[inline(always)]
            fn for_each_ptr<F: FnMut(*mut ffi::PyObject)>(&self, mut f: F) {
                $( self.$idx.for_each_ptr(&mut f); )+
            }
        }
    };
}
impl_arg_source_tuple!(A:0, B:1);
impl_arg_source_tuple!(A:0, B:1, C:2);
impl_arg_source_tuple!(A:0, B:1, C:2, D:3);

struct PyTupleBuilder<'py> {
    py: Python<'py>,
    ptr: *mut ffi::PyObject,
    next_index: ffi::Py_ssize_t,
}
impl<'py> PyTupleBuilder<'py> {
    #[allow(clippy::cast_possible_wrap)]
    #[inline]
    fn from_source<S: ArgSource<'py>>(py: Python<'py>, sources: &S) -> Self {
        let mut builder = Self {
            py,
            ptr: unsafe { ffi::PyTuple_New(sources.source_len() as ffi::Py_ssize_t) },
            next_index: 0,
        };
        sources.for_each_ptr(|ptr| builder.push(ptr));
        builder
    }
    #[inline]
    fn push(&mut self, ptr: *mut ffi::PyObject) {
        unsafe {
            ffi::Py_INCREF(ptr);
            ffi::PyTuple_SetItem(self.ptr, self.next_index, ptr);
        }
        self.next_index += 1;
    }
    #[inline]
    fn finish(self) -> Bound<'py, PyTuple> {
        unsafe { Bound::from_owned_ptr(self.py, self.ptr).cast_into_unchecked::<PyTuple>() }
    }
}

pub trait CallWith<'py> {
    fn call_with(self, others: &Bound<'py, PyTuple>) -> Bound<'py, PyTuple>;
    fn call_with_2(
        self,
        b: &Bound<'py, PyAny>,
        others: &Bound<'py, PyTuple>,
    ) -> Bound<'py, PyTuple>;
}
impl<'py> CallWith<'py> for Bound<'py, PyAny> {
    #[inline(always)]
    fn call_with(self, others: &Bound<'py, PyTuple>) -> Bound<'py, PyTuple> {
        let py = self.py();
        PyTupleBuilder::from_source(py, &(self, others)).finish()
    }
    #[inline]
    fn call_with_2(
        self,
        b: &Bound<'py, PyAny>,
        others: &Bound<'py, PyTuple>,
    ) -> Bound<'py, PyTuple> {
        let py = self.py();
        PyTupleBuilder::from_source(py, &(self, b, others)).finish()
    }
}

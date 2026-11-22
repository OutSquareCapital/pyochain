use std::{cmp::Ordering, convert::Infallible};

use either::Either;
use pyo3::{
    basic::CompareOp,
    ffi,
    prelude::*,
    types::{PyBool, PyFloat, PyInt, PyNotImplemented, PyString, PyTuple},
};
use pyo3_ext::{prelude::*, pystring::PyStringExtMethods};
use pyochain_macros::try_cast;
use tap::Pipe;

type Pair<T> = (T, T);
/// Strategy for ordering and storing the elements added to a sorted list.\
/// [`Identity`] sorts by value, [`WithKey`] sorts by the result of a Python callable.
pub trait KeyFn<'py> {
    /// What gets sorted: the value alone, or the value paired with its key.
    type Item: SortKey;
    type Error: Into<PyErr>;
    fn py(&self) -> Python<'py>;
    /// Calls the key function on `value`. The only place where it's called.
    fn call(&self, value: Bound<'py, PyAny>) -> Result<Self::Item, Self::Error>;
}
pub struct Identity<'py>(pub Python<'py>);
pub struct WithKey<'py>(pub Bound<'py, PyAny>);
impl<'py> KeyFn<'py> for Identity<'py> {
    type Item = Py<PyAny>;
    type Error = Infallible;
    fn py(&self) -> Python<'py> {
        self.0
    }
    fn call(&self, value: Bound<'py, PyAny>) -> Result<Self::Item, Self::Error> {
        Ok(value.unbind())
    }
}
impl<'py> KeyFn<'py> for WithKey<'py> {
    type Item = (Py<PyAny>, Py<PyAny>);
    type Error = PyErr;
    fn py(&self) -> Python<'py> {
        self.0.py()
    }
    fn call(&self, value: Bound<'py, PyAny>) -> PyResult<Self::Item> {
        self.0
            .call1((&value,))
            .map(|key| (key.unbind(), value.unbind()))
    }
}
/// What the sort compares: the item itself, or the key it was paired with.
pub trait SortKey: Sized {
    fn key<'py>(&self) -> &Bound<'py, PyAny>;
    #[inline(always)]
    unsafe fn key_cast<'a, 'py, T>(&'a self, b: &'a Self) -> Pair<&'a Bound<'py, T>> {
        unsafe {
            let a = self.key().cast_unchecked::<T>();
            let b = b.key().cast_unchecked::<T>();
            (a, b)
        }
    }
    #[inline(always)]
    fn long(&self, b: &Self) -> Ordering {
        unsafe {
            let (a, b) = self.key_cast::<PyInt>(b);
            a.compact_value().cmp(&b.compact_value())
        }
    }
    #[inline(always)]
    fn float(&self, b: &Self) -> Ordering {
        unsafe {
            let (a, b) = self.key_cast::<PyFloat>(b);
            a.as_double()
                .partial_cmp(&b.as_double())
                .unwrap_or(Ordering::Equal)
        }
    }
    #[inline(always)]
    fn latin(&self, b: &Self) -> Ordering {
        unsafe {
            let (a, b) = self.key_cast::<PyString>(b);
            a.as_bytes_8().cmp(b.as_bytes_8())
        }
    }
    #[inline(always)]
    fn object(&self, b: &Self) -> PyResult<Ordering> {
        self.key().lt_fast(b.key())
    }
    /// Homogeneous compare: safe for any two comparable objects of the same type.
    /// (ms->key_richcompare is set to ob_type->tp_richcompare in the pre-sort check.)
    /// Note that we can't assert
    ///     res == `PyObject_RichCompareBool(v`, w, `Py_LT`);
    /// because of evil compare functions like this:
    ///     lambda a, b:  `int(random.random()` * 3) - 1)
    /// (which is actually in `test_sort.py`)
    #[inline(always)]
    fn unsafe_object(&self, b: &Self, func: ffi::richcmpfunc) -> PyResult<Ordering> {
        let (a, b) = (self.key(), b.key());
        unsafe {
            /* No assumptions, because we check first: */
            if (*a.get_type_ptr())
                .tp_richcompare
                .is_some_and(|richcmp| std::ptr::fn_addr_eq(richcmp, func))
            {
                let out = func(a.as_ptr(), b.as_ptr(), ffi::Py_LT)
                    .pipe(|ptr| Bound::from_owned_ptr_or_err(a.py(), ptr))?;
                try_cast! {
                    match out {
                        CaseExact::PyNotImplemented(_) => a.lt_fast(b),
                        CaseExact::PyBool(pybool) => {
                            if pybool.is_true() {
                                Ok(Ordering::Less)
                            } else {
                                Ok(Ordering::Equal)
                            }
                        }
                        _ => match out.is_truthy() {
                            Ok(true) => Ok(Ordering::Less),
                            Ok(false) => Ok(Ordering::Equal),
                            Err(e) => Err(e),
                        },
                    }
                }
            } else {
                a.lt_fast(b)
            }
        }
    }
    /// Comment from `CPython`'s listobject.c:\
    /// > Compare *any* two tuples, using ms->tuple_elem_compare to compare the first elements,\
    /// > which is set using the same pre-sort check as we use for ms->key_compare,\
    /// > but run on the list [x[0] for x in L].\
    /// > This allows us to optimize compares on two levels (as long as [x[0] for x in L] is type-homogeneous.)\
    /// > The idea is that most tuple compares don't involve x[1:].
    #[inline(always)]
    fn tuple<F>(&self, b: &Self, elem: F) -> PyResult<Ordering>
    where
        F: Fn(&Py<PyAny>, &Py<PyAny>) -> Ordering,
    {
        tup_cmp(self, b)?
            .map_left(|(item_a, item_b)| elem(item_a.as_unbound(), item_b.as_unbound()))
            .into_inner()
            .pipe(Ok)
    }
    /// Same as [`tuple`], for a comparison of the first items that can fail.
    #[inline(always)]
    fn try_tuple<F>(&self, b: &Self, elem: F) -> PyResult<Ordering>
    where
        F: Fn(&Py<PyAny>, &Py<PyAny>) -> PyResult<Ordering>,
    {
        tup_cmp(self, b)?
            .map_left(|(item_a, item_b)| elem(item_a.as_unbound(), item_b.as_unbound()))
            .map_right(Ok)
            .into_inner()
    }
}

type TupleComp<'a, 'py> = Either<Pair<Borrowed<'a, 'py, PyAny>>, Ordering>;

#[inline(always)]
fn tup_cmp<'r, 'py, T: SortKey>(a: &'r T, b: &'r T) -> PyResult<TupleComp<'r, 'py>> {
    let (a, b) = unsafe { a.key_cast::<PyTuple>(b) };
    let mut i = 0;
    while i < a.len() && i < b.len() {
        let (item_a, item_b) = get_item(a, b, i);
        if !item_a.rich_compare_bool(item_b, CompareOp::Eq)? {
            break;
        }
        i += 1;
    }
    if i >= a.len() || i >= b.len() {
        Ok(Either::Right(a.len().cmp(&b.len())))
    } else {
        let (item_a, item_b) = get_item(a, b, i);
        if i == 0 {
            Ok(TupleComp::Left((item_a, item_b)))
        } else {
            item_a.lt_fast(&item_b).map(TupleComp::Right)
        }
    }
}

impl SortKey for Py<PyAny> {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        let py = unsafe { Python::assume_attached() };
        self.bind(py)
    }
}
impl SortKey for (Py<PyAny>, Py<PyAny>) {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        let py = unsafe { Python::assume_attached() };
        self.0.bind(py)
    }
}

#[inline(always)]
fn get_item<'a, 'py>(
    a: &'a Bound<'py, PyTuple>,
    b: &'a Bound<'py, PyTuple>,
    i: usize,
) -> Pair<Borrowed<'a, 'py, PyAny>> {
    unsafe {
        (
            a.get_borrowed_item_unchecked(i),
            b.get_borrowed_item_unchecked(i),
        )
    }
}

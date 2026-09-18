use core::ffi::c_int;
use pyo3::{
    BoundObject,
    basic::CompareOp,
    conversion::{IntoPyObject, IntoPyObjectExt},
    ffi,
    prelude::*,
};
/// Copy of internal pyo3 macro that works with generic types.
/// Concretely allows to implement `PyAnyMethods` in one line.
#[macro_export]
macro_rules! pyobject_native_type_named (
    ($name:ty $(;$generics:ident)*) => {
        impl pyo3::types::DerefToPyAny for $name {}
    };
);

/// Inplace binary operations API.
#[allow(unused)]
pub trait PyAnyExtMethods<'py> {
    fn iadd<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn isub<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn imul<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn imatmul<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn itruediv<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn ifloordiv<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn irem<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn ilshift<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn irshift<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn iand<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn ior<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn ixor<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>>;
    fn rich_compare_bool<O: IntoPyObject<'py>>(&self, other: O, op: CompareOp) -> PyResult<bool>;
}

macro_rules! implement_inplace_binop {
    ($name:ident, $c_api:path, $op:expr) => {
        #[doc = concat!("Computes `self ", $op, " other`.")]
        fn $name<O: IntoPyObject<'py>>(&self, other: O) -> PyResult<Bound<'py, PyAny>> {
            let py = self.py();
            let other = other.into_pyobject_or_pyerr(py)?.into_any();
            unsafe { Bound::from_owned_ptr_or_err(py, $c_api(self.as_ptr(), other.as_ptr())) }
        }
    };
}

impl<'py> PyAnyExtMethods<'py> for Bound<'py, PyAny> {
    implement_inplace_binop!(iadd, ffi::PyNumber_InPlaceAdd, "+=");
    implement_inplace_binop!(isub, ffi::PyNumber_InPlaceSubtract, "-=");
    implement_inplace_binop!(imul, ffi::PyNumber_InPlaceMultiply, "*=");
    implement_inplace_binop!(imatmul, ffi::PyNumber_InPlaceMatrixMultiply, "@=");
    implement_inplace_binop!(itruediv, ffi::PyNumber_InPlaceTrueDivide, "/=");
    implement_inplace_binop!(ifloordiv, ffi::PyNumber_InPlaceFloorDivide, "//=");
    implement_inplace_binop!(irem, ffi::PyNumber_InPlaceRemainder, "%=");
    implement_inplace_binop!(ilshift, ffi::PyNumber_InPlaceLshift, "<<=");
    implement_inplace_binop!(irshift, ffi::PyNumber_InPlaceRshift, ">>=");
    implement_inplace_binop!(iand, ffi::PyNumber_InPlaceAnd, "&=");
    implement_inplace_binop!(ior, ffi::PyNumber_InPlaceOr, "|=");
    implement_inplace_binop!(ixor, ffi::PyNumber_InPlaceXor, "^=");
    fn rich_compare_bool<O: IntoPyObject<'py>>(&self, other: O, op: CompareOp) -> PyResult<bool> {
        let py = self.py();
        let o = other.into_pyobject_or_pyerr(py)?.into_any();
        let out = unsafe { ffi::PyObject_RichCompareBool(self.as_ptr(), o.as_ptr(), op as c_int) };
        match out {
            -1 => Err(PyErr::fetch(py)),
            0 => Ok(false),
            1 => Ok(true),
            _ => unreachable!(),
        }
    }
}

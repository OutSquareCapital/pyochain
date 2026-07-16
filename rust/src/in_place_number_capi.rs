use pyo3::{
    BoundObject,
    conversion::{IntoPyObject, IntoPyObjectExt},
    ffi,
    prelude::*,
};
#[allow(unused)]
pub trait PyAnyInPlaceMethods<'py> {
    fn iadd<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn isub<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn imul<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn imatmul<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn itruediv<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn ifloordiv<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn irem<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn ilshift<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn irshift<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn iand<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn ior<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;

    fn ixor<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
    where
        O: IntoPyObject<'py>;
}

macro_rules! implement_inplace_binop {
    ($name:ident, $c_api:ident, $op:expr) => {
        #[doc = concat!("Computes `self ", $op, " other`.")]
        fn $name<O>(&self, other: O) -> PyResult<Bound<'py, PyAny>>
        where
            O: IntoPyObject<'py>,
        {
            let py = self.py();
            let other = other.into_pyobject_or_pyerr(py)?.into_any();

            unsafe { Bound::from_owned_ptr_or_err(py, ffi::$c_api(self.as_ptr(), other.as_ptr())) }
        }
    };
}

impl<'py> PyAnyInPlaceMethods<'py> for Bound<'py, PyAny> {
    implement_inplace_binop!(iadd, PyNumber_InPlaceAdd, "+=");
    implement_inplace_binop!(isub, PyNumber_InPlaceSubtract, "-=");
    implement_inplace_binop!(imul, PyNumber_InPlaceMultiply, "*=");
    implement_inplace_binop!(imatmul, PyNumber_InPlaceMatrixMultiply, "@=");
    implement_inplace_binop!(itruediv, PyNumber_InPlaceTrueDivide, "/=");
    implement_inplace_binop!(ifloordiv, PyNumber_InPlaceFloorDivide, "//=");
    implement_inplace_binop!(irem, PyNumber_InPlaceRemainder, "%=");
    implement_inplace_binop!(ilshift, PyNumber_InPlaceLshift, "<<=");
    implement_inplace_binop!(irshift, PyNumber_InPlaceRshift, ">>=");
    implement_inplace_binop!(iand, PyNumber_InPlaceAnd, "&=");
    implement_inplace_binop!(ior, PyNumber_InPlaceOr, "|=");
    implement_inplace_binop!(ixor, PyNumber_InPlaceXor, "^=");
}

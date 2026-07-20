use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyRange, PySet, PyTuple},
};

use crate::{collections, seq};
pub trait PyWrapper: PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync {
    type Inner: PyTypeInfo;
    fn as_inner(&self) -> &Py<Self::Inner>;
    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<&'r Bound<'py, Self::Inner>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().as_inner().bind(py))
            .or_else(|_| value.cast_exact::<Self::Inner>())
            .map_err(|_| {
                let py = value.py();
                let wrapper_name = Self::type_object(py).name().unwrap();
                let inner_name = Self::Inner::type_object(py).name().unwrap();
                let value_name = value.get_type().name().unwrap();
                let txt = format!(
                    "Input must be a '{}'' or a '{}', got '{}'",
                    wrapper_name, inner_name, value_name
                );
                PyTypeError::new_err(txt)
            })
    }
}
/// Implement `PyWrapper` for pyochain types in one line.
macro_rules! impl_py_wrapper {
    ($($wrapper:ty => $inner:ty),* $(,)?) => {
        $(
            impl PyWrapper for $wrapper {
                type Inner = $inner;

                #[inline]
                fn as_inner(&self) -> &Py<Self::Inner> {
                    &self.inner
                }
            }
        )*
    };
}
impl_py_wrapper! {
    seq::Seq => PyTuple,
    seq::Vec => PyList,
    seq::Set => PyFrozenSet,
    seq::SetMut => PySet,
    seq::Range => PyRange,
    seq::Dict => PyDict,
    collections::StableSet => PyDict,
    collections::PyoCounter => PyDict,
}

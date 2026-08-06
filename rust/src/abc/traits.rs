use crate::{abc, collections, dict, iterators, option, pyo3_ext, result, sliceview};
use pyo3::prelude::*;
use pyochain_macros::py_abc;
use tap::prelude::*;
#[py_abc(dict::Dict, collections::PyoCounter)]
pub trait ImplPyoReversible {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>>;
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        self.rev(py)
    }
}
#[py_abc(
    sliceview::SliceViewIterator,
    sliceview::SliceViewReverseIterator,
    iterators::OnceWith,
    iterators::Tail,
    iterators::SequenceIterator,
    iterators::SequenceReverseIterator,
    iterators::ValuesViewIterator,
    iterators::ItemsViewIterator,
    iterators::MapJuxt,
    iterators::UniqueIdentity,
    iterators::UniqueKey,
    iterators::Intersperse,
    iterators::MapWindow,
    iterators::FilterMap,
    iterators::FilterMapStar,
    iterators::Scan,
    iterators::MapWhile,
    iterators::FromFn,
    iterators::Drain,
    iterators::ExtractIf,
    iterators::Successors,
    iterators::FilterStar,
    iterators::WithPosition,
    iterators::ZipLongest,
    iterators::Unzip,
    iterators::GroupBy
)]
pub trait ImplPyoIterator: Sized {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
}
impl ImplPyoIterator for sliceview::SliceViewIterator {}
impl ImplPyoIterator for sliceview::SliceViewReverseIterator {}
impl ImplPyoIterator for iterators::OnceWith {}
impl ImplPyoIterator for iterators::Tail {}
impl ImplPyoIterator for iterators::SequenceIterator {}
impl ImplPyoIterator for iterators::SequenceReverseIterator {}
impl ImplPyoIterator for iterators::ValuesViewIterator {}
impl ImplPyoIterator for iterators::ItemsViewIterator {}
impl ImplPyoIterator for iterators::MapJuxt {}
impl ImplPyoIterator for iterators::UniqueIdentity {}
impl ImplPyoIterator for iterators::UniqueKey {}
impl ImplPyoIterator for iterators::Intersperse {}
impl ImplPyoIterator for iterators::MapWindow {}
impl ImplPyoIterator for iterators::FilterMap {}
impl ImplPyoIterator for iterators::FilterMapStar {}
impl ImplPyoIterator for iterators::Scan {}
impl ImplPyoIterator for iterators::MapWhile {}
impl ImplPyoIterator for iterators::FromFn {}
impl ImplPyoIterator for iterators::Drain {}
impl ImplPyoIterator for iterators::ExtractIf {}
impl ImplPyoIterator for iterators::Successors {}
impl ImplPyoIterator for iterators::FilterStar {}
impl ImplPyoIterator for iterators::WithPosition {}
impl ImplPyoIterator for iterators::ZipLongest {}
impl ImplPyoIterator for iterators::Unzip {}
impl ImplPyoIterator for iterators::GroupBy {}

macro_rules! impl_py_pipe {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            #[pyo3(name = "pipe", signature = (func, *args, **kwargs))]
            fn py_pipe(
                slf: &Bound<'_, Self>,
                func: &Bound<'_, PyAny>,
                args: &pyo3_ext::args::Args<'_>,
                kwargs: Option<&pyo3_ext::args::Kwargs<'_>>,
            ) -> PyResult<Py<PyAny>> {
                (
                    pyo3_ext::args::Concatenate::concat(func, &slf, args, kwargs)?.unbind().pipe(Ok)
                )
            }
        }
    };
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_py_pipe!($first);
        impl_py_pipe!($($rest),+);
    };
}
macro_rules! impl_tap {
    ($type:ty) => {
    #[pymethods]
            impl $type {
    #[pyo3(signature = (f, *args, **kwargs))]
    fn tap(
        slf: &Bound<'_, Self>,
        f: &Bound<'_, PyAny>,
        args: &pyo3_ext::args::Args<'_>,
        kwargs: Option<&pyo3_ext::args::Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        pyo3_ext::args::Concatenate::concat(f, &slf, args, kwargs)?;
        slf.to_owned().into_any().unbind().pipe(Ok)
    }}};
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_tap!($first);
        impl_tap!($($rest),+);
    };
}

macro_rules! impl_mapping_view {
    ($type:ty) => {
        #[pymethods]
        impl $type {

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "{}({:?})",
            slf.get_type().name()?,
            slf.get()._mapping.bind(slf.py())
        ))
    }

    fn __len__(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.get()._mapping.bind(slf.py()).len()
    }}
    };
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_mapping_view!($first);
        impl_mapping_view!($($rest),+);
    };
}
impl_tap!(abc::Fluent, abc::PyoTap, abc::PyoIterable);
impl_py_pipe!(
    option::PySome,
    option::PyNull,
    result::PyoOk,
    result::PyoErr,
    abc::Fluent,
    abc::PyoPipe,
    abc::PyoIterable,
    abc::PyoIterator
);
impl_mapping_view!(
    abc::PyoMappingView,
    abc::PyoKeysView,
    abc::PyoValuesView,
    abc::PyoItemsView
);

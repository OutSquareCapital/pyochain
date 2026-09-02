use crate::{
    abc,
    collections::{self, sorted},
    core::{
        Dict, PyNull, PySome, PyoErr, PyoOk, SliceViewIterator, SliceViewReverseIterator, iterators,
    },
};
use pyo3::{
    PyClass, PyTypeInfo,
    prelude::*,
    types::{DerefToPyAny, PyDict, PyTuple},
};
use pyo3_ext::prelude::*;
use pyochain_macros::py_abc;
use tap::prelude::*;
#[py_abc(Dict, collections::PyoCounter)]
pub trait ImplPyoReversible {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>>;
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        self.rev(py)
    }
}
#[py_abc(
    SliceViewIterator,
    SliceViewReverseIterator,
    abc::PyoIterator,
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
    iterators::GroupBy,
    sorted::iter::SortedIter,
    sorted::iter::SortedIterReverse,
    sorted::iter::SortedIterKey,
    sorted::iter::SortedIterKeyReverse
)]
pub trait ImplPyoIterator: Sized {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
}
impl ImplPyoIterator for SliceViewIterator {}
impl ImplPyoIterator for SliceViewReverseIterator {}
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
impl ImplPyoIterator for sorted::iter::SortedIter {}
impl ImplPyoIterator for sorted::iter::SortedIterReverse {}
impl ImplPyoIterator for sorted::iter::SortedIterKey {}
impl ImplPyoIterator for sorted::iter::SortedIterKeyReverse {}
impl ImplPyoIterator for abc::PyoIterator {}

#[py_abc(
    PySome,
    PyNull,
    PyoOk,
    PyoErr,
    abc::Fluent,
    abc::PyoPipe,
    abc::PyoIterable,
    abc::PyoIterator
)]
trait PipeMethod: PyTypeInfo {
    #[pyo3(name = "pipe", signature = (func, *args, **kwargs))]
    fn py_pipe(
        slf: Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        func.call_concat((slf.as_any(), args), kwargs)?
            .unbind()
            .pipe(Ok)
    }
}
impl TapMethod for abc::Fluent {}
impl TapMethod for abc::PyoTap {}
impl TapMethod for abc::PyoIterable {}
#[py_abc(abc::Fluent, abc::PyoTap, abc::PyoIterable)]
trait TapMethod: PyTypeInfo {
    #[pyo3(signature = (f, *args, **kwargs))]
    fn tap<'py>(
        slf: Bound<'py, Self>,
        f: &Bound<'py, PyAny>,
        args: Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        f.call_concat((slf.as_any(), args), kwargs)?;
        Ok(slf)
    }
}
impl PipeMethod for PySome {}
impl PipeMethod for PyNull {}
impl PipeMethod for PyoOk {}
impl PipeMethod for PyoErr {}
impl PipeMethod for abc::Fluent {}
impl PipeMethod for abc::PyoPipe {}
impl PipeMethod for abc::PyoIterable {}
impl PipeMethod for abc::PyoIterator {}

#[py_abc(
    abc::PyoMappingView,
    abc::PyoKeysView,
    abc::PyoValuesView,
    abc::PyoItemsView,
    collections::sorted::SortedItemsView,
    collections::sorted::SortedKeysView,
    collections::sorted::SortedValuesView,
    collections::sorted::SortedByKeyItemsView,
    collections::sorted::SortedByKeyKeysView,
    collections::sorted::SortedByKeyValuesView
)]
pub trait MappingView:
    Sized
    + PyTypeInfo
    + PyClass<Frozen = pyo3::pyclass::boolean_struct::True>
    + Send
    + Sync
    + DerefToPyAny
{
    type M: Sized;
    #[skip]
    fn mapping(&self) -> &Py<Self::M>;
    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "{}({:?})",
            slf.get_type().name()?,
            slf.get().mapping().bind(slf.py())
        ))
    }
    fn __len__(&self, py: Python<'_>) -> usize;
}

macro_rules! impl_mapping_view {
    ($($t:ty),* $(,)?) => {
        $(
            impl MappingView for $t {
                type M = PyAny;
                fn mapping(&self) -> &Py<Self::M> {
                    &self.0
                }
                fn __len__(&self, py: Python<'_>) -> usize {
                    self.mapping().bind(py).len().expect("Mapping should have a length")
                }
            }
        )*
    };
}

impl_mapping_view!(
    abc::PyoMappingView,
    abc::PyoKeysView,
    abc::PyoValuesView,
    abc::PyoItemsView,
);

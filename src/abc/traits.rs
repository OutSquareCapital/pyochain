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
    iterators::MapWindowStar,
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
    sorted::iter::PyBounded,
    sorted::iter::PyBoundedRev,
    sorted::iter::PyBoundedKey,
    sorted::iter::PyBoundedKeyRev,
    sorted::iter::PyFull,
    sorted::iter::PyFullRev,
    sorted::iter::PyFullKey,
    sorted::iter::PyFullKeyRev
)]
pub trait ImplPyoIterator: Sized {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
}
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

#[py_abc(
    abc::PyoMappingView,
    abc::PyoKeysView,
    abc::PyoValuesView,
    abc::PyoItemsView
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

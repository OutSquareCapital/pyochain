use crate::{
    abc,
    collections::sorted::{self, iter::PySortedIter},
    core::{PyoVec, iterators},
    traits::IntoInit,
};
use either::Either;
use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyNotImplementedError,
    prelude::*,
    pyclass::CompareOp,
    types::{PyBool, PyDict, PyMapping, PySet, PyString, PyTuple, PyType},
};
use pyo3_ext::prelude::*;
use pyo3_ext::types::PyCmpOut;
use pyochain_macros::{py_abc, try_cast_into};
use sorted_rs::{
    Bounds, DictData, InnerGetter, KeysListsData, ListDataGetters, ListDataOwner, ListsData,
    ListsDataMethods, PyRepr, SetComp, SetData, iter as rsiter,
    types::{DictDataRef, IntOrSlice, SeqOrAny},
    views,
};

use std::sync::{Arc, Mutex, MutexGuard};
use std_tools::prelude::*;
use tap::prelude::*;
pub(crate) type Reduced<'py> = PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)>;
pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyoVec>, Bound<'py, PyAny>>>;

pub(super) trait ListGetter:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    type T: ListDataGetters + ListDataOwner;
    type I: PySortedIter + From<rsiter::Bounded<Self::T>>;
    type IRev: PySortedIter + From<rsiter::BoundedRev<Self::T>>;
    type IFull: PySortedIter + From<rsiter::Full<Self::T>>;
    type IFullRev: PySortedIter + From<rsiter::FullRev<Self::T>>;
    fn inner(&self) -> &Arc<Mutex<Self::T>>;
    #[inline(always)]
    fn try_lock(&self) -> MutexGuard<'_, Self::T> {
        self.inner().try_into_inner()
    }
    fn iter_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match (bounds, reverse) {
            (None, _) => iterators::Iter::empty(py).map(Bound::into_super),
            (Some(bounds), true) => rsiter::BoundedRev::new(self.inner().clone(), bounds)
                .conv::<Self::IRev>()
                .into_pyiterator(py),
            (Some(bounds), false) => rsiter::Bounded::new(self.inner().clone(), bounds)
                .conv::<Self::I>()
                .into_pyiterator(py),
        }
    }
}

#[py_abc(
    sorted::SortedList,
    sorted::SortedKeyList,
    sorted::SortedSet,
    sorted::SortedKeySet,
    sorted::SortedDict,
    sorted::SortedKeyDict
)]
pub(super) trait SortedCollectionsMethods: ListGetter
where
    Self::T: ListDataOwner,
{
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py>;
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.inner()
            .clone()
            .pipe(rsiter::Full::new)
            .conv::<Self::IFull>()
            .into_pyiterator(py)
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.inner()
            .clone()
            .pipe(rsiter::FullRev::new)
            .conv::<Self::IFullRev>()
            .into_pyiterator(py)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.try_lock().list_mut().index(&value, start, stop)
    }
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let bounds = self
            .try_lock()
            .list_mut()
            .irange_specs(py, minimum, maximum, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        &self,
        py: Python<'py>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let bounds = self
            .try_lock()
            .list_mut()
            .inner_mut()
            .get_islice_specs(py, start, stop)?;
        self.iter_bounds(py, bounds, reverse)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().list_mut().reset(py, load)
    }
}

#[py_abc(sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict)]
pub(super) trait KeyedSortedCollection: SortedCollectionsMethods
where
    Self::T: ListDataOwner<List = KeysListsData>,
{
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        &self,
        py: Python<'py>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let data = self.try_lock();
        let list = data.list();
        let bounds = Bounds::from_sorted(&list.1, list.maxes(), min_key, max_key, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(key)
    }
    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(key)
    }
}

#[py_abc(sorted::SortedList, sorted::SortedKeyList)]
pub(super) trait SortedListMethods:
    SortedCollectionsMethods + IntoInit + From<<Self::T as ListDataOwner>::List>
{
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let data = self.try_lock();
        let out = match other.cast_exact::<Self>().map(Bound::get) {
            Ok(slf) if Arc::ptr_eq(self.inner(), slf.inner()) => data.list().inner().repeat(py, 2),
            Ok(list) => data
                .list()
                .inner()
                .iter()
                .chain(list.try_lock().list().inner().iter())
                .map(|x| x.clone_ref(py))
                .collect(),
            Err(_) => data
                .list()
                .inner()
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(other.try_iter()?.map(|x| x?.unbind().pipe(Ok)))
                .collect::<PyResult<_>>()?,
        };
        data.list()
            .as_owned_from(py, out)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().eq(other)
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().ne(other)
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().lt(other)
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().gt(other)
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().le(other)
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().ge(other)
    }

    fn __delitem__(&self, index: IntOrSlice<'_>) -> PyResult<()> {
        self.try_lock().list_mut().del_item_or_slice(index)
    }

    fn __getitem__<'py>(&self, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.try_lock()
            .list_mut()
            .get_item_or_slice(index)
            .and_then_left(Bound::try_into_py)
    }
    fn __len__(&self) -> usize {
        self.try_lock().len()
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__add__(other)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.try_lock().list().repr(&Self::type_object(py).name()?)
    }
    fn __rmul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.__mul__(py, num)
    }
    #[allow(unused_variables)]
    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``del sl[index]`` and ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    fn __iadd__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.extend(&other)
    }
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .list()
            .repeat(py, num)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        self.try_lock().list_mut().imul(py, num)
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().list().contains(value)
    }
    #[allow(unused_variables)]
    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().add(value)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().list_mut().clear(py);
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().count(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .list()
            .copy(py)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().discard(value)
    }
    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.try_lock().list_mut().extend(py, values)
    }
    #[allow(unused_variables)]
    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().list_mut().pop(py, index)
    }
    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().remove(value)
    }
    fn reverse(&self) -> PyResult<()> {
        let msg = "use ``sl.rev()`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
}
#[py_abc(sorted::SortedSet, sorted::SortedKeySet)]
pub(super) trait SortedSetMethods:
    SortedCollectionsMethods
    + ListGetter<T = SetData<Self::L>>
    + IntoInit
    + From<SetData<Self::L>>
    + PyTypeInfo
{
    type L: ListsDataMethods;
    #[getter]
    #[inline(always)]
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.try_lock().get_set(py)
    }
    #[skip]
    #[inline]
    fn comp<'py>(&self, value: Bound<'py, PyAny>, op: CompareOp) -> PyCmpOut<bool, 'py> {
        let py = value.py();
        try_cast_into! {
            match value {
                CaseExact::Self(sorted) if Arc::ptr_eq(self.inner(), sorted.get().inner()) => {
                    SetComp::Identity
                }
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => {
                    SetComp::Comparable(self.get_set(py), sorted.get().get_set(py))
                }
                Case::PySet(pyset) => SetComp::Comparable(self.get_set(py), pyset),
                _ => SetComp::NotImplemented(py),
            }
        }
        .comp(op)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_set(value.py()).contains(value)
    }
    fn __getitem__<'py>(&self, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.try_lock()
            .get_item_or_slice(index)
            .and_then_left(Bound::try_into_py)
    }
    fn __delitem__(&self, index: IntOrSlice<'_>) -> PyResult<()> {
        self.try_lock().del_item_or_slice(index)
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Eq)
    }

    fn __ne__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Ne)
    }

    fn __lt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Lt)
    }

    fn __gt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Gt)
    }

    fn __le__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Le)
    }

    fn __ge__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Ge)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.try_lock().__len__(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.try_lock()
            .difference(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __isub__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().difference_update(other.try_into()?)
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.try_lock()
            .intersection(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().intersection_update(other.py(), (other,))
    }

    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().update(other.try_into()?)
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.try_lock()
            .union(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __ror__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(other)
    }
    fn __xor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __rxor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __ixor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        Self::symmetric_difference_update(slf, other).map(|_| ())
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().add(value)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().discard(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.try_lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_disjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_subset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_superset(other)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.try_lock().count(value)
    }

    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().pop(py, index)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        self.try_lock()
            .difference(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }

    #[pyo3(name = "difference_update", signature = (*iterables))]
    fn difference_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().try_lock().difference_update(iterables.into())?;
        Ok(slf)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        self.try_lock()
            .intersection(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get()
            .try_lock()
            .intersection_update(slf.py(), iterables)?;
        Ok(slf)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.try_lock()
            .symmetric_difference(other)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn symmetric_difference_update<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().try_lock().symmetric_difference_update(other)?;
        // NOTE: the clone here is cheap (just an incref) and necessary to return `Self`
        Ok(slf.clone())
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        self.try_lock()
            .union(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }
    #[pyo3(signature = (*iterables))]
    fn update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().try_lock().update(iterables.into())?;
        Ok(slf)
    }
}
impl<T: SortedSetMethods> SortedCollectionsMethods for T {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        PyTuple::new(py, [self.get_set(py).clone()]).map(|tup| (Self::type_object(py), tup))
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(value)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().reset(py, load)
    }
}
#[py_abc(
    sorted::SortedItemsView,
    sorted::SortedKeysView,
    sorted::SortedValuesView,
    sorted::SortedByKeyItemsView,
    sorted::SortedByKeyKeysView,
    sorted::SortedByKeyValuesView
)]
pub trait SortedViewMethods:
    PyClass<BaseType = abc::PyoSequence> + From<DictDataRef<Self::L>>
where
    DictData<Self::L>: PyRepr,
{
    type L: ListsDataMethods;
    const REF_NAME: &'static str;
    #[skip]
    fn mapping(&self) -> MutexGuard<'_, DictData<Self::L>>;
    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py>;
    fn __len__(&self, py: Python<'_>) -> usize {
        self.mapping().__len__(py)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        let values = self.mapping().repr(&PyString::new(py, Self::REF_NAME))?;
        Ok(format!("{name}({values})"))
    }
    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        views::delitem(&mut self.mapping(), index)
    }
}

#[py_abc(sorted::SortedDict, sorted::SortedKeyDict)]
pub(super) trait SortedDictMethods:
    SortedCollectionsMethods + ListGetter<T = DictData<Self::L>> + IntoInit + From<DictData<Self::L>>
where
    DictData<Self::L>: PyRepr,
{
    type L: ListsDataMethods;
    type KView: SortedViewMethods<L = Self::L>;
    type VView: SortedViewMethods<L = Self::L>;
    type IView: SortedViewMethods<L = Self::L>;
    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        self.try_lock().repr(&name)
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().contains(value)
    }
    #[getter]
    fn get_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.try_lock().get_dict().clone_ref(py).into_bound(py)
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::KView>> {
        self.inner().clone().conv::<Self::KView>().into_bound(py)
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::IView>> {
        self.inner().clone().conv::<Self::IView>().into_bound(py)
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::VView>> {
        self.inner().clone().conv::<Self::VView>().into_bound(py)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.try_lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.try_lock().__len__(py)
    }

    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().get_item(key)
    }

    fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().del_item(key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().set_item(key, value)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other.py(), Some(other), None)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .or(value)?
            .conv::<Self>()
            .into_bound(value.py())
    }

    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .ror(value)?
            .conv::<Self>()
            .into_bound(value.py())
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
    #[staticmethod]
    #[pyo3(signature = (iterable, value = None, /))]
    fn from_keys<'py>(
        iterable: &Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, sorted::SortedDict>> {
        DictData::<ListsData>::from_keys(iterable, value)?
            .conv::<sorted::SortedDict>()
            .into_bound(iterable.py())
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        &self,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().pop(key, default)
    }

    #[pyo3(signature = (index = -1))]
    fn popitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.try_lock().popitem(py, index)
    }
    #[pyo3(signature = (index = -1))]
    fn peekitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.try_lock().peekitem(py, index)
    }
    #[pyo3(signature = (key, default = None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.try_lock().setdefault(key, default)
    }
    #[pyo3(signature = (m = None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.try_lock().update(py, m, kwargs)
    }
}

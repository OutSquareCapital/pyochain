use crate::{
    KeysListsData, ListsData,
    getters::ListDataOwner,
    inner::InnerData,
    prelude::*,
    types::{IntOrSlice, ListOrAny},
};
use either::Either;
use pyo3::{
    PyTypeInfo,
    basic::CompareOp,
    prelude::*,
    types::{DerefToPyAny, PyBool, PyList, PyNotImplemented, PySet},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use tap::prelude::*;

pub struct SetData<T>(T, Py<PySet>);
impl<T: InnerGetter> InnerGetter for SetData<T> {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}

impl PyRepr for SetData<ListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let self_repr = self.inner().as_pylist(py)?.repr()?;
        Ok(format!("{name}({self_repr})"))
    }
}
impl PyRepr for SetData<KeysListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let key = format!(", key={}", self.list().2.bind(py).repr()?);
        let list_repr = self.inner().as_pylist(py)?.repr()?;
        Ok(format!("{name}({list_repr}{key})"))
    }
}
impl<T: ListsDataMethods> ListDataOwner for SetData<T> {
    type List = T;

    fn list(&self) -> &Self::List {
        &self.0
    }

    fn list_mut(&mut self) -> &mut Self::List {
        &mut self.0
    }
}

impl SetData<ListsData> {
    pub fn build(py: Python<'_>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        Self::build_inner(py, ListsData::default(), iterable)
    }
}
impl TryFrom<Bound<'_, PyAny>> for SetData<ListsData> {
    type Error = PyErr;
    fn try_from(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::build_inner(iterable.py(), ListsData::default(), Some(iterable))
    }
}
impl SetData<KeysListsData> {
    pub fn build(key: Bound<'_, PyAny>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let py = key.py();
        let list = KeysListsData::new(key.unbind());
        Self::build_inner(py, list, iterable)
    }
}
impl<T: ListsDataMethods> SetData<T> {
    #[inline(always)]
    fn build_inner(
        py: Python<'_>,
        mut list: T,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let set = PySet::empty(py)?;
        if let Some(it) = iterable {
            it.try_iter()?
                .try_for_each(|value| try_add(&mut list, &set, value?))?;
        }
        Ok(Self(list, set.unbind()))
    }
    pub fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.1.clone_ref(py).into_bound(py)
    }
    fn wrap(&self, values: Bound<'_, PySet>) -> PyResult<Self> {
        let py = values.py();
        let list = self
            .list()
            .as_owned_from(py, values.iter().map(Bound::unbind).collect())?;
        Self(list, values.unbind()).pipe(Ok)
    }
    pub fn clear(&mut self, py: Python<'_>) {
        self.0.clear(py);
        self.1.bind(py).clear();
    }
    pub fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.0.reset(py, load)
    }
    pub fn difference(&self, iterables: IntoUpdate<'_>) -> PyResult<Self> {
        self.map_set(iterables, |slf, other| slf.difference((other,)))
            .and_then(|x| self.wrap(x))
    }
    pub fn intersection(&self, iterables: IntoUpdate<'_>) -> PyResult<Self> {
        self.map_set(iterables, |slf, other| slf.intersection((other,)))
            .and_then(|x| self.wrap(x))
    }
    pub fn union(&self, iterables: IntoUpdate<'_>) -> PyResult<Self> {
        self.map_set(iterables, |slf, other| slf.union((other,)))
            .and_then(|x| self.wrap(x))
    }
    pub fn difference_update(&mut self, iterables: IntoUpdate<'_>) -> PyResult<()> {
        self.update_inner(
            iterables,
            |slf, set| slf.difference_update((set,)),
            |slf, set| slf.discard(&set),
        )
    }
    pub fn update(&mut self, other: IntoUpdate<'_>) -> PyResult<()> {
        self.update_inner(other, |slf, other| slf.update((other,)), Self::add)
    }
    pub fn intersection_update(&mut self, iterables: IntoUpdate<'_>) -> PyResult<()> {
        match iterables {
            IntoUpdate::BigSet(pyset) | IntoUpdate::SmallSet(pyset) => {
                self.try_update(pyset.py(), pyset, |set, obj| {
                    set.intersection_update((obj,))
                })
            }
            IntoUpdate::Any(any) => {
                self.try_update(any.py(), any, |set, obj| set.intersection_update((obj,)))
            }
        }
    }
    pub fn get_item_or_slice<'py>(&mut self, index: IntOrSlice<'py>) -> PyResult<ListOrAny<'py>> {
        let py = index.py();
        match index {
            Either::Right(slice) => self
                .inner_mut()
                .get_slice(&slice)?
                .iter()
                .collect_bound::<PyList>(py)
                .map(Either::Left),
            Either::Left(index) => self
                .inner_mut()
                .get_item(py, index.extract()?)
                .map(Either::Right),
        }
    }

    pub fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        let py = index.py();
        match index {
            Either::Right(slice) => {
                let values = self
                    .0
                    .inner_mut()
                    .get_slice(&slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.1.bind(py).difference_update((values,))?;
                self.0.del_slice(&slice)
            }
            Either::Left(int) => {
                let idx = int.extract::<isize>()?;
                let value = self.0.inner_mut().get_item(py, idx)?;
                self.1.bind(py).remove(&value)?;
                self.0.del_item(py, idx)
            }
        }
    }

    pub fn __len__(&self, py: Python<'_>) -> usize {
        self.1.bind(py).len()
    }
    pub fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        try_add(&mut self.0, self.1.bind(value.py()), value)
    }
    pub fn discard(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let set = self.1.bind(value.py());
        match set.contains(value) {
            Ok(true) => {
                set.remove(value)?;
                self.0.remove(value)
            }
            Ok(false) => Ok(()),
            Err(err) => Err(err),
        }
    }
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.1.bind(py).copy().and_then(|x| self.wrap(x))
    }
    pub fn is_disjoint<'py>(&self, other: IntoUpdate<'py>) -> PyResult<Bound<'py, PyBool>> {
        self.map_set(other, Bound::isdisjoint)
    }

    pub fn is_subset<'py>(&self, other: IntoUpdate<'py>) -> PyResult<Bound<'py, PyBool>> {
        self.map_set(other, Bound::issubset)
    }

    pub fn is_superset<'py>(&self, other: IntoUpdate<'py>) -> PyResult<Bound<'py, PyBool>> {
        self.map_set(other, Bound::issuperset)
    }
    pub fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        match self.1.bind(value.py()).contains(value) {
            Ok(true) => Ok(1),
            Ok(false) => Ok(0),
            Err(e) => Err(e),
        }
    }

    pub fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let value = self.0.pop(py, index)?;
        self.1.bind(py).remove(&value)?;
        Ok(value)
    }
    pub fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.1.bind(value.py()).remove(value)?;
        self.0.remove(value)
    }
    pub fn symmetric_difference(&self, other: IntoUpdate<'_>) -> PyResult<Self> {
        self.map_set(other, Bound::symmetric_difference)
            .and_then(|x| self.wrap(x))
    }
    pub fn symmetric_difference_update(&mut self, other: IntoUpdate<'_>) -> PyResult<()> {
        match other {
            IntoUpdate::BigSet(pyset) | IntoUpdate::SmallSet(pyset) => {
                self.try_update(pyset.py(), pyset, Bound::symmetric_difference_update)
            }
            IntoUpdate::Any(any) => {
                self.try_update(any.py(), any, Bound::symmetric_difference_update)
            }
        }
    }
    #[inline(always)]
    fn map_set<'py, R, F: Fn(&Bound<'py, PySet>, Bound<'py, PyAny>) -> R>(
        &self,
        other: IntoUpdate<'py>,
        f: F,
    ) -> R {
        let set = self.1.bind(other.py());
        match other {
            IntoUpdate::BigSet(pyset) | IntoUpdate::SmallSet(pyset) => f(set, pyset.into_any()),
            IntoUpdate::Any(any) => f(set, any),
        }
    }

    fn update_inner<
        'py,
        F1: Fn(&Bound<'py, PySet>, Bound<'py, PySet>) -> PyResult<()>,
        F2: Fn(&mut Self, Bound<'_, PyAny>) -> PyResult<()>,
    >(
        &mut self,
        other: IntoUpdate<'py>,
        set_fn: F1,
        slf_fn: F2,
    ) -> PyResult<()> {
        match other {
            IntoUpdate::BigSet(pyset) => self.try_update(pyset.py(), pyset, set_fn),
            IntoUpdate::SmallSet(pyset) => pyset.iter().try_for_each(|value| slf_fn(self, value)),
            IntoUpdate::Any(any) => any.try_iter()?.try_for_each(|value| slf_fn(self, value?)),
        }
    }

    fn try_update<'py, O, F: Fn(&Bound<'py, PySet>, O) -> PyResult<()>>(
        &mut self,
        py: Python<'py>,
        obj: O,
        set_fn: F,
    ) -> PyResult<()> {
        let set = self.1.bind(py);
        set_fn(set, obj)?;
        self.0.clear(py);
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
    }
}
pub enum SetComp<'py, T, U> {
    /// If both operands are the same object, return true for equality and false for inequality.
    Identity,
    Comparable(Bound<'py, T>, Bound<'py, U>),
    NotImplemented(Python<'py>),
}
impl<'py, T: DerefToPyAny + PyTypeInfo, U: DerefToPyAny + PyTypeInfo> SetComp<'py, T, U> {
    #[inline]
    pub fn comp(self, op: CompareOp) -> PyCmpOut<'py, bool> {
        match self {
            Self::Identity => op.on_identity().pipe(Either::Left).pipe(Ok),
            Self::NotImplemented(py) => PyNotImplemented::from_cmp(py),
            Self::Comparable(a, b) => a.rich_compare_bool(&b, op).map(Either::Left),
        }
    }
}
#[must_use]
pub enum IntoUpdate<'py> {
    SmallSet(Bound<'py, PySet>),
    BigSet(Bound<'py, PySet>),
    Any(Bound<'py, PyAny>),
}
impl<'py> IntoUpdate<'py> {
    fn py(&self) -> Python<'py> {
        match self {
            IntoUpdate::SmallSet(set) | IntoUpdate::BigSet(set) => set.py(),
            IntoUpdate::Any(any) => any.py(),
        }
    }
    #[inline]
    pub fn from_sets(original: &Bound<'py, PySet>, other: Bound<'py, PySet>) -> Self {
        if (4 * other.len()) > original.len() {
            IntoUpdate::BigSet(other)
        } else {
            IntoUpdate::SmallSet(other)
        }
    }
}

#[inline(always)]
fn try_add<T: ListsDataMethods>(
    list: &mut T,
    set: &Bound<'_, PySet>,
    value: Bound<'_, PyAny>,
) -> PyResult<()> {
    match set.contains(&value) {
        Ok(true) => Ok(()),
        Ok(false) => {
            set.add(&value)?;
            list.add(value)
        }
        Err(e) => Err(e),
    }
}

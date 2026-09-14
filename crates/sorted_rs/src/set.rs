use crate::{
    InnerData, InnerGetter, ListsDataMethods,
    getters::ListDataOwner,
    types::{IntOrSlice, ListOrAny},
};
use either::Either;
use pyo3::{
    call::PyCallArgs,
    prelude::*,
    types::{PyBool, PyIterator, PyList, PySet, PyTuple},
};
use pyo3_ext::prelude::*;
use tap::Pipe;

pub struct SetData<T>(T, Py<PySet>);
impl<T: InnerGetter> InnerGetter for SetData<T> {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
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
impl<T: ListsDataMethods> SetData<T> {
    pub fn new(list: T, set: Py<PySet>) -> Self {
        Self(list, set)
    }
    pub fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.1.clone_ref(py).into_bound(py)
    }
    fn wrap(&self, values: Bound<'_, PySet>) -> PyResult<Self> {
        let py = values.py();
        let list = self
            .list()
            .as_owned_from(py, values.iter().map(Bound::unbind).collect())?;
        Self::new(list, values.unbind()).pipe(Ok)
    }
    pub fn clear(&mut self, py: Python<'_>) {
        self.0.clear(py);
        self.1.bind(py).clear();
    }
    pub fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.0.reset(py, load)
    }
    pub fn difference<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Self> {
        self.1
            .bind(py)
            .difference(iterables)
            .and_then(|x| self.wrap(x))
    }
    pub fn intersection<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Self> {
        self.1
            .bind(py)
            .intersection(iterables)
            .and_then(|x| self.wrap(x))
    }
    pub fn union<'py, O: PyCallArgs<'py>>(&self, py: Python<'py>, iterables: O) -> PyResult<Self> {
        self.1.bind(py).union(iterables).and_then(|x| self.wrap(x))
    }
    pub fn difference_update(&mut self, iterables: IntoUpdate<'_>) -> PyResult<()> {
        let py = iterables.py();
        let set = self.1.bind(py);
        let values = iterables.into_set()?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            self.0.clear(py);
            self.0.extend(py, set.iter().map(Bound::unbind).collect())
        } else {
            values.iter().try_for_each(|value| self.discard(&value))
        }
    }
    pub fn update(&mut self, other: IntoUpdate<'_>) -> PyResult<()> {
        let py = other.py();
        let set = self.1.bind(py);
        let values = other.into_set()?;
        if (4 * values.len()) > set.len() {
            set.update((values,))?;
            self.0.clear(py);
            self.0.extend(py, set.iter().map(Bound::unbind).collect())
        } else {
            values.iter().try_for_each(|value| self.add(value))
        }
    }
    pub fn intersection_update<'py, O: PyCallArgs<'py>>(
        &mut self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<()> {
        let set = self.1.bind(py);
        set.intersection_update(iterables)?;
        self.0.clear(py);
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
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
        let py = value.py();
        let set = self.1.bind(py);
        if !set.contains(&value)? {
            set.add(&value)?;
            self.0.add(value)?;
        }
        Ok(())
    }
    pub fn discard(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let set = self.1.bind(value.py());
        if set.contains(value)? {
            set.remove(value)?;
            self.0.remove(value)?;
        }
        Ok(())
    }
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        PySet::new(py, self.1.bind(py).iter()).and_then(|x| self.wrap(x))
    }
    pub fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.1.bind(other.py()).isdisjoint(other)
    }

    pub fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.1.bind(other.py()).issubset(other)
    }

    pub fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.1.bind(other.py()).issuperset(other)
    }

    pub fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        if self.1.bind(value.py()).contains(value)? {
            Ok(1)
        } else {
            Ok(0)
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
    pub fn symmetric_difference(&self, other: Bound<'_, PyAny>) -> PyResult<Self> {
        self.1
            .bind(other.py())
            .symmetric_difference(other)
            .and_then(|x| self.wrap(x))
    }
    pub fn symmetric_difference_update(&mut self, other: Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        let set = self.1.bind(py);
        set.symmetric_difference_update(other)?;
        self.0.clear(py);
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
    }
}

pub enum IntoUpdate<'py> {
    Set(Bound<'py, PySet>),
    Tuple(Bound<'py, PyTuple>),
    Iterable(Bound<'py, PyIterator>),
}
impl<'py> IntoUpdate<'py> {
    #[must_use]
    pub fn py(&self) -> Python<'py> {
        match self {
            Self::Set(pyset) => pyset.py(),
            Self::Tuple(tup) => tup.py(),
            Self::Iterable(any) => any.py(),
        }
    }
    pub fn into_set(self) -> PyResult<Bound<'py, PySet>> {
        let py = self.py();
        match self {
            Self::Tuple(tup) => tup
                .iter()
                .flat_map(|x| x.try_iter().unwrap())
                .try_collect_bound(py),
            Self::Set(pyset) => Ok(pyset),
            Self::Iterable(any) => any.try_collect_bound(py),
        }
    }
}
impl<'py> TryFrom<Bound<'py, PyAny>> for IntoUpdate<'py> {
    type Error = PyErr;
    fn try_from(other: Bound<'py, PyAny>) -> PyResult<Self> {
        if other.is_exact_instance_of::<PySet>() {
            Self::Set(unsafe { other.cast_into_unchecked() }).pipe(Ok)
        } else {
            other.try_iter().map(Self::Iterable)
        }
    }
}
impl<'py> From<Bound<'py, PyTuple>> for IntoUpdate<'py> {
    fn from(other: Bound<'py, PyTuple>) -> Self {
        Self::Tuple(other)
    }
}

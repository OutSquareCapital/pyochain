use crate::{
    InnerData, InnerGetter, ListsDataMethods,
    types::{IntOrSlice, ListOrAny},
};
use either::Either;
use pyo3::{
    call::PyCallArgs,
    prelude::*,
    types::{PyBool, PyList, PySet, PyTuple},
};
use pyo3_ext::prelude::*;
pub struct SetData<T: ListsDataMethods>(T, Py<PySet>);
impl<T: ListsDataMethods> InnerGetter for SetData<T> {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}
#[allow(unused)]
impl<T: ListsDataMethods> SetData<T> {
    pub fn new(list: T, set: Py<PySet>) -> Self {
        Self(list, set)
    }
    pub fn update<'py>(&mut self, py: Python<'py>, other: IntoUpdate<'py>) -> PyResult<()> {
        let set = self.1.bind(py);
        let values = other.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.update((values,))?;
            self.0.clear();
            self.0
                .extend(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())
        } else {
            values
                .iter()
                .map(Bound::unbind)
                .try_for_each(|value| self.add(py, value))
        }
    }
    pub fn difference<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, PySet>> {
        self.1.bind(py).difference(iterables)
    }
    pub fn intersection<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, PySet>> {
        self.1.bind(py).intersection(iterables)
    }
    pub fn union<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, PySet>> {
        self.1.bind(py).union(iterables)
    }
    pub fn difference_update(&mut self, py: Python<'_>, iterables: IntoUpdate<'_>) -> PyResult<()> {
        let set = self.1.bind(py);
        let values = iterables.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            self.0.clear();
            self.0
                .extend(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())
        } else {
            for value in values {
                self.discard(&value)?;
            }
            Ok(())
        }
    }
    pub fn intersection_update<'py, O: PyCallArgs<'py>>(
        &mut self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<()> {
        let set = self.1.bind(py);
        set.intersection_update(iterables)?;
        self.0.clear();
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
    }
    fn __getitem__<'py>(
        &mut self,
        py: Python<'py>,
        index: IntOrSlice<'py>,
    ) -> PyResult<ListOrAny<'py>> {
        match index {
            Either::Right(slice) => self
                .inner_mut()
                .get_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)
                .map(Either::Left),
            Either::Left(index) => self.inner_mut().get_item(py, index).map(Either::Right),
        }
    }
    fn __delitem__(&mut self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => {
                let values = self
                    .0
                    .inner_mut()
                    .get_slice(py, &slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.1.bind(py).difference_update((values,))?;
                self.0.del_slice(py, slice)
            }
            Either::Left(int) => {
                let value = self.0.inner_mut().get_item(py, int)?;
                self.1.bind(py).remove(&value)?;
                self.0.del_item(py, int)
            }
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.1.bind(py).len()
    }
    pub fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let set = self.1.bind(py);
        if !set.contains(&value)? {
            set.add(&value)?;
            self.0.add(py, value)?;
        }
        Ok(())
    }
    fn discard(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let set = self.1.bind(value.py());
        if set.contains(value)? {
            set.remove(value)?;
            self.0.remove(value)?;
        }
        Ok(())
    }
    pub fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.1.bind(py).iter())
    }
    pub fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.1.bind(other.py()).isdisjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
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
    pub fn py_difference_update(&mut self, iterables: &Bound<'_, PyTuple>) -> PyResult<()> {
        let py = iterables.py();
        let set = self.1.bind(py);
        let values = iterables
            .iter()
            .flat_map(|x| x.try_iter().unwrap())
            .try_collect_bound::<PySet>(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            self.0.clear();
            self.0.extend(py, set.iter().map(Bound::unbind).collect())
        } else {
            for value in values {
                self.discard(&value)?;
            }
            Ok(())
        }
    }
    pub fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.1.bind(value.py()).remove(value)?;
        self.0.remove(value)
    }
    pub fn symmetric_difference<'py>(
        &self,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySet>> {
        self.1.bind(other.py()).symmetric_difference(other)
    }
    pub fn symmetric_difference_update(&mut self, other: Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        let set = self.1.bind(py);
        set.symmetric_difference_update(other)?;
        self.0.clear();
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
    }
}

pub enum IntoUpdate<'py> {
    Set(Bound<'py, PySet>),
    Tuple(Bound<'py, PyTuple>),
    Any(Bound<'py, PyAny>),
}
impl<'py> IntoUpdate<'py> {
    pub fn into_set(self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        match self {
            IntoUpdate::Tuple(tup) => tup
                .iter()
                .flat_map(|x| x.try_iter().unwrap())
                .try_collect_bound::<PySet>(py),
            IntoUpdate::Set(pyset) => Ok(pyset),
            IntoUpdate::Any(any) => any.try_iter()?.try_collect_bound::<PySet>(py),
        }
    }
}
impl<'py> From<Bound<'py, PyAny>> for IntoUpdate<'py> {
    fn from(other: Bound<'py, PyAny>) -> Self {
        if other.is_exact_instance_of::<PySet>() {
            Self::Set(unsafe { other.cast_into_unchecked::<PySet>() })
        } else {
            Self::Any(other)
        }
    }
}

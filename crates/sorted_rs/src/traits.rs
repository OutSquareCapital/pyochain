use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
    sync::MutexGuard,
};

use crate::{
    Bounds, Loc, errors,
    indexing::Nb,
    inner::InnerData,
    reprs::PyRepr,
    types::{IntOrSlice, ListOrAny, VecPy},
};
use either::Either;
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyIterator, PyList, PySlice, PySliceIndices},
};
use pyo3_ext::prelude::CollectBoundIterator;
use tap::prelude::*;
pub(super) trait NestedVec<T> {
    fn loc(&self, loc: &Loc) -> &T;
    fn loc_insert(&mut self, loc: &Loc, value: T);
    fn loc_remove(&mut self, loc: &Loc);
    fn loc_push(&mut self, loc: &Loc, value: T);
    fn loc_last(&self, loc: &Loc) -> &T;
    fn loc_len(&self, loc: &Loc) -> usize;
}
impl<T> NestedVec<T> for [Vec<T>] {
    fn loc(&self, loc: &Loc) -> &T {
        &self[loc.pos][loc.idx]
    }
    fn loc_insert(&mut self, loc: &Loc, value: T) {
        self[loc.pos].insert(loc.idx, value);
    }
    fn loc_remove(&mut self, loc: &Loc) {
        self[loc.pos].remove(loc.idx);
    }
    fn loc_push(&mut self, loc: &Loc, value: T) {
        self[loc.pos].push(value);
    }
    fn loc_last(&self, loc: &Loc) -> &T {
        self[loc.pos].last().unwrap()
    }
    fn loc_len(&self, loc: &Loc) -> usize {
        self[loc.pos].len()
    }
}
pub enum ListAdd<'py, T> {
    Identity,
    Sorted(MutexGuard<'py, T>),
    Iterator(Bound<'py, PyIterator>),
}
pub trait ListsDataMethods: Deref<Target = InnerData> + DerefMut + PyRepr + Sized {
    fn irange_specs<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>>;
    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self>;
    fn expand(&mut self, py: Python<'_>, pos: usize);
    fn clear(&mut self, py: Python<'_>);
    fn delete(&mut self, py: Python<'_>, loc: &mut Loc) -> PyResult<()>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()>;
    fn extend(&mut self, py: Python<'_>, values: VecPy) -> PyResult<()>;
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize>;
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect<F: Fn(&[Py<PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>>(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: F,
    ) -> PyResult<usize>;
    fn bisect_left(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_right(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.as_owned_from(py, self.collapse(py))
    }
    fn concat(&self, py: Python<'_>, other: ListAdd<'_, Self>) -> PyResult<Self> {
        let out = match other {
            ListAdd::Identity => self.repeat(py, 2),
            ListAdd::Sorted(list) => self
                .iter()
                .chain(list.iter())
                .map(|x| x.clone_ref(py))
                .collect(),
            ListAdd::Iterator(it) => self
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(it.map(|x| x?.unbind().pipe(Ok)))
                .collect::<PyResult<_>>()?,
        };
        self.as_owned_from(py, out)
    }
    fn get_item_or_slice<'py>(&mut self, index: IntOrSlice<'py>) -> PyResult<ListOrAny<'py>> {
        match index {
            Either::Right(slice) => self
                .get_slice(&slice)?
                .iter()
                .collect_bound::<PyList>(slice.py())
                .map(Either::Left),
            Either::Left(index) => self
                .get_item(index.py(), index.extract()?)
                .map(Either::Right),
        }
    }
    fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        let mut bounds = Loc::default();
        self.set_pos(index, &mut bounds)?;
        self.delete(py, &mut bounds)
    }
    fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.del_slice(&slice),
            Either::Left(index) => self.del_item(index.py(), index.extract()?),
        }
    }
    fn del_slice(&mut self, slice: &Bound<'_, PySlice>) -> PyResult<()> {
        let py = slice.py();
        let length = self.len.cast_signed();
        let mut loc = Loc::default();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step.conv::<Nb>(), start.cmp(&stop)) {
            (Nb::One, Ordering::Less) if start == 0 && stop == length => {
                self.clear(py);
                Ok(())
            }
            (Nb::One, Ordering::Less) if length <= 8 * (stop - start) => {
                let mut values = self.get_slice(&PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice = self.get_slice(&PySlice::new(py, stop, length, 1))?;
                    values.extend(new_slice);
                }
                self.clear(py);
                self.extend(py, values)?;
                Ok(())
            }
            (Nb::Pos | Nb::One, _) => (start..stop)
                .step_by(step.cast_unsigned())
                .rev()
                .try_for_each(|idx| {
                    self.set_pos(idx, &mut loc)?;
                    self.delete(py, &mut loc)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        self.set_pos(idx, &mut loc)?;
                        self.delete(py, &mut loc)
                    })
            }
        }
    }
    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.find(&value)?
            .map_or(Ok(()), |mut loc| self.delete(value.py(), &mut loc))
    }
    fn imul(&mut self, py: Python<'_>, num: usize) -> PyResult<()> {
        let values = self.repeat(py, num);
        self.clear(py);
        self.extend(py, values)
    }
    fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Loc::default();
        if self.len == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }
        let len_last = self.values.last().unwrap().len().cast_signed();
        match index.conv::<Nb>() {
            Nb::NegOne => {
                bounds.pos = self.values.len() - 1;
                bounds.idx = self.values.loc_len(&bounds) - 1_usize;
            }
            Nb::Zero | Nb::One | Nb::Pos if index < self.values[0].len().cast_signed() => {
                bounds.idx = index.cast_unsigned();
            }
            Nb::Neg if -len_last < index => {
                bounds.pos = self.values.len() - 1;
                bounds.idx = (len_last + index).cast_unsigned();
            }
            _ => {
                self.set_pos(index, &mut bounds)?;
            }
        }
        let val = self.values.loc(&bounds).clone_ref(py);
        self.delete(py, &mut bounds)?;
        Ok(val.into_bound(py))
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(mut loc) => self.delete(value.py(), &mut loc),
            None => Err(errors::not_in_list(&value.repr()?)),
        }
    }
    fn as_repeated(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        self.as_owned_from(py, self.repeat(py, num))
    }
    fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.collapse(py);
        self.clear(py);
        self.load = load;
        self.extend(py, values)
    }
}
pub(super) fn update_list_by<T: ListsDataMethods, F: Fn(&Py<PyAny>, &Py<PyAny>) -> Ordering>(
    list: &mut T,
    py: Python<'_>,
    mut values: VecPy,
    func: F,
) -> PyResult<()> {
    values.sort_by(&func);
    if list.maxes.is_empty() {
        list.finalize_update(py, &values)
    } else if 4 * values.len() >= list.len {
        list.values.push(values);
        values = list.collapse(py);
        values.sort_by(func);
        list.clear(py);
        list.finalize_update(py, &values)
    } else {
        values
            .iter()
            .map(|x| x.clone_ref(py).into_bound(py))
            .try_for_each(|val| list.add(val))
    }
}

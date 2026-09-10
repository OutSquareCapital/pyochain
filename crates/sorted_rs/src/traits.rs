use std::cmp::Ordering;

use crate::{
    Bounds, InnerGetter, ListDataGetters, Loc,
    bisect::Bisect,
    errors,
    types::{IntOrSlice, VecPy},
};
use either::Either;
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PySlice, PySliceIndices, PyString},
};
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
pub trait ListsDataMethods: InnerGetter + ListDataGetters {
    fn irange_specs<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>>;
    fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self>;
    fn expand(&mut self, py: Python<'_>, pos: usize);
    fn clear(&mut self);
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
    fn bisect(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: fn(&[pyo3::Py<pyo3::PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>,
    ) -> PyResult<usize>;
    fn repr(&self, py: Python<'_>, name: Bound<'_, PyString>) -> PyResult<String>;
    fn bisect_left(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.bisect(value, Bisect::bisect_left)
    }
    fn bisect_right(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.bisect(value, Bisect::bisect_right)
    }
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.as_owned_from(py, self.inner().collapse(py))
    }
    fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        let mut bounds = Loc::default();
        self.inner_mut().set_pos(index, &mut bounds)?;
        self.delete(py, &mut bounds)
    }
    fn del_item_or_slice(&mut self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.del_slice(py, slice),
            Either::Left(index) => self.del_item(py, index),
        }
    }
    fn del_slice(&mut self, py: Python<'_>, slice: Bound<'_, PySlice>) -> PyResult<()> {
        let length = self.length().cast_signed();
        let mut loc = Loc::default();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step, start.cmp(&stop)) {
            (1, Ordering::Less) if start == 0 && stop == length => {
                self.clear();
                Ok(())
            }
            (1, Ordering::Less) if length <= 8 * (stop - start) => {
                let mut values = self
                    .inner_mut()
                    .get_slice(py, &PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice = self
                        .inner_mut()
                        .get_slice(py, &PySlice::new(py, stop, length, 1))?;
                    values.extend(new_slice);
                }
                self.clear();
                self.extend(py, values)?;
                Ok(())
            }
            _ if step > 0 => (start..stop)
                .step_by(step.cast_unsigned())
                .rev()
                .try_for_each(|idx| {
                    self.inner_mut().set_pos(idx, &mut loc)?;
                    self.delete(py, &mut loc)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        self.inner_mut().set_pos(idx, &mut loc)?;
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
        let values = self.inner().repeat(py, num);
        self.clear();
        self.extend(py, values)
    }
    fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Loc::default();
        if self.length() == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }
        let len_last = self.lists().last().unwrap().len().cast_signed();
        match index {
            -1 => {
                bounds.pos = self.lists().len() - 1;
                bounds.idx = self.lists().loc_len(&bounds) - 1_usize;
            }
            _ if 0 <= index && index < self.lists()[0].len().cast_signed() => {
                bounds.idx = index.cast_unsigned();
            }
            _ if -len_last < index && index < 0 => {
                bounds.pos = self.lists().len() - 1;
                bounds.idx = (len_last + index).cast_unsigned();
            }
            _ => {
                self.inner_mut().set_pos(index, &mut bounds)?;
            }
        }
        let val = self.lists().loc(&bounds).clone_ref(py);
        self.delete(py, &mut bounds)?;
        Ok(val.into_bound(py))
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(mut loc) => self.delete(value.py(), &mut loc),
            None => Err(errors::not_in_list(&value.repr()?)),
        }
    }
    fn repeat(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        self.as_owned_from(py, self.inner().repeat(py, num))
    }
    fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.inner().collapse(py);
        self.clear();
        self.set_load(load);
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
    if list.maxes().is_empty() {
        list.finalize_update(py, &values)
    } else if values.len() * 4 >= list.length() {
        list.lists_mut().push(values);
        values = list.inner().collapse(py);
        values.sort_by(func);
        list.clear();
        list.finalize_update(py, &values)
    } else {
        for val in values {
            list.add(py, val)?;
        }
        Ok(())
    }
}

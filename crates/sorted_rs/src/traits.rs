use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};

use crate::{
    Bounds, Loc, errors,
    indexing::Nb,
    inner::InnerData,
    list_update::ListAdd,
    reprs::PyRepr,
    types::{IntOrSlice, ListOrAny, VecPy},
};
use either::Either;
use itertools::Itertools;
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyList, PySlice, slice::PySliceIndices},
};
use pyo3_ext::prelude::*;
use tap::prelude::*;
pub(super) trait NestedVec {
    fn loc(&self, loc: &Loc) -> &Py<PyAny>;
    fn loc_insert(&mut self, loc: &Loc, value: Py<PyAny>);
    fn loc_remove(&mut self, loc: &Loc);
    fn loc_push(&mut self, loc: &Loc, value: Py<PyAny>);
    fn loc_last(&self, loc: &Loc) -> &Py<PyAny>;
    fn loc_len(&self, loc: &Loc) -> usize;
}
impl NestedVec for [Vec<Py<PyAny>>] {
    fn loc(&self, loc: &Loc) -> &Py<PyAny> {
        &self[loc.pos][loc.idx]
    }
    fn loc_insert(&mut self, loc: &Loc, value: Py<PyAny>) {
        self[loc.pos].insert(loc.idx, value);
    }
    fn loc_remove(&mut self, loc: &Loc) {
        self[loc.pos].remove(loc.idx);
    }
    fn loc_push(&mut self, loc: &Loc, value: Py<PyAny>) {
        self[loc.pos].push(value);
    }
    fn loc_last(&self, loc: &Loc) -> &Py<PyAny> {
        self[loc.pos].last().unwrap()
    }
    fn loc_len(&self, loc: &Loc) -> usize {
        self[loc.pos].len()
    }
}

pub trait ListsDataMethods:
    Deref<Target = InnerData> + DerefMut + PyRepr + Sized + Send + Sync
{
    fn irange_specs<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Bounds>;
    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self>;
    fn expand(&mut self, py: Python<'_>, pos: usize);
    fn clear(&mut self, py: Python<'_>);
    fn delete(&mut self, py: Python<'_>, loc: &mut Loc) -> PyResult<()>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn finalize_update(&mut self, py: Python<'_>, it: impl Iterator<Item = VecPy>) -> PyResult<()>;
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()>;
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
    #[inline]
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
            ListAdd::Vec(values) => values,
            ListAdd::Iterator(it) => self
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(it.map_ok(Bound::unbind))
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
                self.extend(py, values.into())?;
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
        self.extend(py, values.into())
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
        self.extend(py, values.into())
    }
}

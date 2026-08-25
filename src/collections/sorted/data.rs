use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyList, PySlice, PySliceIndices, PyString},
};
use pyo3_ext::prelude::*;
use std::cmp::Ordering;
use tap::prelude::*;

use crate::{
    abc,
    collections::sorted::{
        bisect,
        bounds::{Bounds, Pos},
        traits::BaseSortedList,
    },
    core::{PyoVec, iterators},
    traits::IntoPyochain,
};

#[derive(Default)]
pub(super) struct ListsData {
    pub(super) lists: Vec<Vec<Py<PyAny>>>,
    pub(super) maxes: Vec<Py<PyAny>>,
    pub(super) idx: Vec<usize>,
    pub(super) len: usize,
    pub(super) offset: usize,
}
impl ListsData {
    #[inline]
    pub fn get_value(&self, pos: &Pos) -> &Py<PyAny> {
        &self.lists[pos.pos][pos.idx]
    }
    #[inline]
    pub fn collapse(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.iter().map(|x| x.clone_ref(py)).collect()
    }
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &Py<PyAny>> {
        self.lists.iter().flatten()
    }
    #[inline]
    pub fn py_repr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        self.iter().collect_bound::<PyList>(py)?.repr()
    }
    #[inline]
    pub fn concat(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        self.iter()
            .map(|x| x.clone_ref(py).pipe(Ok))
            .chain(other.try_iter()?.map(|x| x?.unbind().pipe(Ok)))
            .collect::<PyResult<Vec<_>>>()
    }
    #[inline]
    pub fn repeat(&self, py: Python<'_>, num: usize) -> Vec<Py<PyAny>> {
        let values = self.collapse(py);
        (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.lists.clear();
        self.maxes.clear();
        self.idx.clear();
        self.len = 0;
        self.offset = 0;
    }
    #[inline]
    pub fn as_pyovec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyoVec>> {
        self.iter().collect_bound::<PyList>(py)?.into_pyochain()
    }
    pub fn bisect_left(
        &mut self,
        lists: Option<&Vec<Vec<Py<PyAny>>>>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<isize> {
        if self.maxes.is_empty() {
            return Ok(0);
        }
        let mut bound = Pos::new(0, 0);

        bound.pos = bisect::left(&self.maxes, &value)?;

        if bound.pos == self.maxes.len() {
            Ok(self.len as isize)
        } else {
            bound.idx = bisect::left(&lists.unwrap_or(&self.lists)[bound.pos], &value)?;
            bound.loc(self)
        }
    }
    pub fn bisect_right(
        &mut self,
        lists: Option<&Vec<Vec<Py<PyAny>>>>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<isize> {
        if self.maxes.is_empty() {
            return Ok(0);
        }
        let mut bound = Pos::new(0, 0);

        bound.pos = bisect::right(&self.maxes, value)?;

        if bound.pos == self.maxes.len() {
            return Ok(self.len as isize);
        }
        bound.idx = bisect::right(&lists.unwrap_or(&self.lists)[bound.pos], value)?;
        bound.loc(self)
    }

    pub(crate) fn getitem_from_int<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Bounds::default();
        let len_last = self
            .lists
            .last()
            .ok_or(PyIndexError::new_err("list index out of range"))?
            .len() as isize;
        match (index, self.len != 0) {
            (0, true) => self.lists[0][0].clone_ref(py).into_bound(py).pipe(Ok),
            (-1, true) => self
                .lists
                .last()
                .unwrap()
                .last()
                .unwrap()
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, false) => {
                let msg = "list index out of range";
                Err(PyIndexError::new_err(msg))
            }
            (_, true) if 0 <= index && index < self.lists[0].len() as isize => self.lists[0]
                [index as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, true) if -len_last < index && index < 0 => self.lists.last().unwrap()
                [(len_last + index) as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            _ => {
                bounds.min.set_from_pos(index, self)?;
                self.lists[bounds.min.pos][bounds.min.idx]
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    pub(crate) fn getitem_from_slice<'py>(
        &mut self,
        py: Python<'py>,
        slice: &Bound<'py, PySlice>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.len as isize)?;
        let stop_eq_len = stop == self.len as isize;
        let mut bounds = Bounds::default();
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => self.collapse(py).pipe(Ok),
            (1, Ordering::Less) => {
                bounds.min.set_from_pos(start, self)?;
                let start_list = &self.lists[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (stop - start) as usize;
                match (start_list.len() >= bounds.max.idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (true, _) => start_list[bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>()
                        .pipe(Ok),
                    (false, true) => {
                        bounds.max.pos = self.lists.len() - 1;
                        bounds.max.idx = (&self.lists)[bounds.max.pos].len();
                        get_slice(self, bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                    (false, false) => {
                        bounds.max.set_from_pos(stop, self)?;
                        get_slice(self, bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                }
            }
            (-1, Ordering::Greater) => {
                let mut result =
                    self.getitem_from_slice(py, &PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .map(|i| self.getitem_from_int(py, i).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>(),
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (_, Ordering::Less | Ordering::Equal) => Ok(Vec::new()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.getitem_from_int(py, i).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            }
        }
    }

    /// Build a positional index for indexing the sorted list.
    /// Indexes are represented as binary trees in a dense array notation similar to a binary heap.
    /// For example, given a lists representation storing integers:\
    ///     0: [1, 2, 3]
    ///     1: [4, 5]
    ///     2: [6, 7, 8, 9]
    ///     3: [10, 11, 12, 13, 14]
    /// The first transformation maps the sub-lists by their length.\
    /// The first row of the index is the length of the `sub-lists::`
    ///     0: [3, 2, 4, 5]
    /// Each row after that is the sum of consecutive pairs of the previous row:
    ///     1: [5, 9]
    ///     2: [14]
    /// Finally, the index is built by concatenating these lists together:
    ///     _index = [14, 5, 9, 3, 2, 4, 5]
    /// An offset storing the start of the first row is also stored:
    ///     _offset = 3
    /// When built, the index can be used for efficient indexing into the list.
    pub(crate) fn build_index(&mut self) -> PyResult<()> {
        let row0 = self.lists.iter().map(Vec::len).collect::<Vec<usize>>();

        if row0.len() == 1 {
            self.idx.extend(row0);
            self.offset = 0;
            return Ok(());
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            self.idx.clear();
            self.idx.extend(combined);
            self.offset = 1;
            Ok(())
        } else {
            let size = 1usize << ((row1.len() - 1).ilog2() + 1);
            row1.resize(size, 0);

            let mut tree = vec![row0, row1];
            while tree.last().unwrap().len() > 1 {
                let row = tree
                    .last()
                    .unwrap()
                    .chunks(2)
                    .map(|pair| pair.iter().sum())
                    .collect();
                tree.push(row);
            }

            let flat = tree.into_iter().rev().flatten();
            self.idx.extend(flat);
            self.offset = size * 2 - 1;
            Ok(())
        }
    }
    pub(super) fn expand_on_empty_idx(&mut self, pos: usize) -> PyResult<()> {
        let mut child = self.offset + pos;
        while child != 0 {
            self.idx[child] += 1;
            child = (child - 1) >> 1;
        }
        self.idx[0] += 1;
        Ok(())
    }
    pub(super) fn remove_pos(&mut self, bound: &Pos) {
        self.lists.remove(bound.pos);
        self.maxes.remove(bound.pos);
        self.idx.clear();
    }
    pub(super) fn expand_at_pos(
        &mut self,
        pos: usize,
        half: Vec<Py<PyAny>>,
        last_max: Py<PyAny>,
        new_max_at_pos: Py<PyAny>,
    ) {
        self.maxes[pos] = new_max_at_pos;
        self.maxes.insert(pos + 1, last_max);
        self.lists.insert(pos + 1, half);
        self.idx.clear();
    }
    pub(super) fn set_prev_from_removed(&mut self, py: Python<'_>, bounds: &Pos, prev: usize) {
        let mut removed = (self.lists)[bounds.pos]
            .iter()
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>();
        self.lists[prev].append(removed.as_mut());
        self.remove_pos(bounds);
    }
    pub(super) fn delete_on_idx(&mut self, bounds: &Pos, max_at_pos: Py<PyAny>) {
        self.maxes[bounds.pos] = max_at_pos;

        if !self.idx.is_empty() {
            let mut child = self.offset + bounds.pos;
            while child > 0 {
                self.idx[child] -= 1;
                child = (child - 1) >> 1;
            }
            self.idx[0] -= 1;
        }
    }
}
fn get_slice(data: &ListsData, bounds: Bounds) -> impl Iterator<Item = &Py<PyAny>> + '_ {
    data.lists[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(
            data.lists[bounds.min.pos + 1..bounds.max.pos]
                .iter()
                .flatten(),
        )
        .chain(data.lists[bounds.max.pos][0..bounds.max.idx].iter())
}

#[inline]
pub(super) fn reset_list<T: BaseSortedList>(slf: &T, py: Python<'_>, load: usize) -> PyResult<()> {
    let values = slf.get_data().collapse(py);
    slf.clear(py);
    slf.set_load(load);
    slf.update(py, values)
}
#[inline(always)]
pub(super) fn islice_list<T: BaseSortedList>(
    slf: Bound<'_, T>,
    start: Option<isize>,
    stop: Option<isize>,
    reverse: bool,
) -> PyResult<Bound<'_, abc::PyoIterator>> {
    let py = slf.py();
    let specs = Bounds::get_islice_specs(&mut slf.get().get_data(), py, start, stop)?;
    match specs {
        None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
        Some(bounds) => T::islice_iter(slf, bounds, reverse),
    }
}

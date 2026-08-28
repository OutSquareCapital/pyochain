use std::cmp::Ordering;

use crate::{Bounds, Pos, bisect, errors};
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PySlice, PySliceIndices},
};
use tap::Pipe;

pub trait ListDataGetters: Sized {
    fn lists(&self) -> &Vec<Vec<Py<PyAny>>>;
    fn lists_mut(&mut self) -> &mut Vec<Vec<Py<PyAny>>>;
    fn maxes(&self) -> &Vec<Py<PyAny>>;
    fn maxes_mut(&mut self) -> &mut Vec<Py<PyAny>>;
    fn idx(&self) -> &Vec<usize>;
    fn idx_mut(&mut self) -> &mut Vec<usize>;
    fn length(&self) -> usize;
    fn set_len(&mut self, len: usize);
    fn offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
}
pub trait ListsDataMethods: ListDataGetters {
    fn add(&mut self, py: Python<'_>, value: Py<PyAny>, load: usize) -> PyResult<()>;
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool>;
    fn expand(&mut self, py: Python<'_>, pos: usize, load: usize);
    fn delete(&mut self, py: Python<'_>, bounds: &mut Pos, load: usize) -> PyResult<()>;
    fn discard(&mut self, value: Bound<'_, PyAny>, load: usize) -> PyResult<()>;
    fn update(&mut self, py: Python<'_>, values: Vec<Py<PyAny>>, load: usize) -> PyResult<()>;
    fn clear(&mut self);
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn remove(&mut self, value: &Bound<'_, PyAny>, load: usize) -> PyResult<()>;
    #[inline]
    #[must_use]
    fn get_value(&self, pos: &Pos) -> &Py<PyAny> {
        &self.lists()[pos.pos][pos.idx]
    }
    #[inline]
    #[must_use]
    fn collapse(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.iter().map(|x| x.clone_ref(py)).collect()
    }
    #[inline(always)]
    fn iter(&self) -> impl Iterator<Item = &Py<PyAny>> {
        self.lists().iter().flatten()
    }
    #[inline]
    fn concat(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        self.iter()
            .map(|x| x.clone_ref(py).pipe(Ok))
            .chain(other.try_iter()?.map(|x| x?.unbind().pipe(Ok)))
            .collect::<PyResult<Vec<_>>>()
    }
    #[inline]
    #[must_use]
    fn repeat(&self, py: Python<'_>, num: usize) -> Vec<Py<PyAny>> {
        let values = self.collapse(py);
        (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>()
    }

    fn bisect_left(
        &mut self,
        lists: Option<&Vec<Vec<Py<PyAny>>>>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<isize> {
        if self.maxes().is_empty() {
            return Ok(0);
        }
        let mut bound = Pos::new(0, 0);

        bound.pos = bisect::left(self.maxes(), value)?;

        if bound.pos == self.maxes().len() {
            Ok(self.length().cast_signed())
        } else {
            bound.idx = bisect::left(&lists.unwrap_or(self.lists())[bound.pos], value)?;
            bound.loc(self)
        }
    }
    fn bisect_right(
        &mut self,
        lists: Option<&Vec<Vec<Py<PyAny>>>>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<isize> {
        if self.maxes().is_empty() {
            return Ok(0);
        }
        let mut bound = Pos::new(0, 0);

        bound.pos = bisect::right(self.maxes(), value)?;

        if bound.pos == self.maxes().len() {
            return Ok(self.length().cast_signed());
        }
        bound.idx = bisect::right(&lists.unwrap_or(self.lists())[bound.pos], value)?;
        bound.loc(self)
    }

    fn getitem_from_int<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Bounds::default();
        let len_last = self
            .lists()
            .last()
            .ok_or(PyIndexError::new_err("list index out of range"))?
            .len()
            .cast_signed();
        match (index, self.length() != 0) {
            (0, true) => self.lists()[0][0].clone_ref(py).into_bound(py).pipe(Ok),
            (-1, true) => self
                .lists()
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
            (_, true) if 0 <= index && index < self.lists()[0].len().cast_signed() => self.lists()
                [0][index.cast_unsigned()]
            .clone_ref(py)
            .into_bound(py)
            .pipe(Ok),
            (_, true) if -len_last < index && index < 0 => self.lists().last().unwrap()
                [(len_last + index).cast_unsigned()]
            .clone_ref(py)
            .into_bound(py)
            .pipe(Ok),
            _ => {
                self.set_pos(index, &mut bounds.min)?;
                self.lists()[bounds.min.pos][bounds.min.idx]
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    fn getitem_from_slice<'py>(
        &mut self,
        py: Python<'py>,
        slice: &Bound<'py, PySlice>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.length().cast_signed())?;
        let stop_eq_len = stop == self.length().cast_signed();
        let mut bounds = Bounds::default();
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => self.collapse(py).pipe(Ok),
            (1, Ordering::Less) => {
                self.set_pos(start, &mut bounds.min)?;
                let start_list = &self.lists()[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (stop - start).cast_unsigned();
                match (start_list.len() >= bounds.max.idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (true, _) => start_list[bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>()
                        .pipe(Ok),
                    (false, true) => {
                        bounds.max.pos = self.lists().len() - 1;
                        bounds.max.idx = (self.lists())[bounds.max.pos].len();
                        get_slice(self, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                    (false, false) => {
                        self.set_pos(stop, &mut bounds.max)?;
                        get_slice(self, &bounds)
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
                .step_by(step.cast_unsigned())
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
    fn build_index(&mut self) {
        let row0 = self.lists().iter().map(Vec::len).collect::<Vec<usize>>();

        if row0.len() == 1 {
            self.idx_mut().extend(&row0);
            self.set_offset(0);
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            self.idx_mut().clear();
            self.idx_mut().extend(combined);
            self.set_offset(1);
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
            self.idx_mut().extend(flat);
            self.set_offset(size * 2 - 1);
        }
    }
    fn expand_on_empty_idx(&mut self, pos: usize) {
        let mut child = self.offset() + pos;
        while child != 0 {
            self.idx_mut()[child] += 1;
            child = (child - 1) >> 1;
        }
        self.idx_mut()[0] += 1;
    }
    fn remove_pos(&mut self, bound: &Pos) {
        self.lists_mut().remove(bound.pos);
        self.maxes_mut().remove(bound.pos);
        self.idx_mut().clear();
    }
    fn expand_at_pos(
        &mut self,
        pos: usize,
        half: Vec<Py<PyAny>>,
        last_max: Py<PyAny>,
        new_max_at_pos: Py<PyAny>,
    ) {
        self.maxes_mut()[pos] = new_max_at_pos;
        self.maxes_mut().insert(pos + 1, last_max);
        self.lists_mut().insert(pos + 1, half);
        self.idx_mut().clear();
    }

    fn delitem_from_slice(
        &mut self,
        py: Python<'_>,
        slice: Bound<'_, PySlice>,
        load: usize,
    ) -> PyResult<()> {
        let length = self.length().cast_signed();
        let mut bounds = Pos::default();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step, start.cmp(&stop)) {
            (1, Ordering::Less) if start == 0 && stop == length => {
                self.clear();
                Ok(())
            }
            (1, Ordering::Less) if length <= 8 * (stop - start) => {
                let mut values = self.getitem_from_slice(py, &PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice =
                        self.getitem_from_slice(py, &PySlice::new(py, stop, length, 1))?;
                    values.extend(new_slice);
                }
                self.clear();
                self.update(py, values, load)?;
                Ok(())
            }
            _ if step > 0 => (start..stop)
                .step_by(step.cast_unsigned())
                .rev()
                .try_for_each(|idx| {
                    self.set_pos(idx, &mut bounds)?;
                    self.delete(py, &mut bounds, load)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        self.set_pos(idx, &mut bounds)?;
                        self.delete(py, &mut bounds, load)
                    })
            }
        }
    }
    fn delitem_from_int(&mut self, py: Python<'_>, index: isize, load: usize) -> PyResult<()> {
        let mut bounds = Pos::default();
        self.set_pos(index, &mut bounds)?;
        self.delete(py, &mut bounds, load)
    }

    fn delete_on_idx(&mut self, bounds: &Pos, max_at_pos: Py<PyAny>) {
        self.maxes_mut()[bounds.pos] = max_at_pos;

        if !self.idx().is_empty() {
            let mut child = self.offset() + bounds.pos;
            while child > 0 {
                self.idx_mut()[child] -= 1;
                child = (child - 1) >> 1;
            }
            self.idx_mut()[0] -= 1;
        }
    }

    fn set_pos(&mut self, mut idx: isize, bound: &mut Pos) -> PyResult<()> {
        if idx < 0 {
            if idx >= -self.lists().last().unwrap().len().cast_signed() {
                bound.pos = self.lists().len() - 1;
                bound.idx =
                    (self.lists().last().unwrap().len().cast_signed() + idx).cast_unsigned();
                return Ok(());
            }

            idx += self.length().cast_signed();

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= self.length().cast_signed() {
            return errors::out_of_range_err();
        }

        if idx < self.lists()[0].len().cast_signed() {
            bound.pos = 0;
            bound.idx = idx.cast_unsigned();
            return Ok(());
        }

        if self.idx().is_empty() {
            self.build_index();
        }
        let pos = self.idx().pipe_ref_mut(|index| {
            let mut pos = 0;
            let mut child = 1;
            let len_index = index.len();

            while child < len_index {
                let index_child = index[child].cast_signed();

                if idx < index_child {
                    pos = child;
                } else {
                    idx -= index_child;
                    pos = child + 1;
                }

                child = (pos << 1) + 1;
            }
            pos
        });

        bound.pos = pos - self.offset();
        bound.idx = idx.cast_unsigned();
        Ok(())
    }

    fn get_islice_specs(
        &mut self,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<Bounds>> {
        let length = self.length().cast_signed();
        let mut bounds = Bounds::default();

        if length == 0 {
            Ok(None)
        } else {
            //NOTE: Need to investiguate why we need to use PySlice at all. Same pattern in SliceView original code.
            let indices =
                PySlice::new(py, start.unwrap_or(0), stop.unwrap_or(length), 1).indices(length)?;

            if indices.start >= indices.stop {
                Ok(None)
            } else {
                self.set_pos(indices.start, &mut bounds.min)?;

                if indices.stop == length {
                    bounds.max.pos = self.lists().len() - 1;
                    bounds.max.idx = self.lists().last().unwrap().len();
                } else {
                    self.set_pos(indices.stop, &mut bounds.max)?;
                }

                Ok(Some(bounds))
            }
        }
    }
    fn pop<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
        load: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Pos::default();
        if self.length() == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }
        let len_last = self.lists().last().unwrap().len().cast_signed();
        match index {
            -1 => {
                bounds.pos = self.lists().len() - 1;
                bounds.idx = self.lists()[bounds.pos].len() - 1_usize;
            }
            _ if 0 <= index && index < self.lists()[0].len().cast_signed() => {
                bounds.idx = index.cast_unsigned();
            }
            _ if -len_last < index && index < 0 => {
                bounds.pos = self.lists().len() - 1;
                bounds.idx = (len_last + index).cast_unsigned();
            }
            _ => {
                self.set_pos(index, &mut bounds)?;
            }
        }
        let val = self.get_value(&bounds).clone_ref(py);
        self.delete(py, &mut bounds, load)?;
        Ok(val.into_bound(py))
    }
}

fn get_slice<'a, T: ListsDataMethods>(
    data: &'a T,
    bounds: &Bounds,
) -> impl Iterator<Item = &'a Py<PyAny>> + 'a {
    data.lists()[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(
            data.lists()[bounds.min.pos + 1..bounds.max.pos]
                .iter()
                .flatten(),
        )
        .chain(data.lists()[bounds.max.pos][0..bounds.max.idx].iter())
}

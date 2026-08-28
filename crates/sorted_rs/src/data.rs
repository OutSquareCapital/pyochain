use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PySlice, PySliceIndices},
};
use std::cmp::Ordering;
use tap::prelude::*;

use crate::{
    bisect,
    bounds::{Bounds, Indexes, Pos},
    errors, ops,
};

//TODO: This struct is way too big and do way too many things.
// Unfortunately we must first decouple as much as possible code from the main src/ folder into this crate.
#[derive(Default)]
pub struct ListsData {
    pub lists: Vec<Vec<Py<PyAny>>>,
    pub maxes: Vec<Py<PyAny>>,
    pub idx: Vec<usize>,
    pub len: usize,
    pub offset: usize,
}
impl ListsDataMethods for ListsData {
    fn lists(&self) -> &Vec<Vec<Py<PyAny>>> {
        &self.lists
    }
    fn lists_mut(&mut self) -> &mut Vec<Vec<Py<PyAny>>> {
        &mut self.lists
    }
    fn maxes(&self) -> &Vec<Py<PyAny>> {
        &self.maxes
    }
    fn maxes_mut(&mut self) -> &mut Vec<Py<PyAny>> {
        &mut self.maxes
    }
    fn idx(&self) -> &Vec<usize> {
        &self.idx
    }
    fn idx_mut(&mut self) -> &mut Vec<usize> {
        &mut self.idx
    }
    fn length(&self) -> usize {
        self.len
    }
    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
}
pub trait ListsDataMethods: Sized {
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

    #[inline]
    fn clear(&mut self) {
        self.lists_mut().clear();
        self.maxes_mut().clear();
        self.idx_mut().clear();
        self.set_len(0);
        self.set_offset(0);
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
    fn set_prev_from_removed(&mut self, py: Python<'_>, bounds: &Pos, prev: usize) {
        let mut removed = self.lists()[bounds.pos]
            .iter()
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>();
        self.lists_mut()[prev].append(removed.as_mut());
        self.remove_pos(bounds);
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
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let len_ = self.length().cast_signed();

        if len_ == 0 {
            errors::not_in_list_err(value)
        } else {
            let mut indexes = Indexes::new(start, stop, len_);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(value)
            } else {
                let mut bound = Pos {
                    pos: bisect::left(self.maxes(), value)?,
                    idx: 0,
                };
                if bound.pos == self.maxes().len() {
                    errors::not_in_list_err(value)
                } else {
                    bound.idx = bisect::left(&self.lists()[bound.pos], value)?;
                    if self.get_value(&bound).bind(py).ne(value)? {
                        errors::not_in_list_err(value)
                    } else {
                        indexes.stop -= 1;
                        let left = bound.loc(self)?;

                        if indexes.start <= left {
                            if left <= indexes.stop {
                                return Ok(left);
                            }
                        } else {
                            let right = self.bisect_right(None, value)? - 1;

                            if indexes.start <= right {
                                return Ok(indexes.start);
                            }
                        }
                        errors::not_in_list_err(value)
                    }
                }
            }
        }
    }

    fn index_by_key(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
        keys: &[Vec<Py<PyAny>>],
        key: &Bound<'_, PyAny>,
    ) -> PyResult<isize> {
        let py = value.py();
        let length = self.length().cast_signed();
        if length == 0 {
            errors::not_in_list_err(value)
        } else {
            let mut indexes = Indexes::new(start, stop, length);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(value)
            } else {
                let key = key.call1((&value,))?;
                let mut bound = Pos {
                    pos: bisect::left(self.maxes(), &key)?,
                    idx: Default::default(),
                };
                if bound.pos == self.maxes().len() {
                    errors::not_in_list_err(value)
                } else {
                    indexes.stop -= 1;
                    let v_left = &keys[bound.pos];
                    bound.idx = bisect::left(v_left, &key)?;
                    let len_keys = keys.len();
                    let mut len_sublist = v_left.len();

                    loop {
                        if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                            return errors::not_in_list_err(value);
                        }
                        if self.get_value(&bound).bind(py).eq(value)? {
                            let loc = bound.loc(self)?;
                            if indexes.start <= loc && loc <= indexes.stop {
                                return Ok(loc);
                            } else if loc > indexes.stop {
                                break;
                            }
                        }
                        bound.idx += 1;
                        if bound.idx == len_sublist {
                            bound.pos += 1;
                            if bound.pos == len_keys {
                                return errors::not_in_list_err(value);
                            }
                            len_sublist = keys[bound.pos].len();
                            bound.idx = 0;
                        }
                    }

                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut left = Pos::default();
        match ops::Maxes::new(self.maxes(), &mut left, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let mut right = Pos::default();
                left.idx = bisect::left(&self.lists()[left.pos], value)?;
                right.pos = bisect::right(self.maxes(), value)?;

                if right.pos == self.maxes().len() {
                    let left_loc = left.loc(self)?;
                    Ok(self.length() - left_loc.cast_unsigned())
                } else {
                    right.idx = bisect::right(&self.lists()[right.pos], value)?;

                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_loc = right.loc(self)?;
                        let left_loc = left.loc(self)?;
                        Ok((right_loc - left_loc).cast_unsigned())
                    }
                }
            }
        }
    }

    fn count_by_key(
        &mut self,
        value: &Bound<'_, PyAny>,
        keys: &[Vec<Py<PyAny>>],
        key: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let py = value.py();
        let mut bound = Pos::default();
        let key = key.call1((value,))?;
        match ops::Maxes::new(self.maxes(), &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let v_left = &keys[bound.pos];
                bound.idx = bisect::left(v_left, &key)?;
                let mut total = 0;
                let len_keys = keys.len();
                let mut len_sublist = keys[bound.pos].len();
                loop {
                    if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if self.lists()[bound.pos][bound.idx].bind(py).eq(value)? {
                        total += 1;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
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

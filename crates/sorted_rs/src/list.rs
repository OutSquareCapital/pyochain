use pyo3::prelude::*;
use tap::Pipe;

use crate::{
    ListDataGetters,
    bisect::Bisect,
    bounds::{Bounds, Pos},
    cmp::py_cmp,
    errors, impl_inner_getter,
    inner::{InnerData, InnerGetter, VecPy},
    ops,
    traits::{ListsDataMethods, NestedVec, update_list_by},
};

//TODO: This struct is way too big and do way too many things.
// Unfortunately we must first decouple as much as possible code from the main src/ folder into this crate.
#[derive(Default)]
pub struct ListsData(InnerData);
impl ListsData {
    pub fn from_vec(py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::default();
        new_inst.update(py, values)?;
        Ok(new_inst)
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Pos>> {
        match ops::Maxes::left(self.maxes(), value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(None),
            ops::Maxes::LenNEPos(mut bound) => {
                bound.idx = self.lists()[bound.pos].bisect_left(value)?;
                if self.lists().iloc(&bound).bind(value.py()).eq(value)? {
                    Ok(Some(bound))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
impl_inner_getter!(ListsData);
impl ListsDataMethods for ListsData {
    fn irange_specs<'py>(
        &self,
        _py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        Bounds::from_sorted(self.lists(), self.maxes(), minimum, maximum, inclusive)
    }

    fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        match ops::Maxes::right(&self.0.maxes, value.bind(py)) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                self.0.lists.push(vec![value.clone_ref(py)]);
                self.0.maxes.push(value);
            }
            ops::Maxes::LenEQPos(mut bound) => {
                bound.pos -= 1;
                self.0.lists[bound.pos].push(value.clone_ref(py));
                self.0.maxes[bound.pos] = value;
                self.expand(py, bound.pos);
            }
            ops::Maxes::LenNEPos(bound) => {
                let res = self.0.lists[bound.pos].bisect_right(value.bind(py))?;
                self.0.lists[bound.pos].insert(res, value.clone_ref(py));
                self.expand(py, bound.pos);
            }
        }
        self.increment_len();
        Ok(())
    }
    #[inline]
    fn bisect(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: fn(&[pyo3::Py<pyo3::PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>,
    ) -> PyResult<usize> {
        if self.maxes().is_empty() {
            Ok(0)
        } else {
            let mut bound = Pos::new(0, 0);
            bound.pos = func(self.maxes(), value)?;
            if bound.pos == self.maxes().len() {
                Ok(self.length())
            } else {
                bound.idx = func(&self.lists()[bound.pos], value)?;
                Ok(self.inner_mut().loc(&bound))
            }
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.0.clear();
    }
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match ops::Maxes::left(self.maxes(), value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut left) => {
                let mut right = Pos::default();
                left.idx = self.lists()[left.pos].bisect_left(value)?;
                right.pos = self.maxes().bisect_right(value)?;

                if right.pos == self.maxes().len() {
                    let left_loc = self.inner_mut().loc(&left);
                    Ok(self.length() - left_loc)
                } else {
                    right.idx = self.lists()[right.pos].bisect_right(value)?;

                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_loc = self.inner_mut().loc(&right);
                        let left_loc = self.inner_mut().loc(&left);
                        Ok(right_loc - left_loc)
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, bounds: &mut Pos) -> PyResult<()> {
        self.lists_mut()[bounds.pos].remove(bounds.idx);
        self.decrement_len();
        match ops::Delete::new(self.lists(), self.load(), bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.lists()[bounds.pos].last().unwrap().clone_ref(py);
                self.inner_mut().delete_on_idx(bounds, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1;
                }
                let prev = bounds.pos - 1;
                let mut removed = self.lists()[bounds.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.lists_mut()[prev].append(removed.as_mut());
                self.inner_mut().remove_pos(bounds);
                self.maxes_mut()[prev] = self.lists()[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes_mut()[bounds.pos] =
                    self.lists()[bounds.pos].last().unwrap().clone_ref(py);
            }
            ops::Delete::Other => self.inner_mut().remove_pos(bounds),
        }
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.find(&value)?
            .map(|mut bound| self.delete(value.py(), &mut bound))
            .transpose()
            .map(|_| ())
    }

    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self.0.lists[pos].len(), self.0.load, &self.0.idx) {
            ops::Expand::PosLenGtLoad => {
                let half = self.0.lists[pos].split_off(self.0.load);
                let new_max_at_pos = self.lists()[pos].last().unwrap().clone_ref(py);
                let last_max = half.last().unwrap().clone_ref(py);
                self.inner_mut()
                    .expand_at_pos(pos, half, last_max, new_max_at_pos);
            }
            ops::Expand::IdxNotEmpty => self.inner_mut().expand_on_empty_idx(pos),
            ops::Expand::Other => (),
        }
    }

    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        let py = value.py();
        let (mut bound, start, mut stop) =
            ops::Index::new(self.inner(), value, start, stop).into_res()?;
        bound.idx = self.lists()[bound.pos].bisect_left(value)?;
        if self.lists().iloc(&bound).bind(py).ne(value)? {
            errors::not_in_list_err(value)
        } else {
            stop -= 1;
            let left = self.inner_mut().loc(&bound);
            if start <= left {
                if left <= stop {
                    return Ok(left);
                }
            } else {
                let right = self.bisect_right(value)? - 1;

                if start <= right {
                    return Ok(start);
                }
            }
            errors::not_in_list_err(value)
        }
    }

    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(mut bound) => self.delete(value.py(), &mut bound),
            None => errors::not_in_list_err(value),
        }
    }

    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()> {
        self.inner_mut().extend_lists(py, values);
        self.0
            .lists
            .iter()
            .map(|x| x.last().unwrap().clone_ref(py))
            .pipe(|it| self.0.maxes.extend(it));
        self.set_len(values.len());
        self.idx_mut().clear();
        Ok(())
    }
    fn update(&mut self, py: Python<'_>, values: VecPy) -> PyResult<()> {
        update_list_by(self, py, values, |a, b| py_cmp(py, a, b))
    }
}

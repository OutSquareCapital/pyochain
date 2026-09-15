use pyo3::{prelude::*, types::PyString};
use tap::Pipe;

use crate::{
    InnerGetter, ListDataGetters,
    bisect::Bisect,
    bounds::{Bounds, Loc},
    cmp::py_cmp,
    errors, impl_inner_getter,
    inner::InnerData,
    ops,
    traits::{ListsDataMethods, NestedVec, PyRepr, update_list_by},
    types::VecPy,
};

#[derive(Default)]
pub struct ListsData(InnerData);
impl_inner_getter!(ListsData);
impl PyRepr for ListsData {
    fn repr(&self, name: &Bound<'_, PyString>) -> PyResult<String> {
        self.inner()
            .as_pylist(name.py())?
            .repr()
            .map(|repr| format!("{name}({repr})"))
    }
}
impl ListsDataMethods for ListsData {
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::default();
        new_inst.extend(py, values)?;
        Ok(new_inst)
    }
    fn irange_specs<'py>(
        &self,
        _py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        Bounds::from_sorted(self.values(), self.maxes(), minimum, maximum, inclusive)
    }

    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        match ops::Maxes::right(&self.0.maxes, &value) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                let unbounded = value.unbind();
                self.0.values.push(vec![unbounded.clone_ref(py)]);
                self.0.maxes.push(unbounded);
            }
            ops::Maxes::LenEQPos(mut loc) => {
                loc.pos -= 1;
                let unbounded = value.unbind();
                self.0.values.loc_push(&loc, unbounded.clone_ref(py));
                self.0.maxes[loc.pos] = unbounded;
                self.expand(py, loc.pos);
            }
            ops::Maxes::LenNEPos(loc) => {
                let res = self.0.values[loc.pos].bisect_right(&value)?;
                self.0.values[loc.pos].insert(res, value.unbind());
                self.expand(py, loc.pos);
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
            let mut loc = Loc::new(0, 0);
            loc.pos = func(self.maxes(), value)?;
            if loc.pos == self.maxes().len() {
                Ok(self.len())
            } else {
                loc.idx = func(&self.values()[loc.pos], value)?;
                Ok(self.inner_mut().loc(&loc))
            }
        }
    }
    #[inline]
    fn clear(&mut self, _py: Python<'_>) {
        self.0.clear();
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match ops::Maxes::left(self.maxes(), value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut left) => {
                let mut right = Loc::default();
                left.idx = self.values()[left.pos].bisect_left(value)?;
                right.pos = self.maxes().bisect_right(value)?;

                if right.pos == self.maxes().len() {
                    let left_loc = self.inner_mut().loc(&left);
                    Ok(self.len() - left_loc)
                } else {
                    right.idx = self.values()[right.pos].bisect_right(value)?;

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

    fn delete(&mut self, py: Python<'_>, loc: &mut Loc) -> PyResult<()> {
        self.values_mut().loc_remove(loc);
        self.decrement_len();
        match ops::Delete::new(self.values(), self.load(), loc) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.values().loc_last(loc).clone_ref(py);
                self.inner_mut().delete_on_idx(loc, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if loc.pos == 0 {
                    loc.pos += 1;
                }
                let prev = loc.pos - 1;
                let mut removed = self.values()[loc.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.values_mut()[prev].append(removed.as_mut());
                self.inner_mut().remove_pos(loc);
                self.maxes_mut()[prev] = self.values()[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes_mut()[loc.pos] = self.values().loc_last(loc).clone_ref(py);
            }
            ops::Delete::Other => self.inner_mut().remove_pos(loc),
        }
        Ok(())
    }
    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self.0.values[pos].len(), self.0.load, &self.0.idx) {
            ops::Expand::PosLenGtLoad => {
                let half = self.0.values[pos].split_off(self.0.load);
                let new_max_at_pos = self.values()[pos].last().unwrap().clone_ref(py);
                let last_max = half.last().unwrap().clone_ref(py);
                self.inner_mut()
                    .expand_at_pos(pos, half, last_max, new_max_at_pos);
            }
            ops::Expand::IdxNotEmpty => self.inner_mut().expand_on_empty_idx(pos),
            ops::Expand::Other => (),
        }
    }
    fn extend(&mut self, py: Python<'_>, values: VecPy) -> PyResult<()> {
        update_list_by(self, py, values, |a, b| py_cmp(py, a, b))
    }
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        let py = value.py();
        let (mut loc, start, mut stop) =
            ops::Index::new(self.inner(), value, start, stop).into_res()?;
        loc.idx = self.values()[loc.pos].bisect_left(value)?;
        if self.values().loc(&loc).bind(py).ne(value)? {
            Err(errors::not_in_list(&value.repr()?))
        } else {
            stop -= 1;
            let left = self.inner_mut().loc(&loc);
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
            Err(errors::not_in_list(&value.repr()?))
        }
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        match ops::Maxes::left(self.maxes(), value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(None),
            ops::Maxes::LenNEPos(mut loc) => {
                loc.idx = self.values()[loc.pos].bisect_left(value)?;
                if self.values().loc(&loc).bind(value.py()).eq(value)? {
                    Ok(Some(loc))
                } else {
                    Ok(None)
                }
            }
        }
    }
    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()> {
        self.inner_mut().extend_lists(py, values);
        self.0
            .values
            .iter()
            .map(|x| x.last().unwrap().clone_ref(py))
            .pipe(|it| self.0.maxes.extend(it));
        self.set_len(values.len());
        self.idx_mut().clear();
        Ok(())
    }
}

use std::ops::Deref;

use derive_more::Constructor;
use pyo3::prelude::*;

use crate::{
    KeyList,
    bounds::{Bounds, Loc},
    errors, ops,
    traits::ListsMethods,
};
type PyVal<'py> = Bound<'py, PyAny>;
#[derive(Debug, Constructor)]
pub(super) struct Search<'a, 'py, T> {
    list: T,
    loc: Loc,
    value: &'a PyVal<'py>,
    key: &'a PyVal<'py>,
}
impl<T> Search<'_, '_, T>
where
    T: Deref,
    T::Target: ListsMethods,
{
    fn found_in_keys(&self) -> PyResult<bool> {
        let py = self.value.py();
        self.list.keys().loc(&self.loc).bind(py).eq(self.key)
    }
    fn found_in_values(&self) -> PyResult<bool> {
        let py = self.value.py();
        self.list.values.loc(&self.loc).bind(py).eq(self.value)
    }
    #[inline]
    fn increment(&mut self) -> bool {
        self.loc.idx += 1;
        if self.loc.idx == self.list.keys()[self.loc.pos].len() {
            self.loc.pos += 1;
            if self.loc.pos == self.list.keys().len() {
                return true;
            }
            self.loc.idx = 0;
        }
        false
    }
    fn not_in_list(&self) -> PyErr {
        errors::not_in_list(self.value)
    }
}
impl<'a, 'py, T> Search<'a, 'py, &'a mut T>
where
    T: ListsMethods,
{
    pub fn on_index(list: &'a mut T, value: &'a PyVal<'py>, key: &'a PyVal<'py>) -> PyResult<Self> {
        match list.try_bisect_left(key)? {
            ops::Bisect::Empty => unreachable!(),
            ops::Bisect::LenEQPos => Err(errors::not_in_list(value)),
            ops::Bisect::LenNEPos(loc) => Ok(Search::new(list, loc, value, key)),
        }
    }
    pub fn index_by_val(self, bounds: Bounds<usize>) -> PyResult<usize> {
        match self.found_in_values() {
            Ok(true) => match ops::Placement::new(bounds, self.loc.into_index(self.list)) {
                ops::Placement::Within(index) => Ok(index),
                ops::Placement::After => Err(self.not_in_list()),
                ops::Placement::Before => {
                    if bounds.start < self.list.bisect_by_key::<false>(self.value)? {
                        Ok(bounds.start)
                    } else {
                        Err(self.not_in_list())
                    }
                }
            },
            Ok(false) => Err(self.not_in_list()),
            Err(e) => Err(e),
        }
    }
    pub fn index_by_key(mut self, bounds: Bounds<usize>) -> PyResult<usize> {
        loop {
            match self.found_in_keys() {
                Err(e) => return Err(e),
                Ok(false) => return Err(self.not_in_list()),
                Ok(true) => match self.found_in_values() {
                    Ok(true) => match ops::Placement::new(bounds, self.loc.into_index(self.list)) {
                        ops::Placement::Within(index) => return Ok(index),
                        ops::Placement::After => return Err(self.not_in_list()),
                        ops::Placement::Before => {
                            if self.increment() {
                                return Err(self.not_in_list());
                            }
                        }
                    },
                    Ok(false) => {
                        if self.increment() {
                            return Err(self.not_in_list());
                        }
                    }
                    Err(e) => return Err(e),
                },
            }
        }
    }
}
impl<T> Search<'_, '_, T>
where
    T: Deref<Target = KeyList>,
{
    pub fn count(mut self) -> PyResult<usize> {
        let mut total = 0;
        loop {
            if !self.found_in_keys()? {
                return Ok(total);
            }
            if self.found_in_values()? {
                total += 1;
            }
            if self.increment() {
                return Ok(total);
            }
        }
    }
    pub fn find(mut self) -> PyResult<Option<Loc>> {
        loop {
            if !self.found_in_keys()? {
                return Ok(None);
            }
            if self.found_in_values()? {
                return Ok(Some(self.loc));
            }
            if self.increment() {
                return Ok(None);
            }
        }
    }
}

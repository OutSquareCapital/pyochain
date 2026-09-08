//! Logical kinds of operations for sorted lists.\
//! Those enums allow to first make cheap conditional checks, and then match them on clearly named branches in a flat match statement, instead of having nested if statements with complex conditions.\
//! More importantly, they factorize the common logic between the `SortedList` and `SortedKeyList` implementations.\
//! They have very similar conditional checks, but different actions to take on each branch.

use pyo3::prelude::*;

use crate::{bisect::Bisect, bounds::Pos, errors, inner::InnerData};

/// Used in `add`, `discard`, `__contains__`, `count`, and `remove`
pub(super) enum Maxes {
    Empty,
    LenEQPos(Pos),
    LenNEPos(Pos),
    BisectErr(PyErr),
}
impl Maxes {
    pub fn left(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new(maxes, value, Bisect::bisect_left)
    }
    pub fn right(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new(maxes, value, Bisect::bisect_right)
    }
    #[inline(always)]
    fn new<F: Fn(&[Py<PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>>(
        maxes: &[Py<PyAny>],
        value: &Bound<'_, PyAny>,
        func: F,
    ) -> Self {
        if maxes.is_empty() {
            Self::Empty
        } else {
            func(maxes, value)
                .map(Pos::with_pos)
                .map_or_else(Self::BisectErr, |bound| {
                    if bound.pos == maxes.len() {
                        Self::LenEQPos(bound)
                    } else {
                        Self::LenNEPos(bound)
                    }
                })
        }
    }
}

/// Used in `expand`
pub enum Expand {
    PosLenGtLoad,
    IdxNotEmpty,
    Other,
}
impl Expand {
    #[inline(always)]
    #[must_use]
    pub fn new(lists_len: usize, load: usize, idx: &[usize]) -> Self {
        if lists_len > load << 1 {
            Self::PosLenGtLoad
        } else if !idx.is_empty() {
            Self::IdxNotEmpty
        } else {
            Self::Other
        }
    }
}
/// Used in `delete`
pub enum Delete {
    PosSupToLoad,
    DataLenGTOne,
    LenPosNotZero,
    Other,
}
impl Delete {
    #[inline(always)]
    #[must_use]
    pub fn new<T>(lists: &[Vec<T>], load: usize, bounds: &Pos) -> Self {
        let len_pos = lists[bounds.pos].len();
        if len_pos > (load >> 1) {
            Self::PosSupToLoad
        } else if lists.len() > 1 {
            Self::DataLenGTOne
        } else if len_pos != 0 {
            Self::LenPosNotZero
        } else {
            Self::Other
        }
    }
}
pub enum Update {
    EmptyMaxes,
    OtherGESelf,
    OtherLTSelf,
}
impl Update {
    #[inline(always)]
    pub fn new<T, U>(maxes: &[T], length: usize, values: &[U]) -> Self {
        if maxes.is_empty() {
            Self::EmptyMaxes
        } else if values.len() * 4 >= length {
            Self::OtherGESelf
        } else {
            Self::OtherLTSelf
        }
    }
}
/// `Pos`, `start`, and `stop` bounds for a search in a sorted list.
type IdxBounds = (Pos, usize, usize);
pub(super) enum Index<'py, 'a> {
    NotFound(&'a Bound<'py, PyAny>),
    Empty(&'a Bound<'py, PyAny>),
    InvalidRange(&'a Bound<'py, PyAny>),
    BisectErr(PyErr),
    Searchable(IdxBounds),
}
impl<'py, 'a> Index<'py, 'a> {
    pub fn new(
        data: &InnerData,
        value: &'a Bound<'py, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> Self {
        let length = data.len.cast_signed();
        if length == 0 {
            Self::Empty(value)
        } else {
            let mut start = start.unwrap_or(0);
            let mut stop = stop.unwrap_or(length);
            if start < 0 {
                start += length;
            }
            start = start.max(0);
            if stop < 0 {
                stop += length;
            }
            stop = stop.min(length);
            if stop <= start {
                Self::InvalidRange(value)
            } else {
                match data.maxes.bisect_left(value).map(Pos::with_pos) {
                    Ok(bound) if bound.pos == data.maxes.len() => Self::NotFound(value),
                    Ok(bound) => {
                        Self::Searchable((bound, start.cast_unsigned(), stop.cast_unsigned()))
                    }
                    Err(err) => Self::BisectErr(err),
                }
            }
        }
    }
    pub fn into_res(self) -> PyResult<IdxBounds> {
        match self {
            Self::Searchable((bound, start, stop)) => Ok((bound, start, stop)),
            Self::NotFound(value) | Self::Empty(value) | Self::InvalidRange(value) => {
                errors::not_in_list_err(value)
            }
            Self::BisectErr(err) => Err(err),
        }
    }
}

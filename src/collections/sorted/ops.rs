//! Logical kinds of operations for sorted lists.\
//! Those enums allow to first make cheap conditional checks, and then match them on clearly named branches in a flat match statement, instead of having nested if statements with complex conditions.\
//! More importantly, they factorize the common logic between the `SortedList` and `SortedKeyList` implementations.\
//! They have very similar conditional checks, but different actions to take on each branch.

use pyo3::prelude::*;

use crate::collections::sorted::bounds::Pos;

/// Used in `add`, `discard`, `__contains__`, `count`, and `remove`
pub(super) enum Maxes {
    Empty,
    LenEQPos,
    LenNEPos,
}
impl Maxes {
    #[inline(always)]
    pub fn new<F: Fn(&[Py<PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>>(
        maxes: &[Py<PyAny>],
        bound: &mut Pos,
        value: &Bound<'_, PyAny>,
        func: F,
    ) -> PyResult<Self> {
        if maxes.is_empty() {
            Ok(Self::Empty)
        } else {
            bound.pos = func(maxes, value)?;
            if bound.pos == maxes.len() {
                Ok(Self::LenEQPos)
            } else {
                Ok(Self::LenNEPos)
            }
        }
    }
}

/// Used in `expand`
pub(super) enum Expand {
    PosLenGtLoad,
    IdxNotEmpty,
    Other,
}
impl Expand {
    #[inline(always)]
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
pub(super) enum Delete {
    PosSupToLoad,
    DataLenGTOne,
    LenPosNotZero,
    Other,
}
impl Delete {
    #[inline(always)]
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
pub(super) enum Update {
    EmptyMaxes,
    OtherGESelf,
    OtherLTSelf,
}
impl Update {
    #[inline(always)]
    pub fn new<T, U>(maxes: &[T], length: usize, values: &[U]) -> Self {
        if maxes.is_empty() {
            Self::EmptyMaxes
        } else {
            if values.len() * 4 >= length {
                Self::OtherGESelf
            } else {
                Self::OtherLTSelf
            }
        }
    }
}

//! Logical kinds of operations for sorted lists.\
//! Those enums allow to first make cheap conditional checks, and then match them on clearly named branches in a flat match statement, instead of having nested if statements with complex conditions.\
//! More importantly, they factorize the common logic between the `SortedList` and `SortedKeyList` implementations.\
//! They have very similar conditional checks, but different actions to take on each branch.

use std::hint::unreachable_unchecked;

use derive_more::From;
use pyo3::prelude::*;
use tap::Conv;

use crate::{
    bisect::Bisect,
    bounds::{Loc, RangeIdxs},
    inner::InnerList,
    traits::ListsMethods,
    types::VecPy,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum Nb {
    GTOne(usize),
    One,
    Zero,
    MinusOne,
    LTMinusOne(isize),
}
impl From<isize> for Nb {
    fn from(value: isize) -> Self {
        match value {
            0 => Self::Zero,
            1 => Self::One,
            -1 => Self::MinusOne,
            _ if value > 1 => Self::GTOne(value.cast_unsigned()),
            _ => Self::LTMinusOne(value),
        }
    }
}
#[derive(Debug, Clone, Copy, From)]
pub(super) enum Slice {
    #[from]
    Forward(ForwardSlice),
    ForwardStep(usize),
    Backward,
    BackwardStep(usize),
    /// Negative step with nothing to iterate (mirrors Python's `range`, which is empty when `start <= stop` for a negative step).
    NoOp,
}
impl Slice {
    pub fn new(indices: RangeIdxs<usize>, step: Nb, length: usize) -> Self {
        debug_assert!(indices.stop <= length);
        match step {
            // SAFETY: Constructors guarantee that `step` is not zero (slice.indices raises `ValueError` on `step == 0`).
            Nb::Zero => unsafe { unreachable_unchecked() },
            _ if indices.start >= indices.stop => Self::NoOp,
            Nb::One => ForwardSlice::new(indices, length).into(),
            Nb::GTOne(s) => Self::ForwardStep(s),
            Nb::MinusOne => Self::Backward,
            Nb::LTMinusOne(s) => Self::BackwardStep(s.unsigned_abs()),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) enum ForwardSlice {
    /// Whole slice optimization: start to stop slices the whole sorted list.
    Full,
    Partial,
    ToEnd,
    Small,
    SmallToAdjust,
}
impl ForwardSlice {
    pub fn new(indices: RangeIdxs<usize>, length: usize) -> Self {
        match (
            indices.stop == length,
            length <= 8 * (indices.stop - indices.start),
        ) {
            (true, _) if indices.start == 0 => Self::Full,
            (true, true) => Self::Small,
            (true, false) => Self::ToEnd,
            (false, true) => Self::SmallToAdjust,
            (false, false) => Self::Partial,
        }
    }
}
/// Used in `add`, `discard`, `__contains__`, `count`, and `remove`
pub(super) enum Maxes {
    Empty,
    LenEQPos(Loc),
    LenNEPos(Loc),
    BisectErr(PyErr),
}
impl Maxes {
    #[inline(always)]
    pub fn left(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new::<true>(maxes, value)
    }
    #[inline(always)]
    pub fn right(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new::<false>(maxes, value)
    }
    #[inline(always)]
    fn new<const LEFT: bool>(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        if maxes.is_empty() {
            Self::Empty
        } else {
            match maxes.bisect::<LEFT>(value).map(Loc::with_pos) {
                Err(e) => Self::BisectErr(e),
                Ok(loc) if loc.pos == maxes.len() => Self::LenEQPos(loc),
                Ok(loc) => Self::LenNEPos(loc),
            }
        }
    }
}

/// Used in `expand`
pub enum Expand {
    MustSplit,
    IdxBuilt,
    NoOp,
}
impl Expand {
    #[inline(always)]
    #[must_use]
    pub fn new(data: &impl ListsMethods, pos: usize) -> Self {
        if data.keys()[pos].len() > data.load.higher_limit() {
            Self::MustSplit
        } else if data.lengths.is_built() {
            Self::IdxBuilt
        } else {
            Self::NoOp
        }
    }
}
pub enum Update {
    EmptyMaxes,
    Small,
    Big,
}
impl Update {
    pub fn new(list: &InnerList, values: &impl ExactSizeIterator<Item = Py<PyAny>>) -> Self {
        if list.maxes.is_empty() {
            Self::EmptyMaxes
        } else if 4 * values.len() >= list.len() {
            Self::Big
        } else {
            Self::Small
        }
    }
}
/// Used in `delete`
pub enum SubListState {
    LenSupToHalfLoad,
    ManySublists,
    NonEmpty,
    Empty,
}
impl SubListState {
    #[inline(always)]
    #[must_use]
    pub fn new(data: &impl ListsMethods, pos: usize) -> Self {
        let sublist = &data.keys()[pos];
        if sublist.len() > data.load.lower_limit() {
            Self::LenSupToHalfLoad
        } else if data.keys().len() > 1 {
            Self::ManySublists
        } else if sublist.is_empty() {
            Self::Empty
        } else {
            Self::NonEmpty
        }
    }
}

// Used in `pop` and `get_item`
pub(super) enum Locate {
    Empty,
    Last,
    First,
    InFirstList(usize),
    InLastList(isize),
    InOtherList(usize),
    InOtherListNeg(isize),
}
impl Locate {
    #[inline]
    pub fn new(length: usize, values: &[VecPy], index: isize) -> Self {
        match (length, index.conv::<Nb>()) {
            (1, Nb::Zero | Nb::MinusOne) => Self::First,
            (0 | 1, _) => Self::Empty,
            (_, Nb::Zero) => Self::First,
            (_, Nb::MinusOne) => Self::Last,
            (_, Nb::One) if 1 < values[0].len() => Self::InFirstList(1),
            (_, Nb::One) => Self::InOtherList(1),
            (_, Nb::GTOne(i)) if i < values[0].len() => Self::InFirstList(i),
            (_, Nb::LTMinusOne(i)) if -values.last().unwrap().len().cast_signed() < i => {
                Self::InLastList(i)
            }
            (_, Nb::GTOne(i)) => Self::InOtherList(i),
            (_, Nb::LTMinusOne(i)) => Self::InOtherListNeg(i),
        }
    }
}

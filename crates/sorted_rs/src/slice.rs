use std::{cmp::Ordering, hint::unreachable_unchecked};

use itertools::Itertools;
use pyo3::{prelude::*, types::PySlice};
use tap::prelude::*;

use crate::{
    Bounds, InnerData, Loc,
    ops::Nb,
    traits::{ListsDataMethods, NestedVec},
    types::VecPy,
};

#[derive(Clone, Copy)]
pub(super) struct SliceIdxs<'py> {
    py: Python<'py>,
    start: isize,
    stop: isize,
    step: isize,
    /// Length of the referenced sequence.
    ref_len: isize,
}
impl<'py> SliceIdxs<'py> {
    pub fn new(slice: &Bound<'py, PySlice>, length: usize) -> PyResult<Self> {
        let ref_len = length.cast_signed();
        let i = slice.indices(ref_len)?;
        Self {
            py: slice.py(),
            start: i.start,
            stop: i.stop,
            step: i.step,
            ref_len,
        }
        .pipe(Ok)
    }

    pub fn extract_from(&self, list: &mut InnerData) -> PyResult<VecPy> {
        let stop_eq_len = self.stop_cmp_len().is_eq();
        let mut bounds = Bounds::default();
        let py = self.py;
        match self.kind() {
            // SAFETY: Constructors guarantee that `step` is not zero (slice.indices raises `ValueError` on `step == 0`).
            (_, Nb::Zero) => unsafe { unreachable_unchecked() },
            // Whole slice optimization: start to stop slices the whole sorted list.
            (Ordering::Less, Nb::One) if self.start == 0 && stop_eq_len => {
                list.collapse(py).pipe(Ok)
            }
            (Ordering::Less, Nb::One) => {
                list.set_pos(self.start, &mut bounds.min)?;
                let start_list = &list.values[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (self.stop - self.start).cast_unsigned();
                match (start_list.len() >= bounds.max.idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are within the start list.
                    (true, _) => start_list[bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>(),
                    (false, true) => {
                        bounds.max.pos = list.values.len() - 1;
                        bounds.max.idx = list.values.loc_len(&bounds.max);
                        get_slice(&list.values, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                    }
                    (false, false) => {
                        list.set_pos(self.stop, &mut bounds.max)?;
                        get_slice(&list.values, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                    }
                }
                .pipe(Ok)
            }
            (Ordering::Greater, Nb::MinusOne) => {
                let mut result = self
                    .to_new_with(self.stop + 1, self.start + 1)
                    .extract_from(list)?;
                result.reverse();
                Ok(result)
            }
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (Ordering::Less | Ordering::Equal, Nb::MinusOne | Nb::LTMinusOne) => Ok(Vec::new()),
            (Ordering::Greater, Nb::LTMinusOne) => self
                .iter()
                .map(|i| list.get_item(py, i))
                .map_ok(Bound::unbind)
                .collect(),
            // Return a list because a negative step could reverse the order of the items and this could be the desired behavior.
            (_, Nb::One | Nb::GTOne) => self
                .into_range()
                .map(|i| list.get_item(py, i))
                .map_ok(Bound::unbind)
                .collect(),
        }
    }

    pub fn delete_from<T: ListsDataMethods>(&self, list: &mut T) -> PyResult<()> {
        let py = self.py;
        let mut loc = Loc::default();
        match self.kind() {
            // SAFETY: Constructors guarantee that `step` is not zero (slice.indices raises `ValueError` on `step == 0`).
            (_, Nb::Zero) => unsafe { unreachable_unchecked() },
            (Ordering::Less, Nb::One) if self.start == 0 && self.stop_cmp_len().is_eq() => {
                list.clear(py);
                Ok(())
            }
            (Ordering::Less, Nb::One) if self.ref_len <= 8 * (self.stop - self.start) => {
                let mut values = self.to_new_with(0, self.start).extract_from(list)?;
                if self.stop_cmp_len().is_lt() {
                    let mut new_slice = self
                        .to_new_with(self.stop, self.ref_len)
                        .extract_from(list)?;
                    values.append(new_slice.as_mut());
                }
                list.clear(py);
                list.extend(py, values.into())?;
                Ok(())
            }
            (_, Nb::GTOne | Nb::One) => self.into_range().rev().try_for_each(|idx| {
                list.set_pos(idx, &mut loc)?;
                list.delete(py, &mut loc)
            }),
            (Ordering::Greater, Nb::MinusOne | Nb::LTMinusOne) => self.iter().try_for_each(|idx| {
                list.set_pos(idx, &mut loc)?;
                list.delete(py, &mut loc)
            }),
            // Negative step with nothing to delete (mirrors Python's `range`, which is empty when `start <= stop`).
            (Ordering::Less | Ordering::Equal, Nb::MinusOne | Nb::LTMinusOne) => Ok(()),
        }
    }
    fn stop_cmp_len(&self) -> Ordering {
        self.stop.cmp(&self.ref_len)
    }
    fn kind(&self) -> (Ordering, Nb) {
        (self.start.cmp(&self.stop), self.step.conv::<Nb>())
    }
    fn to_new_with(self, start: isize, stop: isize) -> Self {
        Self {
            py: self.py,
            start,
            stop,
            step: 1,
            ref_len: self.ref_len,
        }
    }
    fn iter(&self) -> impl Iterator<Item = isize> + '_ {
        std::iter::successors(Some(self.start), move |&i| {
            (i + self.step > self.stop).then_some(i + self.step)
        })
    }
    fn into_range(self) -> std::iter::StepBy<std::ops::Range<isize>> {
        (self.start..self.stop).step_by(self.step.cast_unsigned())
    }
}

fn get_slice<'a>(lists: &'a [VecPy], bounds: &Bounds) -> impl Iterator<Item = &'a Py<PyAny>> + 'a {
    lists[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(lists[bounds.min.pos + 1..bounds.max.pos].iter().flatten())
        .chain(lists[bounds.max.pos][0..bounds.max.idx].iter())
}

use pyo3::prelude::*;
use tap::prelude::*;

use crate::{bounds::Loc, inner::Values, traits::ListsMethods};

pub enum Strategy {
    LenSupToMin,
    ManySublists(Merge),
    NonEmpty,
    Empty,
}
impl Strategy {
    #[inline(always)]
    #[must_use]
    pub fn new(list: &impl ListsMethods, pos: usize) -> Self {
        let remaining = list.keys()[pos].len() - 1;
        if remaining > list.load.min_size() {
            Self::LenSupToMin
        } else if list.keys().len() > 1 {
            match (pos, remaining) {
                (0, 0) => Merge::NextReplacesFirst,
                (0, _) => Merge::NextIntoFirst,
                _ => Merge::IntoPrev,
            }
            .pipe(Self::ManySublists)
        } else if remaining == 0 {
            Self::Empty
        } else {
            Self::NonEmpty
        }
    }
    #[inline]
    pub fn dispatch<L: ListsMethods>(self, py: Python<'_>, list: &mut L, loc: Loc) {
        match self {
            Self::LenSupToMin => {
                list.loc_remove(loc);
                list.maxes[loc.pos] = list.keys().loc_last(&loc).clone_ref(py);
                if list.lengths.is_built() {
                    list.lengths.delete_at(loc.pos);
                }
            }
            Self::ManySublists(merge) => {
                let prev = merge.remove(&mut list.values, loc);
                list.maxes.remove(prev + 1);
                list.lengths.inner.clear();
                list.delete_merge_keys(loc, merge);
                list.maxes[prev] = list.keys_mut()[prev].last().clone_ref(py);
                list.expand(py, prev);
            }
            Self::NonEmpty => {
                list.loc_remove(loc);
                list.maxes[loc.pos] = list.keys().loc_last(&loc).clone_ref(py);
            }
            Self::Empty => {
                list.maxes.remove(loc.pos);
                list.lengths.inner.clear();
                list.keys_mut().remove(loc.pos);
                list.delete_values(loc);
            }
        }
    }
}
/// How the sublist of the deleted item is merged with a neighbor.
#[derive(Clone, Copy)]
pub enum Merge {
    /// The item's sublist (`pos > 0`) goes into the previous one.
    IntoPrev,
    /// The item is in the first sublist, which keeps other items: the second one goes into it.
    NextIntoFirst,
    /// The item is the only one of the first sublist: the second one replaces it.
    NextReplacesFirst,
}
impl Merge {
    /// Removes the item at `loc`, and merges its sublist with a neighbor as `merge` says.\
    /// Returns the position of the merged sublist. The other one is removed, and was right after it.
    #[inline]
    pub(super) fn remove(self, values: &mut Values, loc: Loc) -> usize {
        match self {
            Self::IntoPrev => {
                let mut rest = values.remove(loc.pos).into_vec();
                rest.remove(loc.idx);
                values[loc.pos - 1].append(&mut rest);
                loc.pos - 1
            }
            Self::NextIntoFirst => {
                let mut next = values.remove(1).into_vec();
                let first = &mut values[0];
                first.remove(loc.idx);
                first.append(&mut next);
                0
            }
            Self::NextReplacesFirst => {
                values[0] = values.remove(1);
                0
            }
        }
    }
}

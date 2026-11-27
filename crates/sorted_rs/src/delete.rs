use pyo3::prelude::*;
use tap::prelude::*;

use crate::{
    bounds::Loc,
    chunks::{Chunks, ChunksMethods},
    inner::InnerList,
};

pub enum Strategy {
    LenSupToMin,
    ManySublists(Merge),
    NonEmpty,
    Empty,
}
impl Strategy {
    #[inline(always)]
    #[must_use]
    pub fn new<C: ChunksMethods>(list: &InnerList<C>, pos: usize) -> Self {
        let remaining = list.chunks.keys()[pos].len() - 1;
        if remaining > list.load.min_size() {
            Self::LenSupToMin
        } else if list.chunks.keys().len() > 1 {
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
    pub fn dispatch<C: ChunksMethods>(self, py: Python<'_>, list: &mut InnerList<C>, loc: Loc) {
        match self {
            Self::LenSupToMin => {
                list.remove_in_chunk(py, loc);
                if list.lengths.is_built() {
                    list.lengths.delete_at(loc.pos);
                }
            }
            Self::ManySublists(merge) => {
                let prev = list.chunks.merge(loc, merge);
                list.maxes.remove(prev + 1);
                list.lengths.inner.clear();
                list.refresh_max(py, prev);
                list.expand(py, prev);
            }
            Self::NonEmpty => list.remove_in_chunk(py, loc),
            Self::Empty => {
                list.maxes.remove(loc.pos);
                list.lengths.inner.clear();
                list.chunks.remove_chunk(loc.pos);
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
    pub(super) fn remove(self, values: &mut Chunks, loc: Loc) -> usize {
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

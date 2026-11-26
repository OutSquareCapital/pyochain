use tap::prelude::*;

use crate::{
    bounds::Loc,
    sublist::{Chunk, Chunks},
};
/// Quoting Grant Jenks:
/// > "a tree of pair-wise sums of the lengths of the lists. The tree is maintained as a dense binary tree."
///
/// See more [here](https://grantjenks.com/docs/sortedcontainers/implementation.html)
#[derive(Debug, Default)]
pub struct Lengths {
    pub(super) inner: Vec<usize>,
    pub(super) offset: usize,
}

impl Lengths {
    pub const fn is_built(&self) -> bool {
        !self.inner.is_empty()
    }
    pub const fn is_not_built(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn delete_at(&mut self, pos: usize) {
        Self::path_to_root(self.offset + pos).for_each(|c| self.inner[c] -= 1);
    }
    pub fn expand_at(&mut self, pos: usize) {
        self.add_at(pos, 1);
    }
    /// Accounts for `count` items added to the sublist at `pos`.
    pub fn add_at(&mut self, pos: usize, count: usize) {
        Self::path_to_root(self.offset + pos).for_each(|c| self.inner[c] += count);
    }
    pub fn loc_at(&self, loc: &Loc) -> usize {
        Self::path_to_root(self.offset + loc.pos)
            .take_while(|&c| c != 0)
            .filter(|c| c.is_multiple_of(2))
            .map(|c| self.inner[c - 1])
            .sum::<usize>()
            + loc.idx
    }
    pub fn pos_at(&self, index: usize) -> Loc {
        std::iter::successors(Some((0, index)), |&(pos, index)| {
            let child = pos * 2 + 1;
            self.inner.get(child).map(|&len| {
                if index < len {
                    (child, index)
                } else {
                    (child + 1, index - len)
                }
            })
        })
        .last()
        .unwrap()
        .pipe(|(pos, index)| Loc::new(pos - self.offset, index))
    }
    pub fn build(&mut self, values: &Chunks) {
        debug_assert!(self.is_not_built());
        let levels = values.len().next_power_of_two().ilog2();
        self.offset = 2_usize.pow(levels) - 1;
        self.inner.resize(self.offset, 0);
        self.inner.extend(values.iter_chunks().map(Chunk::len));
        (0..levels)
            .rev()
            .map(|level| 2_usize.pow(level))
            .for_each(|size| {
                let (parents, children) = self.inner[size - 1..].split_at_mut(size);
                parents
                    .iter_mut()
                    .zip(children.chunks(2))
                    .for_each(|(parent, pair)| *parent = pair.iter().sum());
            });
    }
    /// Indices from `child` up to the root of the tree, both included.
    #[inline]
    fn path_to_root(child: usize) -> impl std::iter::FusedIterator<Item = usize> {
        std::iter::successors(Some(child), |&c| c.checked_sub(1).map(|c| c / 2))
    }
}

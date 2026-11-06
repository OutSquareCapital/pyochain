use crate::{bounds::Loc, types::VecPy};
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
    pub fn is_built(&self) -> bool {
        !self.inner.is_empty()
    }
    pub fn is_not_built(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn delete_at(&mut self, pos: usize) {
        let mut child = self.offset + pos;
        while child > 0 {
            self.inner[child] -= 1;
            child = (child - 1) / 2;
        }
        self.inner[0] -= 1;
    }
    pub fn expand_at(&mut self, pos: usize) {
        let mut child = self.offset + pos;
        while child != 0 {
            self.inner[child] += 1;
            child = (child - 1) / 2;
        }
        self.inner[0] += 1;
    }
    pub fn loc_at(&self, loc: &Loc) -> usize {
        // Increment pos to point in the index to len(self.lists[pos]).
        let mut child = loc.pos + self.offset;
        // Iterate until reaching the root of the index tree at pos = 0.
        let mut total = 0;
        while child != 0 {
            // Right-child nodes are at even indices. At such indices
            // account the total below the left child node.
            if child.is_multiple_of(2) {
                total += self.inner[child - 1];
            }
            // Advance pos to the parent node.
            child = (child - 1) / 2;
        }
        total + loc.idx
    }
    pub fn pos_at(&self, mut index: usize) -> Loc {
        let mut pos = 0;
        let mut child = 1;
        let len_index = self.inner.len();

        while child < len_index {
            let index_child = self.inner[child];

            if index < index_child {
                pos = child;
            } else {
                index -= index_child;
                pos = child + 1;
            }

            child = (pos * 2) + 1;
        }
        Loc::new(pos - self.offset, index)
    }
    //TODO: I'm almost sure this is one of the only places were we concretely diverge from the original implementation. Must check what changes exactly, and how we could further improve this.
    pub fn build<'a>(&mut self, values: impl IntoIterator<Item = &'a VecPy>) {
        let row0 = values.into_iter().map(Vec::len).collect::<Vec<usize>>();

        if row0.len() == 1 {
            self.inner.extend(&row0);
            self.offset = 1;
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            self.inner.clear();
            self.inner.extend(combined);
            self.offset = 1;
        } else {
            let size = row1.len().next_power_of_two();
            debug_assert_ne!(size, 0);
            row1.resize(size, 0);

            let mut tree = vec![row0, row1];
            while tree.last().unwrap().len() > 1 {
                let row = tree
                    .last()
                    .unwrap()
                    .chunks(2)
                    .map(|pair| pair.iter().sum())
                    .collect();
                tree.push(row);
            }

            let flat = tree.into_iter().rev().flatten();
            self.inner.extend(flat);
            self.offset = size * 2 - 1;
        }
    }
}

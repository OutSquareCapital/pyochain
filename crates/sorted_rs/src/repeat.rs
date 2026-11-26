use crate::{
    bounds::Loc,
    chunks::{Chunks, ChunksMethods},
    item::{Item, ItemRef},
    sort::IntoChunked,
    traits::{ItemOf, ListsMethods},
};
use derive_more::Constructor;
use derive_new::new;
use itertools::Itertools;
use pyo3::prelude::*;
use std_tools::prelude::*;
use tap::Pipe;
#[derive(Constructor, Clone, Copy)]
pub(crate) struct Repeated(usize);
impl Repeated {
    pub fn extend_into<L: ListsMethods>(self, py: Python<'_>, list: &L) -> PyResult<L> {
        let mut new_inst = list.new_empty(py);
        new_inst.load = list.load;
        Groups::new(py, list, self.0, list.len * self.0)
            .into_chunked()
            .try_extend_into::<L::Chunks>(py, &mut new_inst)
            .map(|()| new_inst)
    }
}
#[derive(new)]
struct Groups<'py, 'l, L: ListsMethods> {
    py: Python<'py>,
    list: &'l L,
    #[new(default)]
    copies: usize,
    #[new(default)]
    start: Loc,
    #[new(default)]
    end: Loc,
    #[new(default)]
    cur: Loc,
    num: usize,
    len: usize,
}
impl<L: ListsMethods> Groups<'_, '_, L> {
    fn group_end(&self, start: Loc) -> PyResult<Loc> {
        let keys = self.list.chunks.keys();
        std::iter::successors(Some(start), |loc| {
            let next = loc.next(keys);
            (next.pos < keys.len()).then_some(next)
        })
        .tuple_windows()
        .try_find_map(|(a, b)| {
            self.list
                .chunks
                .loc_item(a)
                .cmp(&self.list.chunks.loc_item(b))?
                .is_lt()
                .then_some(b)
                .pipe(PyResult::Ok)
        })?
        .unwrap_or(Loc::with_pos(keys.len()))
        .pipe(Ok)
    }
}
impl<L: ListsMethods> Iterator for Groups<'_, '_, L> {
    type Item = PyResult<ItemOf<L>>;
    fn next(&mut self) -> Option<Self::Item> {
        (self.len > 0).then(|| {
            if self.cur == self.end {
                if self.copies == 0 {
                    self.start = self.end;
                    self.end = self.group_end(self.start)?;
                    self.copies = self.num;
                }
                self.copies -= 1;
                self.cur = self.start;
            }
            let item = self.list.chunks.loc_item(self.cur).owned(self.py);
            self.cur = self.cur.next(self.list.chunks.keys());
            self.len -= 1;
            Ok(item)
        })
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<L: ListsMethods> ExactSizeIterator for Groups<'_, '_, L> {}
impl Loc {
    fn next(self, keys: &Chunks) -> Self {
        if self.idx + 1 < keys.len_of_chunk(&self) {
            Self::new(self.pos, self.idx + 1)
        } else {
            Self::with_pos(self.pos + 1)
        }
    }
}

use crate::{
    item::{Item, ItemRef},
    sort::IntoChunked,
    traits::ListsMethods,
};
use derive_more::Constructor;
use derive_new::new;
use itertools::Itertools;
use pyo3::prelude::*;
use std::{
    cmp::Ordering,
    iter::{Flatten, RepeatN, Take},
};
use tap::Pipe;
#[derive(Constructor, Clone, Copy)]
pub(crate) struct Repeated(usize);
impl Repeated {
    pub fn extend_into<L: ListsMethods>(self, py: Python<'_>, list: &L) -> PyResult<L> {
        let mut new_inst = list.new_empty(py);
        new_inst.load = list.load;
        list.iter_keys()
            .pipe(|items| Groups::new(py, items, self.0, list.len * self.0))
            .into_chunked()
            .try_extend_into(py, &mut new_inst)
            .map(|()| new_inst)
    }
}
#[derive(new)]
struct Groups<'py, I: Iterator + Clone> {
    py: Python<'py>,
    items: I,
    #[new(default)]
    copies: Option<Flatten<RepeatN<Take<I>>>>,
    num: usize,
    len: usize,
}
impl<I> Groups<'_, I>
where
    I: Iterator + Clone,
    I::Item: ItemRef,
{
    fn refresh(&mut self, len: usize) -> Option<PyResult<<I::Item as ItemRef>::Owned>> {
        self.copies = self
            .items
            .clone()
            .take(len)
            .pipe(|it| std::iter::repeat_n(it, self.num))
            .flatten()
            .pipe(Some);
        self.items.nth(len - 1);
        self.next()
    }
}
impl<I> Iterator for Groups<'_, I>
where
    I: Iterator + Clone,
    I::Item: ItemRef,
{
    type Item = PyResult<<I::Item as ItemRef>::Owned>;
    fn next(&mut self) -> Option<Self::Item> {
        use std::ops::ControlFlow as CF;
        self.copies
            .as_mut()
            .and_then(Iterator::next)
            .inspect(|_| self.len -= 1)
            .map(|item| Ok(item.owned(self.py)))
            .or_else(|| {
                (self.len > 0)
                    .then(|| self.items.clone())
                    .and_then(|items| {
                        items
                            .tuple_windows()
                            .try_fold(1, |len, (a, b)| match a.cmp(&b) {
                                Ok(Ordering::Less) => CF::Break(Ok(len)),
                                Ok(_) => CF::Continue(len + 1),
                                Err(e) => CF::Break(Err(e)),
                            })
                            .pipe(|res| match res {
                                CF::Continue(len) | CF::Break(Ok(len)) => self.refresh(len),
                                CF::Break(Err(e)) => Some(Err(e)),
                            })
                    })
            })
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<I> ExactSizeIterator for Groups<'_, I>
where
    I: Iterator + Clone,
    I::Item: ItemRef,
{
}

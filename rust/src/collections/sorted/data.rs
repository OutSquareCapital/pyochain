use pyo3::prelude::*;
pub(super) struct ListsData {
    pub(super) lists: Vec<Vec<Py<PyAny>>>,
    pub(super) maxes: Vec<Py<PyAny>>,
    pub(super) idx: Vec<usize>,
}
impl ListsData {
    pub fn new() -> Self {
        Self {
            lists: Vec::new(),
            maxes: Vec::new(),
            idx: Vec::new(),
        }
    }
    #[inline]
    pub fn collapse(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.lists
            .iter()
            .flatten()
            .map(|x| x.clone_ref(py))
            .collect()
    }
    #[inline]
    pub fn clear(&mut self) {
        self.lists.clear();
        self.maxes.clear();
        self.idx.clear();
    }
}

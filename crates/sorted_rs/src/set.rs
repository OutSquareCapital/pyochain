use crate::{InnerData, InnerGetter, ListsDataMethods, SetDataMethods};
use pyo3::{prelude::*, types::PySet};
pub struct SetData<T: ListsDataMethods>(pub T, pub Py<PySet>);
impl<T: ListsDataMethods> InnerGetter for SetData<T> {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}
impl<T: ListsDataMethods> SetDataMethods<T> for SetData<T> {
    fn get_list(&self) -> &T {
        &self.0
    }
    fn get_list_mut(&mut self) -> &mut T {
        &mut self.0
    }
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.1.clone_ref(py).into_bound(py)
    }
    fn get_set_ref<'py>(&self, py: Python<'py>) -> &Bound<'py, PySet> {
        self.1.bind(py)
    }
}

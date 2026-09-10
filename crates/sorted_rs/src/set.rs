use crate::{InnerData, InnerGetter, KeysListsData, ListsData, SetDataMethods};
use pyo3::{prelude::*, types::PySet};
pub struct SetData(pub ListsData, pub Py<PySet>);
impl SetData {}
pub struct KeySetData(KeysListsData, Py<PySet>);
impl KeySetData {}
impl InnerGetter for SetData {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}
impl SetDataMethods<ListsData> for SetData {
    fn get_list(&self) -> &ListsData {
        &self.0
    }
    fn get_list_mut(&mut self) -> &mut ListsData {
        &mut self.0
    }
    fn get_set(&self) -> &Py<PySet> {
        &self.1
    }
}
impl InnerGetter for KeySetData {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}
impl SetDataMethods<KeysListsData> for KeySetData {
    fn get_list(&self) -> &KeysListsData {
        &self.0
    }
    fn get_list_mut(&mut self) -> &mut KeysListsData {
        &mut self.0
    }
    fn get_set(&self) -> &Py<PySet> {
        &self.1
    }
}

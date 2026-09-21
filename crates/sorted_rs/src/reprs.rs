use crate::{DictData, KeysListsData, ListsData, SetData};
use pyo3::{PyTypeInfo, prelude::*};

pub trait PyRepr {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String>;
}

impl PyRepr for ListsData {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        self.as_pylist(py)?
            .repr()
            .map(|repr| format!("{name}({repr})"))
    }
}
impl PyRepr for KeysListsData {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let key_repr = self.2.bind(py).repr()?;
        self.as_pylist(py)?
            .repr()
            .map(|repr| format!("{name}({repr}, key={key_repr})"))
    }
}
impl PyRepr for SetData<ListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let self_repr = self.as_pylist(py)?.repr()?;
        Ok(format!("{name}({self_repr})"))
    }
}
impl PyRepr for SetData<KeysListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let key = format!(", key={}", self.0.2.bind(py).repr()?);
        let list_repr = self.as_pylist(py)?.repr()?;
        Ok(format!("{name}({list_repr}{key})"))
    }
}
impl PyRepr for DictData<ListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let items = self.values_to_str(name.py())?;
        Ok(format!("{name}({{{items}}})"))
    }
}
impl PyRepr for DictData<KeysListsData> {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let key_arg = self.0.2.bind(py).repr()?;
        let items = self.values_to_str(py)?;
        Ok(format!("{name}({key_arg}, {{{items}}})"))
    }
}

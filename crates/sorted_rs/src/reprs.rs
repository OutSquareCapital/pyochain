use std::fmt::Display;

use crate::{DictData, KeysListsData, ListsData, SetData, traits::ListsDataMethods};
use pyo3::{PyTypeInfo, prelude::*, types::PyString};

pub trait PyRepr {
    fn repr<T: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = T::type_object(py).name()?;
        let values = self.items_repr(py)?;
        match self.key_repr(py) {
            Ok(Some(key)) => Ok(format!("{name}({key}, {values})")),
            Ok(None) => Ok(format!("{name}({values})")),
            Err(e) => Err(e),
        }
    }
    fn items_repr(&self, py: Python<'_>) -> PyResult<impl Display>;
    fn key_repr<'py>(&self, _py: Python<'py>) -> PyResult<Option<Bound<'py, PyString>>>;
}

impl PyRepr for ListsData {
    fn items_repr(&self, py: Python<'_>) -> PyResult<impl Display> {
        self.as_pylist(py)?.repr()
    }
    fn key_repr<'py>(&self, _py: Python<'py>) -> PyResult<Option<Bound<'py, PyString>>> {
        Ok(None)
    }
}
impl PyRepr for KeysListsData {
    fn items_repr(&self, py: Python<'_>) -> PyResult<impl Display> {
        self.as_pylist(py)?.repr()
    }
    fn key_repr<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyString>>> {
        self.2.bind(py).repr().map(Some)
    }
}
impl<T: ListsDataMethods> PyRepr for SetData<T> {
    fn items_repr(&self, py: Python<'_>) -> PyResult<impl Display> {
        self.0.items_repr(py)
    }
    fn key_repr<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyString>>> {
        self.0.key_repr(py)
    }
}
impl<T: ListsDataMethods> PyRepr for DictData<T> {
    fn items_repr(&self, py: Python<'_>) -> PyResult<impl Display> {
        let dict = self.1.bind(py).as_any();
        self.iter()
            .map(|x| x.bind(py))
            .map(|key| {
                dict.get_item(key)
                    .and_then(|value| Ok((key.repr()?, value.repr()?)))
                    .map(|(k, v)| format!("{k}: {v}"))
            })
            .collect::<PyResult<Vec<_>>>()
            .map(|v| v.join(", "))
            .map(|s| format!("{{{s}}}"))
    }
    fn key_repr<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyString>>> {
        self.0.key_repr(py)
    }
}

use pyo3::exceptions::PyKeyError;
use pyo3::ffi;
use pyo3::types::PyDict;
use pyo3::{prelude::*, types::PyMapping};
use pyo3_ext::prelude::*;
use pyo3_ext::types::DictItem;
use pyochain_macros::{try_cast, try_cast_into};
use tap::Pipe;

use crate::{InnerData, InnerGetter, ListDataOwner, ListsData, ListsDataMethods};
pub struct DictData<T: ListsDataMethods>(T, pub Py<PyDict>);

impl<T: ListsDataMethods> InnerGetter for DictData<T> {
    fn inner(&self) -> &InnerData {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> &mut InnerData {
        self.0.inner_mut()
    }
}

impl<T: ListsDataMethods> ListDataOwner for DictData<T> {
    type List = T;

    fn list(&self) -> &Self::List {
        &self.0
    }

    fn list_mut(&mut self) -> &mut Self::List {
        &mut self.0
    }
}
impl DictData<ListsData> {
    #[must_use]
    pub fn empty(py: Python<'_>) -> Self {
        Self::new(ListsData::default(), PyDict::new(py).unbind())
    }
    pub fn from_keys<'py>(
        iterable: &Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let py = iterable.py();
        let value = value.unwrap_or_else(|| py.None().into_bound(py));
        iterable
            .try_iter()?
            .map(|key| Ok((key?, value.clone())))
            .pipe(|v| DictData::empty(py).copy_from_iter(py, v))
    }
}
impl<T: ListsDataMethods> DictData<T> {
    pub fn new(list: T, dict: Py<PyDict>) -> Self {
        Self(list, dict)
    }
    pub fn get_dict(&self) -> &Py<PyDict> {
        &self.1
    }
    pub fn __len__(&self, py: Python<'_>) -> usize {
        self.1.bind(py).len()
    }
    pub fn or(&mut self, value: &Bound<'_, PyMapping>) -> PyResult<Self> {
        let py = value.py();
        let dict = self.1.bind(py).as_any();
        self.0
            .inner()
            .iter()
            .map(|x| x.clone_ref(py).into_bound(py))
            .map(|key| dict.get_item(&key).map(|value| (key, value)))
            .chain(value.items_view()?.iter())
            .pipe(|x| self.copy_from_iter(py, x))
    }

    pub fn ror(&mut self, value: &Bound<'_, PyMapping>) -> PyResult<Self> {
        let py = value.py();
        let dict = self.1.bind(py).as_any();
        value
            .items_view()?
            .iter()
            .chain(
                self.0
                    .inner()
                    .iter()
                    .map(|x| x.clone_ref(py).into_bound(py))
                    .map(|key| dict.get_item(&key).map(|value| (key, value))),
            )
            .pipe(|x| self.copy_from_iter(py, x))
    }
    pub fn copy(&mut self, py: Python<'_>) -> PyResult<Self> {
        let dict = self.1.bind(py).as_any();
        self.0
            .inner()
            .iter()
            .map(|x| x.clone_ref(py).into_bound(py))
            .map(|key| dict.get_item(&key).map(|value| (key, value)))
            .pipe(|v| self.copy_from_iter(py, v))
    }
    pub fn get_item<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.1.bind(key.py()).as_any().get_item(key)
    }
    pub fn set_item(&mut self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        let dict = self.1.bind(py);
        if !dict.contains(&key)? {
            self.0.list_mut().add(py, key.clone().unbind())?;
        }
        dict.set_item(key, value)
    }
    pub fn peekitem<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let key = self.0.list_mut().inner_mut().get_item(py, index)?;
        self.get_item(&key).map(|value| (key, value))
    }
    pub fn setdefault<'py>(
        &mut self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = key.py();
        let dict = self.1.bind(py);
        if dict.contains(&key)? {
            dict.as_any().get_item(&key).map(Some)
        } else {
            dict.set_item(&key, &default)?;
            self.0.list_mut().add(py, key.unbind())?;
            Ok(default)
        }
    }
    pub fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.1.bind(value.py()).contains(value)
    }
    pub fn clear(&mut self, py: Python<'_>) {
        self.1.bind(py).clear();
        self.0.list_mut().clear(py);
    }
    pub fn copy_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        &self,
        py: Python<'py>,
        v: I,
    ) -> PyResult<Self> {
        let inner = PyDict::new(py);
        let unbounded = v
            .into_iter()
            .map(|res| res.and_then(|(key, value)| fill_dict(&inner, py, key, value)))
            .map(|x| x.map(|(k, _)| k.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        let list = self.0.list().as_owned_from(py, unbounded)?;
        DictData::new(list, inner.unbind()).pipe(Ok)
    }

    pub fn del_item(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.1.bind(key.py()).as_any().del_item(key)?;
        self.0.list_mut().remove(key)
    }
    pub fn popitem<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let dict = self.1.bind(py);
        if dict.len() == 0 {
            let msg = "popitem(): dictionary is empty";
            Err(PyKeyError::new_err(msg))
        } else {
            let key = self.0.list_mut().pop(py, index)?;
            let value = dict.pop_or_err(&key).into_pyresult()?;
            Ok((key, value))
        }
    }
    pub fn pop<'py>(
        &mut self,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        let dict = self.1.bind(py);
        if dict.contains(key)? {
            self.0.list_mut().remove(key)?;
            dict.pop_or_err(key).into_pyresult()
        } else {
            default.ok_or_else(|| PyKeyError::new_err(key.to_string()))
        }
    }
    pub fn update(
        &mut self,
        py: Python<'_>,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let inner = self.1.bind(py);
        if inner.len() == 0 {
            if let Some(it) = m {
                try_cast! {
                    match it {
                        CaseExact::PyDict(d) => inner.update(d.as_mapping())?,
                        Case::PyMapping(m) => inner.update(m)?,
                        iterable => inner.update_from_sequence(&iterable)?,
                    }
                }
            }
            if let Some(kw) = kwargs {
                inner.update(kw.as_mapping())?;
            }

            inner
                .iter()
                .map(|(k, _)| k.unbind())
                .collect::<Vec<_>>()
                .pipe(|v| self.0.list_mut().extend(py, v))?;
            Ok(())
        } else {
            let pairs = try_cast_into! {match (m, kwargs) {
                (Some(CaseExact::PyDict(d)), None) => d,
                (Some(CaseExact::PyDict(d)), Some(kw)) => {
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(Case::PyMapping(m)), None) => {
                    let d = PyDict::new(py);
                    d.update(&m)?;
                    d
                }
                (Some(Case::PyMapping(m)), Some(kw)) => {
                    let d = PyDict::new(py);
                    d.update(&m)?;
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(iterable), Some(kw)) => {
                    let d = PyDict::from_sequence(&iterable)?;
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(iterable), None) => PyDict::from_sequence(&iterable)?,
                (None, Some(kw)) => kw,
                (None, None) => PyDict::new(py),
            }};
            if (10 * pairs.len()) > inner.len() {
                inner.update(pairs.as_mapping())?;
                self.0.list_mut().clear(py);
                inner
                    .iter()
                    .map(|(k, _)| k.unbind())
                    .collect::<Vec<_>>()
                    .pipe(|v| self.0.list_mut().extend(py, v))?;
                Ok(())
            } else {
                pairs.keys_view().iter_py().try_for_each(|key| {
                    let k = key?;
                    let new = pairs.as_any().get_item(&k)?;
                    self.set_item(k, new)
                })
            }
        }
    }
}

fn fill_dict<'py>(
    inner: &Bound<'py, PyDict>,
    py: Python<'py>,
    key: Bound<'py, PyAny>,
    value: Bound<'py, PyAny>,
) -> PyResult<DictItem<'py>> {
    match unsafe { ffi::PyDict_SetItem(inner.as_ptr(), key.as_ptr(), value.as_ptr()) } {
        -1 => Err(PyErr::fetch(py)),
        _ => Ok((key, value)),
    }
}

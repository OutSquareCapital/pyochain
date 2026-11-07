use std::ops::Deref;

use crate::{
    prelude::*,
    slice::SliceIdxs,
    types::{Dict, ListOrAny},
};
use derive_more::{Deref, From};
use either::Either;
use itertools::Itertools;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyList, PySlice},
};
use pyo3_ext::{prelude::CollectBoundIterator, tuple, types::PyFrozenClass};
use pyochain_macros::try_cast;

#[derive(From, Deref)]
pub struct OnKeys<M>(Py<M>);
#[derive(From, Deref)]
pub struct OnValues<M>(Py<M>);
#[derive(From, Deref)]
pub struct OnItems<M>(Py<M>);
pub trait ViewMethods: From<Py<Self::Mapping>> + Deref<Target = Py<Self::Mapping>>
where
    Dict<Self::List>: PyRepr,
{
    type List: ListsMethods;
    type Mapping: PyFrozenClass + Deref<Target = RwLock<Dict<Self::List>>>;
    fn get_item<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<ListOrAny<'py>>;
    fn mapping(&self) -> RwLockReadGuard<'_, Dict<Self::List>> {
        self.get().deref().read()
    }
    fn mapping_mut(&self) -> RwLockWriteGuard<'_, Dict<Self::List>> {
        self.get().deref().write()
    }
    fn len(&self, py: Python<'_>) -> usize {
        self.mapping().__len__(py)
    }
    fn repr_from<Wrapper: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = Wrapper::type_object(py).name()?;
        let values = self.mapping().repr::<Self::Mapping>(py)?;
        Ok(format!("{name}({values})"))
    }
    fn del_item(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        let py = index.py();
        let mut mapping = self.mapping_mut();
        let dict = mapping.1.clone_ref(py).into_bound(py);
        try_cast! {
            match index {
                Case::PySlice(pyslice) => {
                    let slice = SliceIdxs::new(pyslice, mapping.len())?;
                    let keys = slice.extract_from(py,&mut mapping)?;
                    slice.delete_from(py,&mut mapping.0)?;
                    for key in keys {
                        dict.del_item(key)?;
                    }
                    Ok(())
                }
                int => {
                    let key = mapping.0.pop(py, int.extract::<isize>()?)?;
                    dict.del_item(key)?;
                    Ok(())
                }
            }
        }
    }
}
impl<M> ViewMethods for OnKeys<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<ListOrAny<'py>> {
        let py = index.py();
        let mut mapping = self.mapping_mut();
        try_cast! {
            match index {
                Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len())?
                    .extract_from(py, &mut mapping)?
                    .iter()
                    .collect_bound::<PyList>(py)
                    .map(Either::Left),
                int => mapping.extract_index(&int).map(Either::Right),
            }
        }
    }
}

impl<M> ViewMethods for OnValues<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<ListOrAny<'py>> {
        let py = index.py();
        let mut mapping = self.mapping_mut();
        let dict = mapping.1.clone_ref(py).into_bound(py).into_any();
        try_cast! {
            match index {
                Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len())?
                    .extract_from(py, &mut mapping)?
                    .iter()
                    .map(|key| dict.get_item(key))
                    .try_collect_bound::<PyList>(py)
                    .map(Either::Left),
                int => dict
                    .get_item(mapping.extract_index(&int)?)
                    .map(Either::Right),
            }
        }
    }
}

impl<M> ViewMethods for OnItems<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<ListOrAny<'py>> {
        let py = index.py();
        let mut mapping = self.mapping_mut();
        let dict = mapping.1.clone_ref(py).into_bound(py).into_any();
        try_cast! {
            match index {
                Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len())?
                    .extract_from(py, &mut mapping)?
                    .iter()
                    .map(|key| tuple!(key.bind(py), &dict.get_item(key)?))
                    .map_ok(Bound::into_any)
                    .try_collect_bound::<PyList>(py)
                    .map(Either::Left),
                int => {
                    let key = mapping.extract_index(&int)?;
                    let value = dict.get_item(&key)?;
                    tuple!(key, value).map(Bound::into_any).map(Either::Right)
                }
            }
        }
    }
}

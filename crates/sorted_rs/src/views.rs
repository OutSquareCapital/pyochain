use std::ops::Deref;

use crate::{
    get_item::Index,
    prelude::*,
    types::{Dict, IndexOut, IntoIndex},
};
use derive_more::{Deref, From};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::{PyTypeInfo, prelude::*};
use pyo3_ext::types::PyFrozenClass;

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
    fn get_item<'py>(&self, py: Python<'py>, index: IntoIndex<'py>) -> PyResult<IndexOut<'py>>;
    fn mapping(&self) -> RwLockReadGuard<'_, Dict<Self::List>> {
        self.get().deref().read()
    }
    fn mapping_mut(&self) -> RwLockWriteGuard<'_, Dict<Self::List>> {
        self.get().deref().write()
    }
    fn length(&self) -> usize {
        self.mapping().list.len()
    }
    fn repr_from<Wrapper: PyTypeInfo>(&self, py: Python<'_>) -> PyResult<String> {
        let name = Wrapper::type_object(py).name()?;
        let values = self.mapping().repr::<Self::Mapping>(py)?;
        Ok(format!("{name}({values})"))
    }
    fn del_item<'py>(&self, py: Python<'py>, index: IntoIndex<'py>) -> PyResult<()> {
        Index::new(py, index).del_from_view(self)
    }
}
impl<M> ViewMethods for OnKeys<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, py: Python<'py>, index: IntoIndex<'py>) -> PyResult<IndexOut<'py>> {
        self.mapping_mut().list.get_item_or_slice(py, index)
    }
}

impl<M> ViewMethods for OnValues<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, py: Python<'py>, index: IntoIndex<'py>) -> PyResult<IndexOut<'py>> {
        Index::new(py, index).get_from_values_view(self)
    }
}

impl<M> ViewMethods for OnItems<M>
where
    M: ListGetter + Deref<Target = RwLock<Dict<M::L>>>,
{
    type List = M::L;
    type Mapping = M;
    fn get_item<'py>(&self, py: Python<'py>, index: IntoIndex<'py>) -> PyResult<IndexOut<'py>> {
        Index::new(py, index).get_from_items_view(self)
    }
}

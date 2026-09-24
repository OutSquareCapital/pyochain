use sorted_rs::{DictData, KeysListsData, ListsData, SetData};
use std::sync::{Arc, RwLock};

use crate::collections::sorted;
macro_rules! impl_from_data {
    ($($from:ty => $into:ty),+ $(,)?) => {
        $(
            impl From<$from> for $into {
                fn from(inner: $from) -> Self {
                    Self(Arc::new(RwLock::new(inner)))
                }
            }
        )+
    };
}
impl_from_data! {
    DictData<ListsData> => sorted::SortedDict,
    DictData<KeysListsData> => sorted::SortedKeyDict,
    KeysListsData => sorted::SortedKeyList,
    ListsData => sorted::SortedList,
    SetData<ListsData> => sorted::SortedSet,
    SetData<KeysListsData> => sorted::SortedKeySet
}

use crate::{DictData, KeysListsData, SetData, inner::InnerData, prelude::*};
use pyo3::prelude::*;
macro_rules! pyassert {
    ($cond:expr) => {
        if !$cond {
            return Err(pyo3::exceptions::PyAssertionError::new_err(""));
        }
    };
}
pub fn check_empty(slf: &InnerData) -> PyResult<()> {
    pyassert!(slf.len == 0);
    pyassert!(slf.maxes.is_empty());
    pyassert!(slf.values.is_empty());
    Ok(())
}
pub fn check_dict<T: ListsDataMethods>(py: Python<'_>, data: &DictData<T>) -> PyResult<()> {
    check_list(py, data.list().inner())?;
    let dict = data.get_dict().bind(py);
    pyassert!(dict.len() == data.len());
    pyassert!(data.list().inner().iter().all(|item| {
        dict.contains(item.bind(py))
            .expect("Failed to check dict membership")
    }));
    Ok(())
}

pub fn check_set_len<T: ListsDataMethods>(py: Python<'_>, checked: &SetData<T>) -> PyResult<()> {
    let set = checked.get_set(py);
    pyassert!(set.len() == checked.len());
    check_list(py, checked.list().inner())?;
    pyassert!(
        checked
            .list()
            .inner()
            .iter()
            .all(|x| set.contains(x).expect("Failed to check set membership"))
    );
    Ok(())
}
pub fn check_list(py: Python<'_>, slf: &InnerData) -> PyResult<()> {
    pyassert!(slf.load >= 4);
    pyassert!(slf.maxes.len() == slf.values.len());
    pyassert!(slf.len == slf.values.iter().map(Vec::len).sum::<usize>());

    // Check all sublists are sorted.

    for sublist in &slf.values {
        for pos in 1..sublist.len() {
            pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?);
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..slf.values.len() {
        pyassert!(
            slf.values[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(slf.values[pos][0].bind(py))?
        );
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..slf.maxes.len() {
        pyassert!(
            slf.maxes[pos]
                .bind(py)
                .eq(slf.values[pos].last().unwrap().bind(py))?
        );
    }

    // Check sublist lengths are less than double load-factor.

    let double = slf.load << 1;
    pyassert!(slf.values.iter().all(|sublist| sublist.len() <= double));

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = slf.load >> 1;
    for pos in 0..slf.values.len().saturating_sub(1) {
        pyassert!(slf.values[pos].len() >= half);
    }

    if !slf.idx.is_empty() {
        pyassert!(slf.len == slf.idx[0]);
        pyassert!(slf.idx.len() == slf.offset + slf.values.len());

        // Check index leaf nodes equal length of sublists.

        for pos in 0..slf.values.len() {
            let leaf = slf.idx[slf.offset + pos];
            pyassert!(leaf.eq(&slf.values[pos].len()));
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..slf.offset {
            let child = (pos << 1) + 1;
            if child >= slf.idx.len() {
                pyassert!(slf.idx[pos].eq(&0));
            } else if child + 1 == slf.idx.len() {
                pyassert!(slf.idx[pos].eq(&slf.idx[child]));
            } else {
                let child_sum = slf.idx[child] + slf.idx[child + 1];
                pyassert!(child_sum.eq(&slf.idx[pos]));
            }
        }
    }

    Ok(()).inspect_err(|e| show_list(py, e, slf))
}

pub fn check_key_list(py: Python<'_>, data: &KeysListsData) -> PyResult<()> {
    let key_fn = data.2.bind(py);
    pyassert!(data.load() >= 4);
    pyassert!(data.maxes().len() == data.values().len() && data.values().len() == data.1.len());
    pyassert!(data.len() == data.values().iter().map(Vec::len).sum::<usize>());

    // Check all sublists are sorted.

    for sublist in &data.1 {
        for pos in 1..sublist.len() {
            pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?);
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..data.1.len() {
        pyassert!(
            data.1[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(data.1[pos][0].bind(py))?
        );
    }

    // Check _keys matches _key mapped to _lists.

    for (val_sublist, key_sublist) in data.values().iter().zip(data.1.iter()) {
        pyassert!(val_sublist.len() == key_sublist.len());
        for (val, key) in val_sublist.iter().zip(key_sublist.iter()) {
            {
                pyassert!(key_fn.call1((&val,))?.eq(key)?);
            }
        }
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..data.maxes().len() {
        pyassert!(
            data.maxes()[pos]
                .bind(py)
                .eq(data.1[pos].last().unwrap().bind(py))?
        );
    }

    // Check sublist lengths are less than double load-factor.

    let double = data.load() << 1;
    pyassert!(data.values().iter().all(|sublist| sublist.len() <= double));

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = data.load() >> 1;
    for pos in 0..data.values().len().saturating_sub(1) {
        pyassert!(data.values()[pos].len() >= half);
    }

    if !data.idx().is_empty() {
        pyassert!(data.len() == data.idx()[0]);
        pyassert!(data.idx().len() == data.offset() + data.values().len());

        // Check index leaf nodes equal length of sublists.

        for pos in 0..data.values().len() {
            let leaf = data.idx()[data.offset() + pos];
            pyassert!(leaf == data.values()[pos].len());
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..data.offset() {
            let child = (pos << 1) + 1;
            if child >= data.idx().len() {
                pyassert!(data.idx()[pos] == 0);
            } else if child + 1 == data.idx().len() {
                pyassert!(data.idx()[pos] == data.idx()[child]);
            } else {
                let child_sum = data.idx()[child] + data.idx()[child + 1];
                pyassert!(child_sum == data.idx()[pos]);
            }
        }
    }
    Ok(()).inspect_err(|e| show_key_list(py, e, data))
}

fn show_list(py: Python<'_>, err: &PyErr, data: &InnerData) {
    let infos = [
        format!("len: {}", data.len),
        format!("load: {}", data.load),
        format!("offset: {}", data.offset),
        format!("len_index: {}", data.idx.len()),
        format!("index: {:?}", data.idx),
        format!("len_maxes: {}", data.maxes.len()),
        format!("maxes: {:?}", data.maxes),
        format!("len_lists: {}", data.values.len()),
        format!("lists: {:?}", data.values),
    ]
    .join("\n");

    err.add_note(py, infos).unwrap();
}

fn show_key_list(py: Python<'_>, err: &PyErr, data: &KeysListsData) {
    show_list(py, err, data.inner());
    let infos = [
        format!("len_keys: {}", data.1.len()),
        format!("keys: {:?}", data.1),
    ];
    err.add_note(py, infos.join("\n")).unwrap();
}

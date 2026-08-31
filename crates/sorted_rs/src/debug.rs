use crate::{KeysListsData, ListDataGetters};
use pyo3::prelude::*;
#[macro_export]
macro_rules! pyassert {
    ($cond:expr) => {
        if !$cond {
            return Err(pyo3::exceptions::PyAssertionError::new_err(""));
        }
    };
}

pub(super) fn check_list(slf: &impl ListDataGetters, py: Python<'_>) -> PyResult<()> {
    pyassert!(slf.load() >= 4);
    pyassert!(slf.maxes().len() == slf.lists().len());
    pyassert!(slf.length() == slf.lists().iter().map(Vec::len).sum::<usize>());

    // Check all sublists are sorted.

    for sublist in slf.lists() {
        for pos in 1..sublist.len() {
            pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?);
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..slf.lists().len() {
        pyassert!(
            slf.lists()[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(slf.lists()[pos][0].bind(py))?
        );
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..slf.maxes().len() {
        pyassert!(
            slf.maxes()[pos]
                .bind(py)
                .eq(slf.lists()[pos].last().unwrap().bind(py))?
        );
    }

    // Check sublist lengths are less than double load-factor.

    let double = slf.load() << 1;
    pyassert!(slf.lists().iter().all(|sublist| sublist.len() <= double));

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = slf.load() >> 1;
    for pos in 0..slf.lists().len().saturating_sub(1) {
        pyassert!(slf.lists()[pos].len() >= half);
    }

    if !slf.idx().is_empty() {
        pyassert!(slf.length() == slf.idx()[0]);
        pyassert!(slf.idx().len() == slf.offset() + slf.lists().len());

        // Check index leaf nodes equal length of sublists.

        for pos in 0..slf.lists().len() {
            let leaf = slf.idx()[slf.offset() + pos];
            pyassert!(leaf.eq(&slf.lists()[pos].len()));
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..slf.offset() {
            let child = (pos << 1) + 1;
            if child >= slf.idx().len() {
                pyassert!(slf.idx()[pos].eq(&0));
            } else if child + 1 == slf.idx().len() {
                pyassert!(slf.idx()[pos].eq(&slf.idx()[child]));
            } else {
                let child_sum = slf.idx()[child] + slf.idx()[child + 1];
                pyassert!(child_sum.eq(&slf.idx()[pos]));
            }
        }
    }

    Ok(()).inspect_err(|e| show_list(py, e, slf))
}

pub fn check_key_list(py: Python<'_>, data: &KeysListsData) -> PyResult<()> {
    let key_fn = data.key.bind(py);
    pyassert!(data.load >= 4);
    pyassert!(data.maxes.len() == data.lists.len() && data.lists.len() == data.keys.len());
    pyassert!(data.len == data.lists.iter().map(Vec::len).sum::<usize>());

    // Check all sublists are sorted.

    for sublist in &data.keys {
        for pos in 1..sublist.len() {
            pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?);
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..data.keys.len() {
        pyassert!(
            data.keys[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(data.keys[pos][0].bind(py))?
        );
    }

    // Check _keys matches _key mapped to _lists.

    for (val_sublist, key_sublist) in data.lists.iter().zip(data.keys.iter()) {
        pyassert!(val_sublist.len() == key_sublist.len());
        for (val, key) in val_sublist.iter().zip(key_sublist.iter()) {
            {
                pyassert!(key_fn.call1((&val,))?.eq(key)?);
            }
        }
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..data.maxes.len() {
        pyassert!(
            data.maxes[pos]
                .bind(py)
                .eq(data.keys[pos].last().unwrap().bind(py))?
        );
    }

    // Check sublist lengths are less than double load-factor.

    let double = data.load << 1;
    pyassert!(data.lists.iter().all(|sublist| sublist.len() <= double));

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = data.load >> 1;
    for pos in 0..data.lists.len().saturating_sub(1) {
        pyassert!(data.lists[pos].len() >= half);
    }

    if !data.idx.is_empty() {
        pyassert!(data.len == data.idx[0]);
        pyassert!(data.idx.len() == data.offset + data.lists.len());

        // Check index leaf nodes equal length of sublists.

        for pos in 0..data.lists.len() {
            let leaf = data.idx[data.offset + pos];
            pyassert!(leaf == data.lists[pos].len());
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..data.offset {
            let child = (pos << 1) + 1;
            if child >= data.idx.len() {
                pyassert!(data.idx[pos] == 0);
            } else if child + 1 == data.idx.len() {
                pyassert!(data.idx[pos] == data.idx[child]);
            } else {
                let child_sum = data.idx[child] + data.idx[child + 1];
                pyassert!(child_sum == data.idx[pos]);
            }
        }
    }
    Ok(()).inspect_err(|e| show_key_list(py, e, data))
}

fn show_list<T: ListDataGetters>(py: Python<'_>, err: &PyErr, data: &T) {
    let infos = [
        format!("len: {}", data.length()),
        format!("load: {}", data.load()),
        format!("offset: {}", data.offset()),
        format!("len_index: {}", data.idx().len()),
        format!("index: {:?}", data.idx()),
        format!("len_maxes: {}", data.maxes().len()),
        format!("maxes: {:?}", data.maxes()),
        format!("len_lists: {}", data.lists().len()),
        format!("lists: {:?}", data.lists()),
    ]
    .join("\n");

    err.add_note(py, infos).unwrap();
}

fn show_key_list(py: Python<'_>, err: &PyErr, data: &KeysListsData) {
    show_list(py, err, data);
    let infos = [
        format!("len_keys: {}", data.keys.len()),
        format!("keys: {:?}", data.keys),
    ];
    err.add_note(py, infos.join("\n")).unwrap();
}

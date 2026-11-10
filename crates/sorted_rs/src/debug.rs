//! Invariants checks ported from the original `sortedcontainers` test suite.
use crate::{
    KeyList,
    inner::InnerList,
    prelude::*,
    types::{Dict, Set},
};
use pyo3::prelude::*;
use std_tools::prelude::*;
macro_rules! pyassert {
    ($cond:expr) => {
        if !$cond {
            let msg = format!("Assertion failed: {}", stringify!($cond));
            Err(pyo3::exceptions::PyAssertionError::new_err(msg))
        } else {
            Ok(())
        }
    };
}
pub fn check_empty(slf: &InnerList) -> PyResult<()> {
    pyassert!(slf.is_empty())?;
    pyassert!(slf.maxes.is_empty())?;
    pyassert!(slf.values.is_empty())
}
pub fn check_dict<T: ListsMethods>(py: Python<'_>, slf: &Dict<T>) -> PyResult<()> {
    check_list(py, &slf.0)?;
    let dict = slf.1.bind(py);
    pyassert!(dict.len() == slf.len())?;
    pyassert!(
        slf.values
            .iter()
            .try_all(|item| { dict.contains(item.bind(py)) })?
    )
}

pub fn check_set<T: ListsMethods>(py: Python<'_>, slf: &Set<T>) -> PyResult<()> {
    let set = slf.get_set(py);
    pyassert!(set.len() == slf.len())?;
    check_list(py, &slf.0)?;
    pyassert!(slf.values.iter().try_all(|x| set.contains(x))?)
}
pub fn check_list<T: ListsMethods>(py: Python<'_>, slf: &T) -> PyResult<()> {
    check_sublists(py, slf).inspect_err(|e| e.add_note(py, list_infos(slf)).unwrap())
}

pub fn check_key_list(py: Python<'_>, slf: &KeyList) -> PyResult<()> {
    check_sublists(py, slf)
        .and_then(|()| check_values_match_keys(py, slf))
        .inspect_err(|e| {
            let infos = [
                list_infos(slf),
                format!("len_keys: {}", slf.keys().len()),
                format!("keys: {:?}", slf.keys()),
            ]
            .join("\n");
            e.add_note(py, infos).unwrap();
        })
}

/// Check keys matches key mapped to lists.
fn check_values_match_keys(py: Python<'_>, data: &KeyList) -> PyResult<()> {
    let key_fn = data.func.bind(py);
    data.values
        .iter_chunks()
        .zip(data.keys().iter_chunks())
        .try_for_each(|(val_sublist, key_sublist)| {
            pyassert!(val_sublist.len() == key_sublist.len())?;
            val_sublist
                .iter()
                .zip(key_sublist.iter())
                .try_for_each(|(val, key)| pyassert!(key_fn.call1((&val,))?.eq(key)?))
        })
}

fn check_sublists<T: ListsMethods>(py: Python<'_>, slf: &T) -> PyResult<()> {
    pyassert!(slf.load.test_limit())?;
    pyassert!(slf.maxes.len() == slf.values.len() && slf.values.len() == slf.keys().len())?;
    pyassert!(slf.len() == slf.values.iter_chunks().map(Vec::len).sum::<usize>())?;
    // Check all sublists are sorted.
    slf.keys().iter_chunks().try_for_each(|sublist| {
        (1..sublist.len())
            .try_for_each(|pos| pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?))
    })?;
    // Check beginning/end of sublists are sorted.
    (1..slf.keys().len()).try_for_each(|pos| {
        pyassert!(
            slf.keys()[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(slf.keys()[pos][0].bind(py))?
        )
    })?;
    // Check _maxes index is the last value of each sublist.
    pyassert!(
        slf.maxes
            .iter()
            .zip(slf.keys().iter_chunks())
            .try_all(|(max, sublist)| max.bind(py).eq(sublist.last().unwrap().bind(py)))?
    )?;
    // Check sublist lengths are less than double load-factor.
    pyassert!(
        slf.values
            .iter_chunks()
            .all(|sublist| sublist.len() <= slf.load.higher_limit())
    )?;
    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.
    pyassert!(
        slf.values
            .iter_chunks()
            .rev()
            .skip(1)
            .all(|sublist| sublist.len() >= slf.load.lower_limit())
    )?;
    if slf.lengths.is_built() {
        pyassert!(slf.len() == slf.lengths.inner[0])?;
        pyassert!(slf.lengths.inner.len() == slf.lengths.offset + slf.values.len())?;
        // Check index leaf nodes equal length of sublists.
        slf.values
            .iter_chunks()
            .enumerate()
            .try_for_each(|(val_idx, val)| {
                let leaf = slf.lengths.inner[slf.lengths.offset + val_idx];
                pyassert!(leaf == val.len())
            })?;
        // Check index branch nodes are the sum of their children.
        for pos in 0..slf.lengths.offset {
            let child = (pos << 1) + 1;
            if child >= slf.lengths.inner.len() {
                pyassert!(slf.lengths.inner[pos] == 0)?;
            } else if child + 1 == slf.lengths.inner.len() {
                pyassert!(slf.lengths.inner[pos] == slf.lengths.inner[child])?;
            } else {
                let child_sum = slf.lengths.inner[child] + slf.lengths.inner[child + 1];
                pyassert!(child_sum == slf.lengths.inner[pos])?;
            }
        }
    }
    Ok(())
}
fn list_infos(slf: &InnerList) -> String {
    [
        format!("len: {}", slf.len()),
        format!("load: {:?}", slf.load),
        format!("offset: {}", slf.lengths.offset),
        format!("len_index: {}", slf.lengths.inner.len()),
        format!("index: {:?}", slf.lengths.inner),
        format!("len_maxes: {}", slf.maxes.len()),
        format!("maxes: {:?}", slf.maxes),
        format!("len_lists: {}", slf.values.len()),
        format!("lists: {:?}", slf.values),
    ]
    .join("\n")
}

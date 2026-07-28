use crate::collections::sorted::traits::InnerSortedGetters;
use crate::collections::{InnerKeyLists, InnerLists};
use crate::seq::PyoVec;
use either::Either;
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
#[pyfunction]
pub fn check_sorted_list(
    py: Python<'_>,
    data: Either<Py<InnerLists>, Py<InnerKeyLists>>,
) -> PyResult<()> {
    data.map_either(
        |x| run_checks(py, x.get()).inspect_err(move |e| show_list(py, &e, x.get())),
        |x| run_checks(py, x.get()).inspect_err(move |e| show_list(py, &e, x.get())),
    )
    .into_inner()
}

fn run_checks(py: Python<'_>, data: &impl InnerSortedGetters) -> PyResult<()> {
    let lists = data.get_lists(py).get().inner.clone_ref(py).into_bound(py);
    let maxes = data.get_maxes(py).get().inner.clone_ref(py).into_bound(py);
    let idx = data.get_idx(py).into_bound(py);
    let offset = data.get_offset();
    let err = |x| PyAssertionError::new_err(x);

    (data.get_load() >= 4)
        .then_some(())
        .ok_or(err("Load factor must be at least 4"))?;
    (maxes.len() == lists.len())
        .then_some(())
        .ok_or(err("Maxes and lists must have the same length"))?;
    (data.get_len()
        == lists
            .iter()
            .map(|sublist| sublist.len().unwrap())
            .sum::<usize>())
    .then_some(())
    .ok_or(err("Data length mismatch"))?;

    // Check all sublists are sorted.

    for sublist in lists.iter().map(|x| {
        unsafe { x.cast_into_unchecked::<PyoVec>() }
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py)
    }) {
        for pos in 1..sublist.len() {
            (sublist.get_item(pos - 1)?.le(sublist.get_item(pos)?)?)
                .then_some(())
                .ok_or(err("Sublists must be sorted"))?;
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..lists.len() {
        (lists
            .get_item(pos - 1)?
            .get_item(-1)?
            .le(lists.get_item(pos)?.get_item(0)?)?)
        .then_some(())
        .ok_or(err("Sublists must be sorted at boundaries"))?;
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..maxes.len() {
        (maxes
            .get_item(pos)?
            .eq(lists.get_item(pos)?.get_item(-1)?)?)
        .then_some(())
        .ok_or(err("Maxes must match last element of sublists"))?;
    }

    // Check sublist lengths are less than double load-factor.

    let double = data.get_load() << 1;
    (lists.iter().all(|sublist| sublist.len().unwrap() <= double))
        .then_some(())
        .ok_or(err("Sublists must not exceed double load factor"))?;

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = data.get_load() >> 1;
    for pos in 0..lists.len().saturating_sub(1) {
        (lists.get_item(pos)?.len()? >= half)
            .then_some(())
            .ok_or(err("Sublists must be at least half load factor"))?;
    }

    if !idx.is_empty() {
        (data.get_len() == idx.get_item(0)?.extract::<usize>()?)
            .then_some(())
            .ok_or(err("Index root must equal total length"))?;
        (idx.len() == offset + lists.len())
            .then_some(())
            .ok_or(err("Index length mismatch"))?;

        // Check index leaf nodes equal length of sublists.

        for pos in 0..lists.len() {
            let leaf = idx.get_item(offset + pos)?;
            (leaf.eq(lists.get_item(pos)?.len()?)?)
                .then_some(())
                .ok_or(err("Index leaf node length mismatch"))?;
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..offset {
            let child = (pos << 1) + 1;
            if child >= idx.len() {
                (idx.get_item(pos)?.eq(0)?)
                    .then_some(())
                    .ok_or(err("Index branch node length mismatch"))?;
            } else if child + 1 == idx.len() {
                (idx.get_item(pos)?.eq(idx.get_item(child)?)?)
                    .then_some(())
                    .ok_or(err("Index branch node length mismatch"))?;
            } else {
                let child_sum = idx.get_item(child)?.add(idx.get_item(child + 1)?)?;
                (child_sum.eq(idx.get_item(pos)?)?)
                    .then_some(())
                    .ok_or(err("Index branch node length mismatch"))?;
            }
        }
    }

    Ok(())
}
fn show_list(py: Python<'_>, err: &PyErr, data: &impl InnerSortedGetters) -> () {
    let infos = [
        format!("len: {}", data.get_len()),
        format!("load: {}", data.get_load()),
        format!("offset: {}", data.get_offset()),
        format!("len_index: {}", data.get_idx(py).bind(py).len()),
        format!("index: {}", data.get_idx(py).bind(py).repr().unwrap()),
        format!("len_maxes: {}", data.get_maxes(py).bind(py).len().unwrap()),
        format!("maxes: {}", data.get_maxes(py).bind(py).repr().unwrap()),
        format!("len_lists: {}", data.get_lists(py).bind(py).len().unwrap()),
        format!("lists: {}", data.get_lists(py).bind(py).repr().unwrap()),
    ]
    .join("\n");

    err.add_note(py, infos).unwrap()
}

use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::{
        keyset::SortedKeySet,
        set::SortedSet,
        traits::{BaseSortedDict, BaseSortedSet, SortedListGetters},
    },
};
use either::Either;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use std::ops::Index;
use tap::prelude::*;
type InnerSorted = Either<Py<SortedList>, Py<SortedKeyList>>;

macro_rules! pyassert {
    ($cond:expr) => {
        if !$cond {
            return Err(PyAssertionError::new_err(""));
        }
    };
}

#[pyfunction]
pub fn check_sorted_dict(
    py: Python<'_>,
    data: Either<Py<SortedDict>, Py<SortedKeyDict>>,
) -> PyResult<()> {
    fn check_dict(x: &impl BaseSortedDict, py: Python<'_>) -> PyResult<()> {
        x.get_list()
            .get()
            .pipe(|list| run_checks(py, list).inspect_err(move |e| show_list(py, e, list)))?;
        let data = x.get_list().get().get_data();
        pyassert!(x.len(py) == data.len);
        pyassert!(data.iter().all(|item| {
            x.contains(item.bind(py))
                .expect("Failed to check dict membership")
        }));
        Ok(())
    }

    data.map_either(|x| check_dict(x.get(), py), |x| check_dict(x.get(), py))
        .into_inner()
}

#[pyfunction]
pub fn check_sorted_set(
    py: Python<'_>,
    data: Either<Py<SortedSet>, Py<SortedKeySet>>,
) -> PyResult<()> {
    fn check_list(x: &impl SortedListGetters, py: Python<'_>) -> PyResult<()> {
        run_checks(py, x).inspect_err(move |e| show_list(py, e, x))
    }
    fn check_len(x: &impl BaseSortedSet, py: Python<'_>) -> PyResult<()> {
        check_list(x.get_list().get(), py)?;
        let set = x.get_set().bind(py);
        let data = x.get_list().get().get_data();
        pyassert!(set.len() == data.len);
        pyassert!(
            data.iter()
                .all(|x| set.contains(x).expect("Failed to check set membership"))
        );
        Ok(())
    }
    data.map_either(|x| check_len(x.get(), py), |x| check_len(x.get(), py))
        .into_inner()?;
    Ok(())
}

#[pyfunction]
pub fn assert_sorted_list_empty(lst: InnerSorted) -> PyResult<()> {
    fn check_empty(x: &impl SortedListGetters) -> PyResult<()> {
        let data = x.get_data();
        pyassert!(data.len == 0);
        pyassert!(data.maxes.is_empty());
        pyassert!(data.lists.is_empty());
        Ok(())
    }
    lst.map_either(|x| check_empty(x.get()), |x| check_empty(x.get()))
        .into_inner()
}
#[pyfunction]
pub fn check_sorted_list(
    py: Python<'_>,
    data: Either<Py<SortedList>, Py<SortedKeyList>>,
) -> PyResult<()> {
    fn check_list(x: &impl SortedListGetters, py: Python<'_>) -> PyResult<()> {
        run_checks(py, x).inspect_err(move |e| show_list(py, e, x))
    }
    data.map_either(|x| check_list(x.get(), py), |x| check_list(x.get(), py))
        .into_inner()
}
#[pyfunction]
pub fn check_sorted_key_list(py: Python<'_>, data: Py<SortedKeyList>) -> PyResult<()> {
    run_key_checks(py, data.get()).inspect_err(move |e| show_key_list(py, e, data.get()))
}

fn run_checks(py: Python<'_>, data: &impl SortedListGetters) -> PyResult<()> {
    let lst_data = data.get_data();
    let err = |x| PyAssertionError::new_err(x);

    (data.get_load() >= 4)
        .then_some(())
        .ok_or(err("Load factor must be at least 4"))?;
    (lst_data.maxes.len() == lst_data.lists.len())
        .then_some(())
        .ok_or(err("Maxes and lists must have the same length"))?;
    (lst_data.len
        == lst_data
            .lists
            .iter()
            .map(|sublist| sublist.len())
            .sum::<usize>())
    .then_some(())
    .ok_or(err("Data length mismatch"))?;

    // Check all sublists are sorted.

    for sublist in lst_data.lists.iter() {
        for pos in 1..sublist.len() {
            (sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?)
                .then_some(())
                .ok_or(err("Sublists must be sorted"))?;
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..lst_data.lists.len() {
        (lst_data.lists[pos - 1]
            .last()
            .unwrap()
            .bind(py)
            .le(lst_data.lists[pos][0].bind(py))?)
        .then_some(())
        .ok_or(err("Sublists must be sorted at boundaries"))?;
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..lst_data.maxes.len() {
        (lst_data.maxes[pos]
            .bind(py)
            .eq(lst_data.lists[pos].last().unwrap().bind(py))?)
        .then_some(())
        .ok_or(err("Maxes must match last element of sublists"))?;
    }

    // Check sublist lengths are less than double load-factor.

    let double = data.get_load() << 1;
    (lst_data.lists.iter().all(|sublist| sublist.len() <= double))
        .then_some(())
        .ok_or(err("Sublists must not exceed double load factor"))?;

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = data.get_load() >> 1;
    for pos in 0..lst_data.lists.len().saturating_sub(1) {
        (lst_data.lists[pos].len() >= half)
            .then_some(())
            .ok_or(err("Sublists must be at least half load factor"))?;
    }

    if !lst_data.idx.is_empty() {
        (&lst_data.len == lst_data.idx.index(0))
            .then_some(())
            .ok_or(err("Index root must equal total length"))?;
        (lst_data.idx.len() == lst_data.offset + lst_data.lists.len())
            .then_some(())
            .ok_or(err("Index length mismatch"))?;

        // Check index leaf nodes equal length of sublists.

        for pos in 0..lst_data.lists.len() {
            let leaf = lst_data.idx.index(lst_data.offset + pos);
            (leaf.eq(&lst_data.lists[pos].len()))
                .then_some(())
                .ok_or(err("Index leaf node length mismatch"))?;
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..lst_data.offset {
            let child = (pos << 1) + 1;
            if child >= lst_data.idx.len() {
                (lst_data.idx.index(pos).eq(&0))
                    .then_some(())
                    .ok_or(err("Index branch node length mismatch"))?;
            } else if child + 1 == lst_data.idx.len() {
                (lst_data.idx.index(pos).eq(lst_data.idx.index(child)))
                    .then_some(())
                    .ok_or(err("Index branch node length mismatch"))?;
            } else {
                let child_sum = lst_data.idx.index(child) + lst_data.idx.index(child + 1);
                pyassert!(child_sum.eq(lst_data.idx.index(pos)));
            }
        }
    }

    Ok(())
}

fn run_key_checks(py: Python<'_>, data: &SortedKeyList) -> PyResult<()> {
    let lst_data = data.get_data();
    let keys = data.get_keys();
    let load = data.get_load();
    let key_fn = data.key.bind(py);
    pyassert!(load >= 4);
    pyassert!(lst_data.maxes.len() == lst_data.lists.len() && lst_data.lists.len() == keys.len());
    pyassert!(
        lst_data.len
            == lst_data
                .lists
                .iter()
                .map(|sublist| sublist.len())
                .sum::<usize>()
    );

    // Check all sublists are sorted.

    for sublist in keys.iter() {
        for pos in 1..sublist.len() {
            pyassert!(sublist[pos - 1].bind(py).le(sublist[pos].bind(py))?);
        }
    }

    // Check beginning/end of sublists are sorted.

    for pos in 1..keys.len() {
        pyassert!(
            keys[pos - 1]
                .last()
                .unwrap()
                .bind(py)
                .le(keys[pos][0].bind(py))?
        );
    }

    // Check _keys matches _key mapped to _lists.

    for (val_sublist, key_sublist) in lst_data.lists.iter().zip(keys.iter()) {
        pyassert!(val_sublist.len() == key_sublist.len());
        for (val, key) in val_sublist.iter().zip(key_sublist.iter()) {
            {
                pyassert!(key_fn.call1((&val,))?.eq(key)?);
            }
        }
    }

    // Check _maxes index is the last value of each sublist.

    for pos in 0..lst_data.maxes.len() {
        pyassert!(
            lst_data.maxes[pos]
                .bind(py)
                .eq(keys[pos].last().unwrap().bind(py))?
        );
    }

    // Check sublist lengths are less than double load-factor.

    let double = load << 1;
    pyassert!(lst_data.lists.iter().all(|sublist| sublist.len() <= double));

    // Check sublist lengths are greater than half load-factor for all
    // but the last sublist.

    let half = load >> 1;
    for pos in 0..lst_data.lists.len().saturating_sub(1) {
        pyassert!(lst_data.lists[pos].len() >= half);
    }

    if !lst_data.idx.is_empty() {
        pyassert!(&lst_data.len == lst_data.idx.index(0));
        pyassert!(lst_data.idx.len() == lst_data.offset + lst_data.lists.len());

        // Check index leaf nodes equal length of sublists.

        for pos in 0..lst_data.lists.len() {
            let leaf = lst_data.idx.index(lst_data.offset + pos);
            pyassert!(leaf == &lst_data.lists[pos].len());
        }

        // Check index branch nodes are the sum of their children.

        for pos in 0..lst_data.offset {
            let child = (pos << 1) + 1;
            if child >= lst_data.idx.len() {
                pyassert!(lst_data.idx.index(pos) == &0);
            } else if child + 1 == lst_data.idx.len() {
                pyassert!(lst_data.idx.index(pos) == lst_data.idx.index(child));
            } else {
                let child_sum = lst_data.idx.index(child) + lst_data.idx.index(child + 1);
                pyassert!(&child_sum == lst_data.idx.index(pos));
            }
        }
    };
    Ok(())
}

fn show_key_list(py: Python<'_>, err: &PyErr, data: &SortedKeyList) {
    show_list(py, err, data);
    let keys = data.get_keys();
    let infos = [
        format!("len_keys: {}", keys.len()),
        format!("keys: {:?}", keys),
    ];
    err.add_note(py, infos.join("\n")).unwrap()
}
fn show_list(py: Python<'_>, err: &PyErr, data: &impl SortedListGetters) {
    let lst_data = data.get_data();
    let infos = [
        format!("len: {}", lst_data.len),
        format!("load: {}", data.get_load()),
        format!("offset: {}", lst_data.offset),
        format!("len_index: {}", lst_data.idx.len()),
        format!("index: {:?}", lst_data.idx),
        format!("len_maxes: {}", lst_data.maxes.len()),
        format!("maxes: {:?}", lst_data.maxes),
        format!("len_lists: {}", lst_data.lists.len()),
        format!("lists: {:?}", lst_data.lists),
    ]
    .join("\n");

    err.add_note(py, infos).unwrap()
}

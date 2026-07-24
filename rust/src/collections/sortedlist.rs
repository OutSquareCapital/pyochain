use crate::seq::PyoVec;
use pyo3::prelude::*;
const DEFAULT_LOAD_FACTOR: usize = 1000;
#[pyclass(generic)]
pub struct InnerLists {
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        self.len = 0;
        self.lists.get().clear(py)?;
        self.maxes.get().clear(py)?;
        self.idx.get().clear(py)?;
        self.offset = 0;
        Ok(())
    }
}

#[pyclass(generic)]
pub struct InnerKeyLists {
    #[pyo3(get, set)]
    key: Py<PyAny>,
    #[pyo3(get, set)]
    keys: Py<PyoVec>,
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerKeyLists {
    #[new]
    fn new(key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = key.py();
        Ok(Self {
            key: key.unbind(),
            keys: PyoVec::new_bound(py)?.unbind(),
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        self.len = 0;
        self.lists.get().clear(py)?;
        self.keys.get().clear(py)?;
        self.maxes.get().clear(py)?;
        self.idx.get().clear(py)?;
        Ok(())
    }
}
/// Module for bisect functions, adapted from the Python standard library's bisect module.\
/// Adapted to only handle `pyochain::PyoVec` for both simplicity and performance.
pub mod bisect {
    use super::*;
    /// The following documentation and code is adapted from the Python standard library's bisect module.
    ///
    /// Insert item x in list a, and keep it sorted assuming a is sorted.

    /// If x is already in a, insert it to the right of the rightmost x.

    /// Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.

    /// A custom key function can be supplied to customize the sort order.
    #[pyfunction(signature = (vec, item, lo=0, hi=None, key=None))]
    pub fn insort_right(
        vec: &Bound<'_, PyoVec>,
        item: &Bound<'_, PyAny>,
        lo: usize,
        hi: Option<usize>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let res = match key {
            None => bisect_right(&vec, &item, lo, hi, None),
            Some(key) => bisect_right(&vec, &key.call1((&item,))?, lo, hi, Some(key)),
        }?;
        vec.get().inner.bind(vec.py()).insert(res, item)
    }

    /// The following documentation and code is adapted from the Python standard library's bisect module.
    ///Return the index where to insert item x in list a, assuming a is sorted.

    ///The return value i is such that all e in a[:i] have e <= x, and all e in
    ///a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    ///insert just after the rightmost x already there.

    ///Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.

    ///A custom key function can be supplied to customize the sort order.
    #[pyfunction(signature = (vec, item, lo=0, hi=None, key=None))]
    pub fn bisect_right(
        vec: &Bound<'_, PyoVec>,
        item: &Bound<'_, PyAny>,
        mut lo: usize,
        hi: Option<usize>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let lst = vec.get().inner.bind(vec.py());
        let mut high = hi.unwrap_or_else(|| lst.len());
        // Note, the comparison uses "<" to match the
        // __lt__() logic in list.sort() and in heapq.
        match key {
            None => {
                while lo < high {
                    let mid = (lo + high) / 2;
                    if item.lt(lst.get_item(mid)?)? {
                        high = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            }
            Some(key) => {
                while lo < high {
                    let mid = (lo + high) / 2;
                    if item.lt(key.call1((lst.get_item(mid)?,))?)? {
                        high = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            }
        };
        Ok(lo)
    }
    /// The following documentation and code is adapted from the Python standard library's bisect module.\
    /// Return the index where to insert item x in list a, assuming a is sorted.\
    /// The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.\
    /// So if x already appears in the list, a.insert(i, x) will insert just before the leftmost x already there.\
    /// Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.\
    /// A custom key function can be supplied to customize the sort order.
    #[pyfunction(signature = (vec, item, lo=0, hi=None, key=None))]
    pub fn bisect_left(
        vec: &Bound<'_, PyoVec>,
        item: &Bound<'_, PyAny>,
        mut lo: usize,
        hi: Option<usize>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let lst = vec.get().inner.bind(vec.py());
        let mut hi = hi.unwrap_or_else(|| lst.len());
        // Note, the comparison uses "<" to match the
        // __lt__() logic in list.sort() and in heapq.
        match key {
            None => {
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if lst.get_item(mid)?.lt(item)? {
                        lo = mid + 1;
                    } else {
                        hi = mid
                    }
                }
            }
            Some(key) => {
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if key.call1((lst.get_item(mid)?,))?.lt(item)? {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
            }
        };
        Ok(lo)
    }
}

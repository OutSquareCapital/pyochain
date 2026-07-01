use std::collections::VecDeque;

use crate::args::{Args, Kwargs};
use crate::option::{PyNull, PySome, option};
use crate::result::{PyoErr, PyoOk};
use crate::{abc, mixins};
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PySequence, PySet, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, ffi, prelude::*};
use tap::prelude::*;
#[pymodule(name = "_tools")]
pub fn tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(retain, m)?)?;
    m.add_class::<UniqueIdentity>()?;
    m.add_class::<UniqueKey>()?;
    m.add_class::<Intersperse>()?;
    m.add_class::<MapWindow>()?;
    m.add_class::<MapJuxt>()?;
    m.add_class::<FilterMap>()?;
    m.add_class::<FilterMapStar>()?;
    m.add_class::<Scan>()?;
    m.add_class::<MapWhile>()?;
    m.add_class::<FromFn>()?;
    m.add_class::<Drain>()?;
    m.add_class::<ExtractIf>()?;
    m.add_class::<Successors>()?;
    m.add_class::<FilterStar>()?;
    m.add_class::<WithPosition>()?;
    m.add_class::<ZipLongest>()?;
    m.add_class::<Unzip>()?;
    m.add_class::<GroupBy>()?;
    m.add_class::<Iter>()?;
    m.add_class::<Peekable>()?;
    Ok(())
}

#[pyfunction]
fn retain(data: Bound<'_, PySequence>, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut write_idx = 0;
    let length = data.len()?;
    for read_idx in 0..length {
        let curr = data.get_item(read_idx)?;
        if predicate.call1((&curr,))?.is_truthy()? {
            data.set_item(write_idx, curr)?;
            write_idx += 1;
        }
    }
    data.del_slice(write_idx, usize::MAX)?;
    Ok(())
}

//TODO: the double collect in `Vec` => `PyTuple` is a performance tax on large Vecs of funcs. Need to optimize.
#[pyclass]
pub struct MapJuxt {
    iterator: Py<PyIterator>,
    funcs: Vec<Py<PyAny>>,
}

#[pymethods]
impl MapJuxt {
    #[new]
    #[pyo3(signature = (iterator, *funcs))]
    pub fn new(iterator: Bound<'_, PyIterator>, funcs: &Bound<'_, PyTuple>) -> Self {
        funcs
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>()
            .pipe(|collected| Self {
                iterator: iterator.unbind(),
                funcs: collected,
            })
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(slf: PyRef<'_, Self>) -> PyResult<Option<Bound<'_, PyTuple>>> {
        let py = slf.py();
        match slf.iterator.clone_ref(py).into_bound(py).next() {
            Some(item) => {
                let args = item?;
                slf.funcs
                    .iter()
                    .map(|func| func.call1(py, (&args,)))
                    .collect::<PyResult<Vec<_>>>()
                    .and_then(|x| PyTuple::new(py, x))
                    .map(Some)
            }
            None => Ok(None),
        }
    }
}
/// TODO: speed is 0.76x compared to the Cython implementation.
/// Saved in `.benchmarks/unique_cy`
#[pyclass(frozen)]
pub struct UniqueIdentity {
    iter: Py<PyIterator>,
    seen: Py<PySet>,
}

#[pymethods]
impl UniqueIdentity {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>) -> PyResult<Self> {
        let py = data.py();
        Self {
            iter: data.unbind(),
            seen: PySet::empty(py)?.unbind(),
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        let seen = slf.seen.bind(py);

        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let item = result?;
                    if seen.contains(&item)? {
                        continue;
                    }
                    seen.add(&item)?;
                    return Ok(Some(item));
                }
            }
        }
    }
}
/// TODO: speed is 0.95x compared to the Cython implementation.
/// Saved in `.benchmarks/unique_cy`
#[pyclass(frozen)]
pub struct UniqueKey {
    iter: Py<PyIterator>,
    key: Py<PyAny>,
    seen: Py<PySet>,
}

#[pymethods]
impl UniqueKey {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>, key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = data.py();
        Self {
            iter: data.unbind(),
            key: key.unbind(),
            seen: PySet::empty(py)?.unbind(),
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        let key = slf.key.bind(py);
        let seen = slf.seen.bind(py);

        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let item = result?;
                    let tag = key.call1((&item,))?;
                    if seen.contains(&tag)? {
                        continue;
                    }
                    seen.add(&tag)?;
                    return Ok(Some(item));
                }
            }
        }
    }
}
/// TODO: speed is 0.44x compared to the Cython implementation.
/// Saved in `.benchmarks/intersperse_cy`
/// Cytoolz median time in us:
/// 256 elements: 10.6
/// 1024 elements: 37.3
/// 4096 elements: 127.7
#[pyclass]
pub struct Intersperse {
    data: Py<PyIterator>,
    element: Py<PyAny>,
    val: Option<Py<PyAny>>,
    must_process: bool,
}

#[pymethods]
impl Intersperse {
    #[new]
    pub fn new(mut data: Bound<'_, PyIterator>, element: Py<PyAny>) -> PyResult<Self> {
        let (val, must_process) = match data.next() {
            None => (None, true),
            Some(item) => (Some(item?.unbind()), false),
        };
        Self {
            data: data.unbind(),
            element,
            val,
            must_process,
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        if slf.must_process {
            match slf.data.clone_ref(py).into_bound(py).next() {
                None => Ok(None),
                Some(item) => {
                    slf.val = Some(item?.unbind());
                    slf.must_process = false;
                    Ok(Some(slf.element.clone_ref(py)))
                }
            }
        } else {
            slf.must_process = true;
            slf.val.as_ref().map(|v| v.clone_ref(py)).pipe(Ok)
        }
    }
}
///TODO: It's actually slower than cytoolz implementation when `n` is small, we should optimize for that case.\
/// Observed speeds:\
/// **0.81x** -> `n=2`\
/// **0.93x** -> `n=8`\
/// **1.17x** -> `n=32`\
/// **1.40x** -> `n=128`\
#[pyclass]
pub struct MapWindow {
    iter: Py<PyIterator>,
    prev: Vec<Py<PyAny>>,
}

#[pymethods]
impl MapWindow {
    #[new]
    pub fn new(mut data: Bound<'_, PyIterator>, n: usize) -> PyResult<Self> {
        let py = data.py();
        let mut prev = (0..n)
            .map(|_| py.None().into_any())
            .collect::<Vec<Py<PyAny>>>();
        for i in 1..n {
            match data.next() {
                None => break,
                Some(item) => prev[i] = item?.unbind(),
            }
        }
        Self {
            iter: data.unbind(),
            prev,
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyTuple>>> {
        let py = slf.py();
        let item = match slf.iter.clone_ref(py).into_bound(py).next() {
            None => return Ok(None),
            Some(result) => result?.unbind(),
        };
        slf.prev.rotate_left(1);
        let last = slf.prev.len() - 1;
        slf.prev[last] = item;
        Ok(Some(PyTuple::new(py, slf.prev.iter())?.into()))
    }
}
#[pyclass]
pub struct FilterMap {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl FilterMap {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> Self {
        Self {
            iter: data.unbind(),
            func: func.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let res = func.call1((result?,))?;
                    match res.cast_into_exact::<PySome>() {
                        Ok(some) => return Ok(Some(some.get().value.clone_ref(py))),
                        Err(_) => continue,
                    }
                }
            }
        }
    }
}
#[pyclass]
pub struct FilterMapStar {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl FilterMapStar {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> Self {
        Self {
            iter: data.unbind(),
            func: func.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let res = func.call1(result?.cast_exact::<PyTuple>()?)?;
                    match res.cast_into_exact::<PySome>() {
                        Ok(some) => return Ok(Some(some.get().value.clone_ref(py))),
                        Err(_) => continue,
                    }
                }
            }
        }
    }
}

#[pyclass]
pub struct Scan {
    iter: Py<PyIterator>,
    initial: Py<PyAny>,
    func: Py<PyAny>,
}
#[pymethods]
impl Scan {
    #[new]
    pub fn new(
        data: Bound<'_, PyIterator>,
        initial: Bound<'_, PyAny>,
        func: Bound<'_, PyAny>,
    ) -> Self {
        Self {
            iter: data.unbind(),
            initial: initial.unbind(),
            func: func.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);

        match iter.next() {
            None => Ok(None),
            Some(result) => {
                let state = slf.initial.clone_ref(py);
                let res = func.call1((state, result?))?;

                match res.cast_exact::<PySome>() {
                    Ok(some) => {
                        let next_state = some.get().value.clone_ref(py);
                        slf.initial = next_state.clone_ref(py);
                        Ok(Some(next_state))
                    }
                    Err(_) => Ok(None),
                }
            }
        }
    }
}

#[pyclass]
pub struct MapWhile {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl MapWhile {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> Self {
        Self {
            iter: data.unbind(),
            func: func.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        match iter.next() {
            None => Ok(None),
            Some(result) => match slf.func.bind(py).call1((result?,))?.cast_exact::<PySome>() {
                Ok(some) => Ok(Some(some.get().value.clone_ref(py))),
                Err(_) => Ok(None),
            },
        }
    }
}

enum FromFnStrategy {
    NoArgs,
    HasArgs(Py<PyTuple>),
    HasKwargs(Py<PyDict>),
    HasBoth(Py<PyTuple>, Py<PyDict>),
}
impl FromFnStrategy {
    fn new(args: &Args<'_>, kwargs: Option<&Kwargs<'_>>) -> Self {
        match (args.is_empty(), kwargs) {
            (true, None) => Self::NoArgs,
            (false, None) => Self::HasArgs(args.to_owned().unbind()),
            (true, Some(kwargs)) => Self::HasKwargs(kwargs.to_owned().unbind()),
            (false, Some(kwargs)) => {
                Self::HasBoth(args.to_owned().unbind(), kwargs.to_owned().unbind())
            }
        }
    }
}
#[pyclass]
pub struct FromFn {
    func: Py<PyAny>,
    strategy: FromFnStrategy,
}
#[pymethods]
impl FromFn {
    #[pyo3(signature = (func, *args, **kwargs))]
    #[new]
    pub fn new(func: Bound<'_, PyAny>, args: &Args<'_>, kwargs: Option<&Kwargs<'_>>) -> Self {
        Self {
            func: func.unbind(),
            strategy: FromFnStrategy::new(args, kwargs),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let py_fn = slf.func.bind(py);
        let res = match slf.strategy {
            FromFnStrategy::NoArgs => py_fn.call0()?,
            FromFnStrategy::HasArgs(ref args) => py_fn.call1(args.bind(py))?,
            FromFnStrategy::HasKwargs(ref kwargs) => py_fn.call((), Some(kwargs.bind(py)))?,
            FromFnStrategy::HasBoth(ref args, ref kwargs) => {
                py_fn.call(args.bind(py), Some(kwargs.bind(py)))?
            }
        };
        match res.cast_into_exact::<PySome>() {
            Ok(some) => Ok(Some(some.get().value.clone_ref(py))),
            Err(_) => Ok(None),
        }
    }
}
#[pyclass]
struct Drain {
    vec: Py<PySequence>,
    start: usize,
    current: usize,
    end: usize,
    done: bool,
}

impl Drain {
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.done {
            return Ok(());
        }
        self.vec.bind(py).del_slice(self.start, self.end)?;
        self.done = true;
        Ok(())
    }
}

#[pymethods]
impl Drain {
    #[new]
    fn new(vec: Bound<'_, PySequence>, start: Option<usize>, end: Option<usize>) -> PyResult<Self> {
        let s = start.unwrap_or_default();
        let e = end.unwrap_or(vec.len()?);
        Self {
            vec: vec.unbind(),
            start: s,
            current: s,
            end: e,
            done: false,
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        if &slf.current >= &slf.end {
            slf.finish(py)?;
            Ok(None)
        } else {
            let val = slf.vec.bind(py).get_item(slf.current)?.unbind();
            slf.current += 1;
            Ok(Some(val))
        }
    }
}

impl Drop for Drain {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        Python::attach(|py| {
            let _ = self.finish(py);
        });
    }
}
#[pyclass]
struct ExtractIf {
    data: Py<PySequence>,
    pred: Py<PyAny>,
    idx: usize,
    end: usize,
    old_len: usize,
    deleted: usize,
    done: bool,
}

impl ExtractIf {
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.done {
            return Ok(());
        }
        let seq = self.data.bind(py);
        if self.deleted > 0 {
            let mut tail_start = self.idx - self.deleted;
            let mut tail_src = self.idx;
            while tail_src < self.old_len {
                let item = seq.get_item(tail_src)?;
                seq.set_item(tail_start, &item)?;
                tail_start += 1;
                tail_src += 1;
            }
            seq.del_slice(self.old_len - self.deleted, self.old_len)?;
        }
        self.done = true;
        Ok(())
    }
}

#[pymethods]
impl ExtractIf {
    #[new]
    fn new(
        data: Bound<'_, PySequence>,
        pred: Bound<'_, PyAny>,
        start: Option<usize>,
        end: Option<usize>,
    ) -> PyResult<Self> {
        let old_len = data.len()?;
        Self {
            data: data.unbind(),
            pred: pred.unbind(),
            old_len,
            idx: start.unwrap_or_default(),
            end: end.unwrap_or(old_len),
            deleted: 0,
            done: false,
        }
        .pipe(Ok)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let seq = slf.data.clone_ref(py).into_bound(py);
        let pred = slf.pred.clone_ref(py).into_bound(py);
        while slf.idx < slf.end {
            let i = slf.idx;
            let item = seq.get_item(i)?;
            slf.idx += 1;

            if pred.call1((&item,))?.is_truthy()? {
                slf.deleted += 1;
                return Ok(Some(item.unbind()));
            }
            if slf.deleted > 0 {
                seq.set_item(i - slf.deleted, &item)?;
            }
        }

        slf.finish(py)?;
        Ok(None)
    }
}

impl Drop for ExtractIf {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        Python::attach(|py| {
            let _ = self.finish(py);
        });
    }
}
#[pyclass]
pub struct Successors {
    succ: Py<PyAny>,
    current: Py<PyAny>,
}
#[pymethods]
impl Successors {
    #[new]
    pub fn new(start: Bound<'_, PyAny>, succ: Bound<'_, PyAny>) -> Self {
        Self {
            current: start.unbind(),
            succ: succ.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let curr = slf.current.clone_ref(py);
        match curr.bind(py).cast_exact::<PySome>() {
            Ok(some) => {
                let unwrapped = some.get().value.clone_ref(py);
                slf.current = slf.succ.bind(py).call1((unwrapped.bind(py),))?.unbind();
                Ok(Some(unwrapped))
            }
            Err(_) => Ok(None),
        }
    }
}
#[pyclass]
pub struct FilterStar {
    iter: Py<PyIterator>,
    predicate: Py<PyAny>,
}

#[pymethods]
impl FilterStar {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>, predicate: Bound<'_, PyAny>) -> Self {
        Self {
            iter: data.unbind(),
            predicate: predicate.unbind(),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        let predicate = slf.predicate.bind(py);
        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let item = result?;
                    if predicate
                        .call1(item.cast_exact::<PyTuple>()?)?
                        .is_truthy()?
                    {
                        return Ok(Some(item.unbind()));
                    }
                }
            }
        }
    }
}

/// Equivalent of a `Literal` in Python.\
/// Defining an Enum here, I feel isn't needed since we only pass around the string values, and we don't need to do anything with them later.
mod position {
    use super::*;
    use pyo3::intern;
    const FIRST: &str = "first";
    const MIDDLE: &str = "middle";
    const LAST: &str = "last";
    const ONLY: &str = "only";
    #[inline(always)]
    pub fn get(did_iter: bool, has_next: bool, py: Python<'_>) -> &Bound<'_, PyString> {
        match (did_iter, has_next) {
            (false, true) => intern!(py, FIRST),
            (false, false) => intern!(py, ONLY),
            (true, true) => intern!(py, MIDDLE),
            (true, false) => intern!(py, LAST),
        }
    }
}

#[pyclass]
pub struct WithPosition {
    iter: Py<PyIterator>,
    did_iter: bool,
    peeked: Option<Py<PyAny>>,
}
#[pymethods]
impl WithPosition {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>) -> Self {
        Self {
            iter: data.unbind(),
            did_iter: false,
            peeked: None,
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(
        mut slf: PyRefMut<'_, Self>,
    ) -> PyResult<Option<(&Bound<'_, PyString>, Bound<'_, PyAny>)>> {
        let py = slf.py();
        let mut iterator = slf.iter.clone_ref(py).into_bound(py);

        let current = match slf.peeked.take() {
            Some(item) => item.into_bound(py),
            None => match iterator.next() {
                Some(item) => item?,
                None => return Ok(None),
            },
        };

        let has_next = match iterator.next() {
            Some(item) => {
                slf.peeked = Some(item?.unbind());
                true
            }
            None => false,
        };
        let position = position::get(slf.did_iter, has_next, py);
        slf.did_iter = true;

        Ok(Some((position, current)))
    }
}
#[pyclass]
pub struct ZipLongest {
    iterator: Py<PyIterator>,
}
#[pymethods]
impl ZipLongest {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>) -> Self {
        Self {
            iterator: data.unbind(),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyTuple>>> {
        let py = slf.py();
        let mut iter = slf.iterator.clone_ref(py).into_bound(py);
        match iter.next() {
            None => Ok(None),
            Some(item) => item
                // SAFETY: we know the passed `PyIterator` is from `itertools::zip_longest`, which yields tuples, so we can safely cast the result to a `PyTuple`.
                .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })?
                .iter()
                .map(|x| option(&x))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|v| PyTuple::new(py, v))
                .map(Bound::unbind)
                .map(Some),
        }
    }
}
#[pyclass]
pub struct Unzip {
    iterator: Py<PyIterator>,
    n: usize,
}
#[pymethods]
impl Unzip {
    #[new]
    pub fn new(data: &Bound<'_, PyTuple>, n: usize) -> Self {
        let iterator = data
            .get_item(n)
            .unwrap()
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
            .unbind();
        Self {
            iterator: iterator,
            n,
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        match slf.iterator.clone_ref(py).into_bound(py).next() {
            Some(item) => item?
                .pipe(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
                .get_item(slf.n)
                .map(Some),
            None => Ok(None),
        }
    }
}
#[pyclass]
pub struct GroupBy {
    iterator: Py<PyIterator>,
}
#[pymethods]
impl GroupBy {
    #[new]
    pub fn new(data: Bound<'_, PyIterator>) -> Self {
        Self {
            iterator: data.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<(Bound<'_, PyAny>, Py<Iter>)>> {
        let py = slf.py();
        match slf.iterator.clone_ref(py).into_bound(py).next() {
            Some(item) => unsafe {
                let tup = item?.cast_into_unchecked::<PyTuple>();
                let (key, group) = (tup.get_item_unchecked(0), tup.get_item_unchecked(1));

                Ok(Some((key, Iter::new(group)?)))
            },
            None => Ok(None),
        }
    }
}
#[pyclass]
pub struct OnceWith {
    func: Py<PyAny>,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    yielded: bool,
}
#[pymethods]
impl OnceWith {
    #[new]
    #[pyo3(signature = (func, *args, **kwargs))]
    pub fn new(func: Bound<'_, PyAny>, args: Args<'_>, kwargs: Option<Kwargs<'_>>) -> Self {
        Self {
            func: func.unbind(),
            args: args.unbind(),
            kwargs: kwargs.map(|k| k.unbind()),
            yielded: false,
        }
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        if !slf.yielded {
            slf.yielded = true;
            let py = slf.py();
            let args = slf.args.bind(py);
            let kwargs = slf.kwargs.as_ref().map(|k| k.bind(py));
            slf.func.bind(py).call(args, kwargs).map(Some)
        } else {
            Ok(None)
        }
    }
}
#[pyclass]
pub struct Tail {
    data: VecDeque<PyResult<Py<PyAny>>>,
}
#[pymethods]
impl Tail {
    /// # Credits
    /// code taken and adapted from [itertools](https://docs.rs/itertools/latest/itertools/)
    #[new]
    pub fn new(iterator: Bound<'_, PyIterator>, n: usize) -> PyResult<Self> {
        match n {
            0 => {
                iterator.last();
                VecDeque::new()
            }
            1 => iterator
                .last()
                .into_iter()
                .map(|item| item.map(|i| i.unbind()))
                .collect(),
            _ => {
                // Skip the starting part of the iterator if possible.
                let (low, _) = iterator.size_hint();
                let mut iter = iterator
                    .fuse()
                    .skip(low.saturating_sub(n))
                    .map(|item| item.map(|i| i.unbind()));
                // TODO: If VecDeque has a more efficient method than
                // `.pop_front();.push_back(val)` in the future then maybe revisit this.
                let mut data = iter.by_ref().take(n).collect::<Vec<_>>();
                // Update `data` cyclically.
                let idx = iter.fold(0, |i, val| {
                    debug_assert_eq!(data.len(), n);
                    data[i] = val;
                    if i + 1 == n { 0 } else { i + 1 }
                });
                // Respect the insertion order, efficiently.
                let mut data = VecDeque::from(data);
                data.rotate_left(idx);
                data
            }
        }
        .pipe(|data| Self { data })
        .pipe(Ok)
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        slf.data
            .pop_front()
            .transpose()?
            .map(|item| item.into_bound(slf.py()))
            .pipe(Ok)
    }
}

#[pyclass(frozen, generic, extends=abc::PyoIterator)]
pub struct Iter {
    inner: Py<PyIterator>,
}
impl Iter {
    /// New constructor for `Iter` in rust.
    /// We do this because `PyClassInitializer` can't be converted to pyobject directly, so we need to wrap it in a `Py` first.
    pub fn new(data: Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let py = data.py();
        let initializer = Self::py_new(data)?;
        Py::new(py, initializer)
    }
}
#[pymethods]
impl Iter {
    #[new]
    fn py_new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        PyClassInitializer::from(mixins::Checkable)
            .add_subclass(abc::PyoIterable {})
            .add_subclass(abc::PyoIterator {})
            .add_subclass(Self {
                inner: data.try_iter()?.unbind(),
            })
            .pipe(Ok)
    }

    fn __iter__<'py>(slf: &Bound<'py, Self>) -> Py<PyIterator> {
        slf.get().inner.clone_ref(slf.py())
    }

    fn __next__<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        slf.get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .next()
            .transpose()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name();
        let inner_repr = slf.get().inner.clone_ref(py).into_bound(py).repr()?;
        Ok(format!("{:?}({:?})", name, inner_repr))
    }
}

// TODO: check if it aligns with `last` logic, AND if it is faster than pyo3 `PyIterator::next()`
#[allow(dead_code)]
pub struct PyUnsafeIterator<'py> {
    py: Python<'py>,
    ptr: *mut ffi::PyObject,
    next: ffi::iternextfunc,
}
#[allow(dead_code)]
impl<'py> PyUnsafeIterator<'py> {
    #[inline(always)]
    pub fn new(iter: &Bound<'py, PyIterator>) -> Self {
        let ptr = iter.as_ptr();

        Self {
            py: iter.py(),
            ptr,
            next: unsafe { (*(*ptr).ob_type).tp_iternext.unwrap_unchecked() },
        }
    }
}
impl<'py> Iterator for PyUnsafeIterator<'py> {
    type Item = PyResult<Bound<'py, PyAny>>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let item = (self.next)(self.ptr);

            if item.is_null() {
                if ffi::PyErr_Occurred().is_null() {
                    return None;
                }

                if ffi::PyErr_ExceptionMatches(ffi::PyExc_StopIteration) != 0 {
                    ffi::PyErr_Clear();
                    return None;
                }

                return Some(Err(PyErr::fetch(self.py)));
            }

            Some(Ok(Bound::from_owned_ptr(self.py, item)))
        }
    }
}
#[pyclass(generic, extends=abc::PyoIterator)]
pub struct Peekable {
    iterator: Py<PyIterator>,
    peeked: Option<Py<PyAny>>,
}
impl Peekable {
    /// New constructor for `Peekable` in rust.
    /// We do this because `PyClassInitializer` can't be converted to pyobject directly, so we need to wrap it in a `Py` first.
    pub fn new(data: Bound<'_, PyIterator>) -> PyResult<Py<Self>> {
        let py = data.py();
        let initializer = Self::py_new(data);
        Py::new(py, initializer)
    }
}
#[pymethods]
impl Peekable {
    #[new]
    fn py_new(iterable: Bound<'_, PyIterator>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(mixins::Checkable)
            .add_subclass(abc::PyoIterable {})
            .add_subclass(abc::PyoIterator {})
            .add_subclass(Self {
                iterator: iterable.unbind(),
                peeked: None,
            })
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.peeked.take() {
            Some(value) => Ok(Some(value.into_bound(py))),
            None => self
                .iterator
                .clone_ref(py)
                .into_bound(py)
                .next()
                .transpose(),
        }
    }

    fn __bool__(&mut self, py: Python<'_>) -> bool {
        self.peek(py)
            .map(|x| x.bind(py).cast_exact::<PySome>().is_ok())
            .unwrap_or(false)
    }

    fn peek(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.peeked.is_none() {
            self.peeked = self
                .iterator
                .clone_ref(py)
                .into_bound(py)
                .next()
                .transpose()?
                .map(Bound::unbind);
        }

        self.peeked
            .as_ref()
            .map(|x| x.clone_ref(py))
            .map(|x| PySome::new(x).into_py_any(py))
            .unwrap_or_else(|| PyNull::get(py).into_py_any(py))
    }

    fn next_if(&mut self, func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = func.py();
        match self
            .iterator
            .clone_ref(py)
            .into_bound(py)
            .next()
            .transpose()?
        {
            Some(matched) if func.call1((&matched,))?.is_truthy()? => {
                matched.unbind().pipe(PySome::new).into_py_any(py)
            }
            other => {
                self.peeked = other.map(Bound::unbind);
                PyNull::get(py).into_py_any(py)
            }
        }
    }

    fn next_if_eq(&mut self, expected: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = expected.py();
        match self
            .iterator
            .clone_ref(py)
            .into_bound(py)
            .next()
            .transpose()?
        {
            Some(nxt) if nxt.eq(expected)? => nxt.unbind().pipe(PySome::new).into_py_any(py),
            other => {
                self.peeked = other.map(Bound::unbind);
                PyNull::get(py).into_py_any(py)
            }
        }
    }

    fn next_if_map(&mut self, f: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = f.py();
        self.peeked = match self.iterator.clone_ref(py).into_bound(py).next() {
            Some(item) => {
                let call_result = f.call1((&item?,))?;
                match call_result.cast_exact::<PyoOk>() {
                    Ok(result) => {
                        return result
                            .get()
                            .value
                            .clone_ref(py)
                            .pipe(PySome::new)
                            .into_py_any(py);
                    }
                    Err(_) => Some(
                        call_result
                            .cast_into_exact::<PyoErr>()?
                            .get()
                            .error
                            .clone_ref(py),
                    ),
                }
            }
            None => None,
        };

        PyNull::get_any_ok(py)
    }
}

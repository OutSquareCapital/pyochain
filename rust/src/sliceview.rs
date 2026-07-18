use crate::cast_match;
use crate::{
    abc::{self, PyoABC, PyoSequence},
    pyo3_ext::pylibs::{PyMutableSequence, PyMutableSequenceMethods, SupportsIndex},
};
use either::Either;
use pyo3::{
    PyTypeInfo,
    call::PyCallArgs,
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyInt, PyRange, PyRangeMethods, PySequence, PySlice, PyTuple},
};
use tap::prelude::*;

struct OpenRange {
    start: isize,
    step: isize,
}
impl OpenRange {
    fn new(start: isize, step: isize) -> Self {
        Self { start, step }
    }
    /// Return a concrete range clamped to the current base length.
    fn resolve<'py>(&self, py: Python<'py>, b_len: isize) -> PyResult<Bound<'py, PyRange>> {
        let stop = if self.step > 0 { b_len } else { -1 };
        PyRange::new_with_step(py, self.start, stop, self.step)
    }
}

/// TODO: See if it make sense to separate mutable vs immutable slices
/// TODO: See if collections should have dedicated slice views methods

trait PyInit<'py, T: PyTypeInfo, A: PyCallArgs<'py>> {
    fn init(py: Python<'py>, args: A) -> PyResult<Bound<'py, T>>;
}
type SliceArgs<'py> = (
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
);
impl<'py> PyInit<'py, PySlice, SliceArgs<'py>> for PySlice {
    fn init(py: Python<'py>, args: SliceArgs<'py>) -> PyResult<Bound<'py, Self>> {
        let (start, stop, step) = args;
        PySlice::type_object(py)
            .call1((start, stop, step))
            .map(|slice| unsafe { slice.cast_into_unchecked::<PySlice>() })
    }
}

#[pyclass(frozen, generic, sequence, extends=abc::PyoSequence)]
pub struct SliceView {
    #[pyo3(get)]
    inner: Py<PySequence>,
    range: Either<Py<PyRange>, OpenRange>,
}
#[pymethods]
impl SliceView {
    #[pyo3(signature = (base, start=None, stop=None, step=None))]
    #[new]
    fn new(
        base: Bound<'_, PySequence>,
        start: Option<Bound<'_, PyAny>>,
        stop: Option<Bound<'_, PyAny>>,
        step: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = base.py();
        let base_len = base.len()? as isize;
        let indices = match start {
            Some(s) => match s.cast_exact::<PySlice>() {
                Ok(slice) => slice.indices(base_len),
                Err(_) => PySlice::init(py, (Some(s), stop, step))?.indices(base_len),
            },
            None => PySlice::init(py, (start, stop, step))?.indices(base_len),
        }?;
        let range = if indices.stop == base_len {
            OpenRange::new(indices.start, indices.step).pipe(Either::Right)
        } else {
            PyRange::new_with_step(py, indices.start, indices.stop, indices.step)?
                .unbind()
                .pipe(Either::Left)
        };

        PyoSequence::build_init()
            .add_subclass(Self {
                inner: base.unbind(),
                range: range,
            })
            .pipe(Ok)
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<SliceViewIterator> {
        SliceViewIterator::new(self._current_range(py)?, self.inner.clone_ref(py))
    }
    #[staticmethod]
    fn _from_range(
        py: Python<'_>,
        inner: Py<PySequence>,
        range: Py<PyRange>,
    ) -> PyResult<Bound<'_, Self>> {
        PyoSequence::build_init()
            .add_subclass(Self {
                inner,
                range: Either::Left(range),
            })
            .pipe(|initializer| Bound::new(py, initializer))
    }

    fn __contains__(slf: Bound<'_, Self>, item: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.try_iter()
            .unwrap()
            .map(|el| item.eq(el?))
            .find_map(|x| match x {
                Ok(true) => Some(Ok(true)),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            })
            .unwrap_or(Ok(false))
    }

    fn __reversed__(&self, py: Python<'_>) -> PyResult<SliceViewReverseIterator> {
        SliceViewReverseIterator::new(self._current_range(py)?, self.inner.clone_ref(py))
    }

    fn __eq__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        other
            .cast_into::<PySequence>()
            .map(|o| {
                let elem_eq = slf
                    .try_iter()
                    .unwrap()
                    .zip(o.try_iter().unwrap())
                    .map(|(a, b)| a?.eq(b?))
                    .find_map(|x| match x {
                        Ok(true) => None,
                        Ok(false) => Some(Ok(false)),
                        Err(e) => Some(Err(e)),
                    })
                    .unwrap_or(Ok(true))?;
                Ok(Self::__len__(slf)? == o.len()? && elem_eq)
            })
            .unwrap_or(Ok(false))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        let repr = self.inner.bind(py).repr()?;
        let cr = self._current_range(py)?;
        Ok(format!(
            "{}({})[{}:{}:{}]",
            name,
            repr,
            cr.start()?,
            cr.stop()?,
            cr.step()?
        ))
    }

    fn __len__(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.get()._current_range(slf.py())?.len()
    }

    fn __getitem__<'py>(
        slf: Bound<'py, Self>,
        index: Bound<'py, PyAny>,
    ) -> PyResult<Either<Bound<'py, Self>, Bound<'py, PyAny>>> {
        let py = slf.py();
        let inner = slf.get().inner.clone_ref(py);
        let current_range = slf.get()._current_range(py)?;
        match index.cast_exact::<PySlice>() {
            Ok(slice) => {
                // Compose slices using Python's range slicing — O(1), exact.
                let range = current_range
                    .get_item(slice)
                    .map(|r| unsafe { r.cast_into_unchecked::<PyRange>() })?
                    .unbind()
                    .pipe(Either::Left);
                PyoSequence::build_init()
                    .add_subclass(Self { inner, range })
                    .pipe(|initializer| Bound::new(py, initializer))
                    .map(Either::Left)
            }
            Err(_) => {
                let length = current_range.len()? as isize;
                let mut idx = index.call_method0("__index__")?.extract::<isize>()?;
                if idx < 0 {
                    idx += length
                };
                if !(0 <= idx && idx < length) {
                    let msg = "sliceview index out of range";
                    Err(PyIndexError::new_err(msg))
                } else {
                    Ok(Either::Right(inner.bind(py).get_item(
                        current_range.get_item(idx)?.extract::<usize>()?,
                    )?))
                }
            }
        }
    }

    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = index.py();

        let inner = self.inner.bind(py);
        let cr = self._current_range(py)?;
        cast_match!((inner, index) {
            (PyMutableSequence, PySlice) => {
                let tr = cr
                    .get_item(index)
                    .map(|r| unsafe { r.cast_into_unchecked::<PyRange>() })?;
                if tr.step()?.abs() != 1 {
                    let values = PyTuple::type_object(py).call1((value,)).map(|t| unsafe { t.cast_into_unchecked::<PyTuple>() })?;
                    let values_len = values.len();
                    let tr_len = tr.len()?;
                    if values_len != tr_len {
                        let msg = format!(
                            "attempt to assign sequence of size {} to slice of size {}", values_len, tr_len
                        );
                        Err(PyValueError::new_err(msg))
                    } else {
                        tr.try_iter()?
                            .zip(values)
                            .try_for_each(|(i, v)| {
                                inner.set_item(i?.extract::<usize>()?, v)?;
                                Ok::<(), PyErr>(())
                            })
                    }
                } else {

                inner.set_slice_with_step(tr.start()?, tr.stop()?, tr.step()?, &value)
                }
            },
            (PyMutableSequence, SupportsIndex) => {
                let length = cr.len()?;

                let mut idx = index
                    .call_method0(intern!(py, "__index__"))?
                    .extract::<isize>()?;
                if idx < 0 {
                    idx += length as isize
                };
                if !(0 <= idx && idx < length as isize) {
                    let msg = "SliceView index out of range";
                    return Err(PyIndexError::new_err(msg));
                };
                inner.set_item(
                    cr.get_item(PyInt::new(py, idx).into_any())?
                        .extract::<usize>()?,
                    value,
                )?;
                Ok(())
            },
            _ => {
                let name = inner.get_type().name()?;
                let msg = format!(
                    "underlying sequence of type '{}' has no '__setitem__'",
                    name
                );
                Err(PyTypeError::new_err(msg))
            }
        })
    }

    fn advance<'py>(&self, n: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        let b_len = self.inner.bind(py).len()?;
        let cr = self._current_range(py)?;
        let new_start = 0.max(cr.start()? + n.extract::<isize>()?.min(b_len as isize));
        let delta = new_start - cr.start()?;
        let new_stop = 0.max(cr.stop()? + delta.min(b_len as isize));
        Self::_from_range(
            py,
            self.inner.clone_ref(py),
            PyRange::new_with_step(py, new_start, new_stop, cr.step()?)?.unbind(),
        )
    }
    fn _current_range<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        self.range
            .as_ref()
            .map_right(|r| r.resolve(py, self.inner.clone_ref(py).bind(py).len()? as isize))
            .map_left(|r| Ok(r.clone_ref(py).into_bound(py)))
            .into_inner()
    }
}
#[pyclass(generic)]
struct SliceViewIterator {
    current_index: usize,
    length: usize,
    range: Py<PyRange>,
    seq: Py<PySequence>,
}
#[pymethods]
impl SliceViewIterator {
    #[new]
    fn new(range: Bound<'_, PyRange>, seq: Py<PySequence>) -> PyResult<Self> {
        let length = range.len()?;
        Ok(Self {
            current_index: 0,
            length,
            range: range.unbind(),
            seq,
        })
    }
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        if slf.current_index >= slf.length {
            Ok(None)
        } else {
            let py = slf.py();
            let base_idx = slf
                .range
                .bind(py)
                .get_item(slf.current_index)?
                .extract::<usize>()?;
            let item = slf.seq.bind(py).get_item(base_idx)?;
            slf.current_index += 1;
            Ok(Some(item))
        }
    }
}

#[pyclass(generic)]
struct SliceViewReverseIterator {
    current_index: usize,
    length: usize,
    range: Py<PyRange>,
    seq: Py<PySequence>,
}
#[pymethods]
impl SliceViewReverseIterator {
    #[new]
    fn new(range: Bound<'_, PyRange>, seq: Py<PySequence>) -> PyResult<Self> {
        let length = range.len()?;
        Ok(Self {
            current_index: 0,
            length,
            range: range.unbind(),
            seq,
        })
    }
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        if slf.current_index >= slf.length {
            Ok(None)
        } else {
            let py = slf.py();
            let rev_idx = slf.length - 1 - slf.current_index;
            let base_idx = slf.range.bind(py).get_item(rev_idx)?.extract::<usize>()?;
            let item = slf.seq.bind(py).get_item(base_idx)?;
            slf.current_index += 1;
            Ok(Some(item))
        }
    }
}

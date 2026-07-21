use crate::abc::PyoABC;
use crate::pyo3_ext::{prelude::*, pylibs};
use crate::{abc, seq, tools};
use bound_from_any::{py_abc, try_cast};
use either::Either;
use pyo3::intern;
use pyo3::{
    BoundObject,
    prelude::*,
    types::{PyList, PyNotImplemented, PyTuple},
};
use tap::Pipe;

#[py_abc(HeapMin, HeapMax)]
trait HeapType: Sized + PyWrapper<PyList> {
    #[new]
    fn new(data: Bound<'_, PyList>) -> PyResult<PyClassInitializer<Self>>;
    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
    fn push<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<()>;
    fn pop<'py>(&self, py: Python<'py>, _index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;

    #[py_skip]
    fn from_ref<'py>(py: Python<'py>, data: Bound<'_, PyList>) -> PyResult<Bound<'py, Self>> {
        Self::new(data).and_then(|init| Bound::new(py, init))
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.as_inner().bind(py).len()
    }

    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.as_inner().bind(index.py()).as_any().get_item(index)
    }

    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.as_inner()
            .bind(index.py())
            .as_any()
            .set_item(index, value)
    }

    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        self.as_inner().bind(index.py()).as_any().del_item(index)
    }

    fn __eq__<'py>(
        &self,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Either<bool, Bound<'py, PyNotImplemented>>> {
        let py = other.py();
        let inner = self.as_inner().bind(py);
        try_cast! {
            match other {
                HeapMax | HeapMin | seq::Vec => inner.eq(other.get().inner.clone_ref(py)).map(Either::Left),
                PyList => inner.eq(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.as_inner()
            .bind(value.py())
            .call_method1("insert", (index, value))?;
        Ok(())
    }

    #[pyo3(signature = (*others, key=None, reverse=false))]
    fn merge(
        &self,
        others: Bound<'_, PyTuple>,
        key: Option<Bound<'_, PyAny>>,
        reverse: bool,
    ) -> PyResult<Py<tools::Iter>> {
        let py = others.py();
        pylibs::heapq::merge(
            others.py(),
            (self.as_inner().bind(py), others),
            key,
            reverse,
        )?
        .into_any()
        .pipe(tools::Iter::new)
    }
    #[pyo3(signature = (n, key=None))]
    fn n_smallest<'py>(
        &self,
        py: Python<'py>,
        n: isize,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        pylibs::heapq::nsmallest(n, self.as_inner().bind(py).as_any(), key)
            .and_then(|x| Self::from_ref(py, x))
    }
    #[pyo3(signature = (n, key=None))]
    fn n_largest<'py>(
        &self,
        py: Python<'py>,
        n: isize,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        pylibs::heapq::nlargest(n, self.as_inner().bind(py).as_any(), key)
            .and_then(|x| Self::from_ref(py, x))
    }
}
/// Present for typing purposes only.
#[pyclass(frozen, generic, sequence, extends = abc::PyoMutableSequence, subclass)]
pub struct Heap;
#[pyclass(frozen, generic, sequence, extends = Heap)]
pub struct HeapMin {
    pub inner: Py<PyList>,
}
#[pymethods]
impl Heap {
    #[allow(unused_variables)]
    #[new]
    fn new(data: Bound<'_, PyList>) -> PyClassInitializer<Self> {
        abc::PyoMutableSequence::build_init().add_subclass(Self)
    }
}
impl HeapType for HeapMin {
    fn new(data: Bound<'_, PyList>) -> PyResult<PyClassInitializer<Self>> {
        pylibs::heapq::heapify(&data)?;
        data.unbind()
            .pipe(|inner| {
                abc::PyoMutableSequence::build_init()
                    .add_subclass(Heap)
                    .add_subclass(Self { inner })
            })
            .pipe(Ok)
    }
    fn push(&self, item: Bound<'_, PyAny>) -> PyResult<()> {
        pylibs::heapq::heappush(self.inner.bind(item.py()), item)
    }
    fn pop<'py>(&self, py: Python<'py>, _index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heappop(self.as_inner().bind(py))
    }
    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heapreplace(self.inner.bind(item.py()), item)
    }

    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heappushpop(self.inner.bind(item.py()), item)
    }
}

#[pyclass(frozen, generic, sequence, extends = Heap)]
pub struct HeapMax {
    pub inner: Py<PyList>,
}
impl HeapType for HeapMax {
    fn new(data: Bound<'_, PyList>) -> PyResult<PyClassInitializer<Self>> {
        pylibs::heapq::heapify_max(&data)?;
        data.unbind()
            .pipe(|inner| {
                abc::PyoMutableSequence::build_init()
                    .add_subclass(Heap)
                    .add_subclass(Self { inner })
            })
            .pipe(Ok)
    }
    fn push(&self, item: Bound<'_, PyAny>) -> PyResult<()> {
        let py = item.py();
        let inner = self.inner.bind(py);
        inner.append(item)?;
        self._siftdown(py, 0, inner.len() - 1)
    }

    fn pop<'py>(&self, py: Python<'py>, _index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.bind(py);
        let lastelt = inner.call_method0(intern!(py, "pop"))?;
        if !(inner.is_empty()) {
            let returnitem = inner.get_item(0)?;
            inner.set_item(0, lastelt)?;
            self._siftup(py, 0)?;
            Ok(returnitem)
        } else {
            Ok(lastelt)
        }
    }

    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = item.py();
        let inner = self.inner.bind(py);
        let returnitem = inner.get_item(0)?; // raises appropriate IndexError if heap is empty
        inner.set_item(0, item)?;
        self._siftup(py, 0)?;
        Ok(returnitem)
    }
    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = item.py();
        let inner = self.inner.bind(py);
        if !(inner.is_empty()) && item.lt(inner.get_item(0)?)? {
            let returnitem = inner.get_item(0)?;
            inner.set_item(0, item)?;
            self._siftup(py, 0)?;
            Ok(returnitem)
        } else {
            Ok(item)
        }
    }
}
impl HeapMax {
    fn _siftdown(&self, py: Python<'_>, startpos: usize, mut pos: usize) -> PyResult<()> {
        let inner = self.inner.bind(py);
        let newitem = inner.get_item(pos)?.extract::<usize>()?;
        // Follow the path to the root, moving parents down until finding a place
        // newitem fits.
        while pos > startpos {
            let parentpos = (pos - 1) >> 1;
            let parent = inner.get_item(parentpos)?.extract::<usize>()?;
            if parent < newitem {
                inner.set_item(pos, parent)?;
                pos = parentpos;
                continue;
            }
            break;
        }
        inner.set_item(pos, newitem)
    }

    fn _siftup(&self, py: Python<'_>, mut pos: usize) -> PyResult<()> {
        let inner = self.inner.bind(py);
        let endpos = inner.len();
        let startpos = pos;
        let newitem = inner.get_item(pos)?;
        // Bubble up the larger child until hitting a leaf.
        let mut childpos = 2 * pos + 1; // leftmost child position
        while childpos < endpos {
            // Set childpos to index of larger child.
            let rightpos = childpos + 1;
            if rightpos < endpos && !(inner.get_item(rightpos)?.lt(inner.get_item(childpos)?)?) {
                childpos = rightpos;
            }
            // Move the larger child up.
            inner.set_item(pos, inner.get_item(childpos)?)?;
            pos = childpos;
            childpos = 2 * pos + 1;
        }
        // The leaf at pos is empty now.  Put newitem there, and bubble it up
        // to its final resting place (by sifting its parents down).
        inner.set_item(pos, newitem)?;
        self._siftdown(py, startpos, pos)
    }
}

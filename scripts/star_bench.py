"""Benchmark: map() with tuples vs NamedTuples vs map_star().

Compares the performance of:
1. .map(lambda x: func(*x))           - Plain tuples unpacking
2. .map(lambda x: func(*x.values()))  - NamedTuple unpacking
3. .map_star(func)                    - Direct starmap approach
"""

import timeit
from typing import NamedTuple

# ============================================================================
# Setup: Test functions
# ============================================================================


def make_sku(color: str, size: str) -> str:
    """Simple 2-arg function."""
    return f"{color}-{size}"


def process_triple(x: int, y: int, z: int) -> int:
    """3-arg function."""
    return x + y * z


def format_record(idx: int, name: str, value: float) -> str:
    """3-arg function with different types."""
    return f"[{idx}] {name}: {value:.2f}"


def compute_complex(a: int, b: int, c: int, d: int, e: int) -> int:
    """5-arg function for heavier computation."""
    return (a + b) * (c - d) + e


# ============================================================================
# NamedTuple definitions
# ============================================================================


class SKUPair(NamedTuple):
    """NamedTuple for SKU data."""

    color: str
    size: str


class Triple(NamedTuple):
    """NamedTuple for 3 integers."""

    x: int
    y: int
    z: int


class Record(NamedTuple):
    """NamedTuple for record data."""

    idx: int
    name: str
    value: float


class ComplexData(NamedTuple):
    """NamedTuple for complex computation."""

    a: int
    b: int
    c: int
    d: int
    e: int


# ============================================================================
# Benchmark functions
# ============================================================================


def benchmark_2_args() -> tuple[float, float, float, float]:
    """Test 2-argument function."""
    setup_tuple = """
import pyochain as pc
colors = ['red', 'blue', 'green', 'yellow', 'purple']
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
data_product = list(pc.Iter(colors).product(sizes).collect())
def make_sku(color, size):
    return f"{color}-{size}"
"""

    setup_namedtuple = """
import pyochain as pc
from typing import NamedTuple
colors = ['red', 'blue', 'green', 'yellow', 'purple']
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
class SKUPair(NamedTuple):
    color: str
    size: str
data_product = [SKUPair(c, s) for c in colors for s in sizes]
def make_sku(color, size):
    return f"{color}-{size}"
"""

    tuple_approach = "pc.Iter(data_product).map(lambda x: make_sku(*x)).collect()"
    namedtuple_approach = "pc.Iter(data_product).map(lambda x: make_sku(*x)).collect()"
    map_star = "pc.Iter(data_product).map_star(make_sku).collect()"

    time_tuple = timeit.timeit(tuple_approach, setup=setup_tuple, number=10000)
    time_namedtuple = timeit.timeit(
        namedtuple_approach, setup=setup_namedtuple, number=10000
    )
    time_map_star = timeit.timeit(map_star, setup=setup_tuple, number=10000)

    return time_tuple, time_namedtuple, time_map_star


def benchmark_3_args() -> tuple[float, float, float]:
    """Test 3-argument function."""
    setup_tuple = """
import pyochain as pc
data_triples = [(i, i + 1, i + 2) for i in range(100)]
def process_triple(x, y, z):
    return x + y * z
"""

    setup_namedtuple = """
import pyochain as pc
from typing import NamedTuple
class Triple(NamedTuple):
    x: int
    y: int
    z: int
data_triples = [Triple(i, i + 1, i + 2) for i in range(100)]
def process_triple(x, y, z):
    return x + y * z
"""

    tuple_approach = "pc.Iter(data_triples).map(lambda x: process_triple(*x)).collect()"
    namedtuple_approach = (
        "pc.Iter(data_triples).map(lambda x: process_triple(*x)).collect()"
    )
    map_star = "pc.Iter(data_triples).map_star(process_triple).collect()"

    time_tuple = timeit.timeit(tuple_approach, setup=setup_tuple, number=10000)
    time_namedtuple = timeit.timeit(
        namedtuple_approach, setup=setup_namedtuple, number=10000
    )
    time_map_star = timeit.timeit(map_star, setup=setup_tuple, number=10000)

    return time_tuple, time_namedtuple, time_map_star


def benchmark_mixed_types() -> tuple[float, float, float]:
    """Test mixed types function."""
    setup_tuple = """
import pyochain as pc
data_records = [(i, f'item_{i}', float(i) * 1.5) for i in range(100)]
def format_record(idx, name, value):
    return f"[{idx}] {name}: {value:.2f}"
"""

    setup_namedtuple = """
import pyochain as pc
from typing import NamedTuple
class Record(NamedTuple):
    idx: int
    name: str
    value: float
data_records = [Record(i, f'item_{i}', float(i) * 1.5) for i in range(100)]
def format_record(idx, name, value):
    return f"[{idx}] {name}: {value:.2f}"
"""

    tuple_approach = "pc.Iter(data_records).map(lambda x: format_record(*x)).collect()"
    namedtuple_approach = (
        "pc.Iter(data_records).map(lambda x: format_record(*x)).collect()"
    )
    map_star = "pc.Iter(data_records).map_star(format_record).collect()"

    time_tuple = timeit.timeit(tuple_approach, setup=setup_tuple, number=10000)
    time_namedtuple = timeit.timeit(
        namedtuple_approach, setup=setup_namedtuple, number=10000
    )
    time_map_star = timeit.timeit(map_star, setup=setup_tuple, number=10000)

    return time_tuple, time_namedtuple, time_map_star


def benchmark_5_args() -> tuple[float, float, float]:
    """Test heavy computation with 5 arguments."""
    setup_tuple = """
import pyochain as pc
data_complex = [(i, i + 1, i + 2, i + 3, i + 4) for i in range(100)]
def compute_complex(a, b, c, d, e):
    return (a + b) * (c - d) + e
"""

    setup_namedtuple = """
import pyochain as pc
from typing import NamedTuple
class ComplexData(NamedTuple):
    a: int
    b: int
    c: int
    d: int
    e: int
data_complex = [ComplexData(i, i + 1, i + 2, i + 3, i + 4) for i in range(100)]
def compute_complex(a, b, c, d, e):
    return (a + b) * (c - d) + e
"""

    tuple_approach = (
        "pc.Iter(data_complex).map(lambda x: compute_complex(*x)).collect()"
    )
    namedtuple_approach = (
        "pc.Iter(data_complex).map(lambda x: compute_complex(*x)).collect()"
    )
    map_star = "pc.Iter(data_complex).map_star(compute_complex).collect()"

    time_tuple = timeit.timeit(tuple_approach, setup=setup_tuple, number=10000)
    time_namedtuple = timeit.timeit(
        namedtuple_approach, setup=setup_namedtuple, number=10000
    )
    time_map_star = timeit.timeit(map_star, setup=setup_tuple, number=10000)

    return time_tuple, time_namedtuple, time_map_star


# ============================================================================
# Run benchmarks
# ============================================================================

print("=" * 80)
print("TEST 1: 2-argument function (make_sku)")
print("=" * 80)
time_tuple, time_namedtuple, time_map_star = benchmark_2_args()
print(f"  Tuples (lambda x: func(*x))         : {time_tuple:.4f}s")
print(f"  NamedTuples (lambda x: func(*x))    : {time_namedtuple:.4f}s")
print(f"  map_star(func)                      : {time_map_star:.4f}s")
imp_tuple = ((time_tuple - time_map_star) / time_tuple) * 100
imp_namedtuple = ((time_namedtuple - time_map_star) / time_namedtuple) * 100
print(f"  📊 Tuple vs map_star                : {imp_tuple:.2f}% faster")
print(f"  📊 NamedTuple vs map_star           : {imp_namedtuple:.2f}% faster")

print("\n" + "=" * 80)
print("TEST 2: 3-argument function (process_triple)")
print("=" * 80)
time_tuple, time_namedtuple, time_map_star = benchmark_3_args()
print(f"  Tuples (lambda x: func(*x))         : {time_tuple:.4f}s")
print(f"  NamedTuples (lambda x: func(*x))    : {time_namedtuple:.4f}s")
print(f"  map_star(func)                      : {time_map_star:.4f}s")
imp_tuple = ((time_tuple - time_map_star) / time_tuple) * 100
imp_namedtuple = ((time_namedtuple - time_map_star) / time_namedtuple) * 100
print(f"  📊 Tuple vs map_star                : {imp_tuple:.2f}% faster")
print(f"  📊 NamedTuple vs map_star           : {imp_namedtuple:.2f}% faster")

print("\n" + "=" * 80)
print("TEST 3: Mixed types (format_record)")
print("=" * 80)
time_tuple, time_namedtuple, time_map_star = benchmark_mixed_types()
print(f"  Tuples (lambda x: func(*x))         : {time_tuple:.4f}s")
print(f"  NamedTuples (lambda x: func(*x))    : {time_namedtuple:.4f}s")
print(f"  map_star(func)                      : {time_map_star:.4f}s")
imp_tuple = ((time_tuple - time_map_star) / time_tuple) * 100
imp_namedtuple = ((time_namedtuple - time_map_star) / time_namedtuple) * 100
print(f"  📊 Tuple vs map_star                : {imp_tuple:.2f}% faster")
print(f"  📊 NamedTuple vs map_star           : {imp_namedtuple:.2f}% faster")

print("\n" + "=" * 80)
print("TEST 4: Heavy computation (5 arguments)")
print("=" * 80)
time_tuple, time_namedtuple, time_map_star = benchmark_5_args()
print(f"  Tuples (lambda x: func(*x))         : {time_tuple:.4f}s")
print(f"  NamedTuples (lambda x: func(*x))    : {time_namedtuple:.4f}s")
print(f"  map_star(func)                      : {time_map_star:.4f}s")
imp_tuple = ((time_tuple - time_map_star) / time_tuple) * 100
imp_namedtuple = ((time_namedtuple - time_map_star) / time_namedtuple) * 100
print(f"  📊 Tuple vs map_star                : {imp_tuple:.2f}% faster")
print(f"  📊 NamedTuple vs map_star           : {imp_namedtuple:.2f}% faster")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Re-run for final summary
t1, nt1, m1 = benchmark_2_args()
t2, nt2, m2 = benchmark_3_args()
t3, nt3, m3 = benchmark_mixed_types()
t4, nt4, m4 = benchmark_5_args()

avg_tuple = (
    (((t1 - m1) / t1) + ((t2 - m2) / t2) + ((t3 - m3) / t3) + ((t4 - m4) / t4)) / 4
) * 100
avg_namedtuple = (
    (((nt1 - m1) / nt1) + ((nt2 - m2) / nt2) + ((nt3 - m3) / nt3) + ((nt4 - m4) / nt4))
    / 4
) * 100

print("\n✅ Average improvement with map_star:")
print(f"   vs Tuples + lambda: {avg_tuple:.2f}%")
print(f"   vs NamedTuples + lambda: {avg_namedtuple:.2f}%")

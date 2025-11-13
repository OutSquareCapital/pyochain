"""Main test file for pyochain."""

import doctester as dt

from tests._performance import test_performance_iter_map

if __name__ == "__main__":
    dt.run_doctester()
    test_performance_iter_map(5)

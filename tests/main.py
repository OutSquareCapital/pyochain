import doctester as dt

from tests.performance import test_performance_iter_map

if __name__ == "__main__":
    dt.run_doctester()
    test_performance_iter_map(5)

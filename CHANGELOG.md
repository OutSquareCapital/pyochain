# Changelog

1. new PyoIterator.fold_star method
2. Moved PyoIterator.for_each in Rust implementation
3. new PyoIterator.for_each_star method
4. Splitted all_equal method into all_equal and all_equal_by (the latter accepts a key function)
5. Added Result.swap method to swap Ok and Err values
6. Added Checkable.{err_or, err_or_else} methods to convert non-empty Iterables into Err Results

import operator
from typing import assert_type

from pyochain import Option, Seq, Some, Vec


def check_constructor_call_context() -> None:
    """Constructors should be handled the same, regardless of the call context.

    Checking with `Seq` and `Vec` to ensure variance doesn't affect the result.

    Warning:
        Currently only passes with `basedpyright`
    """
    x = assert_type(Seq([1]), Seq[int])

    _vec_standard = assert_type(Vec(x), Vec[int])
    _seq_standard = assert_type(Seq(x), Seq[int])

    _vec_called = assert_type(operator.call(Vec, x), Vec[int])
    _seq_called = assert_type(operator.call(Seq, x), Seq[int])

    _vec_piped = assert_type(x.pipe(Vec), Vec[int])
    _seq_piped = assert_type(x.pipe(Seq), Seq[int])

    _seq_collected = assert_type(x.iter().collect(Seq), Seq[int])
    _vec_collected = assert_type(x.iter().collect(Vec), Vec[int])

    x_opt = assert_type(Some(x), Option[Seq[int]])
    _seq_mapped = assert_type(x_opt.map(Vec).unwrap(), Vec[int])
    _vec_mapped = assert_type(x_opt.map(Seq).unwrap(), Seq[int])

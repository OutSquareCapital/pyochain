"""Alternative implementation using TypeGuard for proper type narrowing."""

from __future__ import annotations

from pyochain import NONE, Err, Ok, Option, Result, Some


def test_result_pattern_matching() -> None:
    """Test Result pattern matching."""
    print("=== Testing Result Pattern Matching ===\n")

    def _res_ok() -> Result[int, str]:
        return Ok(42)

    def _res_err() -> Result[int, str]:
        return Err("Something went wrong")

    result_ok = _res_ok()
    match result_ok:
        case Ok(value):
            print(f"✓ Ok pattern: {value}")
        case Err(error):
            print(f"✗ Err pattern: {error}")

    result_err = _res_err()
    match result_err:
        case Ok(value):
            print(f"✗ Ok pattern: {value}")
        case Err(error):
            print(f"✓ Err pattern: {error}")

    print()


def test_option_pattern_matching() -> None:
    """Test Option pattern matching."""
    print("=== Testing Option Pattern Matching ===\n")

    def _opt_some() -> Option[str]:
        return Some("hello")

    def _opt_none() -> Option[str]:
        return NONE

    option_some = _opt_some()
    match option_some:
        case Some(value):
            print(f"✓ Some pattern: {value}")
        case _:
            print("✗ NONE pattern")

    option_none = _opt_none()
    match option_none:
        case Some(value):
            print(f"✗ Some pattern: {value}")
        case _:
            print("✓ NONE pattern")

    print()


def test_nested_pattern_matching() -> None:
    """Test nested Result and Option patterns."""
    print("=== Testing Nested Pattern Matching ===\n")

    def _make_results() -> list[Result[Option[int], str]]:
        return [
            Ok(Some(10)),
            Ok(NONE),
            Err("error occurred"),
        ]

    results = _make_results()

    for i, result in enumerate(results, 1):
        match result:
            case Ok(option):
                match option:
                    case Some(value):
                        print(f"Result {i}: ✓ Ok(Some({value}))")
                    case _:
                        print(f"Result {i}: ✓ Ok(NONE)")
            case Err(error):
                print(f"Result {i}: ✓ Err({error!r})")

    print()


def test_with_guards() -> None:
    """Test pattern matching with guards."""
    print("=== Testing Pattern Matching with Guards ===\n")

    threshold = 10
    results: list[Result[int, str]] = [
        Ok(5),
        Ok(15),
        Ok(25),
        Err("invalid"),
    ]

    for result in results:
        match result:
            case Ok(value):
                print(f"✓ Ok <= {threshold}: {value}")
            case Err(error):
                print(f"✓ Err: {error!r}")

    print()


if __name__ == "__main__":
    test_result_pattern_matching()
    test_option_pattern_matching()
    test_nested_pattern_matching()
    test_with_guards()
    print("All pattern matching tests completed!")

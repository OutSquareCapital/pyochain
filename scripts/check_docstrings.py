"""Check that all code blocks in docstrings are properly closed."""

import ast
import re
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, TypeIs

from rich.table import Table

from pyochain import (
    NONE,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    Result,
    Seq,
    Set,
    SetMut,
    Some,
    Vec,
    option,
)

from ._utils import CONSOLE, Color, Paths, show

CODE_BLOCK_PATTERN = re.compile(r"^```(\w*)", re.MULTILINE)
SKIP_DECORATORS = SetMut.from_ref({"overload", "override", "no_doctest", "wraps"})
type DocumentableNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
type MethodNode = ast.FunctionDef | ast.AsyncFunctionDef


class DocstringInfo(NamedTuple):
    """Information about a docstring."""

    name: str
    line_no: int
    content: str


class ErrorKind(StrEnum):
    """Kind of error found in a docstring."""

    MISSING = "Missing docstring"
    NO_DOCTEST = "Missing doctest: No ```python block found in docstring"
    UNCLOSED = "Unclosed ``` block"
    CLOSED_NOT_OPENED = "Closing ``` block without opening"


class DocstringError(NamedTuple):
    """Error found in a docstring."""

    file_path: Path
    func_name: str
    line_no: int
    error_line_no: int
    errors: Seq[ErrorKind]


class ErrorDetail(NamedTuple):
    """Detail of an error with its line number."""

    line_no: int
    message: ErrorKind


class State(NamedTuple):
    """State during code block traversal."""

    errors: Vec[ErrorDetail]
    stack: Vec[int]


def main() -> None:
    """Check all docstrings in the project."""
    msg = "Checking docstrings for properly closed code blocks..."
    show(msg, style=Color.INFO)

    return (
        _get_files("py")
        .iter()
        .chain(_get_files("pyi"))
        .filter_map(_check_file)
        .flatten()
        .collect()
        .then(_handle_errors)
        .unwrap_or_else(lambda: show("[OK] No issues found!", style=Color.SUCCESS))
    )


def _get_files(pattern: str) -> Seq[Path]:
    return (
        Paths.SRC_DIR
        .iter_rglob(f"*.{pattern}")
        .collect()
        .inspect(
            lambda p: show(f"Checking {p.len()} {pattern} files...", style=Color.INFO)
        )
    )


def _handle_errors(all_errors: Seq[DocstringError]) -> None:
    _show_table(all_errors)
    msg = f"[FAILED] Found {all_errors.len()} issue(s)"
    show(msg, style=Color.ERROR)


def _show_table(all_errors: Seq[DocstringError]) -> None:
    def _add_row(error: DocstringError) -> None:
        file = f"{error.file_path.relative_to(Paths.ROOT.value)}:{error.error_line_no}"
        table.add_row(
            file, error.func_name, error.errors.iter().map(lambda e: e.value).join("\n")
        )

    table: Table = Table(title="Issues Found", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Function", style="magenta")
    table.add_column("Error", style="red")
    all_errors.iter().for_each(_add_row)
    CONSOLE.print(table)


def _check_file(file_path: Path) -> Option[Iter[DocstringError]]:

    def _is_documentable(node: ast.AST) -> TypeIs[DocumentableNode]:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return NONE

    protocol_methods = _get_protocol_methods(tree)

    return Some(
        Iter(ast.walk(tree))
        .filter(_is_documentable)  # We do two filter for `TypeIs` inference
        .filter(lambda node: not _has_skip_decorator(node))
        .filter_map(lambda node: _process_node(file_path, node, protocol_methods))
    )


def _get_protocol_methods(tree: ast.Module) -> Set[int]:
    """Get line numbers of all methods inside Protocol classes.

    Returns:
        Set[int]: A set of line numbers of methods inside Protocol classes.
    """

    def _is_class_def(node: ast.AST) -> TypeIs[ast.ClassDef]:
        return isinstance(node, ast.ClassDef)

    def _is_protocol(expr: ast.expr) -> TypeIs[ast.Name | ast.Attribute]:
        return (isinstance(expr, ast.Name) and expr.id == "Protocol") or (
            isinstance(expr, ast.Attribute) and expr.attr == "Protocol"
        )

    def _is_method(node: ast.AST) -> TypeIs[MethodNode]:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    return (
        Iter(ast.walk(tree))
        .filter(_is_class_def)
        .filter(lambda node: Iter(node.bases).any(_is_protocol))
        .flat_map(lambda node: Iter(ast.walk(node)).filter(_is_method))
        .map(lambda node: node.lineno)
        .collect(Set)
    )


def _process_node(
    file_path: Path, node: DocumentableNode, protocol_methods: Set[int]
) -> Option[DocstringError]:
    match option(ast.get_docstring(node)):
        case Null() if _should_report_missing_docstring(node, protocol_methods):
            return Some(
                DocstringError(
                    file_path,
                    node.name,
                    node.lineno,
                    node.lineno,
                    Seq([ErrorKind.MISSING]),
                )
            )
        case Some(docstring):
            has_decorator = _has_skip_decorator(node)
            docstring_start_line = _get_docstring_start_line(node)
            result = _check_code_blocks(
                docstring, docstring_start_line, node.name, skip_doctest=has_decorator
            )
            match result:
                case Err(errors):
                    err = DocstringError(
                        file_path,
                        node.name,
                        node.lineno,
                        errors.first().line_no,
                        errors.iter().map(lambda e: e.message).collect(),
                    )
                    return Some(err)
                case Ok(_):
                    return NONE
        case _:
            return NONE


def _should_report_missing_docstring(
    node: DocumentableNode, protocol_methods: Set[int]
) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        not node.name.startswith("_")
        and not node.name.istitle()
        and not _has_skip_decorator(node)
        and node.lineno not in protocol_methods
    )


def _get_docstring_start_line(node: DocumentableNode) -> int:
    match node.body:
        case [ast.Expr(value=ast.Constant(value=str())), *_] as body:
            _ = body
            return node.body[0].lineno
        case _:
            return node.lineno


def _check_code_blocks(
    docstring: str, start_line: int, func_name: str, *, skip_doctest: bool = False
) -> Result[tuple[()], Vec[ErrorDetail]]:
    def _process_line(state: State, line_num: int, line: str) -> State:
        marker = "```"
        stripped_line = line.lstrip()
        match = CODE_BLOCK_PATTERN.search(stripped_line)
        if not (match and stripped_line.startswith(marker)):
            return state
        if stripped_line == marker:
            if not state.stack.is_empty():
                _ = state.stack.pop()
                return state
            state.errors.append(
                ErrorDetail(start_line + line_num, ErrorKind.CLOSED_NOT_OPENED)
            )
            return state
        state.stack.append(line_num + 1)
        return state

    lines = Vec.from_ref(docstring.split("\n"))
    state = State(Vec(()), Vec(()))
    block_errors = (
        lines
        .iter()
        .enumerate()
        .fold_star(state, _process_line)
        .errors.iter()
        .chain(
            state.stack.iter().map(
                lambda idx: ErrorDetail(start_line + idx - 1, ErrorKind.UNCLOSED)
            )
        )
        .collect(Vec)
    )
    doctest_errors = lines.into(
        _check_errs, func_name, start_line, has_no_doctest_flag=skip_doctest
    )

    match doctest_errors:
        case Ok(_) if block_errors.is_empty():
            return Ok(())
        case Ok(_):
            return Err(block_errors)
        case Err(errors):
            errors.extend(block_errors)
            return Err(errors)


def _has_skip_decorator(node: DocumentableNode) -> bool:
    return Iter(node.decorator_list).any(
        lambda d: (
            (isinstance(d, ast.Name) and d.id in SKIP_DECORATORS)
            or (isinstance(d, ast.Attribute) and d.attr in SKIP_DECORATORS)
        )
    )


def _check_errs(
    lines: Vec[str], func_name: str, start_line: int, *, has_no_doctest_flag: bool
) -> Result[tuple[()], Vec[ErrorDetail]]:
    should_skip = (
        func_name.startswith("_")
        or func_name.istitle()
        or has_no_doctest_flag
        or lines.iter().any(
            lambda line: bool(
                CODE_BLOCK_PATTERN.search(line.lstrip()) and "python" in line
            )
        )
    )
    if should_skip:
        return Ok(())
    return Err(Vec([ErrorDetail(start_line, ErrorKind.NO_DOCTEST)]))


if __name__ == "__main__":
    main()

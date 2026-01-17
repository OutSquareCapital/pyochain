"""Check that all code blocks in docstrings are properly closed."""

import ast
import re
from pathlib import Path
from typing import NamedTuple, TypeIs

import rich
import rich.table
import rich.text

import pyochain as pc

SRC_DIR = Path().joinpath("src", "pyochain")
CODE_BLOCK_PATTERN = re.compile(r"^```(\w*)", re.MULTILINE)
SKIP_DECORATORS = frozenset({"overload", "override", "no_doctest", "wraps"})


class DocstringInfo(NamedTuple):
    """Information about a docstring."""

    name: str
    line_no: int
    content: str


class DocstringError(NamedTuple):
    """Error found in a docstring."""

    file_path: Path
    func_name: str
    line_no: int
    error_line_no: int
    errors: pc.Seq[str]


class ErrorDetail(NamedTuple):
    """Detail of an error with its line number."""

    line_no: int
    message: str


class State(NamedTuple):
    """State during code block traversal."""

    errors: pc.Seq[ErrorDetail]
    stack: pc.Seq[tuple[int, str]]

    def to_blocks(self, start_line: int) -> pc.Iter[ErrorDetail]:
        """Convert unclosed blocks in the stack to error details."""
        return self.errors.iter().chain(
            self.stack.iter().map_star(
                lambda idx, lang: ErrorDetail(
                    line_no=start_line + idx - 1,
                    message=f"Unclosed ```{lang} block",
                )
            )
        )


def _check_file(file_path: Path) -> pc.Seq[DocstringError]:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return pc.Seq[DocstringError].new()

    protocol_methods = _get_protocol_methods(tree)

    return (
        pc.Iter(ast.walk(tree))
        .filter(_is_documentable)
        .filter(lambda node: not _has_skip_decorator(node))  # type: ignore[arg-type]
        .filter_map(lambda node: _process_node(file_path, node, protocol_methods))
        .collect()
    )


def _get_protocol_methods(tree: ast.Module) -> pc.Set[int]:
    """Get line numbers of all methods inside Protocol classes."""

    def _is_class_def(node: ast.AST) -> TypeIs[ast.ClassDef]:
        return isinstance(node, ast.ClassDef)

    def _is_protocol(expr: ast.expr) -> TypeIs[ast.Name | ast.Attribute]:
        return (isinstance(expr, ast.Name) and expr.id == "Protocol") or (
            isinstance(expr, ast.Attribute) and expr.attr == "Protocol"
        )

    return (
        pc.Iter(ast.walk(tree))
        .filter(_is_class_def)
        .filter(lambda node: pc.Iter(node.bases).any(_is_protocol))
        .flat_map(lambda node: pc.Iter(ast.walk(node)).filter(_is_documentable))
        .map(lambda node: node.lineno)
        .collect(pc.Set)
    )


def _is_documentable(node: ast.AST) -> TypeIs[ast.FunctionDef | ast.AsyncFunctionDef]:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _process_node(
    file_path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    protocol_methods: pc.Set[int],
) -> pc.Option[DocstringError]:
    def _is_public(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return not node.name.startswith("_") and not node.name.istitle()

    docstring = pc.Option(ast.get_docstring(node))
    if docstring.is_none():
        if (
            _is_public(node)
            and not _has_skip_decorator(node)
            and node.lineno not in protocol_methods
        ):
            return pc.Some(
                DocstringError(
                    file_path=file_path,
                    func_name=node.name,
                    line_no=node.lineno,
                    error_line_no=node.lineno,
                    errors=pc.Seq(["Missing docstring"]),
                )
            )
        return pc.NONE

    result = _check_code_blocks(
        docstring.unwrap(),
        node.lineno,
        node.name,
        skip_doctest=_has_skip_decorator(node),
    )
    match result:
        case pc.Err(errors):
            return pc.Some(
                DocstringError(
                    file_path=file_path,
                    func_name=node.name,
                    line_no=node.lineno,
                    error_line_no=errors.first().line_no,
                    errors=errors.map(lambda e: e.message).collect(),
                )
            )
        case _:
            return pc.NONE


def _has_skip_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function has a decorator that should skip docstring check."""
    return pc.Iter(node.decorator_list).any(
        lambda d: (isinstance(d, ast.Name) and d.id in SKIP_DECORATORS)
        or (isinstance(d, ast.Attribute) and d.attr in SKIP_DECORATORS)
    )


def _check_code_blocks(
    docstring: str, start_line: int, func_name: str, *, skip_doctest: bool = False
) -> pc.Result[None, pc.Iter[ErrorDetail]]:
    """Check that all code blocks in docstring are properly closed and that at least one python block exists.

    If skip_doctest is True or docstring contains @no_doctest flag, skips the python block requirement.
    """

    def _process_line(state: State, line_num: int, line: str) -> State:
        """Process a single line and update state."""
        marker = "```"
        match = CODE_BLOCK_PATTERN.search(line)
        if not (match and line.strip().startswith(marker)):
            return state
        language = match.group(1) or "plaintext"
        if line.strip() == marker:
            if not state.stack.is_empty():
                return State(
                    errors=state.errors,
                    stack=state.stack.iter().take(state.stack.length() - 1).collect(),
                )
            return State(
                errors=state.errors.iter()
                .chain(
                    [
                        ErrorDetail(
                            line_no=start_line + line_num,
                            message="Closing block ``` without matching opening",
                        )
                    ]
                )
                .collect(),
                stack=state.stack,
            )
        return State(
            errors=state.errors,
            stack=state.stack.iter().chain([(line_num + 1, language)]).collect(),
        )

    lines = pc.Vec.from_ref(docstring.split("\n"))
    all_block_errors = (
        lines.iter()
        .enumerate()
        .fold_star(State(errors=pc.Seq.new(), stack=pc.Seq.new()), _process_line)
        .to_blocks(start_line)
    )

    return (
        lines.into(
            _check_errs,
            func_name,
            start_line,
            has_no_doctest_flag=docstring.find("@no_doctest") != -1 or skip_doctest,
        )
        .map_err(lambda doctest_errors: all_block_errors.chain(doctest_errors))
        .or_else(lambda _: all_block_errors.ok_or(None).swap())
    )


def _check_errs(
    lines: pc.Vec[str],
    func_name: str,
    start_line: int,
    *,
    has_no_doctest_flag: bool,
) -> pc.Result[None, pc.Seq[ErrorDetail]]:
    should_skip = (
        func_name.startswith("_")
        or func_name.istitle()
        or has_no_doctest_flag
        or lines.any(
            lambda line: bool(CODE_BLOCK_PATTERN.search(line) and "python" in line)
        )
    )
    match should_skip:
        case True:
            return pc.Ok(None)
        case False:
            return pc.Err(
                pc.Iter.once(
                    ErrorDetail(
                        line_no=start_line,
                        message="Missing doctest: No ```python block found in docstring",
                    )
                ).collect()
            )


def main() -> None:
    """Check all docstrings in the project."""
    rich.print(
        rich.text.Text(
            "Checking docstrings for properly closed code blocks...", style="cyan bold"
        )
    )

    def _get_files(pattern: str) -> pc.Seq[Path]:
        return (
            pc.Iter(SRC_DIR.rglob(f"*.{pattern}"))
            .collect()
            .inspect(lambda p: rich.print(f"Checking {p.length()} {pattern} files..."))
        )

    all_errors = (
        _get_files("py")
        .iter()
        .chain(_get_files("pyi").iter())
        .flat_map(_check_file)
        .collect()
    )

    if all_errors.is_empty():
        rich.print(rich.text.Text("[OK] No issues found!", style="green"))
        return

    table = rich.table.Table(title="Issues Found", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Function", style="magenta")
    table.add_column("Error", style="red")
    all_errors.iter().for_each(
        lambda error: table.add_row(
            f"{error.file_path.relative_to(Path())}:{error.error_line_no}",
            error.func_name,
            error.errors.join("\n"),
        )
    )
    rich.print(table)
    rich.print(
        rich.text.Text(f"\n[FAILED] Found {all_errors.length()} issue(s)", style="red")
    )


if __name__ == "__main__":
    main()

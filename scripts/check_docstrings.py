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
SKIP_DECORATORS = frozenset({"overload", "override", "no_doctest"})


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


def _check_file(file_path: Path) -> pc.Seq[DocstringError]:
    def _is_documentable(
        node: ast.AST,
    ) -> TypeIs[ast.FunctionDef | ast.AsyncFunctionDef]:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    def _get_protocol_methods(tree: ast.Module) -> frozenset[int]:
        """Get line numbers of all methods inside Protocol classes."""
        protocol_method_lines: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_protocol = any(
                (isinstance(b, ast.Name) and b.id == "Protocol")
                or (isinstance(b, ast.Attribute) and b.attr == "Protocol")
                for b in node.bases
            )
            if is_protocol:
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        protocol_method_lines.add(child.lineno)
        return frozenset(protocol_method_lines)

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


def _process_node(
    file_path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    protocol_methods: frozenset[int],
) -> pc.Option[DocstringError]:
    def _is_public(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return not node.name.startswith("_") and not node.name.istitle()

    docstring = ast.get_docstring(node)
    if docstring is None:
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
        docstring, node.lineno, node.name, skip_doctest=_has_skip_decorator(node)
    )
    match result:
        case pc.Err(errors):
            return pc.Some(
                DocstringError(
                    file_path=file_path,
                    func_name=node.name,
                    line_no=node.lineno,
                    error_line_no=errors.first().line_no,
                    errors=errors.iter().map(lambda e: e.message).collect(),
                )
            )
        case _:
            return pc.NONE


def _has_skip_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function has a decorator that should skip docstring check."""
    return any(
        (isinstance(d, ast.Name) and d.id in SKIP_DECORATORS)
        or (isinstance(d, ast.Attribute) and d.attr in SKIP_DECORATORS)
        for d in node.decorator_list
    )


def _check_code_blocks(
    docstring: str, start_line: int, func_name: str, *, skip_doctest: bool = False
) -> pc.Result[None, pc.Seq[ErrorDetail]]:
    """Check that all code blocks in docstring are properly closed and that at least one python block exists.

    If skip_doctest is True or docstring contains @no_doctest flag, skips the python block requirement.
    """

    def _process_line(state: State, item: tuple[int, str]) -> State:
        """Process a single line and update state."""
        line_num, line = item
        match = CODE_BLOCK_PATTERN.search(line)
        if not (match and line.strip().startswith("```")):
            return state
        language = match.group(1) or "plaintext"
        if line.strip() == "```":
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
                            line_no=start_line + line_num - 1,
                            message="Closing block ``` without matching opening",
                        )
                    ]
                )
                .collect(),
                stack=state.stack,
            )
        return State(
            errors=state.errors,
            stack=state.stack.iter().chain([(line_num, language)]).collect(),
        )

    lines = pc.Vec.from_ref(docstring.split("\n"))
    final_state = (
        lines.iter()
        .enumerate()
        .fold(
            State(errors=pc.Seq([]), stack=pc.Seq([])),
            lambda state, item: _process_line(state, (item[0] + 1, item[1])),
        )
    )

    # Check for @no_doctest flag in docstring
    has_no_doctest_flag = docstring.find("@no_doctest") != -1 or skip_doctest

    all_errors = (
        final_state.errors.iter()
        .chain(
            final_state.stack.iter().map_star(
                lambda idx, lang: ErrorDetail(
                    line_no=start_line + idx - 1,
                    message=f"Unclosed ```{lang} block",
                )
            )
        )
        .chain(
            []
            if (func_name.startswith("_") or func_name.istitle() or has_no_doctest_flag)
            else (
                [
                    ErrorDetail(
                        line_no=start_line,
                        message="Missing doctest: No ```python block found in docstring",
                    )
                ]
                if not lines.any(
                    lambda line: bool(
                        CODE_BLOCK_PATTERN.search(line) and "python" in line
                    )
                )
                else []
            )
        )
        .collect()
    )
    return all_errors.then_some().map_or_else(
        default=lambda: pc.Ok(None), f=lambda _: pc.Err(all_errors)
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

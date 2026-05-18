"""Generate reference markdown files and update zensical.toml navigation."""

import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, TypeIs

from rich.console import Console
from rich.text import Text

from pyochain import Dict, Iter, Seq, Set, SetMut

type JsonData = dict[str, Any] | list[str] | str  # pyright: ignore[reportExplicitAny]
# Setup paths
DOCS_REF = Path().joinpath("docs", "reference")
ZENSICAL_PATH = Path("zensical.toml")
CONSOLE = Console()


def _generate_markdown(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


def _ensure_docs_dir() -> None:
    """Create docs/reference if it doesn't exist."""
    DOCS_REF.mkdir(parents=True, exist_ok=True)


def _discover_modules(module: ModuleType) -> Seq[ModuleType]:

    def _recurse(mod: ModuleType) -> Iterator[ModuleType]:
        yield mod
        for obj in vars(mod).values():  # pyright: ignore[reportAny]
            match obj:
                case ModuleType() if obj.__name__.startswith(mod.__name__):
                    yield from _recurse(obj)
                case _:  # pyright: ignore[reportAny]
                    return

    return Iter(_recurse(module)).collect()


def _generate_markdown_for_module(module: object, generated_paths: SetMut[str]) -> None:
    """Generate markdown files for all public classes in a module."""
    _ensure_docs_dir()

    public_api = Set(getattr(module, "__all__", []))

    def _write(path: Path, cls_name: str, cls_path: str) -> None:
        generated_paths.add(path.as_posix())
        _ = path.write_text(_generate_markdown(cls_path, cls_name), encoding="utf-8")
        CONSOLE.print(Text(f"✓ Generated {path!s}", style="green"))

    def _is_public_class(obj: tuple[str, Any]) -> TypeIs[tuple[str, type]]:  # pyright: ignore[reportExplicitAny]
        name, cls = obj  # pyright: ignore[reportAny]
        return name in public_api and isinstance(cls, type)

    return (
        Dict.from_object(module)
        .items()
        .iter()
        .filter(_is_public_class)
        .map_star(
            lambda name, cls: (
                DOCS_REF.joinpath(f"{name.lower()}.md"),
                name,
                f"{cls.__module__}.{cls.__name__}".replace("builtins", "pyochain.rs"),
            )
        )
        .filter_star(lambda k, _, _v: k.as_posix() not in generated_paths)
        .for_each_star(_write)
    )


def _check_nav_completeness(config_path: Path = ZENSICAL_PATH) -> None:
    """Check that all generated markdown files are in the navigation."""

    def _collect_paths(acc: SetMut[str], item: JsonData) -> SetMut[str]:
        match item:
            case dict():
                return Iter(item.values()).fold(acc, _collect_paths)
            case list():
                return Iter(item).fold(acc, _collect_paths)
            case str():
                acc.add(item)
                return acc

    txt = config_path.read_text(encoding="utf-8")
    item: JsonData = tomllib.loads(txt)["project"]["nav"]  # pyright: ignore[reportAny]
    nav_paths = _collect_paths(SetMut(()), item)

    return (
        Iter(DOCS_REF.glob("*.md"))
        .map(lambda p: f"reference/{p.name}")
        .filter(lambda p: p not in nav_paths)
        .collect()
        .then(
            lambda missing: CONSOLE.print(
                Text(f"⚠️  Missing files in nav:\n {missing}", style="yellow")
            )
        )
        .unwrap_or_else(
            lambda: CONSOLE.print(Text("✓ Navigation is complete!", style="green"))
        )
    )


def main() -> None:
    """Generate all reference documentation."""
    import pyochain

    generated_paths = SetMut[str].new()

    CONSOLE.print(Text("Generating pyochain documentation...", style="cyan bold"))
    _discover_modules(pyochain).iter().for_each(
        lambda module: _generate_markdown_for_module(module, generated_paths)
    )

    CONSOLE.print("\nChecking navigation completeness...")
    _check_nav_completeness()
    CONSOLE.print(Text("✓ All files generated!", style="cyan bold"))


if __name__ == "__main__":
    main()

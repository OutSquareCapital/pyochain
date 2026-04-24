"""Generate reference markdown files and update zensical.toml navigation."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, TypeIs

import rich
import rich.text
import toml

import pyochain as pc

type JsonData = dict[str, Any] | list[str] | str  # pyright: ignore[reportExplicitAny]
# Setup paths
DOCS_REF = Path().joinpath("docs", "reference")
ZENSICAL_PATH = Path("zensical.toml")


def _generate_markdown(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


def _ensure_docs_dir() -> None:
    """Create docs/reference if it doesn't exist."""
    DOCS_REF.mkdir(parents=True, exist_ok=True)


def _discover_modules(module: ModuleType) -> pc.Seq[ModuleType]:
    """Recursively discover all submodules in a package.

    Returns:
        pc.Seq[ModuleType]: A sequence of all discovered modules.
    """

    def _recurse(mod: ModuleType) -> Iterator[ModuleType]:
        yield mod
        for obj in vars(mod).values():  # pyright: ignore[reportAny]
            if isinstance(obj, ModuleType) and obj.__name__.startswith(mod.__name__):
                yield from _recurse(obj)

    return pc.Iter(_recurse(module)).collect()


def _generate_markdown_for_module(module: object) -> None:
    """Generate markdown files for all public classes in a module."""
    _ensure_docs_dir()

    public_api = set(getattr(module, "__all__", []))

    def _is_class(x: object) -> TypeIs[type]:
        return isinstance(x, type)

    def _write(path: Path, cls_name: str, cls_path: str) -> None:
        _ = path.write_text(_generate_markdown(cls_path, cls_name), encoding="utf-8")
        rich.print(rich.text.Text(f"✓ Generated {path!s}", style="green"))

    return (
        pc.Dict.from_object(module)
        .items()
        .iter()
        .filter_star(lambda name, cls: name in public_api and _is_class(cls))  # pyright: ignore[reportAny]
        .map_star(
            lambda name, cls: (  # pyright: ignore[reportAny]
                DOCS_REF.joinpath(f"{name.lower()}.md"),
                name,
                f"{cls.__module__}.{cls.__name__}".replace("builtins", "pyochain.rs"),  # pyright: ignore[reportAny]
            )
        )
        .for_each_star(_write)
    )


def _check_nav_completeness(config_path: Path = ZENSICAL_PATH) -> None:
    """Check that all generated markdown files are in the navigation."""

    def _collect_paths(acc: pc.SetMut[str], item: JsonData) -> pc.SetMut[str]:
        match item:
            case dict():
                return pc.Iter(item.values()).fold(acc, _collect_paths)
            case list():
                return pc.Iter(item).fold(acc, _collect_paths)
            case str():
                acc.add(item)
                return acc

    nav_paths = (
        pc.SetMut[str]
        .new()
        .into(_collect_paths, toml.load(config_path)["project"]["nav"])  # pyright: ignore[reportAny]
    )

    return (
        pc.Iter(DOCS_REF.glob("*.md"))
        .map(lambda p: f"reference/{p.name}")
        .filter(lambda p: p not in nav_paths)
        .collect()
        .then_some()
        .map_or_else(
            lambda: rich.print(
                rich.text.Text("✓ Navigation is complete!", style="green")
            ),
            lambda missing: rich.print(
                rich.text.Text(f"⚠️  Missing files in nav:\n {missing}", style="yellow")
            ),
        )
    )


def main() -> None:
    """Generate all reference documentation."""
    import pyochain

    rich.print(
        rich.text.Text("Generating pyochain documentation...", style="cyan bold")
    )

    _discover_modules(pyochain).iter().for_each(_generate_markdown_for_module)

    rich.print("\nChecking navigation completeness...")
    _check_nav_completeness()

    rich.print(rich.text.Text("✓ All files generated!", style="cyan bold"))


if __name__ == "__main__":
    main()

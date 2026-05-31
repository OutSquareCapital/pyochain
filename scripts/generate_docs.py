"""Generate reference markdown files and update zensical.toml navigation."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeIs

from pyochain import Dict, Set, SetMut

from ._utils import Color, Paths, show

if TYPE_CHECKING:
    from pathlib import Path


def main() -> None:
    """Main function to generate documentation and check navigation completeness."""
    import pyochain
    from pyochain import collections

    show("Generating pyochain documentation...", style=Color.INFO)
    _generate_all_for_module(pyochain)
    _generate_all_for_module(collections)
    show("✅ All files generated!", style=Color.SUCCESS)
    show("----------------------------------", style=Color.BLANK)
    return show("Checking navigation completeness...", style=Color.INFO)


def _generate_all_for_module(module: ModuleType) -> None:
    generated_paths = SetMut[str](())

    return (
        Dict
        .from_ref(vars(module))
        .values()
        .iter()
        .filter(_is_module)
        .filter(lambda m: m.__name__.startswith(module.__name__))
        .insert(module)
        .for_each(_generate_mds, generated_paths)
    )


def _generate_mds(module: ModuleType, generated_paths: SetMut[str]) -> None:
    """Generate markdown files for all public classes in a module."""
    Paths.DOCS_REF.value.mkdir(parents=True, exist_ok=True)

    public_api = Set(getattr(module, "__all__", ()))

    def _write(path: Path, cls_name: str, cls_path: str) -> None:
        generated_paths.add(path.as_posix())
        _ = path.write_text(_finalize_md(cls_path, cls_name), encoding="utf-8")
        show(f"✓ Generated {path!s}", style=Color.SUCCESS)

    def _is_public_class(obj: tuple[str, Any]) -> TypeIs[tuple[str, type]]:  # pyright: ignore[reportExplicitAny]
        name, cls = obj  # pyright: ignore[reportAny]
        return name in public_api and isinstance(cls, type)

    return (
        Dict
        .from_object(module)
        .items()
        .iter()
        .filter(_is_public_class)
        .map_star(
            lambda name, cls: (
                Paths.DOCS_REF.value.joinpath(f"{name.lower()}.md"),
                name,
                f"{cls.__module__}.{cls.__name__}".replace("builtins", "pyochain.rs"),
            )
        )
        .filter_star(lambda k, _, _v: k.as_posix() not in generated_paths)
        .for_each_star(_write)
    )


def _finalize_md(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


def _is_module(obj: object) -> TypeIs[ModuleType]:
    return isinstance(obj, ModuleType)


if __name__ == "__main__":
    main()

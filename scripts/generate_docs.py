"""Generate reference markdown files and update zensical.toml navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

from pyochain import Dict, Set, SetMut

from ._utils import Color, Paths

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType
SPECIAL_CASES = Set({
    "OptionType",
    "ResultType",
})
"""Those twos types need to be manually handled since they don't exist anywhere else than in the stubs."""


def main() -> None:
    """Main function to generate documentation and check navigation completeness."""
    import pyochain
    from pyochain import abc, collections

    Color.INFO.show("Generating pyochain documentation...")

    generated_paths = SetMut[str](())
    _generate_mds(pyochain, generated_paths)
    _generate_mds(collections, generated_paths)
    _generate_mds(abc, generated_paths)
    return Color.SUCCESS.show("✅ All files generated!")


def _generate_mds(module: ModuleType, generated_paths: SetMut[str]) -> None:
    """Generate markdown files for all public classes in a module."""
    Paths.DOCS_REF.value.mkdir(parents=True, exist_ok=True)

    public_api = Set[str](getattr(module, "__all__", ()))

    def _write(path: Path, cls_name: str, cls_path: str) -> None:
        generated_paths.add(path.as_posix())
        _ = path.write_text(_finalize_md(cls_path, cls_name), encoding="utf-8")
        Color.SUCCESS.show(f"✓ Generated {path!s}")

    def _is_public_class(obj: tuple[str, object]) -> TypeIs[tuple[str, type]]:
        name, cls = obj
        return (
            name in public_api and isinstance(cls, type) and name not in SPECIAL_CASES
        )

    return (
        Dict
        .from_object(module)
        .items()
        .iter()
        .filter(_is_public_class)
        .map_star(_fix_name)
        .filter_star(lambda k, _, _v: k.as_posix() not in generated_paths)
        .for_each_star(_write)
    )


def _fix_name(name: str, cls: type) -> tuple[Path, str, str]:
    cls_path = f"{cls.__module__}.{cls.__name__}".replace(".rs.", ".")

    return Paths.DOCS_REF.value.joinpath(f"{name.lower()}.md"), name, cls_path


def _finalize_md(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


if __name__ == "__main__":
    main()

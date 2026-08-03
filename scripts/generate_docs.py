"""Generate reference markdown files and update zensical.toml navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

from pyochain import Dict, Set

from ._utils import Color, Paths

if TYPE_CHECKING:
    from types import ModuleType


def main() -> None:
    """Main function to generate documentation and check navigation completeness."""
    import pyochain
    from pyochain import abc, collections

    Color.INFO.show("Generating pyochain documentation...")

    _generate_mds(pyochain)
    _generate_mds(collections)
    _generate_mds(abc)
    return Color.SUCCESS.show("✅ All files generated!")


def _generate_mds(module: ModuleType) -> None:
    """Generate markdown files for all public classes in a module."""
    Paths.DOCS_REF.value.mkdir(parents=True, exist_ok=True)

    public_api = Set[str](getattr(module, "__all__", ()))

    def _is_public_class(obj: tuple[str, object]) -> TypeIs[tuple[str, type]]:
        name, cls = obj
        return name in public_api and isinstance(cls, type)

    return (
        Dict
        .from_object(module)
        .items()
        .iter()
        .filter(_is_public_class)
        .for_each_star(_write)
    )


def _write(name: str, cls: type) -> None:
    cls_path = f"{cls.__module__}.{name}".replace(".rs.", ".")

    path = Paths.DOCS_REF.value.joinpath(f"{name.lower()}.md")
    old_content = path.read_text(encoding="utf-8")
    new_content = _finalize_md(cls_path, name)
    if old_content == new_content:
        Color.INFO.show(f"Skipping {path!s} (no changes)")
    else:
        _ = path.write_text(new_content, encoding="utf-8")
        Color.SUCCESS.show(f"Generated {path!s}")


def _finalize_md(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


if __name__ == "__main__":
    main()

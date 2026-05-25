"""Generate reference markdown files and update zensical.toml navigation."""

import re
from pathlib import Path
from types import ModuleType
from typing import Any, TypeIs
from urllib.parse import urlparse

from pyochain import Dict, Iter, Set, SetMut

from ._utils import Color, Paths, show

type JsonData = dict[str, Any] | list[str] | str  # pyright: ignore[reportExplicitAny]
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
DOCS_SITE_URL = "https://outsquarecapital.github.io/pyochain/"


def main() -> None:
    """Main function to generate documentation and check navigation completeness."""
    import pyochain
    from pyochain import collections

    show("Generating pyochain documentation...", style=Color.INFO)
    _generate_all_for_module(pyochain)
    _generate_all_for_module(collections)
    show("✅ All files generated!", style=Color.SUCCESS)
    show("----------------------------------", style=Color.BLANK)
    show("Checking navigation completeness...", style=Color.INFO)
    return _check_nav_completeness()


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
        .for_each(_generate_markdown_for, generated_paths)
    )


def _generate_markdown_for(module: ModuleType, generated_paths: SetMut[str]) -> None:
    """Generate markdown files for all public classes in a module."""
    Paths.DOCS_REF.value.mkdir(parents=True, exist_ok=True)

    public_api = Set(getattr(module, "__all__", ()))

    def _write(path: Path, cls_name: str, cls_path: str) -> None:
        generated_paths.add(path.as_posix())
        _ = path.write_text(_generate_markdown(cls_path, cls_name), encoding="utf-8")
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


def _generate_markdown(full_path: str, class_name: str) -> str:
    return f"""# {class_name}

::: {full_path}
"""


def _check_nav_completeness(config_path: Paths = Paths.ZENSICAL) -> None:
    import tomllib

    txt = config_path.value.read_text(encoding="utf-8")
    config = tomllib.loads(txt)
    docs_dir = config_path.value.parent.joinpath(config["project"]["docs_dir"])  # pyright: ignore[reportAny]
    docs_dir_mds = docs_dir.rglob("*.md")
    docs_ref = docs_dir.joinpath("reference").glob("*.md")
    nav_item: JsonData = config["project"]["nav"]  # pyright: ignore[reportAny]
    nav_paths = _collect_nav_paths(nav_item)

    missing = (
        Iter(docs_ref)
        .map(lambda path: path.relative_to(docs_dir).as_posix())
        .collect(Set)
        .difference(nav_paths)
        .iter()
        .join("\n")
    )
    if missing:
        msg = f"⚠️  Missing generated files in {Paths.ZENSICAL.value.as_posix()}:\n {missing}"
        show(msg, style=Color.WARNING)

    docs_paths = (
        Iter(docs_dir_mds)
        .map(lambda path: path.relative_to(docs_dir).as_posix())
        .collect(Set)
    )
    invalid_nav_paths = nav_paths.difference(docs_paths).iter().join("\n")
    if invalid_nav_paths:
        show(f"⚠️  Invalid nav links:\n {invalid_nav_paths}", style=Color.WARNING)

    invalid_markdown_links = _check_markdown_links(docs_dir)
    if invalid_markdown_links:
        msg = f"⚠️  Invalid markdown links:\n {invalid_markdown_links}"
        show(msg, style=Color.WARNING)
    if missing or invalid_nav_paths or invalid_markdown_links:
        msg = "❌ Please fix the above issues before deploying the documentation."
        return show(msg, style=Color.ERROR)
    return show("✓ Navigation is complete!", style=Color.SUCCESS)


def _collect_nav_paths(item: JsonData) -> SetMut[str]:

    def _collect_paths(acc: SetMut[str], current: JsonData) -> SetMut[str]:
        match current:
            case dict():
                return Iter(current.values()).fold(acc, _collect_paths)
            case list():
                return Iter(current).fold(acc, _collect_paths)
            case str():
                acc.add(current)
                return acc

    return _collect_paths(SetMut(()), item)


def _check_markdown_links(docs_dir: Path) -> str:

    return (
        Iter((Paths.README.value, Paths.CONTRIBUTING.value))
        .flat_map(
            lambda doc_path: Iter(
                MARKDOWN_LINK_RE.finditer(
                    FENCED_CODE_BLOCK_RE.sub("", doc_path.read_text(encoding="utf-8"))
                )
            ).map(
                lambda match: (
                    doc_path.as_posix(),
                    _normalize_link_target(match.group(1)),
                )
            )
        )
        .filter_star(
            lambda doc_path, target: (
                not _is_valid_markdown_link(Path(doc_path), target, docs_dir)
            )
        )
        .map_star(lambda doc_path, target: f"{doc_path} -> {target}")
        .join("\n")
    )


def _is_valid_markdown_link(doc_path: Path, target: str, docs_dir: Path) -> bool:
    normalized_target = _normalize_link_target(target).partition("#")[0]

    match normalized_target:
        case "":
            return True
        case _ if normalized_target.startswith(DOCS_SITE_URL):
            return _is_valid_docs_site_link(normalized_target, docs_dir)
        case _ if normalized_target.startswith(("http://", "https://")):
            return True
        case _:
            return doc_path.parent.joinpath(normalized_target).exists()


def _is_valid_docs_site_link(target: str, docs_dir: Path) -> bool:
    parsed_target = urlparse(target)
    relative_path = parsed_target.path.removeprefix("/pyochain/").strip("/")

    match relative_path:
        case "":
            return docs_dir.joinpath("index.md").exists()
        case _ if parsed_target.path.endswith("/"):
            return docs_dir.joinpath(relative_path).with_suffix(".md").exists()
        case _:
            return docs_dir.joinpath(relative_path).exists()


def _normalize_link_target(target: str) -> str:
    return target.strip().removeprefix("<").removesuffix(">")


def _is_module(obj: object) -> TypeIs[ModuleType]:
    return isinstance(obj, ModuleType)


if __name__ == "__main__":
    main()

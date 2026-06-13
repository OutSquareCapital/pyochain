"""Check that all generated documentation files are included in the navigation and that all markdown links are valid."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlparse

from pyochain import Iter, Set, SetMut

from ._utils import Color, Paths

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyochain.abc import PyoSet

FENCED_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
DOCS_SITE_URL = "https://outsquarecapital.github.io/pyochain/"

type JsonData = dict[str, JsonData] | list[str] | str


class ZensicalConfig(TypedDict):
    """TypedDict for the zensical configuration."""

    docs_dir: str
    nav: JsonData


def main(config_path: Paths = Paths.ZENSICAL) -> None:
    """Main entry point."""
    txt = config_path.value.read_text(encoding="utf-8")
    config: ZensicalConfig = tomllib.loads(txt)["project"]  # pyright: ignore[reportAny]
    docs_dir = config_path.value.parent.joinpath(config["docs_dir"])
    nav_paths = _collect_nav_paths(config["nav"])
    missing = _missing_paths(
        docs_dir.joinpath("reference").glob("*.md"), nav_paths, docs_dir
    )
    if missing:
        msg = f"⚠️  Missing generated files in {Paths.ZENSICAL.value.as_posix()}:\n {missing}"
        Color.WARNING.show(msg)
    invalid_nav_paths = _invalid_paths(docs_dir.rglob("*.md"), nav_paths, docs_dir)
    if invalid_nav_paths:
        Color.WARNING.show(f"⚠️  Invalid nav links:\n {invalid_nav_paths}")

    invalid_markdown_links = _check_markdown_links(docs_dir)
    if invalid_markdown_links:
        msg = f"⚠️  Invalid markdown links:\n {invalid_markdown_links}"
        Color.WARNING.show(msg)
    if missing or invalid_nav_paths or invalid_markdown_links:
        msg = "❌ Please fix the above issues before deploying the documentation."
        return Color.ERROR.show(msg)
    return Color.SUCCESS.show("✓ Navigation is complete!")


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


def _missing_paths(
    docs_ref: Iterator[Path], nav_paths: PyoSet[str], docs_dir: Path
) -> str:

    return (
        Iter(docs_ref)
        .map(lambda path: path.relative_to(docs_dir).as_posix())
        .collect(Set)
        .difference(nav_paths)
        .iter()
        .join("\n")
    )


def _invalid_paths(
    docs_dir_mds: Iterator[Path], nav_paths: SetMut[str], docs_dir: Path
) -> str:

    docs_paths = (
        Iter(docs_dir_mds)
        .map(lambda path: path.relative_to(docs_dir).as_posix())
        .collect(Set)
    )
    return nav_paths.difference(docs_paths).iter().join("\n")


def _check_markdown_links(docs_dir: Path) -> str:

    return (
        Iter((Paths.README.value, Paths.CONTRIBUTING.value))
        .flat_map(
            lambda doc_path: Iter(
                MARKDOWN_LINK.finditer(
                    FENCED_CODE_BLOCK.sub("", doc_path.read_text(encoding="utf-8"))
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

"""Post-process the built site to resolve mkdocstrings cross-reference tags."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from html import escape
from pathlib import Path
from typing import NamedTuple, TypeIs

import rich
import rich.text
import typer

import pyochain as pc

SITE_DIR = Path("site")
"""Default output directory produced by ``zensical build``."""

_AUTOREF_RE = re.compile(
    r"<autoref (?P<attrs>[^>]*?)>(?P<title>.*?)</autoref>",
    re.DOTALL,
)

_ATTR_RE = re.compile(
    r'(?P<key>[\w][\w-]*)(?:="(?P<value>[^"]*)")?',
)

_ID_RE = re.compile(r'\bid="([^"]+)"')

_GENERIC_STRIP_RE = re.compile(r"\[.*")
"""Strip generic type parameters, e.g. ``Iter.collect[R]`` → ``Iter.collect``."""

app = typer.Typer()


def _parse_attrs(attrs_str: str) -> dict[str, str | None]:
    """Parse a space-separated HTML attribute string into a dict.

    Returns:
        dict[str, str | None]: Mapping of attribute keys to their values (or None if no value).
    """
    return (
        pc.Iter(_ATTR_RE.finditer(attrs_str))
        .map(lambda m: (m.group("key"), m.group("value")))
        .collect(dict)
    )


def _relative_url(from_page: str, to_url: str) -> str:
    """Compute the relative URL from *from_page* to *to_url*.

    Returns:
        str: The relative URL from *from_page* to *to_url*.
    """
    split = pc.Iter(to_url.split("#", 1))
    to_url_no_anchor = split.first()

    parts_a = pc.Vec.from_ref(from_page.strip("/").split("/"))
    parts_b = pc.Vec.from_ref(to_url_no_anchor.strip("/").split("/"))

    common = (
        parts_a.iter().zip(parts_b).take_while(lambda pair: pair[0] == pair[1]).length()
    )
    up_count = parts_a.length() - common
    relative = (
        pc.Iter([".."])
        .cycle()
        .take(up_count)
        .chain(parts_b.iter().slice(start=common))
        .join("/")
    ) or "."
    return split.next().map_or_else(
        default=lambda: relative, f=lambda a: f"{relative}#{a}"
    )


def _get_page_url(html_file: Path, site_dir: Path) -> str:
    """Return the absolute-root URL for the page at *html_file*."""
    rel = html_file.parent.relative_to(site_dir)
    page_url = "/" + str(rel).replace("\\", "/").strip("/")
    return page_url + "/" if page_url != "/" else page_url


class _Anchors(NamedTuple):
    id: str
    url: str


def _file_anchors(html_file: Path, site_dir: Path) -> pc.Iter[_Anchors]:
    """Yield ``(anchor_id, absolute_url)`` pairs for all anchors in *html_file*.

    Returns:
        pc.Iter[_Anchors]: An iterator of `_Anchors` for all anchors in the file.
    """
    page_url = _get_page_url(html_file, site_dir)
    content = html_file.read_text(encoding="utf-8")
    return pc.Iter(_ID_RE.findall(content)).map(
        lambda anchor_id: _Anchors(anchor_id, f"{page_url}#{anchor_id}")  # pyright: ignore[reportAny]
    )


def _class_alias(module_path: str, name: str, obj: type) -> pc.Option[_Ids]:
    """Return ``Some(Ids(alias, canonical))`` if the class is re-exported."""
    m = obj.__module__
    canonical_mod = "pyochain.rs" if m == "builtins" else m
    canonical_id = f"{canonical_mod}.{obj.__qualname__}"
    alias_id = f"{module_path}.{name}"
    return (
        pc.NONE if alias_id == canonical_id else pc.Some(_Ids(alias_id, canonical_id))
    )


class _Ids(NamedTuple):
    alias: str
    canonical: str


def _is_alias_target(obj: object) -> TypeIs[type]:
    return isinstance(obj, type)


def _reexport_alias_pairs(anchor_map: Mapping[str, str]) -> pc.Iter[_Ids]:
    public_names = pc.traits.__all__
    mod_name = pc.traits.__name__
    return (
        pc.Iter(pc.traits.__dict__.items())
        .filter_star(lambda k, v: k in public_names and _is_alias_target(v))  # pyright: ignore[reportAny]
        .filter_map_star(lambda name, obj: _class_alias(mod_name, name, obj))  # pyright: ignore[reportAny]
        .filter(lambda ids: ids.canonical in anchor_map and ids.alias not in anchor_map)
        .map(lambda ids: _Ids(ids.alias, anchor_map[ids.canonical]))
    )


def _make_replacer(
    anchor_map: pc.Dict[str, str],
    page_url: str,
) -> Callable[[re.Match[str]], str]:
    """Return a replacement callable for ``re.sub``."""

    def _replace(match: re.Match[str]) -> str:
        attrs = _parse_attrs(match.group("attrs"))
        title: str = match.group("title")
        identifier: str = attrs.get("identifier") or ""
        optional: bool = "optional" in attrs

        stripped = _GENERIC_STRIP_RE.sub("", identifier)
        target = anchor_map.get_item(identifier).or_else(
            lambda: anchor_map.get_item(stripped)
        )
        if "." in identifier:
            parent = identifier.rsplit(".", 1)[0]
            target = target.or_else(lambda: anchor_map.get_item(parent))

        def _unresolved() -> str:
            if not optional:
                rich.print(
                    rich.text.Text(
                        f"WARNING: unresolved cross-reference in {page_url}: {identifier!r}",
                        style="yellow",
                    )
                )
            return f'<span title="{escape(identifier)}">{title}</span>'

        def _resolved(url: str) -> str:
            return (
                f'<a class="autorefs autorefs-internal" href="{escape(_relative_url(page_url, url))}">'
                f"{title}</a>"
            )

        return target.map_or_else(default=_unresolved, f=_resolved)

    return _replace


def _fix_file(html_file: Path, anchor_map: pc.Dict[str, str], site_dir: Path) -> int:
    """Rewrite *html_file* in-place, resolving all ``<autoref>`` tags.

    Returns:
        int: The number of cross-references resolved in the file.
    """
    content = html_file.read_text(encoding="utf-8")
    if "<autoref " not in content:
        return 0

    page_url = _get_page_url(html_file, site_dir)
    new_content, count = _AUTOREF_RE.subn(_make_replacer(anchor_map, page_url), content)
    if count:
        _ = html_file.write_text(new_content, encoding="utf-8")
    return count


@app.command()
def main(site_dir: Path = SITE_DIR) -> None:
    """Run the two-pass fix on every HTML file inside *site_dir*.

    Raises:
        typer.Exit: If *site_dir* does not exist or is not a directory.
    """
    if not site_dir.is_dir():
        rich.print(rich.text.Text(f"Site directory not found: {site_dir}", style="red"))
        raise typer.Exit(code=1)

    rich.print(rich.text.Text("Building anchor map…", style="cyan"))
    file_anchors: pc.Dict[str, str] = (
        pc.Iter(site_dir.rglob("index.html"))
        .sort()
        .iter()
        .flat_map(lambda f: _file_anchors(f, site_dir))
        .collect(pc.Dict)
    )
    anchor_map: pc.Dict[str, str] = (
        file_anchors.items()
        .iter()
        .chain(_reexport_alias_pairs(file_anchors))
        .collect(pc.Dict)
    )
    rich.print(f"  Found {anchor_map.length()} anchors.")

    total = (
        pc.Iter(site_dir.rglob("index.html"))
        .sort()
        .iter()
        .map(lambda f: _fix_file(f, anchor_map, site_dir))
        .sum()
    )
    rich.print(
        rich.text.Text(
            f"Resolved {total} cross-reference(s).",
            style="green",
        )
    )


if __name__ == "__main__":
    app()

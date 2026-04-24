"""Post-process the built site to resolve mkdocstrings cross-reference tags."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from html import escape
from pathlib import Path
from types import ModuleType

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
    """Parse a space-separated HTML attribute string into a dict."""
    return pc.Iter(_ATTR_RE.finditer(attrs_str)).map(
        lambda m: (m.group("key"), m.group("value"))
    ).collect(dict)


def _relative_url(from_page: str, to_url: str) -> str:
    """Compute the relative URL from *from_page* to *to_url*."""
    split = pc.Iter(to_url.split("#", 1))
    to_url_no_anchor = split.first()

    parts_a = pc.Vec.from_ref(from_page.strip("/").split("/"))
    parts_b = pc.Vec.from_ref(to_url_no_anchor.strip("/").split("/"))

    common = (
        parts_a.iter().zip(parts_b)
        .take_while(lambda pair: pair[0] == pair[1])
        .length()
    )
    up_count = parts_a.length() - common
    relative = (
        pc.Iter([".."]).cycle().take(up_count)
        .chain(parts_b.iter().slice(start=common))
        .join("/")
    ) or "."
    return split.next().map_or_else(default=lambda: relative, f=lambda a: f"{relative}#{a}")


def _get_page_url(html_file: Path, site_dir: Path) -> str:
    """Return the absolute-root URL for the page at *html_file*."""
    rel = html_file.parent.relative_to(site_dir)
    page_url = "/" + str(rel).replace("\\", "/").strip("/")
    return page_url + "/" if page_url != "/" else page_url


def _file_anchors(html_file: Path, site_dir: Path) -> pc.Iter[tuple[str, str]]:
    """Yield ``(anchor_id, absolute_url)`` pairs for all anchors in *html_file*."""
    page_url = _get_page_url(html_file, site_dir)
    content = html_file.read_text(encoding="utf-8")
    return pc.Iter(_ID_RE.findall(content)).map(
        lambda anchor_id: (anchor_id, f"{page_url}#{anchor_id}")
    )


def _class_alias(
    module_path: str, name: str, obj: type
) -> pc.Option[tuple[str, str]]:
    """Return ``Some((alias_id, canonical_id))`` if the class is re-exported."""
    canonical_mod = (
        pc.Option(getattr(obj, "__module__", None))
        .map(lambda m: "pyochain.rs" if m == "builtins" else m)
        .unwrap_or("pyochain.rs")
    )
    canonical_id = f"{canonical_mod}.{obj.__qualname__}"
    alias_id = f"{module_path}.{name}"
    return pc.NONE if alias_id == canonical_id else pc.Some((alias_id, canonical_id))


def _module_aliases(module_path: str, module: object) -> pc.Iter[tuple[str, str]]:
    """Yield ``(alias_id, canonical_id)`` pairs for re-exported classes."""
    match getattr(module, "__all__", None):
        case None:
            return pc.Iter[tuple[str, str]].new()
        case public_names:
            return (
                pc.Iter(public_names)
                .map(lambda name: (name, getattr(module, name, None)))
                .filter_star(lambda _name, obj: isinstance(obj, type))
                .map_star(lambda name, obj: _class_alias(module_path, name, obj))
                .filter_map(lambda opt: opt)
            )


def _reexport_alias_pairs(anchor_map: Mapping[str, str]) -> pc.Iter[tuple[str, str]]:
    """Yield ``(alias_id, url)`` pairs for re-exported public names not already in the map."""
    return (
        pc.Iter(vars(pc).values())
        .filter(
            lambda obj: isinstance(obj, ModuleType)
            and getattr(obj, "__name__", "").startswith("pyochain.")
        )
        .flat_map(lambda mod: _module_aliases(mod.__name__, mod))
        .filter_star(lambda _alias, canonical: canonical in anchor_map)
        .filter_star(lambda alias, _canonical: alias not in anchor_map)
        .map_star(lambda alias, canonical: (alias, anchor_map[canonical]))
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
        target = (
            anchor_map.get_item(identifier)
            .or_else(lambda: anchor_map.get_item(stripped))
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
    """Rewrite *html_file* in-place, resolving all ``<autoref>`` tags."""
    content = html_file.read_text(encoding="utf-8")
    if "<autoref " not in content:
        return 0

    page_url = _get_page_url(html_file, site_dir)
    new_content, count = _AUTOREF_RE.subn(_make_replacer(anchor_map, page_url), content)
    if count:
        html_file.write_text(new_content, encoding="utf-8")
    return count


@app.command()
def main(site_dir: Path = SITE_DIR) -> None:
    """Run the two-pass fix on every HTML file inside *site_dir*."""
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
        file_anchors.items().iter()
        .chain(_reexport_alias_pairs(file_anchors))
        .collect(pc.Dict)
    )
    rich.print(f"  Found {len(anchor_map)} anchors.")

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

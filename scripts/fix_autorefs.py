"""Post-process the built site to resolve mkdocstrings cross-reference tags."""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

import pyochain as pc
import rich
import rich.text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITE_DIR = Path("site")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a single <autoref …>…</autoref> tag (non-greedy, dot-all).
_AUTOREF_RE = re.compile(
    r"<autoref (?P<attrs>[^>]*?)>(?P<title>.*?)</autoref>",
    re.DOTALL,
)

# Matches a single HTML attribute in the form  key  or  key="value".
_ATTR_RE = re.compile(
    r'(?P<key>[\w][\w-]*)(?:="(?P<value>[^"]*)")?',
)

# Matches an anchor id attribute anywhere in the HTML source.
_ID_RE = re.compile(r'\bid="([^"]+)"')

# Strips generic type parameters from an identifier, e.g.
#   pyochain._iter.Iter.collect[R]  →  pyochain._iter.Iter.collect
_GENERIC_STRIP_RE = re.compile(r"\[.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_attrs(attrs_str: str) -> dict[str, str | None]:
    """Parse a space-separated HTML attribute string into a dict."""
    result: dict[str, str | None] = {}
    for m in _ATTR_RE.finditer(attrs_str):
        key = m.group("key")
        value = m.group("value")  # None when flag-only
        result[key] = value
    return result


def _relative_url(from_page: str, to_url: str) -> str:
    """Compute the relative URL from *from_page* to *to_url*."""
    to_url_no_anchor, *anchor_parts = to_url.split("#", 1)
    anchor = anchor_parts[0] if anchor_parts else ""

    parts_a = from_page.strip("/").split("/")
    parts_b = to_url_no_anchor.strip("/").split("/")

    while parts_a and parts_b and parts_a[0] == parts_b[0]:
        parts_a.pop(0)
        parts_b.pop(0)

    relative = "/".join([".."] * len(parts_a) + parts_b) or "."
    return f"{relative}#{anchor}" if anchor else relative


def _get_page_url(html_file: Path, site_dir: Path) -> str:
    """Return the absolute-root URL for the page at *html_file*."""
    rel = html_file.parent.relative_to(site_dir)
    page_url = "/" + str(rel).replace("\\", "/").strip("/")
    return page_url + "/" if page_url != "/" else page_url


# ---------------------------------------------------------------------------
# Pass 1 – collect anchor map
# ---------------------------------------------------------------------------


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
    canonical_mod = getattr(obj, "__module__", None) or ""
    if canonical_mod == "builtins":
        canonical_mod = "pyochain.rs"
    canonical_id = f"{canonical_mod}.{obj.__qualname__}"
    alias_id = f"{module_path}.{name}"
    return pc.NONE if alias_id == canonical_id else pc.Some((alias_id, canonical_id))


def _module_aliases(module_path: str, module: object) -> pc.Iter[tuple[str, str]]:
    """Yield ``(alias_id, canonical_id)`` pairs for re-exported classes."""
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        return pc.Iter([])
    return (
        pc.Iter(public_names)
        .map(lambda name: (name, getattr(module, name, None)))
        .filter_star(lambda _name, obj: isinstance(obj, type))
        .map_star(lambda name, obj: _class_alias(module_path, name, obj))
        .filter_map(lambda opt: opt)
    )


def _try_import_module(attr: str) -> pc.Option[tuple[str, object]]:
    """Attempt to import ``pyochain.<attr>`` and return ``Some((path, mod))``."""
    try:
        mod = importlib.import_module(f"pyochain.{attr}")
        return pc.Some((f"pyochain.{attr}", mod))
    except (ImportError, ModuleNotFoundError):
        return pc.NONE


def _add_reexport_aliases(anchor_map: dict[str, str]) -> None:
    """Extend *anchor_map* with aliases for re-exported public names."""
    (
        pc.Iter(dir(pc))
        .filter_map(_try_import_module)
        .flat_map(lambda module_info: _module_aliases(*module_info))
        .filter_star(lambda _alias, canonical: canonical in anchor_map)
        .for_each_star(
            lambda alias, canonical: anchor_map.setdefault(alias, anchor_map[canonical])
        )
    )


# ---------------------------------------------------------------------------
# Pass 2 – rewrite autoref tags
# ---------------------------------------------------------------------------


def _make_replacer(
    anchor_map: dict[str, str],
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
            pc.Option(anchor_map.get(identifier))
            .or_else(lambda: pc.Option(anchor_map.get(stripped)))
        )
        if "." in identifier:
            parent = identifier.rsplit(".", 1)[0]
            target = target.or_else(lambda: pc.Option(anchor_map.get(parent)))

        def _unresolved() -> str:
            if not optional:
                rich.print(
                    rich.text.Text(
                        f"WARNING: unresolved cross-reference in {page_url}: {identifier!r}",
                        style="yellow",
                    )
                )
            return f'<span title="{escape(identifier)}">{title}</span>'

        return target.map_or_else(
            default=_unresolved,
            f=lambda url: (
                f'<a class="autorefs autorefs-internal" href="{escape(_relative_url(page_url, url))}">'
                f"{title}</a>"
            ),
        )

    return _replace


def _fix_file(html_file: Path, anchor_map: dict[str, str], site_dir: Path) -> int:
    """Rewrite *html_file* in-place, resolving all ``<autoref>`` tags."""
    content = html_file.read_text(encoding="utf-8")
    if "<autoref " not in content:
        return 0

    page_url = _get_page_url(html_file, site_dir)
    new_content, count = _AUTOREF_RE.subn(_make_replacer(anchor_map, page_url), content)
    if count:
        html_file.write_text(new_content, encoding="utf-8")
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(site_dir: Path = SITE_DIR) -> None:
    """Run the two-pass fix on every HTML file inside *site_dir*."""
    if not site_dir.is_dir():
        rich.print(rich.text.Text(f"Site directory not found: {site_dir}", style="red"))
        sys.exit(1)

    rich.print(rich.text.Text("Building anchor map…", style="cyan"))
    anchor_map: dict[str, str] = {}
    (
        pc.Iter(site_dir.rglob("index.html"))
        .sort()
        .iter()
        .flat_map(lambda f: _file_anchors(f, site_dir))
        .for_each_star(lambda k, v: anchor_map.setdefault(k, v))
    )
    _add_reexport_aliases(anchor_map)
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
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SITE_DIR)

"""Post-process the built site to resolve mkdocstrings cross-reference tags.

Zensical does not call ``fix_refs`` from ``mkdocs-autorefs``, so the
``<autoref>`` tags emitted by mkdocstrings are left un-resolved in the
generated HTML.  This script performs a two-pass fix:

1. Walk every ``index.html`` file under ``site/`` and collect a mapping of
   anchor ``id`` → absolute-root URL (e.g.
   ``/reference/iter/#pyochain._iter.Iter``).
2. Walk every ``index.html`` file again and replace each ``<autoref …>``
   tag with either a proper ``<a>`` link (when the identifier can be
   resolved) or a ``<span title="…">`` (when the identifier is optional and
   cannot be resolved).

Usage::

    uv run scripts/fix_autorefs.py [site_dir]

The optional ``site_dir`` argument defaults to ``site``.
"""

from __future__ import annotations

import importlib
import re
import sys
from html import escape
from pathlib import Path


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
    """Parse a space-separated HTML attribute string into a dict.

    Bare flags (no value) are stored as ``{key: None}``.
    """
    result: dict[str, str | None] = {}
    for m in _ATTR_RE.finditer(attrs_str):
        key = m.group("key")
        value = m.group("value")  # None when flag-only
        result[key] = value
    return result


def _relative_url(from_page: str, to_url: str) -> str:
    """Compute the relative URL from *from_page* to *to_url*.

    Both paths must be absolute (start with ``/``) and use ``/`` separators.
    ``to_url`` may include a ``#anchor`` fragment.

    Args:
        from_page: The absolute URL of the page that contains the link.
        to_url: The absolute URL of the target (including optional anchor).

    Returns:
        A relative URL string.
    """
    to_url_no_anchor, *anchor_parts = to_url.split("#", 1)
    anchor = anchor_parts[0] if anchor_parts else ""

    parts_a = from_page.strip("/").split("/")
    parts_b = to_url_no_anchor.strip("/").split("/")

    # Remove common prefix segments.
    while parts_a and parts_b and parts_a[0] == parts_b[0]:
        parts_a.pop(0)
        parts_b.pop(0)

    ups = len(parts_a)
    relative_parts = [".."] * ups + parts_b
    relative = "/".join(relative_parts) or "."
    if anchor:
        return f"{relative}#{anchor}"
    return relative


# ---------------------------------------------------------------------------
# Pass 1 – collect anchor map
# ---------------------------------------------------------------------------


def _build_anchor_map(site_dir: Path) -> dict[str, str]:
    """Return a mapping of anchor identifier → absolute-root URL.

    Also adds aliases for re-exported names so that e.g.
    ``pyochain.traits.PyoIterator`` resolves even though the anchor in the
    HTML carries the canonical module path
    ``pyochain.traits._iterable.PyoIterator``.

    Args:
        site_dir: The root of the built site.

    Returns:
        A dict mapping anchor ``id`` values to their absolute URLs
        (e.g. ``/reference/iter/#pyochain._iter.Iter``).
    """
    anchor_map: dict[str, str] = {}

    for html_file in sorted(site_dir.rglob("index.html")):
        rel = html_file.parent.relative_to(site_dir)
        # Convert to a URL path, e.g. "reference/iter" → "/reference/iter/"
        page_url = "/" + str(rel).replace("\\", "/").strip("/")
        if page_url != "/":
            page_url += "/"

        content = html_file.read_text(encoding="utf-8")
        for anchor_id in _ID_RE.findall(content):
            full_url = f"{page_url}#{anchor_id}"
            # First registration wins (keeps stable ordering).
            anchor_map.setdefault(anchor_id, full_url)

    # Add aliases for re-exported names discovered by inspecting the package.
    _add_reexport_aliases(anchor_map)

    return anchor_map


def _add_reexport_aliases(anchor_map: dict[str, str]) -> None:
    """Extend *anchor_map* with aliases for re-exported public names.

    For every module ``M`` that re-exports a class ``C`` originally defined
    in ``M._sub`` (or ``pyochain.rs``), we add the entry
    ``M.C → anchor_map["M._sub.C"]`` so that cross-references using the
    public path are resolved correctly.

    Args:
        anchor_map: The anchor map to extend in-place.
    """
    try:
        import pyochain  # noqa: PLC0415
    except ImportError:
        return  # Can't introspect; skip silently.

    # Collect all pyochain (sub-)modules to inspect.
    modules_to_inspect: list[tuple[str, object]] = []
    for attr in dir(pyochain):
        try:
            mod = importlib.import_module(f"pyochain.{attr}")
            modules_to_inspect.append((f"pyochain.{attr}", mod))
        except (ImportError, ModuleNotFoundError):
            pass

    for module_path, module in modules_to_inspect:
        public_names = getattr(module, "__all__", None)
        if public_names is None:
            continue
        for name in public_names:
            obj = getattr(module, name, None)
            if obj is None or not isinstance(obj, type):
                continue
            canonical_mod = getattr(obj, "__module__", None)
            if canonical_mod is None:
                continue
            # Normalise: builtins → pyochain.rs (Rust extension types).
            if canonical_mod == "builtins":
                canonical_mod = "pyochain.rs"
            canonical_id = f"{canonical_mod}.{obj.__qualname__}"
            alias_id = f"{module_path}.{name}"
            if alias_id != canonical_id and canonical_id in anchor_map:
                anchor_map.setdefault(alias_id, anchor_map[canonical_id])


# ---------------------------------------------------------------------------
# Pass 2 – rewrite autoref tags
# ---------------------------------------------------------------------------


def _make_replacer(
    anchor_map: dict[str, str],
    page_url: str,
) -> re.Pattern[str]:
    """Return a replacement callable for ``re.sub``.

    Args:
        anchor_map: Mapping built by :func:`_build_anchor_map`.
        page_url: The absolute URL of the page being processed.

    Returns:
        A function suitable as the ``repl`` argument of ``re.sub``.
    """

    def _replace(match: re.Match) -> str:
        attrs = _parse_attrs(match.group("attrs"))
        title: str = match.group("title")
        identifier: str = attrs.get("identifier") or ""
        optional: bool = "optional" in attrs

        # Resolution strategy (first match wins):
        #  1. Exact identifier.
        #  2. Strip trailing generic parameters: Iter.collect[R] → Iter.collect
        #  3. Strip trailing attribute name: Err.error → Err
        target_url = anchor_map.get(identifier)
        if target_url is None:
            stripped = _GENERIC_STRIP_RE.sub("", identifier)
            target_url = anchor_map.get(stripped)
        if target_url is None and "." in identifier:
            parent = identifier.rsplit(".", 1)[0]
            target_url = anchor_map.get(parent)

        if target_url is None:
            if optional:
                # Render as a <span> with the identifier as tooltip.
                return f'<span title="{escape(identifier)}">{title}</span>'
            # Non-optional unresolved reference: leave as Markdown cross-ref.
            return f"[{title}][{identifier}]"

        rel = _relative_url(page_url, target_url)
        return (
            f'<a class="autorefs autorefs-internal" href="{escape(rel)}">'
            f"{title}</a>"
        )

    return _replace


def _fix_file(html_file: Path, anchor_map: dict[str, str], site_dir: Path) -> int:
    """Rewrite *html_file* in-place, resolving all ``<autoref>`` tags.

    Args:
        html_file: Path to the HTML file to process.
        anchor_map: Mapping built by :func:`_build_anchor_map`.
        site_dir: The root of the built site.

    Returns:
        The number of ``<autoref>`` tags that were replaced.
    """
    content = html_file.read_text(encoding="utf-8")
    if "<autoref " not in content:
        return 0

    rel = html_file.parent.relative_to(site_dir)
    page_url = "/" + str(rel).replace("\\", "/").strip("/")
    if page_url != "/":
        page_url += "/"

    replacer = _make_replacer(anchor_map, page_url)
    new_content, count = _AUTOREF_RE.subn(replacer, content)
    if count:
        html_file.write_text(new_content, encoding="utf-8")
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(site_dir: Path | None = None) -> None:
    """Run the two-pass fix on every HTML file inside *site_dir*.

    Args:
        site_dir: Root of the built site. Defaults to ``./site``.
    """
    if site_dir is None:
        site_dir = Path("site")

    if not site_dir.is_dir():
        print(f"Site directory not found: {site_dir}", file=sys.stderr)
        sys.exit(1)

    print("Building anchor map…")
    anchor_map = _build_anchor_map(site_dir)
    print(f"  Found {len(anchor_map)} anchors.")

    total = 0
    files = 0
    for html_file in sorted(site_dir.rglob("index.html")):
        replaced = _fix_file(html_file, anchor_map, site_dir)
        if replaced:
            files += 1
            total += replaced

    print(f"Resolved {total} cross-reference(s) across {files} file(s).")


if __name__ == "__main__":
    site_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(site_arg)

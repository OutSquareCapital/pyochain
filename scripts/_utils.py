from enum import Enum, StrEnum
from pathlib import Path

from rich.console import Console
from rich.text import Text

from pyochain import Iter

CONSOLE = Console()


class Paths(Enum):
    ROOT = Path()
    SRC_DIR = ROOT.joinpath("src", "pyochain")
    DOCS = ROOT.joinpath("docs")
    DOCS_REF = DOCS.joinpath("reference")
    ZENSICAL = ROOT.joinpath("zensical.toml")
    README = ROOT.joinpath("README.md")
    CONTRIBUTING = ROOT.joinpath("CONTRIBUTING.md")

    def iter_rglob(
        self,
        pattern: str,
        *,
        case_sensitive: bool | None = None,
        recurse_symlinks: bool = False,
    ) -> Iter[Path]:
        iterator = self.value.rglob(
            pattern,
            case_sensitive=case_sensitive,
            recurse_symlinks=recurse_symlinks,
        )
        return Iter(iterator)


class Color(StrEnum):
    """Enum for consistent console message styling."""

    SUCCESS = "green"
    INFO = "cyan bold"
    WARNING = "yellow"
    ERROR = "red bold"
    BLANK = "white"


def show(msg: str, style: Color) -> None:
    CONSOLE.print(Text(msg, style=style.value))

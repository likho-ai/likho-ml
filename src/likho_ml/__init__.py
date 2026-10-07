"""Likho model service: which speech models exist, how well each does, and what people's
corrections teach them."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("likho-ml")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.2.0"

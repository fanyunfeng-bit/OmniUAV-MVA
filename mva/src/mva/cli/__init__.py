"""Command-line interface for Multi-Video-Analysis.

`python -m mva ...` is the entry point. After `pip install -e .`, the
`mva` script is also on PATH (declared via [project.scripts] in
pyproject.toml).

The QueryService class in mva.cli.query is the public facade for
embedding the runtime into other applications (future web UI, agents,
notebooks). CLI subcommands are thin wrappers around it.
"""
from mva.cli.query import QueryService

__all__ = ["QueryService"]

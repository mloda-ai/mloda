"""``mloda-mcp``: serve the mloda services over MCP stdio, the same as ``mloda mcp serve``.

The ``mcp`` package is an optional extra, imported only once ``serve`` runs.
"""

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

from mloda.core.optional_dependency import require
from mloda.interfaces.cli import main as cli_main
from mloda.interfaces.service.config import Settings

INSTALL_HINT = "the mloda MCP server needs the mcp extra. Install it with: pip install 'mloda[mcp]'"


def mcp_available() -> bool:
    """Is the ``mcp`` package installed?"""
    return importlib.util.find_spec("mcp") is not None


def serve(
    settings: Settings, *, env: Mapping[str, str], transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000
) -> int:
    """Build the services for ``settings`` and serve them over stdio, or over local http, until stopped."""
    require("mcp", "the mloda MCP server")
    from mloda.interfaces.mcp import server

    if transport == "http":
        import uvicorn

        uvicorn.run(server.create_app(settings, env=env, host=host), host=host, port=port)
        return 0
    server.build_server(*cli_main.build_services(settings, env=env)).run("stdio")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run ``mloda-mcp`` and return its exit code."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    return cli_main.main(["mcp", "serve", *arguments], env=env, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    sys.exit(main())

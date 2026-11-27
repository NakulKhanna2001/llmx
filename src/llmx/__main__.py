import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmx", description="llmx provider setup and MCP server")
    parser.add_argument("command", choices=("serve", "setup"), default="serve", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "setup":
        if not sys.stdin.isatty():
            parser.exit(2, "Run llmx setup in an interactive terminal to enter provider keys.\n")
        from .wizard import run_wizard

        try:
            run_wizard()
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Setup failed: {exc}\n")
    else:
        from .server import main as serve

        asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

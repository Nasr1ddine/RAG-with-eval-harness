"""Compatibility entry point for the RAG evaluation CLI."""

from eval.runners.cli import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

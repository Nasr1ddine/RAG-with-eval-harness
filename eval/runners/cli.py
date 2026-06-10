from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from eval.datasets.loader import load_dataset
from eval.runners.runner import METRIC_KEYS, EvalRunner

REGRESSION_THRESHOLD = 0.05
DEFAULT_OUTPUT_DIR = "eval/results"


@click.group()
def cli() -> None:
    """Run and compare RAG evaluation reports."""


@cli.command()
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to a JSONL eval dataset.",
)
@click.option(
    "--api-url",
    required=True,
    type=str,
    help="Base URL for the RAG API.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=str),
    help="Directory where eval reports are written.",
)
@click.option(
    "--parallelism",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of samples to evaluate concurrently.",
)
def run(dataset_path: str, api_url: str, output_dir: str, parallelism: int) -> None:
    """Run an eval dataset against a live API."""
    dataset = load_dataset(dataset_path)
    runner = EvalRunner(
        api_base_url=api_url,
        dataset_path=dataset_path,
        output_dir=output_dir,
        parallelism=parallelism,
    )
    report = asyncio.run(runner.run_eval(dataset))
    if not report.passed:
        raise click.exceptions.Exit(1)


@cli.command()
@click.option(
    "--baseline",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Baseline eval report JSON.",
)
@click.option(
    "--current",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Current eval report JSON.",
)
def compare(baseline: str, current: str) -> None:
    """Compare two eval reports and fail when a metric regresses too far."""
    baseline_report = _load_report(baseline)
    current_report = _load_report(current)
    regressed = _print_diff_table(baseline_report, current_report)
    if regressed:
        raise click.exceptions.Exit(1)


def _load_report(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON report: {path}") from exc

    if not isinstance(payload, dict):
        raise click.ClickException(f"Report must be a JSON object: {path}")
    return payload


def _print_diff_table(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
) -> bool:
    console = Console()
    table = Table(title="RAG Eval Metric Diff")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    any_regressed = False
    for metric in METRIC_KEYS:
        report_key = f"mean_{metric}"
        baseline_value = _metric_value(baseline_report, report_key)
        current_value = _metric_value(current_report, report_key)
        delta = current_value - baseline_value
        regressed = delta < -REGRESSION_THRESHOLD
        any_regressed = any_regressed or regressed
        table.add_row(
            metric,
            f"{baseline_value:.3f}",
            f"{current_value:.3f}",
            f"{delta:+.3f}",
            "REGRESSED" if regressed else _status(delta),
        )

    console.print(table)
    return any_regressed


def _metric_value(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if not isinstance(value, int | float):
        raise click.ClickException(f"Report metric {key!r} must be numeric")
    return float(value)


def _status(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "unchanged"


if __name__ == "__main__":
    cli()

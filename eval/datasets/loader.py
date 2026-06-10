from __future__ import annotations

from pathlib import Path

from eval.datasets.schema import EvalSample


def load_dataset(path: str) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(EvalSample.from_jsonl_line(stripped))
            except ValueError as exc:
                raise ValueError(f"Invalid eval sample at {path}:{line_number}") from exc
    return samples


def save_dataset(samples: list[EvalSample], path: str) -> None:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as dataset_file:
        for sample in samples:
            dataset_file.write(f"{sample.to_jsonl_line()}\n")


def split_dataset(
    samples: list[EvalSample],
    train_ratio: float = 0.8,
) -> tuple[list[EvalSample], list[EvalSample]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0.0 and 1.0")

    split_index = int(len(samples) * train_ratio)
    return samples[:split_index], samples[split_index:]

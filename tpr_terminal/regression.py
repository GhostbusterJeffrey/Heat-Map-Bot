from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class GEXRegressionMetrics:
    matched_cells: int
    sign_accuracy: float
    weighted_absolute_error: float
    weighted_mape: float
    rank_correlation: float
    top_positive_overlap: float
    top_negative_overlap: float
    positive_max_match: bool
    negative_max_match: bool


def build_shadow_comparison(contract_records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the legacy/candidate audit columns into heatmap cells."""

    required = {"expiration", "strike", "legacy_gex", "candidate_gex"}
    missing = required.difference(contract_records.columns)
    if missing:
        raise ValueError(f"Shadow records are missing columns: {', '.join(sorted(missing))}")
    output = contract_records[
        ["expiration", "strike", "legacy_gex", "candidate_gex"]
    ].copy()
    output["expiry"] = pd.to_datetime(output["expiration"]).dt.strftime("%m/%d/%Y")
    return output.groupby(["expiry", "strike"], as_index=False)[
        ["legacy_gex", "candidate_gex"]
    ].sum()


def compare_gex_frames(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    candidate_value: str = "gex",
    reference_value: str = "gex",
    top_k: int = 5,
) -> GEXRegressionMetrics:
    """Compare two strike-by-expiry GEX datasets without over-weighting tiny cells."""

    keys = ["expiry", "strike"]
    left = _normalized_frame(candidate, candidate_value, "candidate")
    right = _normalized_frame(reference, reference_value, "reference")
    merged = left.merge(right, on=keys, how="inner")
    if merged.empty:
        raise ValueError("No matching expiry/strike cells were found for regression.")

    predicted = merged["candidate"].to_numpy(dtype=float)
    actual = merged["reference"].to_numpy(dtype=float)
    nonzero = (predicted != 0.0) | (actual != 0.0)
    sign_accuracy = float(np.mean(np.sign(predicted[nonzero]) == np.sign(actual[nonzero]))) if np.any(nonzero) else 1.0

    absolute_error = np.abs(predicted - actual)
    denominator = max(float(np.sum(np.abs(actual))), 1.0)
    weighted_absolute_error = float(np.sum(absolute_error) / denominator)

    reference_floor = max(float(np.percentile(np.abs(actual), 20)), 1.0)
    weights = np.sqrt(np.maximum(np.abs(actual), reference_floor))
    percentage_error = absolute_error / np.maximum(np.abs(actual), reference_floor)
    weighted_mape = float(np.average(percentage_error, weights=weights))

    if len(merged) > 1 and np.std(predicted) > 0 and np.std(actual) > 0:
        correlation = float(spearmanr(predicted, actual).statistic)
    else:
        correlation = 1.0 if np.allclose(predicted, actual) else 0.0
    if not np.isfinite(correlation):
        correlation = 0.0

    candidate_positive = _top_keys(merged, "candidate", positive=True, count=top_k)
    reference_positive = _top_keys(merged, "reference", positive=True, count=top_k)
    candidate_negative = _top_keys(merged, "candidate", positive=False, count=top_k)
    reference_negative = _top_keys(merged, "reference", positive=False, count=top_k)

    return GEXRegressionMetrics(
        matched_cells=len(merged),
        sign_accuracy=sign_accuracy,
        weighted_absolute_error=weighted_absolute_error,
        weighted_mape=weighted_mape,
        rank_correlation=correlation,
        top_positive_overlap=_overlap(candidate_positive, reference_positive),
        top_negative_overlap=_overlap(candidate_negative, reference_negative),
        positive_max_match=_first(candidate_positive) == _first(reference_positive),
        negative_max_match=_first(candidate_negative) == _first(reference_negative),
    )


def _normalized_frame(frame: pd.DataFrame, value_column: str, output_name: str) -> pd.DataFrame:
    if "expiry" not in frame.columns and "expiration" in frame.columns:
        frame = frame.copy()
        frame["expiry"] = pd.to_datetime(frame["expiration"]).dt.strftime("%m/%d/%Y")
    missing = {"expiry", "strike", value_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Regression input is missing columns: {', '.join(sorted(missing))}")
    result = frame[["expiry", "strike", value_column]].copy()
    result["expiry"] = result["expiry"].astype(str)
    result["strike"] = pd.to_numeric(result["strike"], errors="coerce")
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.dropna(subset=["strike", value_column])
    return (
        result.groupby(["expiry", "strike"], as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: output_name})
    )


def _top_keys(
    frame: pd.DataFrame,
    column: str,
    *,
    positive: bool,
    count: int,
) -> list[tuple[str, float]]:
    eligible = frame.loc[frame[column] > 0] if positive else frame.loc[frame[column] < 0]
    ascending = not positive
    ordered = eligible.sort_values(column, ascending=ascending).head(max(int(count), 1))
    return [(str(row.expiry), float(row.strike)) for row in ordered.itertuples()]


def _overlap(left: list[tuple[str, float]], right: list[tuple[str, float]]) -> float:
    if not left and not right:
        return 1.0
    denominator = max(min(len(left), len(right)), 1)
    return len(set(left).intersection(right)) / denominator


def _first(values: list[tuple[str, float]]) -> tuple[str, float] | None:
    return values[0] if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a TPR GEX export with a reference map.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-value", default="gex")
    parser.add_argument("--candidate-value", default="gex")
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()
    metrics = compare_gex_frames(
        pd.read_csv(arguments.candidate),
        pd.read_csv(arguments.reference),
        candidate_value=arguments.candidate_value,
        reference_value=arguments.reference_value,
        top_k=arguments.top_k,
    )
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

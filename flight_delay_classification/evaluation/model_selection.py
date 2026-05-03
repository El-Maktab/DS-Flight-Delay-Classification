"""
Author: Amir Anwar
Date: 2026-05-03

Description:
    model selection utilities
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SUPPORTED_SELECTION_METRICS = ("balanced_accuracy", "cost")


@dataclass(frozen=True)
class ModelSelectionCandidate:
    model_mode: str
    run_name: str | None
    model_path: str | None
    evaluation_report_path: str | None
    core_metrics: dict[str, float]
    cost_metrics: dict[str, float]


def resolve_holdout_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    holdout_report = report.get("holdout_report")
    if isinstance(holdout_report, Mapping):
        return holdout_report
    return report


def _require_metric_block(
    report: Mapping[str, Any], block_name: str
) -> dict[str, float]:
    block = report.get(block_name)
    if not isinstance(block, Mapping):
        raise ValueError(f"Missing '{block_name}' in evaluation payload")

    return {key: float(value) for key, value in block.items()}


def build_selection_candidate(summary: Mapping[str, Any]) -> ModelSelectionCandidate:
    report_payload = resolve_holdout_report(summary)
    core_metrics = summary.get("metrics")
    if not isinstance(core_metrics, Mapping):
        core_metrics = _require_metric_block(report_payload, "core_metrics")

    cost_metrics = summary.get("cost_metrics")
    if not isinstance(cost_metrics, Mapping):
        cost_metrics = _require_metric_block(report_payload, "cost_metrics")

    model_mode = summary.get("model_mode")
    if not isinstance(model_mode, str) or not model_mode:
        raise ValueError("Missing 'model_mode' for model selection candidate")

    run_name = summary.get("run_name")
    model_path = summary.get("model_path")
    evaluation_report_path = summary.get("evaluation_report_path")

    return ModelSelectionCandidate(
        model_mode=model_mode,
        run_name=run_name if isinstance(run_name, str) else None,
        model_path=model_path if isinstance(model_path, str) else None,
        evaluation_report_path=(
            evaluation_report_path if isinstance(evaluation_report_path, str) else None
        ),
        core_metrics={key: float(value) for key, value in core_metrics.items()},
        cost_metrics={key: float(value) for key, value in cost_metrics.items()},
    )


def load_selection_candidate(report_path: Path) -> ModelSelectionCandidate:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = resolve_holdout_report(payload)
    model_mode = payload.get("model_mode")
    if not isinstance(model_mode, str) or not model_mode:
        model_mode = report_path.stem

    return ModelSelectionCandidate(
        model_mode=model_mode,
        run_name=None,
        model_path=None,
        evaluation_report_path=str(report_path),
        core_metrics=_require_metric_block(report, "core_metrics"),
        cost_metrics=_require_metric_block(report, "cost_metrics"),
    )


def _balanced_accuracy_sort_key(candidate: ModelSelectionCandidate) -> tuple[Any, ...]:
    return (
        -candidate.core_metrics["balanced_accuracy"],
        -candidate.core_metrics["macro_f1"],
        candidate.cost_metrics["average_misclassification_cost"],
        -candidate.cost_metrics["cost_reduction_vs_on_time_baseline"],
        -candidate.core_metrics.get("weighted_f1", 0.0),
        candidate.model_mode,
    )


def _cost_sort_key(candidate: ModelSelectionCandidate) -> tuple[Any, ...]:
    return (
        candidate.cost_metrics["average_misclassification_cost"],
        -candidate.core_metrics["balanced_accuracy"],
        -candidate.core_metrics["macro_f1"],
        -candidate.cost_metrics["cost_reduction_vs_on_time_baseline"],
        -candidate.core_metrics.get("weighted_f1", 0.0),
        candidate.model_mode,
    )


def rank_model_candidates(
    candidates: Iterable[ModelSelectionCandidate],
    primary_metric: str = "balanced_accuracy",
) -> list[ModelSelectionCandidate]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one model selection candidate is required")

    if primary_metric == "balanced_accuracy":
        return sorted(candidate_list, key=_balanced_accuracy_sort_key)

    if primary_metric == "cost":
        return sorted(candidate_list, key=_cost_sort_key)

    raise ValueError(
        f"Unsupported primary_metric '{primary_metric}'. Use one of: {SUPPORTED_SELECTION_METRICS}"
    )


def _selection_policy(primary_metric: str) -> dict[str, Any]:
    if primary_metric == "cost":
        tie_breakers = [
            "balanced_accuracy",
            "macro_f1",
            "cost_reduction_vs_on_time_baseline",
            "weighted_f1",
            "model_mode",
        ]
    else:
        tie_breakers = [
            "macro_f1",
            "average_misclassification_cost",
            "cost_reduction_vs_on_time_baseline",
            "weighted_f1",
            "model_mode",
        ]

    return {
        "evaluation_scope": "holdout_only",
        "primary_metric": primary_metric,
        "tie_breakers": tie_breakers,
    }


def _serialize_candidate(
    candidate: ModelSelectionCandidate,
    rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "model_mode": candidate.model_mode,
        "run_name": candidate.run_name,
        "model_path": candidate.model_path,
        "evaluation_report_path": candidate.evaluation_report_path,
        "core_metrics": candidate.core_metrics,
        "cost_metrics": candidate.cost_metrics,
    }


def build_model_selection_report(
    summaries: Iterable[Mapping[str, Any]],
    primary_metric: str = "balanced_accuracy",
) -> dict[str, Any]:
    ranked_candidates = rank_model_candidates(
        [build_selection_candidate(summary) for summary in summaries],
        primary_metric=primary_metric,
    )
    best_candidate = ranked_candidates[0]

    return {
        "selection_policy": _selection_policy(primary_metric),
        "best_model": _serialize_candidate(best_candidate, rank=1),
        "ranking": [
            _serialize_candidate(candidate, rank=index)
            for index, candidate in enumerate(ranked_candidates, start=1)
        ],
    }


def write_model_selection_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

import h3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (
    ACCIDENTS_CLEAN_PATH,
    CANDIDATE_CELLS_PATH,
    DEFAULT_EVAL_YEAR,
    DEFAULT_TRAIN_YEARS,
    H3_RESOLUTION,
    MODEL_BUNDLE_PATH,
    NEGATIVE_RATIO,
    RANDOM_SEED,
    current_month,
    dow_sin_cos,
    ensure_dirs,
    hour_sin_cos,
    month_sin_cos,
)


@dataclass
class FeatureContext:
    lat_by_cell: dict[str, float]
    lon_by_cell: dict[str, float]
    total_by_cell: dict[str, int]
    hour_by_cell: dict[tuple[str, int], int]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ACCIDENTS_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"missing {ACCIDENTS_CLEAN_PATH}; run build_dataset.py first"
        )
    if not CANDIDATE_CELLS_PATH.exists():
        raise FileNotFoundError(
            f"missing {CANDIDATE_CELLS_PATH}; run build_dataset.py first"
        )
    events = pd.read_csv(ACCIDENTS_CLEAN_PATH)
    cells = pd.read_csv(CANDIDATE_CELLS_PATH)
    return events, cells


def build_context(history: pd.DataFrame, candidate_cells: pd.DataFrame) -> FeatureContext:
    total_by_cell = history.groupby("cell_id").size().astype(int).to_dict()
    hour_by_cell = (
        history.groupby(["cell_id", "hour_of_week"]).size().astype(int).to_dict()
    )
    return FeatureContext(
        lat_by_cell=candidate_cells.set_index("cell_id")["center_lat"].to_dict(),
        lon_by_cell=candidate_cells.set_index("cell_id")["center_lon"].to_dict(),
        total_by_cell=total_by_cell,
        hour_by_cell=hour_by_cell,
    )


def feature_matrix(
    cells: np.ndarray,
    hour_of_week: np.ndarray,
    month: np.ndarray,
    context: FeatureContext,
) -> np.ndarray:
    lats = np.array([context.lat_by_cell.get(cell, 0.0) for cell in cells], dtype=np.float32)
    lons = np.array([context.lon_by_cell.get(cell, 0.0) for cell in cells], dtype=np.float32)
    totals = np.array([context.total_by_cell.get(cell, 0) for cell in cells], dtype=np.float32)
    same_hour = np.array(
        [
            context.hour_by_cell.get((cell, int(frame_idx)), 0)
            for cell, frame_idx in zip(cells, hour_of_week, strict=False)
        ],
        dtype=np.float32,
    )

    dow = hour_of_week // 24 + 1
    hour = hour_of_week % 24

    hour_sin = np.empty(len(hour), dtype=np.float32)
    hour_cos = np.empty(len(hour), dtype=np.float32)
    dow_s = np.empty(len(hour), dtype=np.float32)
    dow_c = np.empty(len(hour), dtype=np.float32)
    month_s = np.empty(len(hour), dtype=np.float32)
    month_c = np.empty(len(hour), dtype=np.float32)
    for idx in range(len(hour)):
        hour_sin[idx], hour_cos[idx] = hour_sin_cos(int(hour[idx]))
        dow_s[idx], dow_c[idx] = dow_sin_cos(int(dow[idx]))
        month_s[idx], month_c[idx] = month_sin_cos(int(month[idx]))

    return np.column_stack(
        [
            lats,
            lons,
            hour_sin,
            hour_cos,
            dow_s,
            dow_c,
            month_s,
            month_c,
            np.log1p(totals),
            np.log1p(same_hour),
            np.divide(same_hour, np.maximum(totals, 1.0), dtype=np.float32),
        ]
    ).astype(np.float32)


def sample_negatives(
    year: int,
    count: int,
    candidate_cells: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_values = candidate_cells["cell_id"].to_numpy()
    sampled_cells = rng.choice(cell_values, size=count, replace=True)
    sampled_hours = rng.integers(0, 24 * 7, size=count, endpoint=False)
    sampled_months = rng.integers(1, 13, size=count, endpoint=False)
    _ = year
    return sampled_cells, sampled_hours.astype(np.int16), sampled_months.astype(np.int8)


def build_split(
    events: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    years: list[int],
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for year in years:
        positives = events.loc[events["year"] == year, ["cell_id", "hour_of_week", "month"]]
        history = events.loc[events["year"] < year]
        context = build_context(history, candidate_cells)

        pos_cells = positives["cell_id"].to_numpy()
        pos_hours = positives["hour_of_week"].to_numpy(dtype=np.int16)
        pos_months = positives["month"].to_numpy(dtype=np.int8)
        x_pos = feature_matrix(pos_cells, pos_hours, pos_months, context)
        y_pos = np.ones(len(x_pos), dtype=np.int8)

        neg_count = len(x_pos) * negative_ratio
        neg_cells, neg_hours, neg_months = sample_negatives(
            year=year,
            count=neg_count,
            candidate_cells=candidate_cells,
            rng=rng,
        )
        x_neg = feature_matrix(neg_cells, neg_hours, neg_months, context)
        y_neg = np.zeros(len(x_neg), dtype=np.int8)

        feature_chunks.extend([x_pos, x_neg])
        label_chunks.extend([y_pos, y_neg])
        print(f"year={year} positives={len(x_pos)} negatives={len(x_neg)}")

    return np.vstack(feature_chunks), np.concatenate(label_chunks)


def build_bundle(
    model: HistGradientBoostingClassifier,
    events: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    metrics: dict[str, float],
    train_years: list[int],
    eval_years: list[int],
) -> dict[str, object]:
    full_context = build_context(events, candidate_cells)
    candidate_ids = candidate_cells["cell_id"].tolist()
    cell_total_counts = np.array(
        [full_context.total_by_cell.get(cell, 0) for cell in candidate_ids],
        dtype=np.float32,
    )
    cell_hour_counts = np.zeros((len(candidate_ids), 24 * 7), dtype=np.float32)
    for idx, cell in enumerate(candidate_ids):
        for frame_idx in range(24 * 7):
            cell_hour_counts[idx, frame_idx] = full_context.hour_by_cell.get(
                (cell, frame_idx), 0
            )

    month = current_month()
    current_frames = np.repeat(np.arange(24 * 7, dtype=np.int16), len(candidate_ids))
    current_cells = np.tile(np.array(candidate_ids, dtype=object), 24 * 7)
    current_months = np.full(len(current_cells), month, dtype=np.int8)
    predicted = model.predict_proba(
        feature_matrix(current_cells, current_frames, current_months, full_context)
    )[:, 1]
    quantiles = np.quantile(predicted, [0.50, 0.80, 0.95]).astype(float).tolist()

    return {
        "model": model,
        "model_version": "0.1.0",
        "resolution": H3_RESOLUTION,
        "train_years": train_years,
        "eval_years": eval_years,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_cells": candidate_ids,
        "candidate_lats": candidate_cells["center_lat"].to_numpy(dtype=np.float32),
        "candidate_lons": candidate_cells["center_lon"].to_numpy(dtype=np.float32),
        "cell_total_counts": cell_total_counts,
        "cell_hour_counts": cell_hour_counts,
        "metrics": metrics,
        "risk_quantiles": quantiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_YEARS,
    )
    parser.add_argument(
        "--eval-years",
        nargs="+",
        type=int,
        default=[DEFAULT_EVAL_YEAR],
    )
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    events, candidate_cells = load_inputs()

    x_train, y_train = build_split(
        events=events,
        candidate_cells=candidate_cells,
        years=args.train_years,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )
    x_eval, y_eval = build_split(
        events=events,
        candidate_cells=candidate_cells,
        years=args.eval_years,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=220,
        min_samples_leaf=40,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    eval_prob = model.predict_proba(x_eval)[:, 1]
    metrics = {
        "val_roc_auc": float(roc_auc_score(y_eval, eval_prob)),
        "val_average_precision": float(average_precision_score(y_eval, eval_prob)),
    }
    print(metrics)

    bundle = build_bundle(
        model=model,
        events=events,
        candidate_cells=candidate_cells,
        metrics=metrics,
        train_years=args.train_years,
        eval_years=args.eval_years,
    )
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"wrote {MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    main()

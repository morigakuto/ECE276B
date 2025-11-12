#!/usr/bin/env python3
"""
Batch evaluation script for the motion planners in Planner.py.

The script runs each selected planner on a set of benchmark maps,
captures runtime/performance statistics, and saves path visualizations.

Example usage (run from repository root):
  python -m starter_code.run_all_planners
  python -m starter_code.run_all_planners --planners MyAStarPlanner MyRRTStarPlanner
  python -m starter_code.run_all_planners --maps maze window --output-root ./my_reports
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import Planner  # noqa: E402
import main  # noqa: E402

plt.ioff()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASE_DIR / "reports"

PLANNERS: Dict[str, type] = {
    "MyPlanner": Planner.MyPlanner,
    "MyPart1SamplingPlanner": Planner.MyPart1SamplingPlanner,
    "MyPart1AABBPlanner": Planner.MyPart1AABBPlanner,
    "MyAStarPlanner": Planner.MyAStarPlanner,
    "MyRRTPlanner": Planner.MyRRTPlanner,
    "MyRRTStarPlanner": Planner.MyRRTStarPlanner,
}

MAP_SPECS: Dict[str, Dict[str, object]] = {
    "single_cube": {
        "map_file": "maps/single_cube.txt",
        "start": [7.0, 7.0, 5.5],
        "goal": [2.3, 2.3, 1.3],
    },
    "maze": {
        "map_file": "maps/maze.txt",
        "start": [0.0, 0.0, 1.0],
        "goal": [12.0, 12.0, 5.0],
    },
    "flappy_bird": {
        "map_file": "maps/flappy_bird.txt",
        "start": [0.5, 4.5, 5.5],
        "goal": [19.5, 1.5, 1.5],
    },
    "pillars": {
        "map_file": "maps/pillars.txt",
        "start": [0.5, 0.5, 0.5],
        "goal": [19.0, 19.0, 9.0],
    },
    "window": {
        "map_file": "maps/window.txt",
        "start": [6.0, -4.9, 2.8],
        "goal": [2.0, 19.5, 5.5],
    },
    "tower": {
        "map_file": "maps/tower.txt",
        "start": [4.0, 2.5, 19.5],
        "goal": [2.5, 4.0, 0.5],
    },
    "room": {
        "map_file": "maps/room.txt",
        "start": [1.0, 5.0, 1.5],
        "goal": [9.0, 7.0, 1.5],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for Planner.py implementations.")
    parser.add_argument(
        "--planners",
        nargs="*",
        default=None,
        help="Subset of planners to run (default: all planners).",
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        default=None,
        help="Subset of maps to evaluate (default: all maps).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for all generated artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional timestamp label for the run directory (default: auto-generated).",
    )
    return parser.parse_args()


def resolve_selection(options: Dict[str, object], requested: Optional[Iterable[str]], kind: str) -> List[Tuple[str, object]]:
    if requested is None:
        return list(options.items())

    requested_list = list(requested)
    if not requested_list:
        return list(options.items())

    selected = []
    for name in requested_list:
        if name not in options:
            raise KeyError(f"Unknown {kind} '{name}'. Available: {', '.join(options.keys())}")
        selected.append((name, options[name]))
    return selected


def evaluate_planner(
    planner_name: str,
    planner_cls: type,
    map_name: str,
    map_spec: Dict[str, object],
) -> Tuple[Dict[str, object], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Runs a planner on a single map and returns metrics, the resulting path (if any),
    and the environment arrays (boundary, blocks).
    """
    map_path = (BASE_DIR / map_spec["map_file"]).resolve()
    boundary, blocks = main.load_map(str(map_path))

    record: Dict[str, object] = {
        "planner": planner_name,
        "map": map_name,
        "map_file": str(map_path),
        "start": list(map_spec["start"]),
        "goal": list(map_spec["goal"]),
        "run_time_sec": None,
        "path_length": None,
        "num_waypoints": 0,
        "nodes_considered": None,  # Add node count tracking
        "nodes_expanded": None,     # For A* specifically
        "iterations": None,         # For RRT/RRT*
        "goal_reached": False,
        "collision": False,
        "success": False,
        "error": None,
    }

    start = np.asarray(map_spec["start"], dtype=float)
    goal = np.asarray(map_spec["goal"], dtype=float)

    try:
        planner = planner_cls(boundary, blocks)
    except Exception as exc:
        record["error"] = f"initialization failed: {exc}"
        return record, None, boundary, blocks

    t0 = time.perf_counter()
    try:
        path = planner.plan(start.copy(), goal.copy())
    except Exception as exc:
        record["error"] = f"plan() raised: {exc}"
        record["run_time_sec"] = round(time.perf_counter() - t0, 6)
        return record, None, boundary, blocks

    elapsed = time.perf_counter() - t0
    record["run_time_sec"] = round(elapsed, 6)

    # Extract node statistics if available
    if hasattr(planner, 'stats') and planner.stats:
        if 'nodes_considered' in planner.stats:
            record["nodes_considered"] = planner.stats['nodes_considered']
        if 'nodes_expanded' in planner.stats:
            record["nodes_expanded"] = planner.stats['nodes_expanded']
        if 'iterations' in planner.stats:
            record["iterations"] = planner.stats['iterations']

    if path is None:
        record["error"] = "plan() returned None"
        return record, None, boundary, blocks

    path = np.asarray(path, dtype=float)
    if path.ndim == 1:
        if path.size == 3:
            path = path.reshape(1, 3)
        else:
            record["error"] = f"unexpected path shape {path.shape}"
            return record, None, boundary, blocks
    if path.ndim != 2 or path.shape[1] != 3:
        record["error"] = f"unexpected path shape {path.shape}"
        return record, None, boundary, blocks

    record["num_waypoints"] = int(path.shape[0])

    if path.shape[0] >= 2:
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        record["path_length"] = float(np.sum(segment_lengths))
    else:
        record["path_length"] = 0.0

    collision = False
    for i in range(path.shape[0] - 1):
        for block in blocks:
            if Planner.segment_block_collision(path[i], path[i + 1], block):
                collision = True
                break
        if collision:
            break

    goal_reached = bool(np.sum((path[-1] - goal) ** 2) <= 0.1)
    record["collision"] = collision
    record["goal_reached"] = goal_reached
    record["success"] = bool(goal_reached and not collision)

    return record, path, boundary, blocks


def save_figure(
    dest: Path,
    boundary: np.ndarray,
    blocks: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    path: Optional[np.ndarray],
    title: str,
) -> None:
    fig, ax, _, _, _ = main.draw_map(boundary, blocks, start, goal)
    ax.set_title(title)
    if path is not None and path.size > 0:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-", linewidth=2.0, label="Path")
        ax.legend(loc="best")
    fig.savefig(dest, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(run_dir: Path, summary: List[Dict[str, object]]) -> None:
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    fieldnames = [
        "planner",
        "map",
        "success",
        "run_time_sec",
        "path_length",
        "num_waypoints",
        "nodes_considered",  # Add new fields
        "nodes_expanded",
        "iterations",
        "goal_reached",
        "collision",
        "path_file",
        "figure_file",
        "metrics_file",
        "error",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary:
            row = {key: record.get(key, "") for key in fieldnames}
            writer.writerow(row)


def main_cli() -> None:
    args = parse_args()

    try:
        selected_planners = resolve_selection(PLANNERS, args.planners, "planner")
        selected_maps = resolve_selection(MAP_SPECS, args.maps, "map")
    except KeyError as exc:
        raise SystemExit(str(exc))

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, object]] = []

    for planner_name, planner_cls in selected_planners:
        planner_dir = run_dir / planner_name
        planner_dir.mkdir(parents=True, exist_ok=True)

        for map_name, map_spec in selected_maps:
            print(f"[RUN] Planner={planner_name} Map={map_name}")
            record, path, boundary, blocks = evaluate_planner(planner_name, planner_cls, map_name, map_spec)
            start = np.asarray(map_spec["start"], dtype=float)
            goal = np.asarray(map_spec["goal"], dtype=float)

            figure_path = planner_dir / f"{map_name}.png"
            save_figure(
                figure_path,
                boundary,
                blocks,
                start,
                goal,
                path,
                f"{planner_name} on {map_name}",
            )

            path_file: Optional[Path] = None
            if path is not None and record.get("error") is None:
                path_file = planner_dir / f"{map_name}_path.csv"
                np.savetxt(path_file, path, delimiter=",", header="x,y,z", comments="")
                record["path_file"] = str(path_file.relative_to(run_dir))
            else:
                record["path_file"] = None

            record["figure_file"] = str(figure_path.relative_to(run_dir))

            metrics_path = planner_dir / f"{map_name}_metrics.json"
            record["metrics_file"] = str(metrics_path.relative_to(run_dir))
            with metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)

            summary_records.append(record)

    write_summary(run_dir, summary_records)
    print(f"\nArtifacts saved to: {run_dir}")
    print("Summary files:")
    print(f"  - {run_dir / 'summary.json'}")
    print(f"  - {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main_cli()

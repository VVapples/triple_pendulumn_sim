"""Parallel 3D initial-angle-space visualization for a triple pendulum."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import product
from pathlib import Path
import time
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from triple_pendulum import TriplePendulumParameters, TriplePendulumSimulator


def _simulate_one(
    task: tuple[int, tuple[float, float, float], float, float, float | None]
) -> tuple[int, tuple[float, float, float], np.ndarray, float]:
    task_index, angles, t_max, dt, max_job_seconds = task
    started_at = time.perf_counter()
    simulator = TriplePendulumSimulator()
    final_angles = simulator.simulate_final(
        angles, t_max=t_max, dt=dt, max_seconds=max_job_seconds
    )
    return task_index, angles, final_angles, time.perf_counter() - started_at


def build_chaos_volume(
    grid_size: int = 5,
    angle_min: float = -np.pi,
    angle_max: float = np.pi,
    t_max: float = 3.0,
    dt: float = 0.03,
    workers: int | None = None,
    checkpoint: str | Path | None = "triple_pendulum_checkpoint.npz",
    max_job_seconds: float | None = None,
    failures_file: str | Path = "triple_pendulum_failures.json",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the initial-angle grid in parallel and compute six-neighbor chaos."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3 for neighbor differences.")
    started_at = time.perf_counter()
    axis = np.linspace(angle_min, angle_max, grid_size)
    angle_grid = np.array(list(product(axis, axis, axis)), dtype=float)
    tasks = [
        (index, tuple(point), t_max, dt, max_job_seconds)
        for index, point in enumerate(angle_grid)
    ]
    total = len(tasks)
    final_angles = np.full((grid_size, grid_size, grid_size, 3), np.nan)
    completed_mask = np.zeros(total, dtype=bool)
    failures: dict[str, str] = {}
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_candidates = [
            path for path in (
                checkpoint_path,
                Path(f"{checkpoint_path}.tmp.npz"),
            )
            if path.exists()
        ]
        if checkpoint_candidates:
            checkpoint_to_load = max(
                checkpoint_candidates,
                key=lambda path: path.stat().st_mtime,
            )
            saved = np.load(checkpoint_to_load, allow_pickle=False)
            matches = (
                int(saved["grid_size"]) == grid_size
                and np.isclose(float(saved["angle_min"]), angle_min)
                and np.isclose(float(saved["angle_max"]), angle_max)
                and np.isclose(float(saved["t_max"]), t_max)
                and np.isclose(float(saved["dt"]), dt)
            )
            if matches:
                final_angles = saved["final_angles"]
                completed_mask = saved["completed_mask"]
                print(
                    f"[resume] Loaded {int(completed_mask.sum())}/{total} jobs "
                    f"from {checkpoint_to_load}",
                    flush=True,
                )
    pending_tasks = [task for task in tasks if not completed_mask[task[0]]]
    print(
        f"[1/4] Preparing {total} initial conditions "
        f"(grid={grid_size}^3, t_max={t_max:g}, dt={dt:g}, "
        f"pending={len(pending_tasks)})",
        flush=True,
    )

    if workers == 1:
        results = []
        for task in pending_tasks:
            try:
                result = _simulate_one(task)
            except Exception as exc:
                failures[str(task[0])] = f"{type(exc).__name__}: {exc}"
                print(f"\n[fail] task #{task[0]} angles={task[1]}: {exc}", flush=True)
                continue
            results.append(result[1:3])
            completed_mask[result[0]] = True
            _store_result(final_angles, axis, result[1], result[2])
            _save_checkpoint(
                checkpoint, final_angles, completed_mask,
                grid_size, angle_min, angle_max, t_max, dt
            )
            print(f" (last job {result[3]:.1f} s)", flush=True)
            _print_progress(
                "[2/4] Simulating", int(completed_mask.sum()), total, started_at
            )
    else:
        print(
            f"[2/4] Simulating in parallel "
            f"(workers={workers or 'default'})...",
            flush=True,
        )
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(_simulate_one, task): task for task in pending_tasks
            }
            pending = set(future_to_task)
            completed = int(completed_mask.sum())
            last_heartbeat = time.perf_counter()
            while pending:
                done, pending = wait(
                    pending,
                    timeout=10.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    now = time.perf_counter()
                    oldest_task = min(
                        future_to_task[future][0] for future in pending
                    )
                    print(
                        f"\n[2/4] Still working: {completed}/{total} complete, "
                        f"{len(pending)} pending, "
                        f"waiting {now - last_heartbeat:.0f} s since last update "
                        f"(oldest pending task #{oldest_task})",
                        flush=True,
                    )
                    last_heartbeat = now
                    continue
                for future in done:
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        angles = task[1]
                        failures[str(task[0])] = f"{type(exc).__name__}: {exc}"
                        print(
                            f"\n[fail] task #{task[0]} angles={angles}: {exc}",
                            flush=True,
                        )
                        continue
                    results.append(result[1:3])
                    completed_mask[result[0]] = True
                    _store_result(final_angles, axis, result[1], result[2])
                    _save_checkpoint(
                        checkpoint, final_angles, completed_mask,
                        grid_size, angle_min, angle_max, t_max, dt
                    )
                    completed += 1
                    _print_progress(
                        "[2/4] Simulating", completed, total, started_at
                    )
                    print(
                        f" last job #{result[0]} took {result[3]:.1f} s",
                        flush=True,
                    )

    if failures:
        Path(failures_file).write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[fail] Wrote failures to {failures_file}", flush=True)
    if not np.all(completed_mask):
        raise RuntimeError(
            f"{int((~completed_mask).sum())} simulations did not complete. "
            "Rerun with the same checkpoint after fixing the failures."
        )

    print("[3/4] Computing six-neighbor chaos values...", flush=True)

    chaos = np.full((grid_size, grid_size, grid_size), np.nan, dtype=float)
    for i, j, k in product(range(1, grid_size - 1), repeat=3):
        center = final_angles[i, j, k]
        neighbor_values = [
            final_angles[i - 1, j, k],
            final_angles[i + 1, j, k],
            final_angles[i, j - 1, k],
            final_angles[i, j + 1, k],
            final_angles[i, j, k - 1],
            final_angles[i, j, k + 1],
        ]
        chaos[i, j, k] = np.mean(
            [
                np.linalg.norm(
                    TriplePendulumSimulator.wrapped_angle_difference(
                        neighbor, center
                    )
                )
                for neighbor in neighbor_values
            ]
        )
    print(
        f"[4/4] Finished volume computation in "
        f"{time.perf_counter() - started_at:.1f} s",
        flush=True,
    )
    return angle_grid, final_angles, chaos


def _store_result(
    final_angles: np.ndarray,
    axis: np.ndarray,
    point: tuple[float, float, float],
    final: np.ndarray,
) -> None:
    indices = tuple(int(np.argmin(np.abs(axis - value))) for value in point)
    final_angles[indices] = final


def _save_checkpoint(
    checkpoint: str | Path | None,
    final_angles: np.ndarray,
    completed_mask: np.ndarray,
    grid_size: int,
    angle_min: float,
    angle_max: float,
    t_max: float,
    dt: float,
) -> None:
    if checkpoint is None:
        return
    checkpoint_path = Path(checkpoint)
    temporary_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.tmp"
    )
    np.savez(
        temporary_path,
        final_angles=final_angles,
        completed_mask=completed_mask,
        grid_size=grid_size,
        angle_min=angle_min,
        angle_max=angle_max,
        t_max=t_max,
        dt=dt,
    )
    generated_path = Path(f"{temporary_path}.npz")
    generated_path.replace(checkpoint_path)


def _print_progress(
    label: str, completed: int, total: int, started_at: float
) -> None:
    """Print a single-line progress update without buffering."""
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 * completed / total
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else float("inf")
    if np.isfinite(remaining):
        eta = f", ETA {remaining:.0f} s"
    else:
        eta = ""
    print(
        f"\r{label}: {completed}/{total} ({percentage:5.1f}%), "
        f"elapsed {elapsed:.0f} s{eta}",
        end="",
        flush=True,
    )
    if completed == total:
        print(flush=True)


def plot_chaos_volume(
    angle_grid: np.ndarray,
    chaos: np.ndarray,
    output: str | Path = "triple_pendulum_chaos.png",
) -> None:
    """Plot finite-time sensitivity as opacity and color in angle space."""
    print("[plot] Preparing 3D scatter plot...", flush=True)
    grid_size = chaos.shape[0]
    axis = np.unique(angle_grid[:, 0])
    coordinates = np.meshgrid(axis, axis, axis, indexing="ij")
    x, y, z = (value.ravel() for value in coordinates)
    values = chaos.ravel()
    valid = np.isfinite(values)
    values = values[valid]
    if values.size == 0:
        raise ValueError("No interior grid points were available for plotting.")

    normalized = (values - values.min()) / (np.ptp(values) + 1.0e-12)
    colors = plt.get_cmap("plasma")(normalized)
    colors[:, 3] = 0.08 + 0.92 * normalized

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        x[valid],
        y[valid],
        z[valid],
        c=colors,
        s=24,
        depthshade=False,
    )
    axes.set_xlabel(r"$\theta_1(0)$")
    axes.set_ylabel(r"$\theta_2(0)$")
    axes.set_zlabel(r"$\theta_3(0)$")
    axes.set_title(
        "Triple pendulum: finite-time sensitivity in initial-angle space"
    )
    figure.tight_layout()
    print(f"[plot] Saving image to {output}...", flush=True)
    figure.savefig(output, dpi=180)
    print("[plot] Done. Opening the plot window.", flush=True)
    plt.show()


def save_volume_csv(
    angle_grid: np.ndarray,
    final_angles: np.ndarray,
    chaos: np.ndarray,
    output: str | Path = "triple_pendulum_volume.csv",
) -> None:
    """Save every initial-condition point and derived value as a CSV file."""
    output_path = Path(output)
    grid_size = chaos.shape[0]
    axis = np.unique(angle_grid[:, 0])

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "i",
                "j",
                "k",
                "theta1_initial_rad",
                "theta2_initial_rad",
                "theta3_initial_rad",
                "theta1_final_rad",
                "theta2_final_rad",
                "theta3_final_rad",
                "chaos_value",
                "chaos_value_normalized",
                "alpha",
                "is_interior",
            ]
        )

        finite_chaos = chaos[np.isfinite(chaos)]
        if finite_chaos.size:
            chaos_min = float(np.min(finite_chaos))
            chaos_max = float(np.max(finite_chaos))
        else:
            chaos_min = 0.0
            chaos_max = 0.0

        for i, j, k in product(range(grid_size), repeat=3):
            chaos_value = chaos[i, j, k]
            is_interior = bool(np.isfinite(chaos_value))
            if is_interior and chaos_max > chaos_min:
                normalized = (chaos_value - chaos_min) / (
                    chaos_max - chaos_min
                )
                alpha = float(np.clip(0.08 + 0.92 * normalized, 0.08, 1.0))
            elif is_interior:
                normalized = 0.0
                alpha = 1.0
            else:
                normalized = ""
                alpha = ""

            final = final_angles[i, j, k]
            writer.writerow(
                [
                    i,
                    j,
                    k,
                    f"{axis[i]:.17g}",
                    f"{axis[j]:.17g}",
                    f"{axis[k]:.17g}",
                    f"{final[0]:.17g}",
                    f"{final[1]:.17g}",
                    f"{final[2]:.17g}",
                    f"{chaos_value:.17g}" if is_interior else "",
                    f"{normalized:.17g}" if normalized != "" else "",
                    f"{alpha:.17g}" if alpha != "" else "",
                    int(is_interior),
                ]
            )
    print(f"[csv] Saved {grid_size ** 3} rows to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--angle-min", type=float, default=-np.pi)
    parser.add_argument("--angle-max", type=float, default=np.pi)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        default="triple_pendulum_checkpoint.npz",
        help="checkpoint path; use 'none' to disable",
    )
    parser.add_argument(
        "--max-job-seconds",
        type=float,
        default=None,
        help="stop one initial condition after this wall-clock time",
    )
    parser.add_argument(
        "--failures-file",
        default="triple_pendulum_failures.json",
    )
    parser.add_argument(
        "--csv",
        default="triple_pendulum_volume.csv",
        help="CSV output path; use 'none' to disable",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="save the CSV without opening the Matplotlib window",
    )
    args = parser.parse_args()
    checkpoint = None if args.checkpoint.lower() == "none" else args.checkpoint
    csv_output = None if args.csv.lower() == "none" else args.csv
    data = build_chaos_volume(
        grid_size=args.grid_size,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        t_max=args.t_max,
        dt=args.dt,
        workers=args.workers,
        checkpoint=checkpoint,
        max_job_seconds=args.max_job_seconds,
        failures_file=args.failures_file,
    )
    if csv_output is not None:
        save_volume_csv(data[0], data[1], data[2], csv_output)
    if not args.no_plot:
        plot_chaos_volume(data[0], data[2])


if __name__ == "__main__":
    main()

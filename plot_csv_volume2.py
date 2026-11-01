"""Plot triple-pendulum CSV data as semi-transparent voxels."""

# run with: python plot_csv_volume2.py [--output OUTPUT] [--chaos-threshold CHAOS_THRESHOLD] [CSV]
# example: python plot_csv_volume2.py --output none --chaos-threshold 0.5 triple_pendulum_chaos.csv

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_csv_volume import load_csv


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Return voxel boundaries from regularly spaced cell-center coordinates."""
    if centers.size < 2:
        raise ValueError("At least two distinct coordinates are required.")
    differences = np.diff(centers)
    if not np.allclose(differences, differences[0]):
        raise ValueError("CSV coordinates do not form a regular grid.")
    half_step = differences[0] / 2.0
    return np.concatenate(
        ([centers[0] - half_step], centers[:-1] + half_step, [centers[-1] + half_step])
    )


def _cube_faces(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> list[list[tuple[float, float, float]]]:
    """Return all six faces of one voxel, including internal faces."""
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [vertices[index] for index in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        )
    ]


def plot_csv_voxels(
    data: dict[str, np.ndarray],
    output: str | Path | None = "triple_pendulum_voxels.png",
    chaos_threshold: float = 0.75,
) -> None:
    """Create a voxel plot, hiding values above a normalized threshold."""
    if not 0.0 <= chaos_threshold <= 1.0:
        raise ValueError("chaos_threshold must be between 0 and 1.")
    theta1 = np.unique(data["theta1"])
    theta2 = np.unique(data["theta2"])
    theta3 = np.unique(data["theta3"])
    shape = (theta1.size, theta2.size, theta3.size)

    chaos_grid = np.full(shape, np.nan, dtype=float)
    for row in range(data["chaos"].size):
        has_indices = all(
            name in data and data[name].size == data["chaos"].size
            for name in ("i", "j", "k")
        )
        if has_indices:
            i = int(data["i"][row])
            j = int(data["j"][row])
            k = int(data["k"][row])
        else:
            i = int(np.argmin(np.abs(theta1 - data["theta1"][row])))
            j = int(np.argmin(np.abs(theta2 - data["theta2"][row])))
            k = int(np.argmin(np.abs(theta3 - data["theta3"][row])))
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            raise ValueError(f"CSV grid index is out of range at row {row + 2}.")
        if data["interior"][row] == 1:
            chaos_grid[i, j, k] = data["chaos"][row]

    valid = np.isfinite(chaos_grid)
    if not np.any(valid):
        raise ValueError("CSV contains no valid interior voxels.")

    values = chaos_grid[valid]
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    normalized = np.zeros_like(chaos_grid)
    if maximum > minimum:
        normalized[valid] = (chaos_grid[valid] - minimum) / (maximum - minimum)

    visible = valid & (normalized < chaos_threshold)
    if not np.any(visible):
        raise ValueError(
            "The chaos threshold hides every voxel. Increase --chaos-threshold."
        )

    # High chaos means more transparent; low chaos means more opaque.
    alpha = np.zeros_like(chaos_grid)
    alpha[visible] = 0.08 + 0.92 * (1.0 - normalized[visible])
    colors = plt.get_cmap("plasma")(normalized)
    colors[..., 3] = alpha

    x_edges = _cell_edges(theta1)
    y_edges = _cell_edges(theta2)
    z_edges = _cell_edges(theta3)
    print(
        f"[plot] Grid={shape}, valid voxels={int(valid.sum())}, "
        f"visible voxels={int(visible.sum())}, "
        f"hidden voxels={int((valid & ~visible).sum())}, "
        f"chaos range=[{minimum:.6g}, {maximum:.6g}], "
        f"normalized threshold={chaos_threshold:g}"
    )

    figure = plt.figure(figsize=(11, 9))
    axes = figure.add_subplot(111, projection="3d")
    for i, j, k in zip(*np.where(visible)):
        collection = Poly3DCollection(
            _cube_faces(
                x_edges[i],
                x_edges[i + 1],
                y_edges[j],
                y_edges[j + 1],
                z_edges[k],
                z_edges[k + 1],
            ),
            facecolors=[colors[i, j, k]],
            edgecolors=[(0.15, 0.15, 0.15, 0.35)],
            linewidths=0.25,
        )
        axes.add_collection3d(collection)
    axes.set_xlabel(r"$\theta_1(0)$ [rad]")
    axes.set_ylabel(r"$\theta_2(0)$ [rad]")
    axes.set_zlabel(r"$\theta_3(0)$ [rad]")
    axes.set_title(
        "Triple pendulum: chaos indicator as voxels\n"
        f"higher chaos = more transparent, hidden above {chaos_threshold:g}"
    )

    color_map = plt.get_cmap("plasma")
    color_norm = plt.Normalize(vmin=minimum, vmax=maximum)
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes,
        pad=0.12,
        shrink=0.75,
    )
    color_bar.set_label("chaos value")
    figure.text(
        0.02,
        0.02,
        "Each voxel is an initial-angle condition. "
        "Low chaos: opaque; high chaos: transparent. "
        "Boundary points are omitted because they lack six neighbors.",
        fontsize=9,
    )
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, dpi=180)
        print(f"[plot] Saved voxel image to {output}")
    plt.show()


def choose_csv_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Open triple-pendulum CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        help="CSV path; if omitted, a file-selection dialog opens",
    )
    parser.add_argument(
        "--output",
        default="triple_pendulum_voxels.png",
        help="PNG output path; use 'none' to skip saving",
    )
    parser.add_argument(
        "--chaos-threshold",
        type=float,
        default=0.75,
        help="hide normalized chaos values at or above this value (0 to 1)",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv) if args.csv else choose_csv_file()
    if csv_path is None:
        return
    output = None if args.output.lower() == "none" else args.output
    try:
        plot_csv_voxels(
            load_csv(csv_path),
            output,
            chaos_threshold=args.chaos_threshold,
        )
    except (OSError, ValueError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not plot CSV", str(exc))
        root.destroy()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

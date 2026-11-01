"""Open a triple-pendulum CSV and recreate the 3D chaos plot."""

from __future__ import annotations

import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "theta1_initial_rad",
    "theta2_initial_rad",
    "theta3_initial_rad",
    "chaos_value",
    "is_interior",
}


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load numeric columns needed for the initial-angle chaos plot."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_text}")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    def numeric_column(name: str) -> np.ndarray:
        values = []
        for row_number, row in enumerate(rows, start=2):
            value = row.get(name, "")
            if value == "":
                values.append(np.nan)
                continue
            try:
                values.append(float(value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid number in column '{name}' at row {row_number}: {value}"
                ) from exc
        return np.asarray(values, dtype=float)

    return {
        "i": numeric_column("i") if "i" in columns else np.array([]),
        "j": numeric_column("j") if "j" in columns else np.array([]),
        "k": numeric_column("k") if "k" in columns else np.array([]),
        "theta1": numeric_column("theta1_initial_rad"),
        "theta2": numeric_column("theta2_initial_rad"),
        "theta3": numeric_column("theta3_initial_rad"),
        "chaos": numeric_column("chaos_value"),
        "interior": numeric_column("is_interior"),
    }


def plot_csv_volume(
    data: dict[str, np.ndarray],
    output: str | Path | None = "triple_pendulum_chaos_from_csv.png",
) -> None:
    """Create the same opacity/color 3D scatter plot from CSV data."""
    valid = (data["interior"] == 1) & np.isfinite(data["chaos"])
    if not np.any(valid):
        raise ValueError("CSV contains no valid interior points to plot.")

    chaos = data["chaos"][valid]
    minimum = float(np.min(chaos))
    maximum = float(np.max(chaos))
    normalized = (chaos - minimum) / (maximum - minimum + 1.0e-12)
    colors = plt.get_cmap("plasma")(normalized)
    colors[:, 3] = 0.08 + 0.92 * normalized

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        data["theta1"][valid],
        data["theta2"][valid],
        data["theta3"][valid],
        c=colors,
        s=24,
        depthshade=False,
    )
    axes.set_xlabel(r"$\theta_1(0)$")
    axes.set_ylabel(r"$\theta_2(0)$")
    axes.set_zlabel(r"$\theta_3(0)$")
    axes.set_title(
        "Triple pendulum: chaos strength in initial-angle space"
    )
    color_map = plt.get_cmap("plasma")
    color_norm = plt.Normalize(vmin=minimum, vmax=maximum)
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes,
        pad=0.12,
        shrink=0.75,
    )
    color_bar.set_label(
        "chaos value: mean final-angle difference from six neighbors"
    )
    figure.text(
        0.02,
        0.02,
        "Each point = one initial-angle condition. "
        "Color and opacity: low = more similar nearby outcomes, "
        "high = more sensitive/chaotic outcomes. "
        "Angles are radians.",
        fontsize=9,
    )
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, dpi=180)
        print(f"[plot] Saved image to {output}")
    plt.show()


def choose_csv_file() -> Path | None:
    """Open a native file-selection dialog and return the selected CSV."""
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
        default="triple_pendulum_chaos_from_csv.png",
        help="PNG output path; use 'none' to skip saving",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else choose_csv_file()
    if csv_path is None:
        return

    output = None if args.output.lower() == "none" else args.output
    try:
        data = load_csv(csv_path)
        plot_csv_volume(data, output)
    except (OSError, ValueError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not open CSV", str(exc))
        root.destroy()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

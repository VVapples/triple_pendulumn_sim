"""Create a browser-based WebGL voxel viewer from a triple-pendulum CSV."""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from pathlib import Path


REQUIRED = {
    "theta1_initial_rad",
    "theta2_initial_rad",
    "theta3_initial_rad",
    "chaos_value",
    "is_interior",
}


def load_rows(path: Path) -> dict[str, list[float] | list[dict[str, float | int]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED.issubset(reader.fieldnames):
            missing = sorted(REQUIRED - set(reader.fieldnames or ()))
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = []
        axes = [set(), set(), set()]
        for row_number, row in enumerate(reader, 2):
            try:
                coordinates = [
                    float(row["theta1_initial_rad"]),
                    float(row["theta2_initial_rad"]),
                    float(row["theta3_initial_rad"]),
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinate at CSV row {row_number}.") from exc
            for axis, value in enumerate(coordinates):
                axes[axis].add(value)
            if int(float(row["is_interior"])) != 1:
                continue
            try:
                rows.append(
                    {
                        "i": int(float(row.get("i", 0))),
                        "j": int(float(row.get("j", 0))),
                        "k": int(float(row.get("k", 0))),
                        "x": float(row["theta1_initial_rad"]),
                        "y": float(row["theta2_initial_rad"]),
                        "z": float(row["theta3_initial_rad"]),
                        "chaos": float(row["chaos_value"]),
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at CSV row {row_number}.") from exc
    if not rows:
        raise ValueError("CSV contains no interior voxels.")
    return {"axes": [sorted(axis) for axis in axes], "voxels": rows}


def create_viewer(csv_path: Path, output_path: Path, open_browser: bool) -> None:
    rows = load_rows(csv_path)
    html_path = Path(__file__).with_name("triple_pendulum_voxel_viewer.html")
    template = html_path.read_text(encoding="utf-8")
    embedded = json.dumps(rows, separators=(",", ":"))
    output_path.write_text(
        template.replace(
            '/* EMBEDDED_CSV_ROWS */{"axes":[[],[],[]],"voxels":[]}',
            f"/* EMBEDDED_CSV_ROWS */{embedded}",
        ),
        encoding="utf-8",
    )
    print(f"[web] saved {output_path} ({len(rows)} interior voxels)")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="input triple-pendulum CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triple_pendulum_voxels.html"),
        help="HTML output path",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    create_viewer(args.csv, args.output, not args.no_browser)


if __name__ == "__main__":
    main()

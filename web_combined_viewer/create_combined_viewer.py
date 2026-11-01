"""Create a combined voxel and realtime-pendulum HTML viewer."""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from pathlib import Path


def load_rows(path: Path) -> dict:
    required = {
        "theta1_initial_rad",
        "theta2_initial_rad",
        "theta3_initial_rad",
        "chaos_value",
        "is_interior",
    }
    axes = [set(), set(), set()]
    voxels = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        for row in reader:
            coordinates = [
                float(row["theta1_initial_rad"]),
                float(row["theta2_initial_rad"]),
                float(row["theta3_initial_rad"]),
            ]
            for axis, value in enumerate(coordinates):
                axes[axis].add(value)
            if int(float(row["is_interior"])) != 1:
                continue
            chaos = float(row["chaos_value"])
            if not (chaos == chaos):
                continue
            voxels.append(
                {
                    "i": int(float(row.get("i", 0))),
                    "j": int(float(row.get("j", 0))),
                    "k": int(float(row.get("k", 0))),
                    "x": coordinates[0],
                    "y": coordinates[1],
                    "z": coordinates[2],
                    "chaos": chaos,
                }
            )
    if not voxels:
        raise ValueError("CSV contains no interior voxels.")
    return {"axes": [sorted(axis) for axis in axes], "voxels": voxels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("triple_pendulum_combined.html"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    template = Path(__file__).with_name("triple_pendulum_combined.html").read_text(
        encoding="utf-8"
    )
    data = json.dumps(load_rows(args.csv), separators=(",", ":"))
    output = template.replace(
        '/* EMBEDDED_DATA */{"axes":[[],[],[]],"voxels":[]}',
        f"/* EMBEDDED_DATA */{data}",
    )
    args.output.write_text(output, encoding="utf-8")
    print(f"[web] saved {args.output}")
    if not args.no_browser:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()

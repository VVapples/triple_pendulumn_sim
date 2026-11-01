"""Compress a triple-pendulum volume CSV into a compact binary .tpvol file.

The original CSV is never modified. Only the information needed to display the
volume is kept: one quantized chaos value per voxel on a regular grid. Initial
angles are not stored per voxel; they are reconstructed from the grid index and
the angle range saved in the header.

Large grids can be block-reduced (e.g. 1024^3 -> 256^3) so the result fits in a
browser. Each output voxel then covers a factor^3 block of original grid points.

Inputs:
  * a volume CSV (columns i, j, k, chaos_value, theta*_initial_rad), or
  * the CUDA program's temporary ``<csv>.chaos.bin`` cache (much faster).

File layout (little endian):
  8 bytes   magic  b"TPVOL\\x00\\x01\\x00"
  4 bytes   uint32 header length H
  H bytes   UTF-8 JSON header (space padded so the data starts 8-byte aligned)
  N bytes   uint8 or uint16 voxel codes, C order [i][j][k] (theta1, theta2, theta3)
            code 0 = no data (boundary / NaN); 1..qmax maps linearly to
            chaos_min..chaos_max given in the header.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = b"TPVOL\x00\x01\x00"
ROW = re.compile(rb"^(\d+),(\d+),(\d+),(?:[^,\n]*,){6}([^,\r\n]+),", re.M)
READ_BLOCK = 64 * 1024 * 1024
TASK_BYTES = 512 * 1024 * 1024


# ---------------------------------------------------------------- CSV metadata
def _first_data_row(path: Path) -> tuple[list[str], list[str]]:
    with path.open("rb") as fh:
        header = fh.readline().decode("utf-8-sig").strip().split(",")
        first = fh.readline().decode().strip().split(",")
    return header, first


def _last_data_row(path: Path) -> list[str]:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 8192))
        tail = fh.read().strip().splitlines()
    return tail[-1].decode().strip().split(",")


def inspect_csv(path: Path, args) -> dict:
    header, first = _first_data_row(path)
    last = _last_data_row(path)
    expected = ["i", "j", "k", "theta1_initial_rad", "theta2_initial_rad",
                "theta3_initial_rad", "theta1_final_rad", "theta2_final_rad",
                "theta3_final_rad", "chaos_value"]
    if header[: len(expected)] != expected:
        raise SystemExit(
            "Unexpected CSV columns. Expected the first columns to be:\n  "
            + ",".join(expected) + "\nGot:\n  " + ",".join(header))
    grid = args.grid_size or int(last[0]) + 1
    if not args.grid_size and not (int(last[0]) == int(last[1]) == int(last[2])):
        raise SystemExit("Could not infer --grid-size from the last row; pass it explicitly.")
    angle_min = args.angle_min if args.angle_min is not None else float(first[3])
    angle_max = args.angle_max if args.angle_max is not None else float(last[3])
    return {"grid": grid, "angle_min": angle_min, "angle_max": angle_max}


# ----------------------------------------------------------- slab accumulation
class SlabAccumulator:
    """Dense accumulator over a growing range of output i-slabs."""

    def __init__(self, ny: int, nz: int, reduce: str):
        self.ny, self.nz, self.reduce = ny, nz, reduce
        self.base = None
        self.a = None  # sums or maxima
        self.c = None  # counts (mean only)

    def _ensure(self, lo: int, hi: int) -> None:
        if self.base is None:
            self.base = lo
            self.a, self.c = self._alloc(hi - lo + 1)
            return
        new_lo, new_hi = min(lo, self.base), max(hi, self.base + len(self.a) - 1)
        if new_lo == self.base and new_hi == self.base + len(self.a) - 1:
            return
        a, c = self._alloc(new_hi - new_lo + 1)
        off = self.base - new_lo
        a[off: off + len(self.a)] = self.a
        if c is not None:
            c[off: off + len(self.c)] = self.c
        self.base, self.a, self.c = new_lo, a, c

    def _alloc(self, n: int):
        shape = (n, self.ny * self.nz)
        if self.reduce == "mean":
            return np.zeros(shape, np.float64), np.zeros(shape, np.uint32)
        return np.full(shape, -np.inf, np.float64), None

    def add(self, bi, bj, bk, values) -> None:
        if len(values) == 0:
            return
        lo, hi = int(bi.min()), int(bi.max())
        self._ensure(lo, hi)
        n = len(self.a)
        flat = (bi - self.base) * (self.ny * self.nz) + bj * self.nz + bk
        size = n * self.ny * self.nz
        if self.reduce == "mean":
            self.a.reshape(-1)[:] += np.bincount(flat, weights=values, minlength=size)
            self.c.reshape(-1)[:] += np.bincount(flat, minlength=size).astype(np.uint32)
        else:
            order = np.argsort(flat, kind="stable")
            flat, values = flat[order], values[order]
            starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
            maxima = np.maximum.reduceat(values, starts)
            target = self.a.reshape(-1)
            idx = flat[starts]
            target[idx] = np.maximum(target[idx], maxima)


def _scan_task(task):
    path, start, end, grid, factor, ny, nz, reduce = task
    acc = SlabAccumulator(ny, nz, reduce)
    rows = 0
    with open(path, "rb") as fh:
        fh.seek(start)
        remaining = end - start
        carry = b""
        while remaining > 0 or carry:
            chunk = fh.read(min(READ_BLOCK, remaining)) if remaining > 0 else b""
            remaining -= len(chunk)
            data = carry + chunk
            if remaining > 0:
                cut = data.rfind(b"\n") + 1
                data, carry = data[:cut], data[cut:]
            else:
                carry = b""
            found = ROW.findall(data)
            if not found:
                continue
            arr = np.array(found, dtype="S32")
            i = arr[:, 0].astype(np.int64)
            j = arr[:, 1].astype(np.int64)
            k = arr[:, 2].astype(np.int64)
            v = arr[:, 3].astype(np.float64)
            ok = np.isfinite(v)
            if not ok.all():
                i, j, k, v = i[ok], j[ok], k[ok], v[ok]
            if len(v) and (i.max() >= grid or j.max() >= grid or k.max() >= grid):
                raise ValueError("grid index exceeds --grid-size")
            rows += len(v)
            acc.add(i // factor, j // factor, k // factor, v)
    return end - start, rows, acc.base, acc.a, acc.c


def _line_aligned_splits(path: Path, size: int, task_bytes: int) -> list[int]:
    with path.open("rb") as fh:
        fh.readline()  # skip the header
        points = [fh.tell()]
        pos = points[0] + task_bytes
        while pos < size:
            fh.seek(pos)
            fh.readline()
            aligned = fh.tell()
            if aligned >= size:
                break
            points.append(aligned)
            pos = aligned + task_bytes
    points.append(size)
    return points


def reduce_csv(path: Path, grid: int, factor: int, dims, reduce: str, workers: int,
               task_bytes: int = TASK_BYTES):
    nx, ny, nz = dims
    size = path.stat().st_size
    splits = _line_aligned_splits(path, size, task_bytes)
    tasks = [(str(path), a, b, grid, factor, ny, nz, reduce)
             for a, b in zip(splits[:-1], splits[1:])]
    total_a = (np.zeros((nx, ny * nz), np.float64) if reduce == "mean"
               else np.full((nx, ny * nz), -np.inf, np.float64))
    total_c = np.zeros((nx, ny * nz), np.uint32) if reduce == "mean" else None
    done_bytes = rows = 0
    t0 = time.perf_counter()
    print(f"[scan] {size / 1e9:.1f} GB in {len(tasks)} pieces with {workers} workers",
          flush=True)
    with mp.Pool(workers) as pool:
        for nbytes, nrows, base, a, c in pool.imap_unordered(_scan_task, tasks):
            done_bytes += nbytes
            rows += nrows
            if base is not None:
                sl = slice(base, base + len(a))
                if reduce == "mean":
                    total_a[sl] += a
                    total_c[sl] += c
                else:
                    np.maximum(total_a[sl], a, out=total_a[sl])
            elapsed = time.perf_counter() - t0
            rate = done_bytes / max(elapsed, 1e-9)
            eta = (size - done_bytes) / max(rate, 1)
            print(f"[scan] {100 * done_bytes / size:5.1f}%  {rate / 1e6:7.0f} MB/s  "
                  f"rows={rows:,}  eta {eta / 60:5.1f} min", flush=True)
    if reduce == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(total_c > 0, total_a / np.maximum(total_c, 1), np.nan)
    else:
        out = np.where(np.isfinite(total_a), total_a, np.nan)
    return out.reshape(nx, ny, nz), rows


def reduce_chaos_bin(path: Path, grid: int, factor: int, dims, reduce: str):
    nx, ny, nz = dims
    volume = np.memmap(path, dtype="<f8", mode="r", shape=(grid, grid, grid))
    out = np.full(dims, np.nan, np.float64)
    pad = nx * factor - grid
    t0 = time.perf_counter()
    rows = 0
    import warnings
    for bi in range(nx):
        slab = np.array(volume[bi * factor: (bi + 1) * factor], dtype=np.float64)
        if slab.shape[0] < factor or pad:
            full = np.full((factor, ny * factor, nz * factor), np.nan)
            full[: slab.shape[0], :grid, :grid] = slab
            slab = full
        blocks = slab.reshape(factor, ny, factor, nz, factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if reduce == "mean":
                out[bi] = np.nanmean(blocks, axis=(0, 2, 4))
            else:
                out[bi] = np.nanmax(blocks, axis=(0, 2, 4))
        rows += int(np.isfinite(slab).sum())
        if bi % max(1, nx // 50) == 0 or bi == nx - 1:
            el = time.perf_counter() - t0
            print(f"[scan] {bi + 1}/{nx} slabs  {el:6.1f}s", flush=True)
    return out, rows


# --------------------------------------------------------------------- output
def quantize(values: np.ndarray, bits: int):
    finite = np.isfinite(values)
    if not finite.any():
        raise SystemExit("No finite chaos values found.")
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    qmax = 255 if bits == 8 else 65535
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    span = vmax - vmin if vmax > vmin else 1.0
    codes = np.zeros(values.shape, dtype)
    scaled = np.rint((values[finite] - vmin) / span * (qmax - 1)) + 1
    codes[finite] = scaled.astype(dtype)
    return codes, vmin, vmax, qmax


def write_tpvol(path: Path, codes: np.ndarray, header: dict) -> None:
    text = json.dumps(header, indent=1).encode("utf-8")
    pad = (-(len(MAGIC) + 4 + len(text))) % 8
    text += b" " * pad
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(len(text).to_bytes(4, "little"))
        fh.write(text)
        fh.write(np.ascontiguousarray(codes).tobytes())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="volume CSV or <csv>.chaos.bin")
    parser.add_argument("-o", "--output", type=Path,
                        help="output .tpvol (default: next to input, <name>_<N>.tpvol)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--target-size", type=int, default=256,
                      help="max voxels per axis in the output (default 256)")
    size.add_argument("--factor", type=int, help="explicit block size per axis")
    parser.add_argument("--reduce", choices=["mean", "max"], default="mean",
                        help="how to combine a block of chaos values (default mean)")
    parser.add_argument("--bits", type=int, choices=[8, 16], default=8,
                        help="quantization depth (8 = 1 byte/voxel, 16 = 2 bytes)")
    parser.add_argument("--grid-size", type=int, help="override/inferred source grid size")
    parser.add_argument("--angle-min", type=float, help="override angle minimum (rad)")
    parser.add_argument("--angle-max", type=float, help="override angle maximum (rad)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="parallel CSV parser processes")
    parser.add_argument("--chunk-mb", type=int, default=TASK_BYTES >> 20,
                        help="CSV megabytes per parallel work piece (default 512)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.is_file():
        raise SystemExit(f"Input not found: {src}")
    is_bin = src.name.endswith(".chaos.bin")

    if is_bin:
        cells = src.stat().st_size // 8
        grid = args.grid_size or round(cells ** (1 / 3))
        if grid ** 3 * 8 != src.stat().st_size:
            raise SystemExit("chaos.bin size is not grid^3 doubles; pass --grid-size.")
        meta = {"grid": grid,
                "angle_min": args.angle_min if args.angle_min is not None else -math.pi,
                "angle_max": args.angle_max if args.angle_max is not None else math.pi}
    else:
        meta = inspect_csv(src, args)
    grid = meta["grid"]
    factor = args.factor or max(1, math.ceil(grid / args.target_size))
    n = math.ceil(grid / factor)
    dims = (n, n, n)

    if args.output:
        out = args.output.resolve()
    else:
        stem = src.name[: -len(".csv.chaos.bin")] if is_bin else src.stem
        out = src.with_name(f"{stem}_{n}.tpvol")
    if out == src:
        raise SystemExit("Refusing to overwrite the input file.")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists (use --force to overwrite).")

    print(f"[info] source grid {grid}^3, angles [{meta['angle_min']:.6f}, "
          f"{meta['angle_max']:.6f}] rad", flush=True)
    print(f"[info] block factor {factor} -> output {n}^3 ({args.reduce}, {args.bits}-bit)",
          flush=True)

    t0 = time.perf_counter()
    if is_bin:
        values, rows = reduce_chaos_bin(src, grid, factor, dims, args.reduce)
    else:
        values, rows = reduce_csv(src, grid, factor, dims, args.reduce, args.workers,
                                  args.chunk_mb << 20)
    codes, vmin, vmax, qmax = quantize(values, args.bits)

    header = {
        "format": "tpvol",
        "version": 1,
        "dims": list(dims),
        "order": "C [i][j][k]; i=theta1, j=theta2, k=theta3",
        "bits": args.bits,
        "qmax": qmax,
        "source": src.name,
        "source_grid": grid,
        "factor": factor,
        "reduce": args.reduce,
        "angle_min": meta["angle_min"],
        "angle_max": meta["angle_max"],
        "chaos_min": vmin,
        "chaos_max": vmax,
        "valid_voxels": int((codes > 0).sum()),
        "source_rows": rows,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_tpvol(out, codes, header)
    mb = out.stat().st_size / 1e6
    print(f"[done] wrote {out} ({mb:.1f} MB, {header['valid_voxels']:,} voxels, "
          f"chaos {vmin:.6g}..{vmax:.6g}) in {time.perf_counter() - t0:.1f}s")
    print(f"[done] original kept: {src}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

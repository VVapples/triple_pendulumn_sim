# Compressed chaos volume (`.tpvol`) + viewer

Large CSV volumes (the 1024^3 run is ~206 GB) cannot be opened by the browser
viewers. These two tools solve that:

| File | Purpose |
| --- | --- |
| `compress_volume.py` | Reads a volume CSV (or the CUDA `.chaos.bin` cache) and writes a small `.tpvol` file. The original file is never modified. |
| `tpvol_viewer.html` | Opens `.tpvol` files and renders the volume on the GPU. |

## What is kept

The viewer only needs the chaos value of each voxel; initial angles follow
from the grid index and the angle range. A `.tpvol` file therefore stores:

- a small JSON header (grid size, block factor, angle range, chaos min/max, ...)
- one 8-bit (or 16-bit) code per voxel: `0` = no data (boundary), `1..qmax`
  linearly maps to `chaos_min..chaos_max`.

Final angles and the derived CSV columns (`chaos_value_normalized`, `alpha`,
`is_interior`) are dropped; they are not needed for display.

For grids larger than `--target-size` (default 256), blocks of
`factor^3` source points are combined into one voxel (`--reduce mean` or `max`).
1024^3 -> 256^3 at 8 bits is about **16 MB**.

## Compress

From `tripple_pendulumn_sim`:

```powershell
# from the CSV (parallel; speed is mostly limited by disk read speed)
..\.venv\Scripts\python.exe compressed_volume\compress_volume.py `
  data\triple_pendulum_volume_cuda_1024.csv --target-size 256

# much faster, if the CUDA cache still exists (8 GB instead of 206 GB)
..\.venv\Scripts\python.exe compressed_volume\compress_volume.py `
  data\triple_pendulum_volume_cuda_1024.csv.chaos.bin --target-size 256
```

Output defaults to `<name>_<N>.tpvol` next to the input
(e.g. `data\triple_pendulum_volume_cuda_1024_256.tpvol`).

Options:

| Option | Meaning |
| --- | --- |
| `--target-size N` | maximum voxels per axis (default 256) |
| `--factor F` | explicit block size instead of `--target-size` |
| `--reduce mean\|max` | combine a block by its mean (default) or maximum chaos |
| `--bits 8\|16` | value precision (16 doubles the size) |
| `--workers N` | CSV parser processes (default: CPU count - 2) |
| `-o FILE` | output path; existing files need `--force` |
| `--grid-size`, `--angle-min`, `--angle-max` | override inferred metadata (needed for `.chaos.bin` from non-default angle ranges) |

256^3 renders smoothly on most GPUs. 512^3 (128 MB) also works on a
discrete GPU; the browser's 3D-texture limit is usually 2048 per axis.

## View

Open `tpvol_viewer.html` in Chrome/Edge (WebGL2, internet access for the
three.js CDN), then choose or drag-and-drop a `.tpvol` file.

- **Hide chaos above / below**: show only a normalized chaos band; the
  histogram shows the distribution (log scale) and the hidden part dimmed.
- **Opacity scale**, **Smooth colors** (default: 16 bands like the old viewer).
- **Resolution-independent opacity**: scales per-voxel opacity so a 256^3
  volume looks similar in density to a 64^3 one. Uncheck for the raw
  per-voxel opacity of the old viewer.
- **Clip box**: cut the volume along theta1/theta2/theta3 to look inside.
- **Hover a voxel**: shows its initial angles (centre and range for block
  voxels), chaos value, indices in the source grid, and a drawing of the
  initial pendulum configuration (angles from the downward vertical).

When served over HTTP, `tpvol_viewer.html?src=file.tpvol` loads a file directly.


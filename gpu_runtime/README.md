# CUDA runtime

This folder contains a native CUDA implementation of the triple-pendulum
volume calculation. It does not require Python or CuPy at runtime.

## What is optimized

- One CUDA thread owns one initial condition and keeps its complete six-value
  pendulum state in device memory.
- The Gonzalez discrete-gradient/Newton solver runs entirely on the GPU.
- Chaos values are calculated by a second GPU kernel.
- The host performs one input transfer before simulation and one result
  transfer after the kernels finish.

This removes the Python/CUDA array-management overhead and avoids CPU
synchronization inside Newton iterations.

## Build on Windows

Install the NVIDIA CUDA Toolkit and ensure `nvcc` is on `PATH`. On Windows,
use the Visual Studio generator because a normal PowerShell may not provide
`nmake` and `cl`:

```powershell
cmake -S . -B build-vs -G "Visual Studio 18 2026" -A x64
cmake --build build-vs --config Release --parallel 4
```

If the project is under a path containing Japanese characters and CUDA/MSBuild
cannot find its temporary compiler-identification source, map the project root
to an ASCII drive letter first:

```powershell
subst X: "C:\path\to\tripple_pendulumn_sim"
cmake -S "X:\gpu runtime" -B "X:\gpu runtime\build-vs" `
  -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "X:\gpu runtime\build-vs" --config Release --parallel 4
```

Alternatively:

```powershell
nvcc -O3 --use_fast_math -std=c++17 triple_pendulum_cuda.cu `
  -o triple_pendulum_cuda.exe
```

The default architecture is `120` for RTX 50-series/Blackwell GPUs. Override it
when configuring for another GPU, for example
`-DCMAKE_CUDA_ARCHITECTURES=89` for many RTX 40-series GPUs. If `nvcc` reports
an unsupported architecture, use the compute capability supported by the
installed CUDA Toolkit.

## Run

```powershell
.\build-vs\Release\triple_pendulum_cuda.exe `
  --grid-size 32 `
  --t-max 3 `
  --dt 0.03 `
  --csv ..\data\triple_pendulum_volume_cuda.csv
```

The CSV columns match the Python volume output, so it can be opened by the
existing voxel and combined web viewers.

## Tiled large-grid runs

Large grids are processed in `i`-slice tiles, so the complete volume is not
held in GPU or host memory. `--tile-size` controls how many slices are sent to
the GPU at once. The program prints progress during simulation, chaos
calculation, and CSV writing.

For example, for a 1024^3 run:

```powershell
.\build-vs\Release\triple_pendulum_cuda.exe `
  --grid-size 1024 `
  --t-max 8 `
  --dt 0.02 `
  --tile-size 1 `
  --csv ..\data\triple_pendulum_volume_cuda_1024.csv
```

Increase `--tile-size` only while there is enough free GPU memory. A tile size
of 1 is the safest starting point for a 16 GiB GPU. Temporary files are created
next to the CSV during the run and removed after successful completion.

## Resuming an interrupted run

If a run is stopped (Ctrl+C, crash, reboot), the completed slices remain in
`<csv>.final.bin`. Re-run with the **same** `--grid-size`, `--angle-min`,
`--angle-max`, `--t-max`, `--dt` and `--csv`, plus `--resume`:

```powershell
.\build-vs\Release\triple_pendulum_cuda.exe `
  --grid-size 1024 --t-max 8 --dt 0.02 --tile-size 1 `
  --csv ..\data\triple_pendulum_volume_cuda_1024.csv --resume
```

The program counts the complete slices in the file, discards any partially
written slice, and continues from there. `--tile-size` may differ from the
original run. Parameters are saved to `<csv>.meta.txt` and checked on resume.
Without `--resume`, an existing `.final.bin` is overwritten.

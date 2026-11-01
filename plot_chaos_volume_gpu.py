"""CUDA/CuPy batch version of the triple-pendulum volume simulation.

This keeps the Gonzalez discrete-gradient method, but evaluates many initial
conditions simultaneously on the GPU. It is intended for NVIDIA GPUs such as
the RTX 5070 Ti.
"""

# run with: python plot_chaos_volume_gpu.py [--grid-size GRID_SIZE] [--angle-min ANGLE_MIN] [--angle-max ANGLE_MAX] [--t-max T_MAX] [--dt DT] [--batch-size BATCH_SIZE] [--csv CSV] [--plot]
# example: python plot_chaos_volume_gpu.py --grid-size 9 --angle-min -3.14159 --angle-max 3.14159 --t-max 8.0 --dt 0.02 --batch-size 1024 --csv triple_pendulum_volume_gpu.csv --plot

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np

from plot_chaos_volume import plot_chaos_volume, save_volume_csv
from triple_pendulum import TriplePendulumSimulator


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is not installed. For an RTX 5070 Ti, install a CUDA 12 "
            "CuPy wheel with: py -3.10 -m pip install cupy-cuda12x"
        ) from exc
    try:
        device = cp.cuda.Device()
        device.use()
        free_memory, total_memory = cp.cuda.Device().mem_info
    except Exception as exc:
        raise RuntimeError(
            "CuPy could not initialize CUDA. Check the NVIDIA driver and CUDA "
            "runtime, then run `python -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"`."
        ) from exc
    print(
        f"[gpu] device={device.id}, free={free_memory / 2**30:.1f} GiB, "
        f"total={total_memory / 2**30:.1f} GiB",
        flush=True,
    )
    return cp


def _explain_cuda_library_error(exc):
    if isinstance(exc, ImportError) and "cublas" in str(exc).lower():
        raise RuntimeError(
            "CuPy found the GPU, but its cuBLAS DLL could not be loaded. "
            "This usually means that the CuPy CUDA wheel and installed CUDA "
            "runtime do not match. In the same Python environment, run:\n"
            "  py -3.10 -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x\n"
            "  py -3.10 -m pip install --no-cache-dir cupy-cuda12x\n"
            "Then restart the terminal and retry. Also make sure the NVIDIA "
            "driver is up to date."
        ) from exc
    raise exc


def _mass_matrix(cp, q):
    lengths = cp.asarray((1.0, 1.0, 1.0))
    mass_sums = cp.asarray((3.0, 2.0, 1.0))
    matrix = cp.empty((q.shape[0], 3, 3), dtype=cp.float64)
    for a in range(3):
        for b in range(3):
            matrix[:, a, b] = (
                lengths[a]
                * lengths[b]
                * mass_sums[max(a, b)]
                * cp.cos(q[:, a] - q[:, b])
            )
    return matrix


def _batched_solve(cp, matrix, rhs, regularize=False):
    """Solve A x = b for a batch of vector right-hand sides."""
    if regularize:
        scale = cp.maximum(
            1.0, cp.max(cp.abs(matrix), axis=(1, 2))
        )
        diagonal = (1.0e-12 * scale)[:, None, None]
        identity = cp.eye(matrix.shape[1], dtype=matrix.dtype)[None, :, :]
        matrix = matrix + diagonal * identity
    return cp.linalg.solve(matrix, rhs[..., None])[..., 0]


def _gradient(cp, state):
    q = state[:, :3]
    p = state[:, 3:]
    mass = _mass_matrix(cp, q)
    omega = _batched_solve(cp, mass, p)
    gradient_q = cp.zeros_like(q)
    gravity = 9.8
    lengths = (1.0, 1.0, 1.0)
    mass_sums = (3.0, 2.0, 1.0)
    for coordinate in range(3):
        d_mass = cp.zeros_like(mass)
        for a in range(3):
            for b in range(3):
                if coordinate == a and coordinate == b:
                    continue
                if coordinate != a and coordinate != b:
                    continue
                sign = 1.0 if coordinate == a else -1.0
                d_mass[:, a, b] = (
                    lengths[a]
                    * lengths[b]
                    * mass_sums[max(a, b)]
                    * (-cp.sin(q[:, a] - q[:, b]))
                    * sign
                )
        gradient_q[:, coordinate] = (
            -0.5 * cp.einsum("ni,nij,nj->n", omega, d_mass, omega)
            + gravity * lengths[coordinate] * mass_sums[coordinate]
            * cp.sin(q[:, coordinate])
        )
    return cp.concatenate((gradient_q, omega), axis=1)


def _hamiltonian(cp, state):
    q = state[:, :3]
    p = state[:, 3:]
    mass = _mass_matrix(cp, q)
    omega = _batched_solve(cp, mass, p)
    kinetic = 0.5 * cp.sum(p * omega, axis=1)
    potential = 9.8 * (
        3.0 * (1.0 - cp.cos(q[:, 0]))
        + 2.0 * (1.0 - cp.cos(q[:, 1]))
        + 1.0 * (1.0 - cp.cos(q[:, 2]))
    )
    return kinetic + potential


def _rhs(cp, state):
    gradient = _gradient(cp, state)
    return cp.concatenate((gradient[:, 3:], -gradient[:, :3]), axis=1)


def _discrete_gradient(cp, state0, state1, hamiltonian0=None):
    delta = state1 - state0
    midpoint = 0.5 * (state0 + state1)
    gradient = _gradient(cp, midpoint)
    if hamiltonian0 is None:
        hamiltonian0 = _hamiltonian(cp, state0)
    norm_squared = cp.sum(delta * delta, axis=1)
    correction = (
        _hamiltonian(cp, state1)
        - hamiltonian0
        - cp.sum(gradient * delta, axis=1)
    ) / cp.maximum(norm_squared, 1.0e-28)
    return gradient + correction[:, None] * delta


def _residual(cp, state0, state1, step, hamiltonian0=None):
    gradient = _discrete_gradient(cp, state0, state1, hamiltonian0)
    return state1 - state0 - step * cp.concatenate(
        (gradient[:, 3:], -gradient[:, :3]), axis=1
    )


def _rk4(cp, state, step):
    k1 = _rhs(cp, state)
    k2 = _rhs(cp, state + 0.5 * step * k1)
    k3 = _rhs(cp, state + 0.5 * step * k2)
    k4 = _rhs(cp, state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _try_discrete_gradient_step(
    cp, state0, step, tolerance, max_iterations
):
    state1 = _rk4(cp, state0, step)
    hamiltonian0 = _hamiltonian(cp, state0)
    residual = _residual(cp, state0, state1, step, hamiltonian0)
    active = cp.ones(state0.shape[0], dtype=bool)
    for iteration in range(max_iterations):
        residual_norm = cp.max(cp.abs(residual), axis=1)
        active = active & (residual_norm >= tolerance)

        jacobian = cp.empty((state0.shape[0], 6, 6), dtype=cp.float64)
        for column in range(6):
            epsilon = 2.0e-7 * cp.maximum(1.0, cp.abs(state1[:, column]))
            plus = state1.copy()
            minus = state1.copy()
            plus[:, column] += epsilon
            minus[:, column] -= epsilon
            jacobian[:, :, column] = (
                _residual(cp, state0, plus, step, hamiltonian0)
                - _residual(cp, state0, minus, step, hamiltonian0)
            ) / (2.0 * epsilon[:, None])
        correction = _batched_solve(
            cp, jacobian, residual, regularize=True
        )
        finite = cp.all(cp.isfinite(correction), axis=1)
        active = active & finite
        correction = cp.where(active[:, None], correction, 0.0)
        damping = cp.ones(state0.shape[0], dtype=cp.float64)
        best_state = state1
        best_norm = residual_norm
        for _ in range(8):
            candidate = state1 - damping[:, None] * correction
            candidate_norm = cp.max(
                cp.abs(
                    _residual(cp, state0, candidate, step, hamiltonian0)
                ),
                axis=1,
            )
            accepted = (~active) | (
                cp.isfinite(candidate_norm) & (candidate_norm < best_norm)
            )
            best_state = cp.where(accepted[:, None], candidate, best_state)
            best_norm = cp.where(accepted, candidate_norm, best_norm)
            damping = cp.where(accepted & active, damping, 0.5 * damping)
        state1 = best_state
        residual = _residual(cp, state0, state1, step, hamiltonian0)
        # Check convergence on the host only occasionally. The mask itself
        # remains on the GPU, avoiding one synchronization per Newton step.
        if (iteration + 1) % 4 == 0 and not bool(cp.any(active)):
            return state1, cp.zeros(state0.shape[0], dtype=bool)
    residual_norm = cp.max(cp.abs(residual), axis=1)
    return state1, (~cp.isfinite(residual_norm)) | (residual_norm >= tolerance)


def _discrete_gradient_step(
    cp, state0, step, tolerance, max_iterations, subdivision_depth=0
):
    state1, failed = _try_discrete_gradient_step(
        cp, state0, step, tolerance, max_iterations
    )
    if not bool(cp.any(failed)):
        return state1
    if subdivision_depth >= 12:
        raise RuntimeError(
            "GPU Newton iteration failed even after adaptive step subdivision. "
            "The GPU solver reached its local subdivision limit."
        )

    result = state1.copy()
    failed_state = state0[failed]
    half_step = 0.5 * step
    midpoint = _discrete_gradient_step(
        cp, failed_state, half_step, tolerance, max_iterations,
        subdivision_depth + 1,
    )
    midpoint = _discrete_gradient_step(
        cp, midpoint, half_step, tolerance, max_iterations,
        subdivision_depth + 1,
    )
    result[failed] = midpoint
    return result


def _initial_states(cp, angles):
    q = cp.asarray(angles, dtype=cp.float64)
    # The requested initial angular velocity is zero, so p = M(q) * omega is
    # exactly zero without constructing M or calling cuBLAS.
    return cp.concatenate((q, cp.zeros_like(q)), axis=1)


def _simulate_batch(cp, angles, t_max, dt, tolerance=2.0e-10):
    state = _initial_states(cp, angles)
    steps = int(np.ceil(t_max / dt))
    for _ in range(steps):
        state = _discrete_gradient_step(cp, state, dt, tolerance, 30)
    return cp.asnumpy(state[:, :3])


def _simulate_batch_with_fallback(cp, angles, t_max, dt):
    try:
        return _simulate_batch(cp, angles, t_max, dt)
    except RuntimeError as gpu_error:
        print(
            "[gpu] GPU Newton failed for this batch; using the CPU "
            "discrete-gradient solver for these initial conditions.",
            flush=True,
        )
        simulator = TriplePendulumSimulator()
        results = []
        for angle in angles:
            try:
                results.append(
                    simulator.simulate_final(angle, t_max=t_max, dt=dt)
                )
            except RuntimeError as cpu_error:
                raise RuntimeError(
                    "Both GPU and CPU solvers failed for initial angles "
                    f"{np.asarray(angle)}. GPU error: {gpu_error}; "
                    f"CPU error: {cpu_error}"
                ) from cpu_error
        return np.asarray(results, dtype=float)


def build_gpu_volume(
    grid_size=9,
    angle_min=-np.pi,
    angle_max=np.pi,
    t_max=8.0,
    dt=0.02,
    batch_size=1024,
):
    cp = _require_cupy()
    axis = np.linspace(angle_min, angle_max, grid_size)
    angle_grid = np.asarray(list(product(axis, axis, axis)), dtype=float)
    total = len(angle_grid)
    final_angles = np.empty((total, 3), dtype=float)
    started_at = time.perf_counter()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        try:
            final_angles[start:stop] = _simulate_batch_with_fallback(
                cp, angle_grid[start:stop], t_max, dt
            )
        except Exception as exc:
            failing_angles = angle_grid[start:stop]
            print(
                "[gpu] failed batch initial-angle range: "
                f"min={failing_angles.min(axis=0)}, "
                f"max={failing_angles.max(axis=0)}",
                flush=True,
            )
            _explain_cuda_library_error(exc)
        print(
            f"[gpu] {stop}/{total} ({100 * stop / total:.1f}%), "
            f"elapsed {time.perf_counter() - started_at:.1f} s",
            flush=True,
        )

    final_grid = final_angles.reshape((grid_size, grid_size, grid_size, 3))
    chaos = np.full((grid_size, grid_size, grid_size), np.nan)
    for i, j, k in product(range(1, grid_size - 1), repeat=3):
        center = final_grid[i, j, k]
        neighbors = (
            final_grid[i - 1, j, k], final_grid[i + 1, j, k],
            final_grid[i, j - 1, k], final_grid[i, j + 1, k],
            final_grid[i, j, k - 1], final_grid[i, j, k + 1],
        )
        chaos[i, j, k] = np.mean([
            np.linalg.norm(np.arctan2(np.sin(neighbor - center),
                                       np.cos(neighbor - center)))
            for neighbor in neighbors
        ])
    return angle_grid, final_grid, chaos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--angle-min", type=float, default=-np.pi)
    parser.add_argument("--angle-max", type=float, default=np.pi)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--csv", default="triple_pendulum_volume_gpu.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    angle_grid, final_angles, chaos = build_gpu_volume(
        args.grid_size, args.angle_min, args.angle_max,
        args.t_max, args.dt, args.batch_size
    )
    save_volume_csv(angle_grid, final_angles, chaos, args.csv)
    if args.plot:
        plot_chaos_volume(angle_grid, chaos)


if __name__ == "__main__":
    main()

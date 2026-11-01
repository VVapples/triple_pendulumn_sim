"""Energy-preserving simulation of a planar triple pendulum.

The three angles are absolute angles measured from the downward vertical.
The state is ``z = (q1, q2, q3, p1, p2, p3)``, where ``p`` are canonical
momenta.  The implementation uses Gonzalez's discrete gradient method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TriplePendulumParameters:
    """Physical and solver parameters."""

    masses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gravity: float = 9.8
    newton_tolerance: float = 2.0e-10
    newton_max_iterations: int = 30
    max_step_subdivisions: int = 6


@dataclass
class SimulationResult:
    """Time history returned by :meth:`TriplePendulumSimulator.simulate`."""

    time: Array
    angles: Array
    angular_velocities: Array
    energy: Array

    @property
    def energy_error(self) -> Array:
        return self.energy - self.energy[0]


class TriplePendulumSimulator:
    """Simulator and finite-time chaos estimator for a planar triple pendulum."""

    def __init__(self, parameters: TriplePendulumParameters | None = None):
        self.parameters = parameters or TriplePendulumParameters()
        self._masses = np.asarray(self.parameters.masses, dtype=float)
        self._lengths = np.asarray(self.parameters.lengths, dtype=float)
        if np.any(self._masses <= 0.0) or np.any(self._lengths <= 0.0):
            raise ValueError("masses and lengths must be positive.")

    def mass_matrix(self, angles: Array) -> Array:
        """Return the 3x3 generalized mass matrix M(q)."""
        q = np.asarray(angles, dtype=float)
        matrix = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                start = max(i, j)
                matrix[i, j] = sum(
                    self._masses[k]
                    * self._lengths[i]
                    * self._lengths[j]
                    * np.cos(q[i] - q[j])
                    for k in range(start, 3)
                )
        return matrix

    def potential_energy(self, angles: Array) -> float:
        """Return potential energy, with the lowest configuration at zero."""
        q = np.asarray(angles, dtype=float)
        return float(
            sum(
                self._masses[k]
                * self.parameters.gravity
                * self._lengths[i]
                * (1.0 - np.cos(q[i]))
                for i in range(3)
                for k in range(i, 3)
            )
        )

    def hamiltonian(self, state: Array) -> float:
        """Return H(q,p) = 1/2 p.T M(q)^-1 p + V(q)."""
        q = np.asarray(state[:3], dtype=float)
        p = np.asarray(state[3:], dtype=float)
        velocity = np.linalg.solve(self.mass_matrix(q), p)
        kinetic = 0.5 * float(p @ velocity)
        return kinetic + self.potential_energy(q)

    def gradient_hamiltonian(self, state: Array) -> Array:
        """Return the analytic gradient of the Hamiltonian."""
        q = np.asarray(state[:3], dtype=float)
        p = np.asarray(state[3:], dtype=float)
        velocity = np.linalg.solve(self.mass_matrix(q), p)
        gradient_q = np.zeros(3, dtype=float)

        for i in range(3):
            d_mass = np.zeros((3, 3), dtype=float)
            for a in range(3):
                for b in range(3):
                    if i == a or i == b:
                        sign = 1.0 if i == a else -1.0
                        if i == a and i == b:
                            sign = 0.0
                        start = max(a, b)
                        d_mass[a, b] = sum(
                            self._masses[k]
                            * self._lengths[a]
                            * self._lengths[b]
                            * (-np.sin(q[a] - q[b]))
                            * sign
                            for k in range(start, 3)
                        )
            d_kinetic = -0.5 * float(velocity @ d_mass @ velocity)
            d_potential = (
                self.parameters.gravity
                * self._lengths[i]
                * sum(self._masses[i:])
                * np.sin(q[i])
            )
            gradient_q[i] = d_kinetic + d_potential

        return np.concatenate((gradient_q, velocity))

    def initial_state(
        self, angles: Iterable[float], angular_velocities: Iterable[float]
    ) -> Array:
        """Convert angles and angular velocities to canonical state variables."""
        q = np.asarray(tuple(angles), dtype=float)
        omega = np.asarray(tuple(angular_velocities), dtype=float)
        if q.shape != (3,) or omega.shape != (3,):
            raise ValueError("angles and angular_velocities must each have length 3.")
        return np.concatenate((q, self.mass_matrix(q) @ omega))

    def angular_velocities(self, state: Array) -> Array:
        """Return angular velocities from a canonical state."""
        return np.linalg.solve(self.mass_matrix(state[:3]), state[3:])

    def positions(self, angles: Array) -> Array:
        """Return pivot and three bob positions as an array of shape (4, 2)."""
        q = np.asarray(angles, dtype=float)
        x = np.concatenate(([0.0], np.cumsum(self._lengths * np.sin(q))))
        y = np.concatenate(([0.0], -np.cumsum(self._lengths * np.cos(q))))
        return np.column_stack((x, y))

    def _discrete_gradient(self, state0: Array, state1: Array) -> Array:
        delta = state1 - state0
        midpoint = 0.5 * (state0 + state1)
        gradient = self.gradient_hamiltonian(midpoint)
        norm_squared = float(delta @ delta)
        if norm_squared < 1.0e-28:
            return gradient
        correction = (
            self.hamiltonian(state1)
            - self.hamiltonian(state0)
            - float(gradient @ delta)
        ) / norm_squared
        return gradient + correction * delta

    def _residual(self, state1: Array, state0: Array, step: float) -> Array:
        gradient = self._discrete_gradient(state0, state1)
        symplectic_gradient = np.concatenate((gradient[3:], -gradient[:3]))
        return state1 - state0 - step * symplectic_gradient

    def _rk4_step(self, state: Array, step: float) -> Array:
        def rhs(value: Array) -> Array:
            gradient = self.gradient_hamiltonian(value)
            return np.concatenate((gradient[3:], -gradient[:3]))

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _step_once(self, state: Array, step: float) -> Array:
        """Solve one implicit discrete-gradient step without subdivision."""
        if step <= 0.0:
            raise ValueError("step must be positive.")
        state1 = self._rk4_step(np.asarray(state, dtype=float), step)
        tolerance = self.parameters.newton_tolerance

        for _ in range(self.parameters.newton_max_iterations):
            residual = self._residual(state1, state, step)
            if np.linalg.norm(residual, ord=np.inf) < tolerance:
                return state1

            jacobian = np.empty((6, 6), dtype=float)
            for column in range(6):
                epsilon = 2.0e-7 * max(1.0, abs(state1[column]))
                plus = state1.copy()
                minus = state1.copy()
                plus[column] += epsilon
                minus[column] -= epsilon
                jacobian[:, column] = (
                    self._residual(plus, state, step)
                    - self._residual(minus, state, step)
                ) / (2.0 * epsilon)
            try:
                correction = np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Newton法のヤコビ行列が特異です。") from exc
            residual_norm = np.linalg.norm(residual, ord=np.inf)
            damping = 1.0
            while damping >= 1.0 / 128.0:
                candidate = state1 - damping * correction
                candidate_norm = np.linalg.norm(
                    self._residual(candidate, state, step), ord=np.inf
                )
                if candidate_norm < residual_norm:
                    state1 = candidate
                    break
                damping *= 0.5
            else:
                raise RuntimeError(
                    "Newton法の更新で残差を減少させられませんでした。"
                    "dtを小さくしてください。"
                )

        raise RuntimeError(
            "離散勾配法のNewton反復が収束しませんでした。"
            "dtまたは初期角速度を小さくしてください。"
        )

    def step(self, state: Array, step: float) -> Array:
        """Advance with automatic local subdivision after Newton failure.

        A difficult state is retried using two half-steps. Each accepted
        substep still uses the same energy-preserving discrete-gradient
        equation; subdivision only improves the Newton initial guess.
        """
        if step <= 0.0:
            raise ValueError("step must be positive.")
        initial = np.asarray(state, dtype=float)
        substep = step
        for _ in range(self.parameters.max_step_subdivisions + 1):
            current = initial.copy()
            try:
                count = max(1, int(round(step / substep)))
                actual_substep = step / count
                for _ in range(count):
                    current = self._step_once(current, actual_substep)
                return current
            except RuntimeError:
                substep *= 0.5
        raise RuntimeError(
            "離散勾配法が局所的な時間刻み分割後も収束しませんでした。"
            f"要求dt={step:g}, 最大分割回数={self.parameters.max_step_subdivisions}."
        )

    def simulate(
        self,
        angles: Iterable[float],
        angular_velocities: Iterable[float] = (0.0, 0.0, 0.0),
        t_max: float = 10.0,
        dt: float = 0.02,
        max_seconds: float | None = None,
    ) -> SimulationResult:
        """Simulate and return the full time history."""
        if t_max <= 0.0 or dt <= 0.0:
            raise ValueError("t_max and dt must be positive.")
        if max_seconds is not None and max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive when provided.")
        started_at = time.perf_counter()
        steps = int(np.ceil(t_max / dt))
        sim_time = np.arange(steps + 1, dtype=float) * dt
        state = self.initial_state(angles, angular_velocities)
        states = np.empty((steps + 1, 6), dtype=float)
        states[0] = state
        for index in range(steps):
            if max_seconds is not None and time.perf_counter() - started_at > max_seconds:
                raise TimeoutError(
                    f"simulation exceeded the {max_seconds:g}-second limit"
                )
            states[index + 1] = self.step(states[index], dt)
        return SimulationResult(
            time=sim_time,
            angles=states[:, :3],
            angular_velocities=np.array(
                [self.angular_velocities(value) for value in states]
            ),
            energy=np.array([self.hamiltonian(value) for value in states]),
        )

    def simulate_final(
        self,
        angles: Iterable[float],
        angular_velocities: Iterable[float] = (0.0, 0.0, 0.0),
        t_max: float = 10.0,
        dt: float = 0.02,
        max_seconds: float | None = None,
    ) -> Array:
        """Simulate while retaining only the final angle vector."""
        result = self.simulate(
            angles, angular_velocities, t_max, dt, max_seconds=max_seconds
        )
        return result.angles[-1].copy()

    @staticmethod
    def wrapped_angle_difference(first: Array, second: Array) -> Array:
        """Return componentwise angular difference in the interval [-pi, pi]."""
        delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
        return np.arctan2(np.sin(delta), np.cos(delta))

    def finite_time_sensitivity(
        self,
        angles: Iterable[float],
        angular_velocities: Iterable[float] = (0.0, 0.0, 0.0),
        t_max: float = 10.0,
        dt: float = 0.02,
        perturbation: float = 1.0e-5,
    ) -> float:
        """Estimate local chaos from six perturbed initial-angle trajectories."""
        if perturbation <= 0.0:
            raise ValueError("perturbation must be positive.")
        center = np.asarray(tuple(angles), dtype=float)
        final_center = self.simulate_final(center, angular_velocities, t_max, dt)
        differences = []
        for axis in range(3):
            for direction in (-1.0, 1.0):
                neighbor = center.copy()
                neighbor[axis] += direction * perturbation
                final_neighbor = self.simulate_final(
                    neighbor, angular_velocities, t_max, dt
                )
                differences.append(
                    np.linalg.norm(
                        self.wrapped_angle_difference(final_neighbor, final_center)
                    )
                    / perturbation
                )
        return float(np.mean(differences))

    def finite_time_sensitivity_from_state(
        self,
        state: Array,
        dt: float,
        perturbation: float = 1.0e-5,
        horizon_steps: int = 1,
    ) -> float:
        """Estimate local sensitivity around an already evolved state.

        Six copies are perturbed in the angle directions, advanced for a
        short finite horizon, and compared with the unperturbed copy. This is
        useful for real-time diagnostics because the current state need not be
        simulated from time zero again.
        """
        if dt <= 0.0 or perturbation <= 0.0 or horizon_steps < 1:
            raise ValueError("dt, perturbation, and horizon_steps must be positive.")
        center = np.asarray(state, dtype=float).copy()
        center_final = center
        for _ in range(horizon_steps):
            center_final = self.step(center_final, dt)

        differences = []
        for axis in range(3):
            for direction in (-1.0, 1.0):
                neighbor = np.asarray(state, dtype=float).copy()
                neighbor[axis] += direction * perturbation
                for _ in range(horizon_steps):
                    neighbor = self.step(neighbor, dt)
                differences.append(
                    np.linalg.norm(
                        self.wrapped_angle_difference(
                            neighbor[:3], center_final[:3]
                        )
                    )
                    / perturbation
                )
        return float(np.mean(differences))

"""Real-time animation of one triple pendulum.

Run this file directly from the directory containing ``triple_pendulum.py``.
The animation advances the same discrete-gradient solver one step at a time,
so a Newton failure is shown in the window instead of being hidden.
"""

from __future__ import annotations

import argparse
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from triple_pendulum import TriplePendulumParameters, TriplePendulumSimulator


class RealtimeTriplePendulum:
    """Interactive real-time viewer for a single planar triple pendulum."""

    def __init__(
        self,
        initial_angles: tuple[float, float, float],
        dt: float = 0.005,
        steps_per_frame: int = 4,
        interval_ms: int = 25,
        chaos_perturbation: float = 1.0e-5,
        chaos_horizon_steps: int = 1,
    ) -> None:
        if dt <= 0.0 or steps_per_frame < 1 or interval_ms < 1:
            raise ValueError("dt, steps_per_frame, and interval_ms must be positive.")
        if chaos_perturbation <= 0.0 or chaos_horizon_steps < 1:
            raise ValueError(
                "chaos_perturbation and chaos_horizon_steps must be positive."
            )

        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.interval_ms = interval_ms
        self.chaos_perturbation = chaos_perturbation
        self.chaos_horizon_steps = chaos_horizon_steps
        self.initial_angles = np.asarray(initial_angles, dtype=float)
        parameters = TriplePendulumParameters()
        self.simulator = TriplePendulumSimulator(parameters)
        self.state = self.simulator.initial_state(self.initial_angles, (0.0, 0.0, 0.0))
        self.time = 0.0
        self.running = True
        self.error_message = ""
        self.chaos_value = 0.0
        self.chaos_values = [0.0]

        self.root = tk.Tk()
        self.root.title("三重振り子 - 離散勾配法リアルタイム表示")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.axes = self.figure.add_subplot(121)
        self.energy_axes = self.figure.add_subplot(122)
        self.axes.set_aspect("equal", adjustable="box")
        limit = sum(parameters.lengths) + 0.3
        self.axes.set_xlim(-limit, limit)
        self.axes.set_ylim(-limit, limit)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.set_title("physical space")
        self.axes.grid(True, alpha=0.25)
        self.energy_axes.set_title("energy error")
        self.energy_axes.set_xlabel("time")
        self.energy_axes.set_ylabel(r"$H(t)-H(0)$")
        self.energy_axes.grid(True, alpha=0.25)

        self.rod_line, = self.axes.plot([], [], "o-", lw=2.5, color="tab:blue")
        self.bob_points, = self.axes.plot([], [], "o", ms=8, color="tab:red")
        self.energy_line, = self.energy_axes.plot([], [], color="tab:green")
        self.status = self.figure.text(0.02, 0.02, "", fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(controls, text="Pause / Resume", command=self.toggle).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Reset", command=self.reset).pack(side=tk.LEFT)
        tk.Button(controls, text="Close", command=self.close).pack(side=tk.RIGHT)

        self.times = [0.0]
        self.energy_errors = [0.0]
        self.animation = FuncAnimation(
            self.figure,
            self.update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def reset(self) -> None:
        """Reset the state and clear the displayed history."""
        self.state = self.simulator.initial_state(self.initial_angles, (0.0, 0.0, 0.0))
        self.time = 0.0
        self.times = [0.0]
        self.energy_errors = [0.0]
        self.chaos_value = 0.0
        self.chaos_values = [0.0]
        self.running = True
        self.error_message = ""

    def toggle(self) -> None:
        self.running = not self.running

    def update(self, _frame: int) -> None:
        if self.running and not self.error_message:
            try:
                for _ in range(self.steps_per_frame):
                    self.state = self.simulator.step(self.state, self.dt)
                    self.time += self.dt
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                self.running = False
                self.error_message = f"{type(exc).__name__}: {exc}"

        positions = self.simulator.positions(self.state[:3])
        self.rod_line.set_data(positions[:, 0], positions[:, 1])
        self.bob_points.set_data(positions[1:, 0], positions[1:, 1])
        energy_error = self.simulator.hamiltonian(self.state) - self.simulator.hamiltonian(
            self.simulator.initial_state(self.initial_angles, (0.0, 0.0, 0.0))
        )
        if not self.error_message:
            self.times.append(self.time)
            self.energy_errors.append(energy_error)
        self.energy_line.set_data(self.times, self.energy_errors)
        self.energy_axes.relim()
        self.energy_axes.autoscale_view()

        if not self.error_message:
            try:
                self.chaos_value = (
                    self.simulator.finite_time_sensitivity_from_state(
                        self.state,
                        dt=self.dt,
                        perturbation=self.chaos_perturbation,
                        horizon_steps=self.chaos_horizon_steps,
                    )
                )
                self.chaos_values.append(self.chaos_value)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                self.running = False
                self.error_message = f"{type(exc).__name__}: {exc}"
        chaos_min = min(self.chaos_values)
        chaos_max = max(self.chaos_values)
        if chaos_max > chaos_min:
            chaos_alpha = (self.chaos_value - chaos_min) / (chaos_max - chaos_min)
        else:
            chaos_alpha = 1.0
        chaos_alpha = float(np.clip(chaos_alpha, 0.08, 1.0))
        self.rod_line.set_alpha(chaos_alpha)
        self.bob_points.set_alpha(chaos_alpha)

        if self.error_message:
            status = "STOPPED: " + self.error_message
        else:
            status = (
                f"t = {self.time:.3f} s | dt = {self.dt:g} s | "
                f"energy error = {energy_error:.3e} | "
                f"chaos = {self.chaos_value:.3e} "
                f"(min={chaos_min:.3e}, max={chaos_max:.3e}, "
                f"alpha={chaos_alpha:.2f})"
            )
        self.status.set_text(status)
        self.status.set_color("crimson" if self.error_message else "black")
        self.canvas.draw_idle()

    def close(self) -> None:
        self.animation.event_source.stop()
        self.root.destroy()

    def show(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        default=(1.2, 1.8, 2.4),
        metavar=("THETA1", "THETA2", "THETA3"),
        help="initial absolute angles in radians",
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--interval-ms", type=int, default=25)
    parser.add_argument("--chaos-perturbation", type=float, default=1.0e-5)
    parser.add_argument("--chaos-horizon-steps", type=int, default=1)
    args = parser.parse_args()
    RealtimeTriplePendulum(
        tuple(args.angles),
        dt=args.dt,
        steps_per_frame=args.steps_per_frame,
        interval_ms=args.interval_ms,
        chaos_perturbation=args.chaos_perturbation,
        chaos_horizon_steps=args.chaos_horizon_steps,
    ).show()


if __name__ == "__main__":
    main()

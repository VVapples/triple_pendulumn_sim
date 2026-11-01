"""Real-time triple-pendulum viewer with speed control and a fading endpoint trace."""

# run with: python realtime_triple_pendulum2.py [--angles ANGLES] [--dt DT] [--steps-per-frame STEPS_PER_FRAME] [--interval-ms INTERVAL_MS] [--fade-start-dt FADE_START_DT] [--fade-time FADE_TIME]
# example: python realtime_triple_pendulum2.py --angles 1.2 1.8 2.4 --dt 0.005 --steps-per-frame 4 --interval-ms 25 --fade-start-dt 10.0 --fade-time 10.0

from __future__ import annotations

import argparse
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from triple_pendulum import TriplePendulumParameters, TriplePendulumSimulator


class RealtimeTriplePendulum2:
    """Display one triple pendulum with an adjustable simulation speed."""

    def __init__(
        self,
        initial_angles: tuple[float, float, float],
        dt: float = 0.005,
        steps_per_frame: int = 4,
        interval_ms: int = 25,
        fade_start_dt: float = 10.0,
        fade_time: float = 10.0,
    ) -> None:
        if dt <= 0 or steps_per_frame < 1 or interval_ms < 1:
            raise ValueError("dt, steps_per_frame, and interval_ms must be positive.")
        if fade_start_dt < 0 or fade_time <= 0:
            raise ValueError("fade_start_dt must be non-negative and fade_time positive.")

        self.dt = dt
        self.base_steps_per_frame = steps_per_frame
        self.steps_per_frame = steps_per_frame
        self.interval_ms = interval_ms
        self.fade_start_dt = fade_start_dt
        self.fade_time = fade_time
        self.speed_factor = 1.0
        self.initial_angles = np.asarray(initial_angles, dtype=float)
        self.simulator = TriplePendulumSimulator(TriplePendulumParameters())
        self.initial_state = self.simulator.initial_state(
            self.initial_angles, (0.0, 0.0, 0.0)
        )
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.running = True
        self.error_message = ""
        self.trace_times: list[float] = [0.0]
        self.trace_points: list[np.ndarray] = [
            self.simulator.positions(self.state[:3])[-1].copy()
        ]

        self.root = tk.Tk()
        self.root.title("Triple pendulum - speed and fading trace")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.axes = self.figure.add_subplot(121)
        self.energy_axes = self.figure.add_subplot(122)
        parameters = self.simulator.parameters
        limit = sum(parameters.lengths) + 0.3
        self.axes.set_aspect("equal", adjustable="box")
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

        self.trace = LineCollection([], linewidths=1.5)
        self.axes.add_collection(self.trace)
        self.rod_line, = self.axes.plot([], [], "o-", lw=2.5, color="tab:blue")
        self.bob_points, = self.axes.plot([], [], "o", ms=8, color="tab:red")
        self.energy_line, = self.energy_axes.plot([], [], color="tab:green")
        self.status = self.figure.text(0.02, 0.02, "", fontsize=9)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(controls, text="Pause / Resume", command=self.toggle).pack(side=tk.LEFT)
        tk.Button(controls, text="Slower", command=self.slower).pack(side=tk.LEFT)
        tk.Button(controls, text="Faster", command=self.faster).pack(side=tk.LEFT)
        tk.Button(controls, text="Reset", command=self.reset).pack(side=tk.LEFT)
        tk.Button(controls, text="Clear trace", command=self.clear_trace).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Close", command=self.close).pack(side=tk.RIGHT)
        self.root.bind("<Left>", lambda _event: self.slower())
        self.root.bind("<Right>", lambda _event: self.faster())
        self.root.bind("<space>", lambda _event: self.toggle())

        self.times = [0.0]
        self.energy_errors = [0.0]
        self.animation = FuncAnimation(
            self.figure,
            self.update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def faster(self) -> None:
        self.speed_factor = min(64.0, self.speed_factor * 2.0)
        self.steps_per_frame = max(
            1, int(round(self.base_steps_per_frame * self.speed_factor))
        )

    def slower(self) -> None:
        self.speed_factor = max(0.125, self.speed_factor / 2.0)
        self.steps_per_frame = max(
            1, int(round(self.base_steps_per_frame * self.speed_factor))
        )

    def toggle(self) -> None:
        self.running = not self.running

    def clear_trace(self) -> None:
        endpoint = self.simulator.positions(self.state[:3])[-1].copy()
        self.trace_times = [self.time]
        self.trace_points = [endpoint]

    def reset(self) -> None:
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.times = [0.0]
        self.energy_errors = [0.0]
        self.running = True
        self.error_message = ""
        self.clear_trace()

    def _update_trace(self) -> None:
        now = self.time
        points = np.asarray(self.trace_points)
        times = np.asarray(self.trace_times)
        if len(points) < 2:
            self.trace.set_segments([])
            return

        ages = now - times[:-1]
        keep = ages <= self.fade_start_dt + self.fade_time
        points = points[np.r_[keep, True]]
        times = times[np.r_[keep, True]]
        self.trace_points = list(points)
        self.trace_times = list(times)
        if len(points) < 2:
            self.trace.set_segments([])
            return

        segments = np.stack((points[:-1], points[1:]), axis=1)
        segment_ages = now - times[:-1]
        alpha = np.ones(len(segments))
        fading = segment_ages > self.fade_start_dt
        alpha[fading] = 1.0 - (
            segment_ages[fading] - self.fade_start_dt
        ) / self.fade_time
        alpha = np.clip(alpha, 0.0, 1.0)
        colors = np.tile(np.array([0.05, 0.35, 0.85, 1.0]), (len(segments), 1))
        colors[:, 3] = alpha
        self.trace.set_segments(segments)
        self.trace.set_color(colors)

    def update(self, _frame: int) -> None:
        if self.running and not self.error_message:
            try:
                for _ in range(self.steps_per_frame):
                    self.state = self.simulator.step(self.state, self.dt)
                    self.time += self.dt
                    endpoint = self.simulator.positions(self.state[:3])[-1].copy()
                    self.trace_times.append(self.time)
                    self.trace_points.append(endpoint)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                self.running = False
                self.error_message = f"{type(exc).__name__}: {exc}"

        positions = self.simulator.positions(self.state[:3])
        self.rod_line.set_data(positions[:, 0], positions[:, 1])
        self.bob_points.set_data(positions[1:, 0], positions[1:, 1])
        initial_energy = self.simulator.hamiltonian(self.initial_state)
        energy_error = self.simulator.hamiltonian(self.state) - initial_energy
        if not self.error_message:
            self.times.append(self.time)
            self.energy_errors.append(energy_error)
        self.energy_line.set_data(self.times, self.energy_errors)
        self.energy_axes.relim()
        self.energy_axes.autoscale_view()
        self._update_trace()

        if self.error_message:
            text = "STOPPED: " + self.error_message
        else:
            text = (
                f"t={self.time:.2f}s | speed={self.speed_factor:g}x | "
                f"steps/frame={self.steps_per_frame} | "
                f"trace fade starts after {self.fade_start_dt:g}s, "
                f"fades over {self.fade_time:g}s | "
                f"energy error={energy_error:.3e}"
            )
        self.status.set_text(text)
        self.status.set_color("crimson" if self.error_message else "black")
        self.canvas.draw_idle()

    def close(self) -> None:
        self.animation.event_source.stop()
        self.root.destroy()

    def show(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", nargs=3, type=float, default=(1.2, 1.8, 2.4))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--interval-ms", type=int, default=25)
    parser.add_argument(
        "--fade-start-dt",
        type=float,
        default=10.0,
        help="simulation seconds before old trace segments start fading",
    )
    parser.add_argument(
        "--fade-time",
        type=float,
        default=10.0,
        help="simulation seconds over which trace segments fade to zero",
    )
    args = parser.parse_args()
    RealtimeTriplePendulum2(
        tuple(args.angles),
        dt=args.dt,
        steps_per_frame=args.steps_per_frame,
        interval_ms=args.interval_ms,
        fade_start_dt=args.fade_start_dt,
        fade_time=args.fade_time,
    ).show()


if __name__ == "__main__":
    main()

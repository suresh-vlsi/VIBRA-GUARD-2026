"""Stable harmonic feed-forward active vibration rejection.

This is a simulation/research controller, not a certified industrial safety
controller. It preserves the ActiveRejector interface used by Agent.
"""
import numpy as np
from scipy import signal
from scipy.optimize import least_squares

from .config import *
from .simulator import _accel_tf


def model_eval(p, f):
    """Evaluate the identified secondary-path model at frequency f (Hz)."""
    fn, z, gain, delay = p
    b, a = _accel_tf(fn, z)
    _, h = signal.freqz(
        b, a,
        worN=np.atleast_1d(2.0 * np.pi * f / FS)
    )
    return (gain * h * np.exp(-2j * np.pi * f * delay / FS))[0]


class ActiveRejector:
    def __init__(
        self,
        mu=0.04,
        leakage=0.999,
        max_weight=None,
        max_update=0.05,
        min_secondary_gain=1e-3,
    ):
        # Conservative normalized adaptation; tune only using controlled tests.
        self.mu = float(mu)
        self.leakage = float(leakage)
        self.max_weight = (
            float(max_weight) if max_weight is not None
            else ACT_LIMIT / 2.0
        )
        self.max_update = float(max_update)
        self.min_secondary_gain = float(min_secondary_gain)

        self.enabled = False
        self.harm = (1, 2)
        self.w = {}
        self._ph = None
        self._fr = 0.0
        self.fn_est = FN_HEALTHY
        self.p = None

        self.last_output_peak = 0.0
        self.last_update_peak = 0.0
        self.saturation_count = 0
        self.skipped_updates = 0
        self.unstable_count = 0

    # ----------------------------------------------------------
    # Secondary-path identification
    # ----------------------------------------------------------

    def identify(self, machine):
        """Identify actuator-to-IMU path from a machine probe.

        The probe implementation and its amplitude are controlled by
        Machine.probe(); this routine only estimates the resulting FRF.
        """
        u, y = machine.probe()

        nperseg = min(2048, len(u))
        f, Pxy = signal.csd(u, y, fs=FS, nperseg=nperseg)
        _, Pxx = signal.welch(u, fs=FS, nperseg=nperseg)

        H = Pxy / (Pxx + 1e-12)
        band = (f > 30) & (f < 140)

        if not np.any(band):
            raise RuntimeError("Insufficient frequency range for path identification")

        fn0 = float(f[band][np.argmax(np.abs(H[band]))])

        def model(params, freq):
            fn, zeta, gain, delay = params
            b, a = _accel_tf(fn, zeta)
            _, response = signal.freqz(
                b, a, worN=2.0 * np.pi * freq / FS
            )
            return gain * response * np.exp(
                -2j * np.pi * freq * delay / FS
            )

        def residual(params):
            difference = model(params, f[band]) - H[band]
            return np.r_[difference.real, difference.imag]

        fit = least_squares(
            residual,
            x0=[fn0, 0.03, 1.0, 0.0],
            bounds=(
                [20.0, 0.003, 0.2, -2.0],
                [200.0, 0.3, 5.0, 3.0],
            ),
            max_nfev=300,
        )

        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise RuntimeError("Secondary-path model fitting failed")

        self.p = fit.x
        self.fn_est = float(self.p[0])
        return self.fn_est

    def S(self, frequency):
        """Complex secondary-path response at frequency in Hz."""
        if self.p is None:
            raise RuntimeError("Call identify(machine) before enabling control")
        return complex(model_eval(self.p, frequency))

    # ----------------------------------------------------------
    # Controller lifecycle
    # ----------------------------------------------------------

    def enable(self, harmonics=(1, 2)):
        harmonics = tuple(int(k) for k in harmonics)
        if not harmonics or any(k <= 0 for k in harmonics):
            raise ValueError("Harmonics must be positive integers")
        if self.p is None:
            raise RuntimeError("Secondary path must be identified before control")

        self.harm = harmonics
        self.w = {k: 0.0j for k in self.harm}
        self.enabled = True
        self.last_output_peak = 0.0
        self.last_update_peak = 0.0

    def disable(self):
        self.enabled = False
        self.w = {}
        self._ph = None
        self._fr = 0.0

    def reset(self):
        """Reset adaptive coefficients while retaining the identified path."""
        self.w = {k: 0.0j for k in self.harm}

    # ----------------------------------------------------------
    # Actuator output
    # ----------------------------------------------------------

    def output(self, machine, n):
        """Generate bounded harmonic anti-vibration force for next block."""
        if not self.enabled:
            return None

        self._ph = machine.peek_phase(n)
        self._fr = float(machine.rpm / 60.0)

        output = np.zeros(n, dtype=float)

        for harmonic, weight in self.w.items():
            phase = harmonic * self._ph
            output += (
                weight.real * np.cos(phase)
                - weight.imag * np.sin(phase)
            )

        peak_before = float(np.max(np.abs(output))) if output.size else 0.0
        bounded = np.clip(output, -ACT_LIMIT, ACT_LIMIT)

        if peak_before > ACT_LIMIT:
            self.saturation_count += 1

        self.last_output_peak = (
            float(np.max(np.abs(bounded))) if bounded.size else 0.0
        )
        return bounded

    # ----------------------------------------------------------
    # Bounded normalized FxLMS adaptation
    # ----------------------------------------------------------

    def update(self, error):
        """Adapt harmonic coefficients using the measured block error.

        Uses a normalized complex secondary-path update, a bounded per-block
        coefficient change, leakage, and coefficient magnitude limits.
        """
        if not self.enabled or self._ph is None:
            return

        error = np.asarray(error, dtype=float)
        if error.ndim != 1 or error.size == 0:
            self.skipped_updates += 1
            return

        if not np.all(np.isfinite(error)):
            self.skipped_updates += 1
            self.unstable_count += 1
            return

        error = error - np.mean(error)
        n = len(error)
        max_delta_seen = 0.0

        for harmonic in self.harm:
            if self._fr <= 0:
                continue

            phase = harmonic * self._ph
            error_phasor = (
                2.0 / n
                * np.sum(error * np.exp(-1j * phase))
            )

            secondary = self.S(harmonic * self._fr)
            secondary_power = abs(secondary) ** 2

            if (
                not np.isfinite(secondary_power)
                or secondary_power < self.min_secondary_gain ** 2
            ):
                self.skipped_updates += 1
                continue

            # Normalized complex FxLMS gradient:
            # E / S == E * conj(S) / |S|^2.
            gradient = (
                error_phasor * np.conj(secondary)
                / (secondary_power + 1e-8)
            )

            delta = self.mu * gradient

            delta_magnitude = abs(delta)
            if not np.isfinite(delta_magnitude):
                self.skipped_updates += 1
                self.unstable_count += 1
                continue

            # Bound coefficient movement in one update.
            if delta_magnitude > self.max_update:
                delta *= self.max_update / delta_magnitude
                delta_magnitude = self.max_update

            candidate = (
                self.leakage * self.w.get(harmonic, 0.0j) - delta
            )

            # Bound each harmonic coefficient. The total waveform is also
            # independently clipped in output().
            candidate_magnitude = abs(candidate)
            if candidate_magnitude > self.max_weight:
                candidate *= self.max_weight / candidate_magnitude

            if not (
                np.isfinite(candidate.real)
                and np.isfinite(candidate.imag)
            ):
                self.skipped_updates += 1
                self.unstable_count += 1
                continue

            self.w[harmonic] = candidate
            max_delta_seen = max(max_delta_seen, delta_magnitude)

        self.last_update_peak = max_delta_seen

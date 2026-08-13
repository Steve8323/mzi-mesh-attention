"""OTDM front end: the MRR pulse train, one-slot delay ladder, and full
frequency-domain propagation of the pulse trains through the (wavelength-
dependent) calibrated mesh.

Physical constants pinned to the silica repo measurements:
  slot period  T_s = 1/FSR,  FSR = c/(n_g * 2 pi R),  R = 16 um ring,
               n_g = 1.4948 at the zero-GVD width w = 0.991 um
               -> FSR = 1.995 THz, T_s = 501.26 fs
  pulse        T0 = 9.86 fs Gaussian (FWHM 16.4 fs), the shortest 5 mm-safe
               pulse from pulse_5mm.py (bandwidth 26.8 THz)
  one-slot delay = v_g * T_s = 100.53 um of bus — exactly one ring
               circumference of matched waveguide per tap.

Everything propagates as a complex envelope on a uniform time grid; the mesh
is applied per frequency bin, M(lambda) built from the FDTD-calibrated
kappa(lambda), node amplitude a(lambda), and 1/lambda electrode phases.
kappa data cover 1470-1630 nm; outside, np.interp clamps to the edge value
(flagged: the extreme +-3 dB tails of the 16.4 fs spectrum reuse edge kappa).
"""
import json
import os

import numpy as np

from node import NodeCal, coupler, LAM0
from clements import mesh_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299792458.0
NG = 1.4947983969526526          # dispersion.json @ w=0.991
R_RING = 16e-6
FSR = C0 / (NG * 2 * np.pi * R_RING)      # 1.9950e12 Hz
T_SLOT = 1.0 / FSR                        # 501.26 fs
T0_PULSE = 9.859442324056654e-15          # pulse_5mm.json, w=0.991
NU0 = C0 / 1.550e-6


def time_grid(n_slots, dt=1e-15, guard=4):
    """Uniform grid covering n_slots + guard slots."""
    T = (n_slots + guard) * T_SLOT
    n = int(np.ceil(T / dt))
    return np.arange(n) * dt


def make_train(tokens, t, t0=None, T0=T0_PULSE):
    """Complex envelope: one Gaussian pulse per slot, amplitude tokens[k]."""
    if t0 is None:
        t0 = 2 * T_SLOT
    E = np.zeros(len(t), dtype=complex)
    for k, s in enumerate(tokens):
        E += s * np.exp(-((t - t0 - k * T_SLOT) ** 2) / (2 * T0 ** 2))
    return E


def delayed_inputs(tokens, t, N, extra_phase=None, T0=T0_PULSE):
    """Channel j = the SAME train delayed by j slots (j = 0..N-1).
    extra_phase[j] adds a static path offset (radians at the carrier,
    implemented as frequency-scaled phase of a physical path length)."""
    X = np.zeros((N, len(t)), dtype=complex)
    for j in range(N):
        X[j] = make_train(tokens, t, t0=2 * T_SLOT + j * T_SLOT, T0=T0)
    if extra_phase is not None:
        f = np.fft.fftfreq(len(t), d=t[1] - t[0])
        for j, psi in enumerate(extra_phase):
            if psi:
                ph = psi * (NU0 + f) / NU0        # dispersionless path
                X[j] = np.fft.ifft(np.fft.fft(X[j]) * np.exp(1j * ph))
    return X


def propagate(X, t, transfer):
    """Apply an NxN frequency-dependent transfer to the channel envelopes.

    transfer(lam_um) -> NxN complex matrix. Evaluated on the FFT grid."""
    f = np.fft.fftfreq(len(t), d=t[1] - t[0])
    lam = C0 / np.clip(NU0 + f, 1e12, None) * 1e6   # um; clip: empty bins
    Xf = np.fft.fft(X, axis=1)
    Ms = transfer(lam)                       # shape (nbin, N, N)
    Yf = np.einsum("fij,jf->if", Ms, Xf)
    return np.fft.ifft(Yf, axis=1)


def mesh_transfer_fn(nodes, d, N, cal, phase_screen=None):
    """Vectorized wavelength-dependent mesh transfer (nbin, N, N).

    phase_screen(lam_um) -> radians added to every node's INTERNAL arm
    phase: the unmodeled smooth MZI phase dispersion measured by the
    two-input quadrature FDTD (node2/coupler2 runs): ~ -0.253 deg/nm,
    zero at 1550 after the static per-node calibration."""
    def fn(lam):
        lam = np.asarray(lam)
        kap = cal.kappa(lam)
        amp = cal.amp(lam)
        scale = LAM0 / lam
        scr = phase_screen(lam) if phase_screen is not None else 0.0
        nb = len(lam)
        M = np.broadcast_to(np.eye(N, dtype=complex), (nb, N, N)).copy()
        for nd in nodes:
            th = nd["theta"] * scale + scr
            ph = nd["phi"] * scale
            c, s = np.cos(kap), np.sin(kap)
            e = np.exp(1j * th)
            A = e * c ** 2 - s ** 2
            B = 1j * s * c * (e + 1.0)
            Dd = c ** 2 - s ** 2 * e
            ep = np.exp(1j * ph)
            m = nd["m"]
            cols = M[:, :, m].copy(), M[:, :, m + 1].copy()
            M[:, :, m] = amp[:, None] * (cols[0] * (A * ep)[:, None]
                                         + cols[1] * (B * ep)[:, None])
            M[:, :, m + 1] = amp[:, None] * (cols[0] * B[:, None]
                                             + cols[1] * Dd[:, None])
        # output phase screen: row i multiplied by exp(i * ang_i(lam))
        M = M * np.exp(1j * (np.angle(d)[None, :, None]
                             * scale[:, None, None]))
        return M
    return fn


def sample_slots(Y, t, n_tokens, N):
    """Peak-sample every output channel at each slot center.

    Slot t of the computation lands at time  2*T_SLOT + t*T_SLOT  (channel-0
    timing; all channels share it because the mesh has equal path lengths).
    Returns (N, n_out) complex samples for windows t = 0..n_out-1."""
    n_out = n_tokens + N - 1
    idx = [np.argmin(np.abs(t - (2 * T_SLOT + k * T_SLOT)))
           for k in range(n_out)]
    return Y[:, idx]


def slot_energies(E, t, n_out):
    """Per-slot integrated |E|^2 (what a slot-gated detector reports)."""
    dt = t[1] - t[0]
    half = T_SLOT / 2
    out = np.zeros(n_out)
    for k in range(n_out):
        c = 2 * T_SLOT + k * T_SLOT
        w = (t > c - half) & (t <= c + half)
        out[k] = np.sum(np.abs(E[w]) ** 2) * dt
    return out


def coupler_transfer_fn(cal, psi_static=0.0):
    """2x2: just the calibrated 50/50 coupler (the score head)."""
    def fn(lam):
        lam = np.asarray(lam)
        # kappa_grid is per single coupler (s0 = cos(4 kappa) counts both
        # couplers of the calibration MZI), so use it directly here
        kap = cal.kappa(lam)
        c, s = np.cos(kap), np.sin(kap)
        nb = len(lam)
        M = np.zeros((nb, 2, 2), dtype=complex)
        M[:, 0, 0] = c
        M[:, 0, 1] = 1j * s
        M[:, 1, 0] = 1j * s
        M[:, 1, 1] = c
        return M
    return fn


def score_band(tokens, cal, lag=1, dt=1e-15, T0=T0_PULSE,
               leak_correct=True):
    """Photonic attention scores: interfere the train with its lag-slot
    delayed copy (quarter-wave static offset) in the calibrated coupler,
    balanced-detect per slot.

    The FDTD coupler is 45/55, not 50/50, so the balanced output leaks an
    intensity-imbalance term (t^2-r^2)(|a|^2-|b|^2).  leak_correct=True
    subtracts the two SINGLE-ARM balanced measurements (a physically real
    3-measurement calibration; each is just the same receiver with one
    input blocked), which removes that term exactly.

    Returns (measured, expected, gain): expected[k] = tokens[k]*tokens[k-lag]
    scaled by the one fitted global gain."""
    n = len(tokens)
    t = time_grid(n + lag + 2, dt)
    f = np.fft.fftfreq(len(t), d=dt)
    qwave = np.exp(1j * (np.pi / 2) * (NU0 + f) / NU0)
    # arm 0: train delayed by 'lag' slots -> its slot k carries token k-lag
    # arm 1: undelayed train (+ quarter-wave path) -> slot k carries token k
    arm0 = make_train(tokens, t, t0=2 * T_SLOT + lag * T_SLOT, T0=T0)
    arm1 = np.fft.ifft(np.fft.fft(
        make_train(tokens, t, t0=2 * T_SLOT, T0=T0)) * qwave)
    n_out = n + lag
    fn = coupler_transfer_fn(cal)

    def balanced(a, b):
        X2 = np.vstack([a, b])
        Y = propagate(X2, t, fn)
        return (slot_energies(Y[0], t, n_out)
                - slot_energies(Y[1], t, n_out))

    meas = balanced(arm0, arm1)
    if leak_correct:
        z = np.zeros_like(arm0)
        meas = meas - balanced(arm0, z) - balanced(z, arm1)
    exp = np.zeros(n_out)
    for k in range(n_out):
        if 0 <= k - lag and k < n:
            exp[k] = tokens[k] * tokens[k - lag]
    g = (np.dot(meas, exp) / np.dot(exp, exp)) if np.any(exp) else 1.0
    return meas, exp * g, float(g)

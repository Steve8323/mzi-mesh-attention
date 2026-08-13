#!/usr/bin/env python
"""Optimization harness: evaluates one system configuration across the
metrics that matter, with the error budget DECOMPOSED so that 'noise' is
attributed, not assumed.

Metrics per config:
  bits            rms of peak-sampled outputs vs ideal Toeplitz (64 tokens)
  isi_leak        single-token impulse response: energy landing OUTSIDE
                  the N-slot span that the convolution is supposed to
                  occupy, relative to in-span energy  (THE pulse-overlap /
                  dispersion 'noise' the user asked about)
  isi_leak_win    same but slot-window-integrated (what a detector sees)
  broadening      output pulse FWHM / input FWHM (port-averaged)
  walkoff_fs      differential bar/cross group delay per node from the
                  measured phase-screen slope, x N columns (closed form)
  score_rms       balanced-homodyne score error at lag 1 and lag N-1
  density_xB300   from density.py area model

Attribution (fraction of the full-model error variance removed when a
component is idealized):
  screen (node phase dispersion) | kappa(lambda) drift | guide beta2/beta3
  over the delay lines | per-node loss
Knobs: N, T0, slot_fs, prechirp_fs2 (quadratic spectral phase at source).
"""
import json
import os

import numpy as np

from node import NodeCal, LAM0
from clements import decompose, mesh_matrix, haar
import otdm
from otdm import (time_grid, delayed_inputs, propagate, mesh_transfer_fn,
                  sample_slots, score_band, NU0, C0)
from density import mesh_area_mm2, delay_area_mm2, B300_DENS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
cal = NodeCal()

# measured constants
BETA2 = 2.2552113674743342e-28      # s^2/m at w=0.991 (dispersion.json)
BETA3 = -1.9738717909545214e-40     # s^3/m
NG = 1.4947983969526526
CC = json.load(open(os.path.join(OUT, "compact_cal.json")))
SCREEN_C = CC["screen_deg_per_nm"]  # +0.098 deg/nm compact node
KAPPA_C = CC["kappa_1550"]
L_NODE_C = CC["node_len_um"]


class FrozenKappaCal:
    """Wrap a cal but freeze kappa (and optionally loss) at 1550."""

    def __init__(self, base, freeze_kappa=False, lossless=False):
        self.base, self.fk, self.ll = base, freeze_kappa, lossless

    def kappa(self, lam):
        if self.fk:
            return np.full_like(np.asarray(lam, dtype=float),
                                float(self.base.kappa(LAM0)))
        return self.base.kappa(lam)

    def amp(self, lam):
        if self.ll:
            return np.ones_like(np.asarray(lam, dtype=float))
        return self.base.amp(lam)


def guide_phase(lam_um, extra_len_m):
    """beta2/beta3 spectral phase for propagation over extra_len_m."""
    om = 2 * np.pi * C0 / (lam_um * 1e-6)
    dom = om - 2 * np.pi * NU0
    return (BETA2 / 2 * dom ** 2 + BETA3 / 6 * dom ** 3) * extra_len_m


class CompactNodeCal:
    """Consistent COMPACT-node calibration (critic MED-10): kappa(lambda)
    and amp(lambda) from the compact FDTD runs, matching the compact
    screen slope used in the harness."""

    def __init__(self):
        import compact_cal as _cc
        c = _cc.CompactCal()
        self.lam_grid, self.kappa_grid = c.lam_grid, c.kappa_grid
        self.amp_grid = c.amp_grid

    def kappa(self, lam):
        return np.interp(lam, self.lam_grid, self.kappa_grid)

    def amp(self, lam):
        return np.interp(lam, self.lam_grid, self.amp_grid)


_ccal = None


def compact_node_cal():
    global _ccal
    if _ccal is None:
        _ccal = CompactNodeCal()
    return _ccal


def run_config(N=5, T0=120e-15, slot_fs=501.26, prechirp_fs2=0.0,
               use_screen=True, use_kappa_disp=True, use_guide_disp=True,
               use_loss=True, n_tokens=64, seed=42, want_score=True,
               phase_set_rms=0.0, kappa_sigma=0.0, delay_phase_rms=0.0,
               ceo_jitter_rms=0.0, seed_err=None, receiver="peak",
               gate_fs=250.0, sample_offset_fs=0.0, filter_kind=None,
               filter_thz=2.21, kappa_aware=False, kappa_char_rms=0.0,
               comb_linewidth_hz=0.0, amp_scale=0.97, return_arrays=False):
    """Error terms (critic-2 structure):
      phase_set_rms [rad]  per-node theta/phi LUT error (iid)
      kappa_sigma  [rad]   per-node coupler fab disorder, applied EXACTLY
      delay_phase_rms      per-channel trim/thermal residual, scaled by j
                           (channel j gets sigma = j * delay_phase_rms)
      ceo_jitter_rms       comb carrier phase jitter, also j-scaled
      seed_err             separate RNG stream for the error draws (paired
                           comparisons); default: derived from seed
    Receiver: 'peak' (complex sample at slot center + sample_offset_fs) or
      'gate' (complex field integrated over gate_fs around slot center —
      a finite-gate homodyne; washes out chirp).
    filter_kind: None | 'gauss' | 'ring1' | 'ring2' applied to the source
      spectrum (FWHM filter_thz THz): the on-chip bandwidth filter arm."""
    rng = np.random.default_rng(seed)
    rng_e = np.random.default_rng(
        seed_err if seed_err is not None else 10_000 + seed)
    old_slot, old_fsr = otdm.T_SLOT, otdm.FSR
    otdm.T_SLOT = slot_fs * 1e-15
    otdm.FSR = 1.0 / otdm.T_SLOT
    try:
        ncal = compact_node_cal()
        base = FrozenKappaCal(ncal, freeze_kappa=not use_kappa_disp,
                              lossless=not use_loss)
        U = haar(N, rng)
        n_nodes = N * (N - 1) // 2
        dkappa = (rng_e.normal(0, kappa_sigma, n_nodes)
                  if kappa_sigma else None)
        # kappa-aware compile: decompose against the MEASURED per-node
        # couplers.  kappa_char_rms = in-situ characterization error
        # (critic HIGH-3: without it the recovery is a tautology)
        if kappa_aware and dkappa is not None:
            dk_meas = dkappa + (rng_e.normal(0, kappa_char_rms, n_nodes)
                                if kappa_char_rms else 0.0)
            kap_compile = float(base.kappa(LAM0)) + dk_meas
        else:
            kap_compile = float(base.kappa(LAM0))
        nodes, dvec, _ = decompose(U, kap_compile)
        if phase_set_rms:
            for nd in nodes:
                nd["theta"] += rng_e.normal(0, phase_set_rms)
                nd["phi"] += rng_e.normal(0, phase_set_rms)

        def screen(lam_um):
            if not use_screen:
                return np.zeros_like(np.asarray(lam_um, dtype=float))
            return np.deg2rad(SCREEN_C) * (lam_um - 1.55) * 1000.0

        L_slot_m = C0 / NG * otdm.T_SLOT

        # j-scaled channel phase errors: trim/thermal + CEO jitter
        dphase = np.zeros(N)
        for j in range(N):
            if delay_phase_rms:
                dphase[j] += rng_e.normal(0, j * delay_phase_rms)
            if ceo_jitter_rms:
                dphase[j] += rng_e.normal(0, j * ceo_jitter_rms)

        def filter_H(f):
            if filter_kind is None:
                return 1.0
            G = filter_thz * 1e12                 # intensity FWHM
            if filter_kind == "gauss":
                # field |H| such that |H|^2 has FWHM = G (critic-3 MED-4)
                return np.exp(-np.log(2) / 2 * (2 * f / G) ** 2)
            if filter_kind == "ring1":            # 1-pole field Lorentzian
                return 1.0 / (1 + 2j * f / G)
            if filter_kind == "ring2":            # two cascaded rings
                g2 = G / np.sqrt(np.sqrt(2) - 1)  # match intensity FWHM
                return (1.0 / (1 + 2j * f / g2)) ** 2
            raise ValueError(filter_kind)

        def apply_frontend(X, t):
            """source filter + delay-line guide dispersion (channel j:
            j slots of guide) + per-channel carrier phase error +
            optional source prechirp, in the spectral domain."""
            f = np.fft.fftfreq(len(t), d=t[1] - t[0])
            lam = C0 / np.clip(NU0 + f, 1e12, None) * 1e6
            dom = 2 * np.pi * f
            H = filter_H(f)
            for j in range(X.shape[0]):
                ph = np.full_like(f, dphase[j])
                if use_guide_disp:
                    ph = ph + guide_phase(lam, j * L_slot_m)
                if prechirp_fs2:
                    ph = ph + 0.5 * prechirp_fs2 * 1e-30 * dom ** 2
                X[j] = np.fft.ifft(np.fft.fft(X[j]) * H
                                   * np.exp(1j * ph))
            return X

        tf = mesh_transfer_fn(nodes, dvec, N, base, phase_screen=screen,
                              dkappa=dkappa)

        def read_out(Y, t, n_out):
            """The receiver (critic-2 HIGH-1): 'peak' = complex sample at
            slot center (+ offset); 'gate' = complex field integrated
            over a gate_fs homodyne window (chirp washes out here)."""
            off = sample_offset_fs * 1e-15
            S = np.zeros((N, n_out), dtype=complex)
            for k in range(n_out):
                c = 2 * otdm.T_SLOT + k * otdm.T_SLOT + off
                if receiver == "peak":
                    idx = np.argmin(np.abs(t - c))
                    S[:, k] = Y[:, idx]
                else:
                    w = (t > c - gate_fs * 1e-15 / 2) & \
                        (t <= c + gate_fs * 1e-15 / 2)
                    S[:, k] = np.sum(Y[:, w], axis=1)
            return S

        # ---- bits: random tokens ------------------------------------
        tokens = rng.uniform(-amp_scale, amp_scale, n_tokens)
        # per-pulse comb phase noise: random walk along the train,
        # sigma_slot = sqrt(2 pi linewidth T_slot) (critic-3 HIGH-2;
        # NOT trimmable, unlike the deleted static term)
        if comb_linewidth_hz:
            sig = np.sqrt(2 * np.pi * comb_linewidth_hz * otdm.T_SLOT)
            walk = np.cumsum(rng_e.normal(0, sig, n_tokens))
            tokens_ph = tokens * np.exp(1j * walk)
        else:
            tokens_ph = tokens
        t = time_grid(n_tokens + N + 2)
        X = apply_frontend(delayed_inputs(tokens_ph, t, N, T0=T0), t)
        Y = propagate(X, t, tf)
        S = read_out(Y, t, n_tokens + N - 1)
        ref = np.zeros_like(S)
        for i in range(N):
            ref[i] = np.convolve(tokens, U[i], mode="full")[:S.shape[1]]
        g = np.vdot(ref, S) / np.vdot(ref, ref)
        err = np.sqrt(np.mean(np.abs(S / g - ref) ** 2)
                      / np.mean(np.abs(ref) ** 2))
        bits = float(-np.log2(max(err, 1e-12)))
        # per-row complex gain refit (a legitimate post-detection rescale:
        # one complex number per output port) -- separates compilable
        # differential loss from irreducible error (critic HIGH-4)
        S_r = S.copy()
        for i in range(N):
            gi = np.vdot(ref[i], S[i]) / max(np.vdot(
                ref[i], ref[i]).real, 1e-30)
            S_r[i] = S[i] / gi
        err_row = np.sqrt(np.mean(np.abs(S_r - ref) ** 2)
                          / np.mean(np.abs(ref) ** 2))
        bits_row = float(-np.log2(max(err_row, 1e-12)))

        # ---- ISI: single-token impulse response ---------------------
        n_imp = 4 * N + 8
        k0 = 2 * N
        imp = np.zeros(n_imp)
        imp[k0] = 1.0
        t2 = time_grid(n_imp + N + 2)
        X2 = apply_frontend(delayed_inputs(imp, t2, N, T0=T0), t2)
        Y2 = propagate(X2, t2, tf)
        S2 = read_out(Y2, t2, n_imp + N - 1)
        span = range(k0, k0 + N)
        e_in = sum(np.sum(np.abs(S2[:, k]) ** 2) for k in span)
        e_out = (np.sum(np.abs(S2) ** 2) - e_in)
        isi = float(e_out / e_in)
        # window-integrated version
        dt2 = t2[1] - t2[0]
        half = otdm.T_SLOT / 2
        e_in_w = e_out_w = 0.0
        for k in range(S2.shape[1]):
            c = 2 * otdm.T_SLOT + k * otdm.T_SLOT
            w = (t2 > c - half) & (t2 <= c + half)
            e = float(np.sum(np.abs(Y2[:, w]) ** 2) * dt2)
            if k in span:
                e_in_w += e
            else:
                e_out_w += e
        isi_w = float(e_out_w / max(e_in_w, 1e-30))

        # ---- broadening: the single pulse in the strongest in-span
        # slot of the impulse response, intensity FWHM -----------------
        def fwhm_window(tr, k):
            c = 2 * otdm.T_SLOT + k * otdm.T_SLOT
            w = (t2 > c - otdm.T_SLOT / 2) & (t2 <= c + otdm.T_SLOT / 2)
            I = np.abs(tr[w]) ** 2
            if I.max() <= 0:
                return np.nan
            I = I / I.max()
            ab = np.where(I >= 0.5)[0]
            return (ab[-1] - ab[0]) * (t2[1] - t2[0]) if len(ab) > 1 else 0
        i_best, k_best = np.unravel_index(
            np.argmax(np.abs(S2[:, list(span)])), (N, N))
        w_out = fwhm_window(Y2[i_best], k0 + k_best)
        w_in = 2 * np.sqrt(np.log(2)) * T0          # intensity FWHM
        broadening = float(w_out / w_in)

        # ---- walk-off (closed form from the measured screen) --------
        m = np.deg2rad(abs(SCREEN_C)) * 1e9          # rad/m of lambda
        dtau = m * (1.55e-6) ** 2 / (2 * np.pi * C0)  # s per node
        walkoff_fs = float(dtau * N * 1e15)

        # ---- score band ---------------------------------------------
        score = {}
        if want_score:
            for lag in (1, N - 1):
                tk = rng.uniform(-0.97, 0.97, 32)
                meas, exp, gsc = score_band(tk, cal, lag=lag, T0=T0)
                ok = np.abs(exp) > 0
                score[lag] = float(np.sqrt(np.mean(
                    (meas[ok] / gsc - exp[ok] / gsc) ** 2)))

        # ---- density -------------------------------------------------
        area = (mesh_area_mm2(N, L_NODE_C, 4.0)
                + delay_area_mm2(N, otdm.FSR, 4.0) + 0.005
                + (N - 1) * 250e-6) * 1.15
        dens = N ** 2 * otdm.FSR / area / B300_DENS

        if return_arrays:
            return {"tokens": tokens, "S": S / g, "ref": ref,
                    "S_rowgain": S_r, "g": g}
        return {"N": N, "T0_fs": T0 * 1e15,
                "fwhm_fs": 2 * np.sqrt(np.log(2)) * T0 * 1e15,
                "slot_fs": slot_fs, "prechirp_fs2": prechirp_fs2,
                "flags": {"screen": use_screen, "kappa": use_kappa_disp,
                          "guide": use_guide_disp, "loss": use_loss},
                "rms_err": float(err), "bits": bits,
                "rms_err_rowgain": float(err_row), "bits_rowgain": bits_row,
                "isi_leak": isi, "isi_leak_win": isi_w,
                "broadening": broadening, "walkoff_fs": walkoff_fs,
                "score_rms": score, "density_xB300": float(dens)}
    finally:
        otdm.T_SLOT, otdm.FSR = old_slot, old_fsr


def score_full(tokens, T0, lag=1, dkap=0.0, comb_linewidth_hz=0.0,
               seed_err=0, leak_correct=True):
    """Score head under the FULL error model (critic-3 MED-6): compact
    coupler + kappa disorder dkap on ITS coupler + per-pulse comb phase
    walk.  Truth = real token products.  Returns relative rms."""
    from otdm import (make_train, slot_energies, propagate as prop,
                      time_grid as tg)
    rng_e = np.random.default_rng(20_000 + seed_err)
    n = len(tokens)
    if comb_linewidth_hz:
        sig = np.sqrt(2 * np.pi * comb_linewidth_hz * otdm.T_SLOT)
        walk = np.cumsum(rng_e.normal(0, sig, n))
        tokens_ph = tokens * np.exp(1j * walk)
    else:
        tokens_ph = tokens.astype(complex)
    ncal = compact_node_cal()

    class KapShift:
        def kappa(self, lam):
            return ncal.kappa(lam) + dkap

        def amp(self, lam):
            return ncal.amp(lam)

    from otdm import coupler_transfer_fn
    fn = coupler_transfer_fn(KapShift())
    t = tg(n + lag + 2, 1e-15)
    f = np.fft.fftfreq(len(t), d=1e-15)
    qwave = np.exp(1j * (np.pi / 2) * (NU0 + f) / NU0)
    arm0 = make_train(tokens_ph, t, t0=2 * otdm.T_SLOT
                      + lag * otdm.T_SLOT, T0=T0)
    arm1 = np.fft.ifft(np.fft.fft(
        make_train(tokens_ph, t, t0=2 * otdm.T_SLOT, T0=T0)) * qwave)
    n_out = n + lag

    def balanced(a, b):
        Y = prop(np.vstack([a, b]), t, fn)
        return (slot_energies(Y[0], t, n_out)
                - slot_energies(Y[1], t, n_out))

    meas = balanced(arm0, arm1)
    if leak_correct:
        z = np.zeros_like(arm0)
        meas = meas - balanced(arm0, z) - balanced(z, arm1)
    exp = np.zeros(n_out)
    for k in range(n_out):
        if lag <= k < n:
            exp[k] = np.real(tokens[k]) * np.real(tokens[k - lag])
    g = np.dot(meas, exp) / np.dot(exp, exp)
    ok = np.abs(exp) > 0
    return float(np.sqrt(np.mean((meas[ok] / g - exp[ok]) ** 2))
                 / np.sqrt(np.mean(exp[ok] ** 2)))


def attribution(N=5, T0=120e-15, slot_fs=501.26):
    """Error with everything on, then with one component idealized at a
    time -> fraction of error variance each component explains."""
    full = run_config(N=N, T0=T0, slot_fs=slot_fs, want_score=False)
    out = {"full": full, "components": {}}
    for name, kw in (("screen", {"use_screen": False}),
                     ("kappa_disp", {"use_kappa_disp": False}),
                     ("guide_disp", {"use_guide_disp": False}),
                     ("loss", {"use_loss": False})):
        r = run_config(N=N, T0=T0, slot_fs=slot_fs, want_score=False, **kw)
        out["components"][name] = {
            "rms_err": r["rms_err"], "isi_leak": r["isi_leak"],
            "err_var_explained": float(
                1 - (r["rms_err"] / full["rms_err"]) ** 2),
            "isi_explained": float(
                1 - r["isi_leak"] / max(full["isi_leak"], 1e-30))}
    return out


if __name__ == "__main__":
    import sys
    r = run_config()
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("flags",)}, indent=1))

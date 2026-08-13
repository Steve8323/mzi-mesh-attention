"""The four numerical claims of ANALYSIS.md, run end-to-end on the
calibrated mesh + OTDM front end.

  A. Toeplitz identity: peak-sampled mesh outputs == conv(U rows, tokens).
  B. Useful fixed mixer: U = DFT_N -> sliding 5-point DFT (chirp spectrogram).
  C. Expressivity gap: best Toeplitz/linear fit to softmax attention is bad;
     the same fit to a Hyena-style causal conv is exact.
  D. Photonic attention scores: balanced-homodyne band s_t*s_{t-k} from the
     delay ladder + calibrated coupler; electronic softmax closes the loop.

Writes out/demo.json + arrays for figs.py.
"""
import json
import os

import numpy as np

from node import NodeCal, LAM0
from clements import decompose, mesh_matrix, fidelity, haar, dft
from otdm import (T_SLOT, FSR, T0_PULSE, time_grid, delayed_inputs,
                  propagate, mesh_transfer_fn, sample_slots, score_band)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(42)
cal = NodeCal()
KAPPA0 = float(cal.kappa(LAM0))
N = 5


SCREEN_DEG_PER_NM = -0.253      # measured: node2 quadrature minus control


def phase_screen(lam_um):
    """Unmodeled per-node MZI phase dispersion, zeroed at 1550 nm by the
    static calibration (see fdtd_coupler2.py control run)."""
    return np.deg2rad(SCREEN_DEG_PER_NM) * (lam_um - 1.55) * 1000.0


def run_mesh_on_train(U, tokens, N, T0=T0_PULSE, screen=True):
    nodes, d, resid = decompose(U, KAPPA0)
    M0 = mesh_matrix(nodes, d, N, cal=cal)
    t = time_grid(len(tokens) + N + 2)
    X = delayed_inputs(tokens, t, N, T0=T0)
    Y = propagate(X, t, mesh_transfer_fn(
        nodes, d, N, cal, phase_screen=phase_screen if screen else None))
    S = sample_slots(Y, t, len(tokens), N)
    return M0, S, (t, Y), resid


def toeplitz_apply(U, tokens):
    """y[i,k] = sum_j U[i,j] tokens[k-j]  (the claim: mesh == this)."""
    n_out = len(tokens) + U.shape[0] - 1
    Yt = np.zeros((U.shape[0], n_out), dtype=complex)
    for i in range(U.shape[0]):
        Yt[i] = np.convolve(tokens, U[i], mode="full")[:n_out]
    return Yt


# ---------------------------------------------------------------- A ------
def _otdm_errors(U, tokens, T0, screen=True):
    M0, S, _, resid = run_mesh_on_train(U, tokens, N, T0=T0, screen=screen)
    ref_ideal = toeplitz_apply(U, tokens)
    ref_cal = toeplitz_apply(M0, tokens)
    g = np.vdot(ref_cal, S) / np.vdot(ref_cal, ref_cal)
    err_cal = np.sqrt(np.mean(np.abs(S / g - ref_cal) ** 2)
                      / np.mean(np.abs(ref_cal) ** 2))
    g2 = np.vdot(ref_ideal, S) / np.vdot(ref_ideal, ref_ideal)
    err_ideal = np.sqrt(np.mean(np.abs(S / g2 - ref_ideal) ** 2)
                        / np.mean(np.abs(ref_ideal) ** 2))
    return M0, S, ref_ideal, float(err_cal), float(err_ideal), resid, g


T0_OP = 120e-15                  # recommended operating point (200 fs FWHM)


def demo_A():
    tokens = rng.uniform(-0.97, 0.97, 64)
    U = haar(N, rng)
    M0, S, ref_ideal, err_cal, err_ideal, resid, g = _otdm_errors(
        U, tokens, T0_OP)
    e_mesh, T = fidelity(M0, U)
    # pulse-width sweep: the slot is 501 fs, so the mesh prefers the LONGEST
    # pulse the guard band allows -- opposite of the compression goal
    sweep = []
    for T0 in (T0_PULSE, 20e-15, 40e-15, 80e-15, 120e-15):
        _, _, _, ec, ei, _, _ = _otdm_errors(U, tokens, T0)
        _, _, _, _, ei_ns, _, _ = _otdm_errors(U, tokens, T0, screen=False)
        sweep.append({"T0_fs": T0 * 1e15, "fwhm_fs": T0 * 1e15 * 1.665,
                      "err_vs_calibrated": ec, "err_vs_ideal": ei,
                      "err_vs_ideal_noscreen": ei_ns,
                      "bits_vs_ideal": float(-np.log2(max(ei, 1e-12)))})
    return {
        "mesh_rms_err_vs_U": e_mesh, "mesh_power_T": T,
        "decomp_resid": resid,
        "trace_fwhm_fs": T0_OP * 1e15 * 1.665,
        "otdm_vs_calibrated_toeplitz_rms": err_cal,
        "otdm_vs_ideal_toeplitz_rms": err_ideal,
        "bits_vs_ideal": float(-np.log2(max(err_ideal, 1e-12))),
        "pulse_sweep": sweep,
        "tokens": tokens.tolist(),
        "S_real": np.real(S).tolist(), "S_imag": np.imag(S).tolist(),
        "ref_real": np.real(ref_ideal).tolist(),
        "ref_imag": np.imag(ref_ideal).tolist(),
        "gain_abs": float(abs(g)),
    }


# ---------------------------------------------------------------- B ------
def demo_B():
    """Chirped token sequence through the DFT_5 mesh = sliding spectrogram."""
    n = 96
    k = np.arange(n)
    fr = 0.05 + 0.35 * k / n            # normalized token-rate chirp
    tokens = 0.9 * np.cos(2 * np.pi * np.cumsum(fr))
    U = dft(N)
    # run at the recommended operating point (long pulse, see demo A sweep)
    M0, S, _, resid = run_mesh_on_train(U, tokens, N, T0=120e-15)
    P = np.abs(S) ** 2                   # what N slow detectors would time-bin
    ref = np.abs(toeplitz_apply(U, tokens)) ** 2
    err = np.sqrt(np.mean((P / P.max() - ref / ref.max()) ** 2))
    return {"spectrogram": P.tolist(), "reference": ref.tolist(),
            "tokens": tokens.tolist(), "rms_err_norm": float(err),
            "mesh_fid": fidelity(M0, U)[0], "resid": resid}


# ---------------------------------------------------------------- C ------
def softmax_attention(s, W, beta=1.0):
    """Causal window-W scalar self-attention (q=k=v=s), per sequence."""
    n = len(s)
    y = np.zeros(n)
    for t in range(n):
        lo = max(0, t - W + 1)
        sc = beta * s[lo:t + 1] * s[t]
        a = np.exp(sc - sc.max())
        a /= a.sum()
        y[t] = np.dot(a, s[lo:t + 1])
    return y


def vector_attention(X, W, Wq, Wk, wv, beta=1.0):
    """Causal window-W attention on d-dim tokens X (n, d), scalar values."""
    n = X.shape[0]
    q, k = X @ Wq.T, X @ Wk.T
    v = X @ wv
    y = np.zeros(n)
    for t in range(n):
        lo = max(0, t - W + 1)
        sc = beta * (k[lo:t + 1] @ q[t]) / np.sqrt(len(q[t]))
        a = np.exp(sc - sc.max())
        a /= a.sum()
        y[t] = np.dot(a, v[lo:t + 1])
    return y


def fit_toeplitz_vec(X, Y, W):
    """Best causal multi-channel FIR (W taps x d channels) X -> Y."""
    feats, targs = [], []
    for x, y in zip(X, Y):
        n, dd = x.shape
        F = np.zeros((n, W * dd))
        for k in range(W):
            F[k:, k * dd:(k + 1) * dd] = x[:n - k]
        feats.append(F)
        targs.append(y)
    F = np.vstack(feats)
    yv = np.concatenate(targs)
    w, *_ = np.linalg.lstsq(F, yv, rcond=None)
    pred = F @ w
    return float(1 - np.sum((pred - yv) ** 2)
                 / np.sum((yv - yv.mean()) ** 2))


def fit_toeplitz(X, Y, W):
    """Best causal FIR kernel (len W) mapping token seqs X -> targets Y."""
    feats, targs = [], []
    for s, y in zip(X, Y):
        n = len(s)
        F = np.zeros((n, W))
        for k in range(W):
            F[k:, k] = s[:n - k]
        feats.append(F)
        targs.append(y)
    F = np.vstack(feats)
    yv = np.concatenate(targs)
    w, *_ = np.linalg.lstsq(F, yv, rcond=None)
    pred = F @ w
    ss = 1 - np.sum((pred - yv) ** 2) / np.sum((yv - yv.mean()) ** 2)
    return float(ss)


def demo_C():
    nseq, L, W, d = 400, 64, N, 4
    X = [rng.uniform(-0.97, 0.97, L) for _ in range(nseq)]
    kern = rng.standard_normal(W)
    kern /= np.linalg.norm(kern)
    Y_conv = [np.convolve(s, kern, mode="full")[:L] for s in X]
    out = {"R2_toeplitz_to_conv": fit_toeplitz(X, Y_conv, W)}
    for beta in (1.0, 4.0):
        Y = [softmax_attention(s, W, beta) for s in X]
        out[f"R2_toeplitz_to_attention_b{beta:g}"] = fit_toeplitz(X, Y, W)
        out[f"R2_toeplitz_to_attention_b{beta:g}_W32"] = fit_toeplitz(
            X, Y, 32)
    # d-dim tokens, distinct random Q/K/V projections (the generic case)
    Wq, Wk = rng.standard_normal((d, d)), rng.standard_normal((d, d))
    wv = rng.standard_normal(d)
    Xv = [rng.uniform(-0.97, 0.97, (L, d)) for _ in range(nseq)]
    for beta in (1.0, 4.0):
        Yv = [vector_attention(x, W, Wq, Wk, wv, beta) for x in Xv]
        out[f"R2_toeplitz_to_vec_attention_b{beta:g}"] = fit_toeplitz_vec(
            Xv, Yv, W)
    return out


# ---------------------------------------------------------------- D ------
def demo_D():
    tokens = rng.uniform(-0.97, 0.97, 48)
    n = len(tokens)
    W = N
    # photonic score band: lag 0 from direct detection (|s_t|^2 with sign
    # known non-negative), lags 1..4 from balanced homodyne
    bands = {}
    band_errs = {}
    band_errs_uncorrected = {}
    for lag in range(1, W):
        meas, exp, g = score_band(tokens, cal, lag=lag)
        mu, eu, gu = score_band(tokens, cal, lag=lag, leak_correct=False)
        # align: entry k of meas = s_k * s_{k-lag}
        sc = np.full(n, np.nan)
        scu = np.full(n, np.nan)
        for k in range(lag, n):
            sc[k] = meas[k] / g
            scu[k] = mu[k] / gu
        bands[lag] = sc
        ok = ~np.isnan(sc)
        truth = tokens * np.roll(tokens, lag)
        band_errs[lag] = float(np.sqrt(np.mean((sc[ok] - truth[ok]) ** 2)))
        band_errs_uncorrected[lag] = float(np.sqrt(np.mean(
            (scu[ok] - truth[ok]) ** 2)))
    bands[0] = tokens ** 2                     # direct detection of the train
    # electronic softmax + reweight  (photonic scores -> attention output)
    y_ph = np.zeros(n)
    y_ref = softmax_attention(tokens, W)
    for t in range(n):
        lags = [k for k in range(min(t + 1, W))]
        sc = np.array([bands[k][t] if k > 0 else bands[0][t] for k in lags])
        a = np.exp(sc - sc.max())
        a /= a.sum()
        y_ph[t] = np.dot(a, [tokens[t - k] for k in lags])
    err = np.sqrt(np.mean((y_ph - y_ref) ** 2) / np.mean(y_ref ** 2))
    return {"band_rms_err": band_errs,
            "band_rms_err_uncorrected": band_errs_uncorrected,
            "attention_pipeline_rms": float(err),
            "tokens": tokens.tolist(),
            "y_photonic": y_ph.tolist(), "y_exact": y_ref.tolist(),
            "band1_meas": np.where(np.isnan(bands[1]), None,
                                   bands[1]).tolist()}


if __name__ == "__main__":
    res = {"A_toeplitz": demo_A(), "B_stft": demo_B(),
           "C_expressivity": demo_C(), "D_scores": demo_D(),
           "constants": {"FSR_THz": FSR / 1e12, "T_slot_fs": T_SLOT * 1e15,
                         "pulse_T0_fs": T0_PULSE * 1e15, "N": N,
                         "kappa0": KAPPA0}}
    json.dump(res, open(os.path.join(OUT, "demo.json"), "w"))
    A, B, C, D = (res["A_toeplitz"], res["B_stft"], res["C_expressivity"],
                  res["D_scores"])
    print(f"[A] mesh vs U rms {A['mesh_rms_err_vs_U']:.4f}, "
          f"OTDM vs ideal-Toeplitz rms {A['otdm_vs_ideal_toeplitz_rms']:.4f} "
          f"({A['bits_vs_ideal']:.1f} bits), power T {A['mesh_power_T']:.3f}")
    for row in A["pulse_sweep"]:
        print(f"    pulse FWHM {row['fwhm_fs']:6.1f} fs -> "
              f"rms vs ideal {row['err_vs_ideal']:.4f} "
              f"({row['bits_vs_ideal']:.1f} bits), "
              f"vs calibrated {row['err_vs_calibrated']:.4f}")
    print(f"[B] sliding-DFT spectrogram rms (norm) {B['rms_err_norm']:.4f}")
    print("[C] R2 of best causal FIR fit:")
    for k, v in C.items():
        print(f"    {k}: {v:.4f}")
    print(f"[D] score band rms per lag (leak-corrected): "
          + ", ".join(f"{k}:{v:.2e}" for k, v in D['band_rms_err'].items()))
    print(f"    uncorrected: "
          + ", ".join(f"{k}:{v:.2e}"
                      for k, v in D['band_rms_err_uncorrected'].items())
          + f"; full attention pipeline rms "
            f"{D['attention_pipeline_rms']:.2e}")

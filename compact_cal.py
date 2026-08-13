#!/usr/bin/env python
"""Assemble the compact-node calibration from the fdtd_compact runs and
recompute the OTDM token-mixing accuracy with it.

Steps (all measured, nothing assumed):
  1. kappa_c(lambda) per coupler from the compact coupler runs
     (cross fraction = sin^2 kappa), interpolated at the chosen LC.
  2. Node amplitude a_c(lambda) and s0(lambda) from the compact MZI dn=0
     run; consistency check s0 vs cos(4 kappa_c).
  3. Geometric-phase validation: the dL=0.3003 um run (theta = pi/2 by
     design) against the model prediction.
  4. Quadrature phase screen of the compact node from the mzi2 run
     (same extraction as the original node) -> deg/nm slope.
  5. OTDM accuracy of the compact 5x5 at 200 fs and 125 fs pulses with
     the measured screen.
Writes out/compact_cal.json (consumed by density.py).
"""
import glob
import json
import os

import numpy as np

from node import coupler as coupler_mat
from clements import decompose, mesh_matrix
from otdm import (time_grid, delayed_inputs, propagate, mesh_transfer_fn,
                  sample_slots)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LAM0 = 1.55
LC_PICK = 5.8


def jload(p):
    return json.load(open(os.path.join(OUT, p)))


class CompactCal:
    """Same interface as node.NodeCal (kappa, amp) but from compact runs."""

    def __init__(self):
        # per-coupler kappa(lambda): interpolate cross-fraction across the
        # LC sweep to LC_PICK, per wavelength
        runs = []
        for f in sorted(glob.glob(os.path.join(OUT, "compact_coupler_*.json"))):
            runs.append(json.load(open(f)))
        runs.sort(key=lambda r: r["LC"])
        lam = np.array(runs[0]["lam_nm"]) / 1000.0
        order = np.argsort(lam)
        self.lam_grid = lam[order]
        fracs = []
        for r in runs:
            pb, pc = np.array(r["P_bar"]), np.array(r["P_cross"])
            fracs.append((pc / (pb + pc))[order])
        LCs = [r["LC"] for r in runs]
        f_pick = np.array([np.interp(LC_PICK, LCs, [f[i] for f in fracs])
                           for i in range(len(self.lam_grid))])
        self.kappa_grid = np.arcsin(np.sqrt(np.clip(f_pick, 0, 1)))
        # node amplitude from the full compact MZI dn=0 run
        mz = jload(f"compact_mzi_LC{LC_PICK:g}_dL0.json")
        l2 = np.array(mz["lam_nm"]) / 1000.0
        o2 = np.argsort(l2)
        pb = np.array(mz["P_bar"])[o2]
        pc = np.array(mz["P_cross"])[o2]
        pi = np.array(mz["P_in"])[o2]
        self.amp_grid = np.interp(self.lam_grid, l2[o2],
                                  np.sqrt(np.clip((pb + pc) / pi, 1e-6,
                                                  None)))
        self.s0_meas = np.interp(self.lam_grid, l2[o2],
                                 ((pb - pc) / (pb + pc)))
        self.node_len = mz["node_len_um"]

    def kappa(self, lam):
        return np.interp(lam, self.lam_grid, self.kappa_grid)

    def amp(self, lam):
        return np.interp(lam, self.lam_grid, self.amp_grid)


def s_model(theta, kap):
    C = coupler_mat(kap)
    M = C @ np.diag([np.exp(1j * theta), 1.0]) @ C
    pb, pc = abs(M[0, 0]) ** 2, abs(M[1, 0]) ** 2
    return (pb - pc) / (pb + pc)


def screen_slope(cal):
    """Quadrature phase screen (deg/nm) of the compact node from mzi2."""
    d = jload(f"compact_mzi2_LC{LC_PICK:g}_dL0.json")
    lam = np.array(d["lam_nm"]) / 1000.0
    pb, pc, pi = (np.array(d["P_bar"]), np.array(d["P_cross"]),
                  np.array(d["P_in"]))
    sel = (lam > 1.48) & (lam < 1.62)
    out = []
    for l, mb, mc in zip(lam[sel], (pb / pi)[sel], (pc / pi)[sel]):
        kap = float(cal.kappa(l))
        C = coupler_mat(kap)
        M = float(cal.amp(l)) * (C @ C)
        best = (9e9, 0.0)
        for dd in np.deg2rad(np.linspace(-60, 60, 481)):
            v = M @ np.array([1.0, 0.7 * np.exp(1j * (np.pi / 2 + dd))])
            e = ((abs(v[0]) ** 2 / 1.49 - mb) ** 2
                 + (abs(v[1]) ** 2 / 1.49 - mc) ** 2)
            if e < best[0]:
                best = (e, dd)
        out.append((l * 1000, np.rad2deg(best[1])))
    out = np.array(out)
    p = np.polyfit(out[:, 0], out[:, 1], 1)
    return float(p[0]), float(np.polyval(p, 1550.0))


def otdm_bits(ccal, slope_deg_nm, T0, n_slots=64, seed=42):
    rng = np.random.default_rng(seed)
    from clements import haar
    N = 5
    U = haar(N, rng)
    kap0 = float(ccal.kappa(LAM0))
    nodes, dvec, _ = decompose(U, kap0)
    tokens = rng.uniform(-0.97, 0.97, n_slots)
    t = time_grid(n_slots + N + 2)
    X = delayed_inputs(tokens, t, N, T0=T0)

    def screen(lam_um):
        return np.deg2rad(slope_deg_nm) * (lam_um - 1.55) * 1000.0

    Y = propagate(X, t, mesh_transfer_fn(nodes, dvec, N, ccal,
                                         phase_screen=screen))
    S = sample_slots(Y, t, n_slots, N)
    ref = np.zeros_like(S)
    for i in range(N):
        ref[i] = np.convolve(tokens, U[i], mode="full")[:S.shape[1]]
    g = np.vdot(ref, S) / np.vdot(ref, ref)
    err = np.sqrt(np.mean(np.abs(S / g - ref) ** 2)
                  / np.mean(np.abs(ref) ** 2))
    return float(-np.log2(max(err, 1e-12))), float(err)


if __name__ == "__main__":
    cal = CompactCal()
    i0 = np.argmin(abs(cal.lam_grid - LAM0))
    kap0 = float(cal.kappa_grid[i0])
    # consistency: s0 measured vs cos(4 kappa) from the coupler sweep
    s0_pred = np.cos(4 * cal.kappa_grid[i0])
    # geometric-phase validation at 1550
    geo = jload(f"compact_mzi_LC{LC_PICK:g}_dL0.3003.json")
    lamg = np.array(geo["lam_nm"])
    ig = np.argmin(abs(lamg - 1550))
    NEFF = 1.2905
    th_geo = 2 * np.pi * NEFF * 0.3003 / 1.55
    s_geo_pred = s_model(th_geo, kap0)
    s_geo_meas = geo["s"][ig]
    slope, off1550 = screen_slope(cal)
    b200, e200 = otdm_bits(cal, slope, 120e-15)
    b125, e125 = otdm_bits(cal, slope, 75e-15)
    res = {
        "LC": LC_PICK, "node_len_um": cal.node_len,
        "kappa_1550": kap0, "power_split_1550": float(np.sin(kap0) ** 2),
        "node_T_1550": float(cal.amp_grid[i0] ** 2),
        "s0_meas_1550": float(cal.s0_meas[i0]),
        "s0_pred_from_coupler": float(s0_pred),
        "geo_phase_s_meas": float(s_geo_meas),
        "geo_phase_s_pred": float(s_geo_pred),
        "screen_deg_per_nm": slope, "screen_offset_1550_deg": off1550,
        "bits_200fs": b200, "err_200fs": e200,
        "bits_125fs": b125, "err_125fs": e125,
    }
    json.dump(res, open(os.path.join(OUT, "compact_cal.json"), "w"),
              indent=1)
    print(f"compact node: len {cal.node_len:.1f} um, kappa {kap0:.4f} "
          f"({np.sin(kap0)**2*100:.1f}% split), node T "
          f"{cal.amp_grid[i0]**2:.4f}")
    print(f"s0 meas {cal.s0_meas[i0]:+.4f} vs cos4k {s0_pred:+.4f}")
    print(f"geometric phase: s meas {s_geo_meas:+.4f} vs pred "
          f"{s_geo_pred:+.4f}")
    print(f"screen: {slope:+.3f} deg/nm (offset {off1550:+.1f} deg @1550)")
    print(f"OTDM 5x5: {b200:.1f} bits @200 fs, {b125:.1f} bits @125 fs")

#!/usr/bin/env python
"""Joint recalibration of the node model against ALL data at 1550:
the single-input s(dn) staircase (9 FDTD points) PLUS the two coherent
two-input runs (psi = 0 and 90 deg, bar+cross fractions).

Hypothesis: the physical MZI carries a static internal-arm phase offset
EPS (accumulated in the s-bend/coupler transitions) that power-only data
cannot separate from kappa; the quadrature (psi=90) run exposes it.
Free parameters: kappa, eps, LP_eff, and one input-launch screen delta
for the two-input runs only. In a mesh, eps and delta are absorbed by the
standard in-situ phase calibration (theta_set = theta_target - eps; input
screen into first-column phis) -- so the mesh results stand IF this fit
works; the fit VALIDATES the absorbable-offset picture.
Writes out/node2_joint.json.
"""
import itertools
import json
import os

import numpy as np

from node import NodeCal, coupler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
cal = NodeCal()
LAM = 1.55


def node_eps(theta, kappa, eps, amp=1.0):
    C = coupler(kappa)
    return amp * (C @ np.diag([np.exp(1j * (theta + eps)), 1.0]) @ C)


# data ---------------------------------------------------------------
dn_tab, s_tab = cal.dn_tab, cal.s_tab
runs = []
for psi in (0, 90):
    d = json.load(open(os.path.join(OUT, f"node2_DN0.01_a0.7_p{psi}.json")))
    lam = np.array(d["lam_nm"])
    i0 = int(np.argmin(abs(lam - 1550)))
    runs.append({"psi": np.deg2rad(psi), "a2": d["amp2"], "dn": d["dn"],
                 "bar": d["meas_bar"][i0], "cross": d["meas_cross"][i0]})
AMP = float(cal.amp(LAM))


def staircase_err(kappa, eps, LP):
    e = []
    for dn, s in zip(dn_tab, s_tab):
        th = 2 * np.pi * dn * LP / LAM
        M = node_eps(th, kappa, eps)
        pb, pc = abs(M[0, 0]) ** 2, abs(M[1, 0]) ** 2
        e.append((pb - pc) / (pb + pc) - s)
    return np.array(e)


def twoinput_err(kappa, eps, LP, delta):
    e = []
    for r in runs:
        th = 2 * np.pi * r["dn"] * LP / LAM
        M = node_eps(th, kappa, eps, AMP)
        v = M @ np.array([1.0, r["a2"] * np.exp(1j * (r["psi"] + delta))])
        norm = 1 + r["a2"] ** 2
        e += [abs(v[0]) ** 2 / norm - r["bar"],
              abs(v[1]) ** 2 / norm - r["cross"]]
    return np.array(e)


def total(kappa, eps, LP, delta):
    return float(np.sum(staircase_err(kappa, eps, LP) ** 2)
                 + np.sum(twoinput_err(kappa, eps, LP, delta) ** 2))


# coarse grid then refine --------------------------------------------
best = None
for kappa in np.linspace(0.68, 0.80, 25):
    for eps in np.deg2rad(np.linspace(-40, 40, 41)):
        for LP in np.linspace(23.0, 24.4, 15):
            for delta in np.deg2rad(np.linspace(-30, 30, 31)):
                v = total(kappa, eps, LP, delta)
                if best is None or v < best[0]:
                    best = (v, kappa, eps, LP, delta)
for _ in range(3):
    v0, k0, e0, L0, d0 = best
    for kappa in k0 + np.linspace(-0.01, 0.01, 21):
        for eps in e0 + np.deg2rad(np.linspace(-2, 2, 21)):
            for LP in L0 + np.linspace(-0.1, 0.1, 11):
                for delta in d0 + np.deg2rad(np.linspace(-2, 2, 21)):
                    v = total(kappa, eps, LP, delta)
                    if v < best[0]:
                        best = (v, kappa, eps, LP, delta)

v, kappa, eps, LP, delta = best
st_rms = float(np.sqrt(np.mean(staircase_err(kappa, eps, LP) ** 2)))
ti_rms = float(np.sqrt(np.mean(twoinput_err(kappa, eps, LP, delta) ** 2)))
# baseline: original model (eps=delta=0)
st0 = float(np.sqrt(np.mean(staircase_err(float(cal.kappa(LAM)), 0.0,
                                          cal.LP_eff) ** 2)))
ti0 = float(np.sqrt(np.mean(twoinput_err(float(cal.kappa(LAM)), 0.0,
                                         cal.LP_eff, 0.0) ** 2)))
rows = []
for r in runs:
    th = 2 * np.pi * r["dn"] * LP / LAM
    M = node_eps(th, kappa, eps, AMP)
    v = M @ np.array([1.0, r["a2"] * np.exp(1j * (r["psi"] + delta))])
    n = 1 + r["a2"] ** 2
    v0 = node_eps(2 * np.pi * r["dn"] * cal.LP_eff / LAM,
                  float(cal.kappa(LAM)), 0.0, AMP) @ np.array(
        [1.0, r["a2"] * np.exp(1j * r["psi"])])
    rows.append({"psi_deg": float(np.rad2deg(r["psi"])),
                 "meas_bar_1550": r["bar"], "meas_cross_1550": r["cross"],
                 "pred_bar_fit": float(abs(v[0]) ** 2 / n),
                 "pred_cross_fit": float(abs(v[1]) ** 2 / n),
                 "pred_bar_raw": float(abs(v0[0]) ** 2 / n),
                 "pred_cross_raw": float(abs(v0[1]) ** 2 / n)})
res = {"kappa": kappa, "kappa_orig": float(cal.kappa(LAM)),
       "eps_deg": float(np.rad2deg(eps)), "LP_eff": LP,
       "delta_deg": float(np.rad2deg(delta)),
       "staircase_rms": st_rms, "twoinput_rms": ti_rms,
       "staircase_rms_orig": st0, "twoinput_rms_orig": ti0,
       "rows": rows}
json.dump(res, open(os.path.join(OUT, "node2_joint.json"), "w"), indent=1)
print(f"kappa {cal.kappa(LAM):.4f} -> {kappa:.4f}   "
      f"eps = {np.rad2deg(eps):+.1f} deg   LP {LP:.2f}   "
      f"launch delta = {np.rad2deg(delta):+.1f} deg")
print(f"staircase rms {st0:.4f} -> {st_rms:.4f}")
print(f"two-input rms {ti0:.4f} -> {ti_rms:.4f}")

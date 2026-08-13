#!/usr/bin/env python
"""Control experiment: coherent TWO-INPUT drive of a SINGLE coupler
(same geometry as fdtd_score.py, i.e. the calibrated 50/50 with s-bends),
broadband Gaussian, flux spectra on both outputs.  If the lambda-dependent
phase screen seen in the full-MZI two-input runs (out/node2_*.json)
reproduces here, the systematic lives in the coupler/launch, not in the
phase-arm sections.

usage: fdtd_coupler2.py PSI_DEG      writes out/coupler2_p<psi>.json
"""
import json
import os
import sys

import numpy as np
import meep as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node import NodeCal  # noqa: E402

PSI = np.deg2rad(float(sys.argv[1])) if len(sys.argv) > 1 else np.pi / 2
AMP2 = 0.7

W = 0.991
NSIL = 1.444024
RES = 25
GAP_C = 0.30
LC = 5.35
LB = 14.0
D_C = W + GAP_C
D_A = 6.0
PADX = 4.0
LIN = 3.0
LOUT = 8.0

seg = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB), ("out", LOUT)]
edges = np.cumsum([0] + [s[1] for s in seg])
LTOT = edges[-1]
segx = {name: (edges[i], edges[i + 1]) for i, (name, _) in enumerate(seg)}


def y_top(x):
    for name, (a, b) in segx.items():
        if a <= x <= b:
            t = (x - a) / (b - a)
            c = (1 - np.cos(np.pi * t)) / 2
            if name in ("in", "out"):
                return D_A / 2
            if name == "b1":
                return D_A / 2 - (D_A - D_C) / 2 * c
            if name == "c1":
                return D_C / 2
            return D_C / 2 + (D_A - D_C) / 2 * c
    return D_A / 2


def material(p):
    x = min(max(p.x + LTOT / 2, 0.0), LTOT)
    yc = y_top(x)
    if abs(p.y - yc) <= W / 2 or abs(p.y + yc) <= W / 2:
        return mp.Medium(index=NSIL)
    return mp.Medium(index=1.0)


fcen = 1 / 1.55
df = 0.10 * fcen
sx = LTOT + 2 * PADX
sy = D_A + W + 2 * 4.0
src_x = -sx / 2 + PADX * 0.55
mon_x = sx / 2 - PADX * 0.55

sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=RES,
                    boundary_layers=[mp.PML(1.5)],
                    material_function=material,
                    default_material=mp.Medium(index=1.0))
sim.sources = [
    mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df),
                       center=mp.Vector3(src_x, D_A / 2),
                       size=mp.Vector3(0, 3.6), eig_band=1,
                       eig_parity=mp.ODD_Z, eig_match_freq=True),
    mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df),
                       center=mp.Vector3(src_x, -D_A / 2),
                       size=mp.Vector3(0, 3.6), eig_band=1,
                       eig_parity=mp.ODD_Z, eig_match_freq=True,
                       amplitude=AMP2 * np.exp(1j * PSI)),
]
nfreq = 81
fl_bar = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(mon_x, D_A / 2), size=mp.Vector3(0, 3.6)))
fl_cross = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(mon_x, -D_A / 2), size=mp.Vector3(0, 3.6)))
fl_in1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(src_x + 1.2, D_A / 2), size=mp.Vector3(0, 3.6)))
fl_in2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(src_x + 1.2, -D_A / 2), size=mp.Vector3(0, 3.6)))
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    60, mp.Ez, mp.Vector3(mon_x, D_A / 2), 1e-8))

freqs = np.array(mp.get_flux_freqs(fl_bar))
Pb = np.array(mp.get_fluxes(fl_bar))
Pc = np.array(mp.get_fluxes(fl_cross))
Pi = np.array(mp.get_fluxes(fl_in1)) + np.array(mp.get_fluxes(fl_in2))
lam = 1.0 / freqs

# model + per-lambda phase-screen extraction
cal = NodeCal()
delta_fit, pred0_bar = [], []
for l, mb, mc in zip(lam, Pb / Pi, Pc / Pi):
    k = float(cal.kappa(l))
    c, s = np.cos(k), np.sin(k)
    best = (9e9, 0.0)
    for dd in np.deg2rad(np.linspace(-60, 60, 481)):
        v = np.array([[c, 1j * s], [1j * s, c]]) @ np.array(
            [1.0, AMP2 * np.exp(1j * (PSI + dd))])
        n = 1 + AMP2 ** 2
        e = (abs(v[0]) ** 2 / n - mb) ** 2 + (abs(v[1]) ** 2 / n - mc) ** 2
        if e < best[0]:
            best = (e, dd)
    delta_fit.append(float(np.rad2deg(best[1])))
    v = np.array([[c, 1j * s], [1j * s, c]]) @ np.array(
        [1.0, AMP2 * np.exp(1j * PSI)])
    pred0_bar.append(abs(v[0]) ** 2 / (1 + AMP2 ** 2))

i0 = int(np.argmin(np.abs(freqs - fcen)))
json.dump(dict(psi_deg=float(np.rad2deg(PSI)), amp2=AMP2,
               lam_nm=list(1000 * lam), meas_bar=list(Pb / Pi),
               meas_cross=list(Pc / Pi), pred_bar_nodelta=pred0_bar,
               delta_fit_deg=delta_fit),
          open(f"out/coupler2_p{np.rad2deg(PSI):.0f}.json", "w"))
print(f"single-coupler 2-input @1550: bar {Pb[i0]/Pi[i0]:.4f} "
      f"(model, no screen: {pred0_bar[i0]:.4f}), "
      f"delta_fit = {delta_fit[i0]:+.1f} deg")
sel = (lam > 1.48) & (lam < 1.62)
d = np.array(delta_fit)[sel]
ln = lam[sel] * 1000
p = np.polyfit(ln, d, 1)
print(f"delta(lambda): {d[-1]:+.1f} deg @ {ln[-1]:.0f} nm ... "
      f"{d[0]:+.1f} deg @ {ln[0]:.0f} nm; slope {p[0]:+.3f} deg/nm")

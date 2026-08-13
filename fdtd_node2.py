#!/usr/bin/env python
"""FDTD check that the calibrated MZI works as a MESH NODE, i.e. under
COHERENT TWO-INPUT drive (the original silica-repo calibration only ever
drove one input).  Geometry byte-identical to mzi.py (LC=5.35, gap 0.30,
w=0.991, LB=14, LP=24); second EigenModeSource on the bottom guide with a
complex amplitude sets the relative phase.

usage: fdtd_node2.py DN AMP2 PSI_DEG   (e.g. 0.01 0.7 90)
Writes out/node2_DN<dn>_a<amp>_p<psi>.json with P_bar/P_cross spectra and
the node-model prediction at every wavelength.
"""
import json
import os
import sys

import numpy as np
import meep as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node import NodeCal, node_matrix  # noqa: E402

DN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
AMP2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
PSI = np.deg2rad(float(sys.argv[3])) if len(sys.argv) > 3 else np.pi / 2

W = 0.991
NSIL = 1.444024
RES = 25
GAP_C = 0.30
LC = 5.35
LB = 14.0
LP = 24.0
D_C = W + GAP_C
D_A = 6.0
PADX = 4.0
LIN = 3.0

seg = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB),
       ("ph", LP), ("b3", LB), ("c2", LC), ("b4", LB), ("out", LIN)]
edges = np.cumsum([0] + [s[1] for s in seg])
LTOT = edges[-1]
segx = {name: (edges[i], edges[i + 1]) for i, (name, _) in enumerate(seg)}


def y_top(x):
    for name, (a, b) in segx.items():
        if a <= x <= b:
            t = (x - a) / (b - a)
            c = (1 - np.cos(np.pi * t)) / 2
            if name in ("in", "ph", "out"):
                return D_A / 2
            if name in ("b1", "b3"):
                return D_A / 2 - (D_A - D_C) / 2 * c
            if name in ("c1", "c2"):
                return D_C / 2
            return D_C / 2 + (D_A - D_C) / 2 * c
    return D_A / 2


def material(p):
    x = min(max(p.x + LTOT / 2, 0.0), LTOT)
    yc = y_top(x)
    if abs(p.y - yc) <= W / 2:
        dn = DN if segx["ph"][0] <= x <= segx["ph"][1] else 0.0
        return mp.Medium(index=NSIL + dn)
    if abs(p.y + yc) <= W / 2:
        return mp.Medium(index=NSIL)
    return mp.Medium(index=1.0)


fcen = 1 / 1.55
df = 0.10 * fcen
sx = LTOT + 2 * PADX
sy = D_A + W + 2 * 4.0
cell = mp.Vector3(sx, sy)
src_x = -sx / 2 + PADX * 0.55
mon_x = sx / 2 - PADX * 0.55

sim = mp.Simulation(cell_size=cell, resolution=RES,
                    boundary_layers=[mp.PML(1.5)],
                    material_function=material,
                    extra_materials=[mp.Medium(index=NSIL + DN + 1e-4)],
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

# node-model prediction, same input state, every wavelength
cal = NodeCal()
lam = 1.0 / freqs
pred_bar, pred_cross = [], []
for l in lam:
    th = cal.theta_of_dn(DN, l)
    M = node_matrix(th, 0.0, float(cal.kappa(l)), float(cal.amp(l)))
    out = M @ np.array([1.0, AMP2 * np.exp(1j * PSI)])
    pred_bar.append(abs(out[0]) ** 2)
    pred_cross.append(abs(out[1]) ** 2)
pred_bar = np.array(pred_bar) / (1 + AMP2 ** 2)
pred_cross = np.array(pred_cross) / (1 + AMP2 ** 2)

meas_bar, meas_cross = Pb / Pi, Pc / Pi
i0 = np.argmin(np.abs(freqs - fcen))
tag = f"DN{DN:g}_a{AMP2:g}_p{np.rad2deg(PSI):.0f}"
json.dump(dict(dn=DN, amp2=AMP2, psi_deg=float(np.rad2deg(PSI)),
               lam_nm=list(1000 * lam), P_bar=list(Pb), P_cross=list(Pc),
               P_in=list(Pi), meas_bar=list(meas_bar),
               meas_cross=list(meas_cross), pred_bar=list(pred_bar),
               pred_cross=list(pred_cross)),
          open(f"out/node2_{tag}.json", "w"))
print(f"[{tag}] at 1550:  bar meas {meas_bar[i0]:.4f} pred {pred_bar[i0]:.4f}"
      f" | cross meas {meas_cross[i0]:.4f} pred {pred_cross[i0]:.4f}")

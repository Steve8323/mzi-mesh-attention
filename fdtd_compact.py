#!/usr/bin/env python
"""FDTD calibration of the COMPACT node for the density push.

Geometry changes vs the original silica-repo MZI (mzi.py):
  port pitch D_A 6 -> 4 um        (crosstalk at 3 um edge gap: extrapolated
                                   Lpi ~ 0.4 m -> <1% amplitude over a mesh)
  s-bends    LB 14 -> 8 um        (min bend radius ~9.7 um; ring data say
                                   R>=8 um is low-loss on this platform)
  phase arm  LP 24 -> TRIM 6 um   (fixed-weight mesh: theta is set
                                   GEOMETRICALLY by an arm-length offset dL,
                                   the 6 um stub is the electrical trim)
  LIN 3 -> 1.5 um
Node length = 2*LC + 41 um  (vs 96.7 um original).

usage:
  fdtd_compact.py coupler LC          -> splitting of one compact coupler
  fdtd_compact.py mzi LC DL [DN]      -> full compact MZI, arm offset DL um
                                         (extra length on TOP arm trim stub)
  fdtd_compact.py mzi2 LC DL          -> same MZI, coherent 2-input drive
                                         (1, 0.7 e^{i pi/2}) for the
                                         quadrature phase-screen check
Writes out/compact_<mode>_<tag>.json
"""
import json
import os
import sys

import numpy as np
import meep as mp

MODE = sys.argv[1]
LC = float(sys.argv[2])
DL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
DN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

W = 0.991
NSIL = 1.444024
RES = 25
GAP_C = 0.30
LB = 8.0
TRIM = 6.0
D_C = W + GAP_C
D_A = 4.0
PADX = 4.0
LIN = 1.5

COUPLER_ONLY = MODE == "coupler"
TWO_INPUT = MODE == "mzi2"
seg = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB)]
if COUPLER_ONLY:
    seg += [("out", 6.0)]
else:
    seg += [("ph", TRIM), ("b3", LB), ("c2", LC), ("b4", LB), ("out", LIN)]
edges = np.cumsum([0] + [s[1] for s in seg])
LTOT = edges[-1]
segx = {name: (edges[i], edges[i + 1]) for i, (name, _) in enumerate(seg)}

# the geometric arm offset: implemented as a raised-cosine LENGTH bump on
# the top trim stub -- physically a slightly meandered/longer top arm.  In
# 2D we emulate the extra optical path with an equivalent index bump over
# the stub: dn_geo * TRIM = n_eff * DL  (exact for a length change at fixed
# mode index; validated against the phase model in the analysis).
NEFF = 1.2905          # slab TE0 n_eff at w=0.991 (dispersion.json)
DN_GEO = NEFF * DL / TRIM


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
        dn = 0.0
        if not COUPLER_ONLY and segx["ph"][0] <= x <= segx["ph"][1]:
            dn = DN + DN_GEO
        return mp.Medium(index=NSIL + dn)
    if abs(p.y + yc) <= W / 2:
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
                    extra_materials=[mp.Medium(index=NSIL + DN + DN_GEO
                                               + 1e-4)],
                    default_material=mp.Medium(index=1.0))
sim.sources = [mp.EigenModeSource(
    mp.GaussianSource(fcen, fwidth=df),
    center=mp.Vector3(src_x, D_A / 2), size=mp.Vector3(0, 2.8),
    eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]
if TWO_INPUT:
    sim.sources.append(mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(src_x, -D_A / 2), size=mp.Vector3(0, 2.8),
        eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True,
        amplitude=0.7j))
nfreq = 81
fl_bar = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(mon_x, D_A / 2), size=mp.Vector3(0, 2.8)))
fl_cross = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(mon_x, -D_A / 2), size=mp.Vector3(0, 2.8)))
fl_in = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(src_x + 1.0, D_A / 2), size=mp.Vector3(0, 2.8)))
fl_in2 = (sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
    center=mp.Vector3(src_x + 1.0, -D_A / 2), size=mp.Vector3(0, 2.8)))
    if TWO_INPUT else None)
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    60, mp.Ez, mp.Vector3(mon_x, D_A / 2), 1e-8))

freqs = np.array(mp.get_flux_freqs(fl_bar))
Pb = np.array(mp.get_fluxes(fl_bar))
Pc = np.array(mp.get_fluxes(fl_cross))
Pi = np.array(mp.get_fluxes(fl_in))
if TWO_INPUT:
    Pi = Pi + np.array(mp.get_fluxes(fl_in2))
s = (Pb - Pc) / (Pb + Pc)
i0 = int(np.argmin(np.abs(freqs - fcen)))
tag = (f"LC{LC:g}" if COUPLER_ONLY else f"LC{LC:g}_dL{DL:g}"
       + (f"_dn{DN:g}" if DN else ""))
json.dump(dict(mode=MODE, LC=LC, dL=DL, dn=DN, dn_geo=DN_GEO,
               node_len_um=float(LTOT),
               lam_nm=list(1000 / freqs), P_bar=list(Pb),
               P_cross=list(Pc), P_in=list(Pi), s=list(s)),
          open(f"out/compact_{MODE}_{tag}.json", "w"))
if COUPLER_ONLY:
    print(f"[compact coupler LC={LC:g}] cross frac at 1550 = "
          f"{Pc[i0]/(Pb[i0]+Pc[i0]):.4f}  T = {(Pb[i0]+Pc[i0])/Pi[i0]:.4f}")
else:
    print(f"[compact mzi LC={LC:g} dL={DL:g}] s(1550) = {s[i0]:+.4f}  "
          f"T = {(Pb[i0]+Pc[i0])/Pi[i0]:.4f}  node len {LTOT:.1f} um")

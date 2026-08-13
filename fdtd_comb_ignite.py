#!/usr/bin/env python
"""Comb ignition from a SINGLE pump line — the "only 1550 nm comes in"
requirement, tested in FDTD.

One CW pump on a ring resonance (the master ring is EO-locked to the
laser on the real chip), chi3 ARTIFICIALLY BOOSTED (0.05, labeled — real
silica chi3=2.5e-22 m^2/V^2 needs the literature Q~1e8 regime, see the
silica repo README), plus a femtosecond noise burst at t=0 with amplitude
1e-3 of the pump — the explicit stand-in for the vacuum fluctuations that
seed real Kerr-comb ignition.  Desired effect: sidebands at +-FSR grow
out of the noise floor by degenerate FWM: 2*f_pump -> (f_pump+FSR) +
(f_pump-FSR).

Writes out/comb_ignite.json + fig/snap_comb.png (spectrum + ring field).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, OUT = os.path.join(HERE, "fig"), os.path.join(HERE, "out")

F_PUMP = 0.64574          # ring resonance (mrr_comb.py lineage), 1548.6 nm
FSR = 0.65909 - 0.64574   # adjacent-mode spacing of the same ring
CHI3 = 0.05
RUNTIME = 4000.0
R, GAP, W = 8.0, 0.40, 0.991
NSIL = 1.444024
RES = 20

pad = 3.0
sx = 2 * (R + W) + 2 * pad + 4
sy = 2 * (R + W) + 2 * pad + GAP + W + 2
ring_cy = -sy / 2 + pad + W + R + W / 2
bus_y = ring_cy + R + W / 2 + GAP + W / 2
src_x, mon_x = -sx / 2 + 1.8, sx / 2 - 1.8

sil_nl = mp.Medium(index=NSIL, chi3=CHI3)
geom = [
    mp.Cylinder(radius=R + W / 2, center=mp.Vector3(0, ring_cy),
                material=sil_nl),
    mp.Cylinder(radius=R - W / 2, center=mp.Vector3(0, ring_cy),
                material=mp.Medium(index=1.0)),
    mp.Block(size=mp.Vector3(mp.inf, W, mp.inf),
             center=mp.Vector3(0, bus_y), material=sil_nl),
]
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry=geom,
                    resolution=RES, boundary_layers=[mp.PML(1.5)],
                    default_material=mp.Medium(index=1.0))
sim.sources = [
    mp.EigenModeSource(mp.ContinuousSource(F_PUMP, width=100),
                       amplitude=1.0, center=mp.Vector3(src_x, bus_y),
                       size=mp.Vector3(0, 2.4), eig_band=1,
                       eig_parity=mp.ODD_Z, eig_match_freq=True),
    # vacuum-noise surrogate: weak broadband femtosecond burst
    mp.EigenModeSource(mp.GaussianSource(F_PUMP, fwidth=0.10 * F_PUMP),
                       amplitude=1e-3, center=mp.Vector3(src_x, bus_y),
                       size=mp.Vector3(0, 2.4), eig_band=1,
                       eig_parity=mp.ODD_Z, eig_match_freq=True),
]
ts = []


def rec(sim):
    ts.append(np.real(sim.get_field_point(mp.Ez,
                                          mp.Vector3(mon_x, bus_y))))


snap = {}


def grab(sim):
    snap["field"] = np.abs(sim.get_array(component=mp.Ez,
                                         center=mp.Vector3(),
                                         size=mp.Vector3(sx, sy)))


sim.run(mp.at_every(0.05, rec), mp.at_end(grab), until=RUNTIME)

sig = np.array(ts)
dt = 0.05
# analyze only the LATE window (t > RUNTIME/3): the seed burst and its
# direct spectral pedestal have left the cell; what remains at +-FSR is
# ring-stored FWM light, so the comparison floor is honest
i_late = len(sig) // 3
sig = sig[i_late:]
n = len(sig)
win = np.hanning(n)
S = np.abs(np.fft.rfft(sig * win)) ** 2
fr = np.fft.rfftfreq(n, d=dt)
sel = (fr > F_PUMP - 3.2 * FSR) & (fr < F_PUMP + 3.2 * FSR)


def line_power(f0, half=0.15 * FSR):
    m = (fr > f0 - half) & (fr < f0 + half)
    return float(S[m].max()) if m.any() else 0.0


def local_pedestal(f0, inner=0.15 * FSR, outer=0.45 * FSR):
    m = ((abs(fr - f0) > inner) & (abs(fr - f0) < outer))
    return float(np.median(S[m])) if m.any() else 1e-30


pump = line_power(F_PUMP)
side = {k: line_power(F_PUMP + k * FSR) for k in (-2, -1, 1, 2)}
ped = {k: local_pedestal(F_PUMP + k * FSR) for k in (-2, -1, 1, 2)}
res = {"chi3": CHI3, "runtime": RUNTIME, "f_pump": F_PUMP, "FSR": FSR,
       "pump_power": pump, "sidebands": side, "local_pedestal": ped,
       "sideband_dB_above_local": {str(k): 10 * np.log10(
           max(side[k], 1e-30) / ped[k]) for k in side},
       "sideband_dBc": {str(k): 10 * np.log10(max(v, 1e-30) / pump)
                        for k, v in side.items()},
       "spectrum_f_over_FSR": list((fr[sel] - F_PUMP) / FSR),
       "spectrum_S_over_pump": list(S[sel] / pump)}
json.dump(res, open(os.path.join(OUT, "comb_ignite.json"), "w"))

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), dpi=150,
                        gridspec_kw={"width_ratios": [1.4, 1]})
ax = axs[0]
lam = 1000 / fr[sel]
ax.semilogy(fr[sel] / FSR - F_PUMP / FSR, S[sel] / pump, color="#0072B2",
            lw=0.9)
for k in (-2, -1, 1, 2):
    ax.axvline(k, color="#E69F00", lw=0.8, ls="--", alpha=0.7)
ax.set_xlabel("(f − f_pump) / FSR")
ax.set_ylabel("through-port PSD / pump")
ax.set_title("COMB IGNITION from ONE line (late-window spectrum): ±FSR "
             f"lines {res['sideband_dB_above_local']['1']:.0f}/"
             f"{res['sideband_dB_above_local']['-1']:.0f} dB above LOCAL "
             "pedestal — boosted χ³, noise-seeded (labeled)", fontsize=9)
ax = axs[1]
f = snap["field"]
ax.imshow((f.T / f.max()) ** 0.5, cmap="magma", origin="lower",
          aspect="equal")
ax.set_title("|Ez| at end of run — pump circulating in the ring",
             fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG, "snap_comb.png"), bbox_inches="tight")
plt.close(fig)
print("pump", f"{pump:.3e}", "sidebands dBc:",
      {k: round(v, 1) for k, v in res["sideband_dBc"].items()})

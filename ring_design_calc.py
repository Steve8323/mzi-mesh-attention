#!/usr/bin/env python
"""Industry-standard MRR design — closed form, no FDTD sweeps.

Inputs are MEASURED quantities from this repo's calibrations:
  gamma   evanescent decay constant of the coupler gap, fit from the
          coupler_design.json Lpi(gap) table (silica repo FDTD)
  n_g     group index at w = 0.991 um (dispersion.json)
  Q_int   intrinsic (bend-limited) Q vs R from the silica repo ring runs

Standard formulas (Bogaerts et al., Laser Photon. Rev. 6, 47 (2012)):
  round-trip time     T = n_g L / c,  L = 2 pi R
  FSR                 = 1/T (Hz)  = lam^2/(n_g L) (m)
  point coupling      kappa = sin( (pi / 2 Lpi(g)) * L_eff ),
                      L_eff = sqrt(2 pi R / gamma)   (curved-bus overlap)
  per-trip amplitude  a_rt = sqrt( (1-|kappa|^2) * exp(-omega T / Q_int) )
  loaded Q            1/Q_L = 1/Q_int + 1/Q_c,
                      Q_c = -omega T / ln(1-|kappa|^2)
  finesse             = FSR / linewidth = pi a_rt / (1 - a_rt^2) (approx)
  linewidth           = f / Q_L
  ring-down           tau = Q_L / omega
  EO gate requirement dn > n_g * linewidth/f / Gamma (one linewidth shift)

Two design sheets:
  1. FDTD demo ring for fdtd_e2e.py mrr: round trip = 150 fs slot
  2. real-chip master ring: R = 16 um, FSR 1.995 THz
Writes out/ring_design.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
C0 = 299792458.0
NG = 1.4947983969526526
LAM = 1.55e-6
OMEGA = 2 * np.pi * C0 / LAM
GAMMA_EO = 0.97                       # ring EO overlap (silica repo)

# gamma from the measured Lpi(gap) table (log-linear fit)
tab = json.load(open(os.path.join(HERE, "calib", "coupler_design.json")))
gaps = np.array([r["gap"] for r in tab])
lpis = np.array([r["Lpi"] for r in tab])
GAMMA = float(np.polyfit(gaps, np.log(lpis), 1)[0])       # 1/um
LPI0 = float(np.exp(np.polyfit(gaps, np.log(lpis), 1)[1]))  # Lpi at g=0

# measured intrinsic Q anchor points (silica repo ring_q runs)
QI_R = {5.0: 2.6e3, 8.0: 1.3e5, 16.0: 1.1e7}


def q_int(R):
    """log-interpolate measured bend-limited Q(R)."""
    rs = np.array(sorted(QI_R))
    qs = np.array([QI_R[r] for r in rs])
    return float(np.exp(np.interp(R, rs, np.log(qs))))


def lpi(gap):
    return LPI0 * np.exp(GAMMA * gap)


def design(R_um, target_k2=None, gap=None):
    R = R_um * 1e-6
    L = 2 * np.pi * R
    T = NG * L / C0
    fsr = 1.0 / T
    L_eff = np.sqrt(2 * np.pi * R_um / GAMMA)          # um
    if gap is None:
        # solve gap for target power coupling
        arg = np.arcsin(np.sqrt(target_k2))
        lpi_needed = np.pi * L_eff / (2 * arg)
        gap = float(np.log(lpi_needed / LPI0) / GAMMA)
    k = np.sin(np.pi / (2 * lpi(gap)) * L_eff)
    k2 = float(k ** 2)
    qi = q_int(R_um)
    a_int = np.exp(-OMEGA * T / qi / 2)
    a_rt = float(np.sqrt(1 - k2) * a_int)
    qc = -OMEGA * T / np.log(max(1 - k2, 1e-12))
    ql = 1.0 / (1.0 / qi + 1.0 / qc)
    lw = C0 / LAM / ql
    dn_gate = NG * (lw / (C0 / LAM)) / GAMMA_EO
    return {
        "R_um": R_um, "gap_um": float(gap), "L_eff_um": float(L_eff),
        "round_trip_fs": T * 1e15, "FSR_THz": fsr / 1e12,
        "kappa2": k2, "Q_int": qi, "Q_coupling": float(qc),
        "Q_loaded": float(ql), "linewidth_GHz": float(lw / 1e9),
        "finesse": float(np.pi * a_rt / (1 - a_rt ** 2)),
        "ringdown_ps": float(ql / OMEGA * 1e12),
        "per_trip_amplitude": a_rt,
        "train_amps_5": [float(a_rt ** n) for n in range(5)],
        "dn_for_one_linewidth": float(dn_gate),
    }


if __name__ == "__main__":
    print(f"measured evanescent constant gamma = {GAMMA:.3f} /um "
          f"(Lpi = {LPI0:.2f} um * e^(gamma*gap))")
    # 1. FDTD demo clock ring: round trip = the 150 fs scale-model slot
    R_demo = 150e-15 * C0 / NG / (2 * np.pi) * 1e6
    demo = design(R_demo, target_k2=0.10)
    print(f"\nDEMO clock ring (one pulse per round trip = one slot):")
    for k, v in demo.items():
        print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 4)}")
    # 2. real-chip master ring R=16
    chip = design(16.0, gap=0.40)
    print(f"\nREAL master ring (R=16, gap 0.40):")
    for k, v in chip.items():
        print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 4)}")
    json.dump({"gamma_per_um": GAMMA, "demo": demo, "chip": chip},
              open(os.path.join(OUT, "ring_design.json"), "w"), indent=1)

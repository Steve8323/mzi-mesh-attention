#!/usr/bin/env python
"""FTIR intensity gate, biased to trip just above the AVERAGE beam level.

Geometry: the repo's collinear FTIR barrier — the silica guide (w=0.991,
n=1.444) interrupted by a sub-wavelength AIR GAP. Below threshold the gap
frustrates transmission (evanescent tunneling); the Kerr index rise at
high intensity (chi3, ARTIFICIALLY BOOSTED and labeled, as in
mrr_comb.py) weakens the frustration -> transmission rises with
intensity. The barrier gap is THE knob (sister-project result): it sets
where the knee sits relative to the average intensity.

The user's claim under test: 'pulse peak intensity, in proportion to the
rest of the beam, scales with beam strength — stronger beams create
sharper pulses'. E3 measures output-vs-input contrast of a weak pulse
bump riding on a CW background, at three drive strengths.

Reality mapping (honest): with chi3_boost = 0.05 and the demo drive
amplitudes A ~ 1, the induced dn ~ (3 chi3/8n) |E|^2 ~ 5e-3. Real silica
chi3 = 2.5e-22 m^2/V^2 gives the same dn at |E|^2 ~ 8e19 V^2/m^2, i.e.
intensity ~ n eps0 c |E|^2/2 ~ 1.5e13 W/cm^2 — far above silica damage
(~kW peak in 0.5 um^2 is ~1e11 W/cm^2). A REAL silica gate at this
contrast needs resonant assistance (put the barrier inside the ring) or
a higher-chi3 overlay; as a bare element it is a >=100x-too-weak Kerr
device. The demo establishes the MECHANISM and the bias rule only.

usage: ftir_gate.py static|knee|pulse|all
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")
W = 0.991
NSIL = 1.444024
RES = 25
FCEN = 1 / 1.55
CHI3 = 0.05                       # BOOSTED (labeled); real silica 2.5e-22
L_IN, L_OUT = 16.0, 14.0
SY = 10.0


def run_gate(gap, amp, chi3, tmax=800.0, src_env=None):
    """Guide - air gap - guide. Returns (t, Pin(t), Pout(t)) point powers
    and the steady flux ratio T (via time-integrated |Ez|^2 tail)."""
    sx = L_IN + gap + L_OUT + 8
    mat = mp.Medium(index=NSIL, chi3=chi3) if chi3 else \
        mp.Medium(index=NSIL)
    geom = [
        mp.Block(size=mp.Vector3(L_IN, W, mp.inf),
                 center=mp.Vector3(-sx / 2 + 4 + L_IN / 2, 0),
                 material=mat),
        mp.Block(size=mp.Vector3(L_OUT + 4, W, mp.inf),
                 center=mp.Vector3(-sx / 2 + 4 + L_IN + gap
                                   + (L_OUT + 4) / 2, 0),
                 material=mat),
    ]
    sim = mp.Simulation(cell_size=mp.Vector3(sx, SY), geometry=geom,
                        resolution=RES, boundary_layers=[mp.PML(1.5)],
                        default_material=mp.Medium(index=1.0))
    if src_env is None:
        src = mp.ContinuousSource(FCEN, width=40)
    else:
        src = mp.CustomSource(src_env, start_time=0, end_time=tmax,
                              center_frequency=FCEN)
    sim.sources = [mp.EigenModeSource(
        src, amplitude=amp, center=mp.Vector3(-sx / 2 + 2.0, 0),
        size=mp.Vector3(0, 3.0), eig_band=1, eig_parity=mp.ODD_Z,
        eig_match_freq=True)]
    x_in = -sx / 2 + 4 + L_IN - 2.0
    x_out = -sx / 2 + 4 + L_IN + gap + 6.0
    fl_out = sim.add_flux(FCEN, 0.02 * FCEN, 3, mp.FluxRegion(
        center=mp.Vector3(x_out + 4.0, 0), size=mp.Vector3(0, 3.0)))
    rec = {"t": [], "pin": [], "pout": []}
    ys = np.linspace(-1.0, 1.0, 9)

    def grab(sim):
        rec["t"].append(sim.meep_time())
        rec["pin"].append(sum(np.real(sim.get_field_point(
            mp.Ez, mp.Vector3(x_in, y))) ** 2 for y in ys))
        rec["pout"].append(sum(np.real(sim.get_field_point(
            mp.Ez, mp.Vector3(x_out, y))) ** 2 for y in ys))
    sim.run(mp.at_every(0.25, grab), until=tmax)
    t = np.array(rec["t"])
    pin, pout = np.array(rec["pin"]), np.array(rec["pout"])
    flux = float(np.array(mp.get_fluxes(fl_out))[1])
    return t, pin, pout, flux


def static():
    _, _, _, f_ref = run_gate(0.0, 1.0, 0.0, tmax=500)   # no-gap reference
    rows = []
    for gap in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        _, _, _, f = run_gate(gap, 1.0, 0.0, tmax=500)
        T = f / f_ref
        rows.append({"gap": gap, "T_linear": T})
        print(f"  gap {gap:.2f} um  T = {T:.4f}")
    json.dump(rows, open(os.path.join(OUT, "ftir_gate_static.json"), "w"),
              indent=1)
    return rows


def knee(gap=0.30, a_avg=1.0):
    rows = []
    for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        _, _, _, f_ref = run_gate(0.0, a * a_avg, 0.0, tmax=700)
        _, _, _, f = run_gate(gap, a * a_avg, CHI3, tmax=700)
        T = f / f_ref                      # per-drive linear reference
        rows.append({"amp_over_avg": a, "T": T})
        print(f"  A/A_avg {a:4.2f}  T = {T:.4f}")
    # knee metrics at the average point
    Ts = {r["amp_over_avg"]: r["T"] for r in rows}
    dTdI = (Ts[1.25] - Ts[0.75]) / (1.25 ** 2 - 0.75 ** 2)
    res = {"gap": gap, "chi3_boosted": CHI3, "rows": rows,
           "T_at_avg": Ts[1.0],
           "dT_dI_at_avg_normalized": float(dTdI / max(Ts[1.0], 1e-9)),
           "contrast_20pct_above": float(Ts[1.25] / Ts[1.0])}
    json.dump(res, open(os.path.join(OUT, "ftir_gate_knee.json"), "w"),
              indent=1)
    print(f"  knee: T(avg)={Ts[1.0]:.4f}, +20% intensity -> "
          f"T x{Ts[1.25]/Ts[1.0]:.2f}, dT/dI|avg (norm) = "
          f"{dTdI/max(Ts[1.0],1e-9):.2f}")
    return res


def pulse(gap=0.30):
    """CW background + weak pulse bump (peak 1.3x background field),
    three drive strengths -> output vs input contrast."""
    T0 = 45.0            # ~150 fs in meep t.u.
    TC = 400.0

    def env(t):
        return (1.0 + 0.3 * np.exp(-((t - TC) ** 2) / (2 * T0 ** 2))) \
            * np.exp(-2j * np.pi * FCEN * t)

    rows = []
    for a in (0.7, 1.0, 1.4):
        t, pin, pout, _ = run_gate(gap, a, CHI3, tmax=800, src_env=env)
        # smooth the carrier out
        k = max(int(2.0 / (t[1] - t[0])), 1)
        ker = np.ones(k) / k
        pi_s = np.convolve(pin, ker, "same")
        po_s = np.convolve(pout, ker, "same")
        sel_bg = (t > 150) & (t < 250)
        sel_pk = (t > TC - 30) & (t < TC + 30)
        cin = 1.3 ** 2                 # source design contrast (known);
        # the x_in trace sits in a standing wave and is NOT usable
        cout = float(po_s[sel_pk].max() / po_s[sel_bg].mean())
        rows.append({"drive": a, "contrast_in": cin, "contrast_out": cout,
                     "discrimination": cout / cin,
                     "t": t[::4].tolist(), "pin": pi_s[::4].tolist(),
                     "pout": po_s[::4].tolist()})
        print(f"  drive {a:.1f}: contrast in {cin:.3f} -> out {cout:.3f} "
              f"(discrimination x{cout/cin:.3f})")
    json.dump({"gap": gap, "chi3_boosted": CHI3, "rows": rows},
              open(os.path.join(OUT, "ftir_gate_pulse.json"), "w"))
    return rows


def figures():
    st = json.load(open(os.path.join(OUT, "ftir_gate_static.json")))
    kn = json.load(open(os.path.join(OUT, "ftir_gate_knee.json")))
    pu = json.load(open(os.path.join(OUT, "ftir_gate_pulse.json")))
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=150)
    ax = axs[0]
    ax.semilogy([r["gap"] for r in st], [r["T_linear"] for r in st],
                "o-", color="#0072B2", lw=2)
    ax.axvline(kn["gap"], color="#D55E00", ls="--", lw=1)
    ax.set_xlabel("barrier gap (µm)")
    ax.set_ylabel("linear transmission")
    ax.set_title("E1: the barrier knob — bias gap chosen\nat the knee "
                 "foot (dashed)", fontsize=9)
    ax = axs[1]
    rows = kn["rows"]
    ax.plot([r["amp_over_avg"] ** 2 for r in rows],
            [r["T"] for r in rows], "o-", color="#009E73", lw=2)
    ax.axvline(1.0, color="#8a8a97", ls="--", lw=1)
    ax.set_xlabel("intensity / average")
    ax.set_ylabel("transmission (χ³ boosted, labeled)")
    ax.set_title(f"E2: the knee — +20% over average → T ×"
                 f"{kn['contrast_20pct_above']:.2f}", fontsize=9)
    ax = axs[2]
    for r, col in zip(pu["rows"], ("#56B4E9", "#0072B2", "#D55E00")):
        ax.plot([r["drive"]], [r["discrimination"]], "o", ms=10,
                color=col, label=f"drive {r['drive']:.1f}")
    ax.plot([r["drive"] for r in pu["rows"]],
            [r["discrimination"] for r in pu["rows"]], "-",
            color="#8a8a97", lw=1)
    ax.axhline(1.0, color="#8a8a97", ls=":", lw=1)
    ax.set_xlabel("beam strength (× knee drive)")
    ax.set_ylabel("output/input pulse contrast")
    ax.set_title("E3: the user's claim — discrimination vs\nbeam "
                 "strength", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ftir_gate_knee.png"),
                bbox_inches="tight")
    # time traces
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    for r, col in zip(pu["rows"], ("#56B4E9", "#0072B2", "#D55E00")):
        t = np.array(r["t"])
        po = np.array(r["pout"])
        ax.plot(t * 3.33564 / 1000, po / max(po[(t > 150) & (t < 250)].mean(), 1e-30),
                color=col, lw=1.5,
                label=f"drive {r['drive']:.1f} (out contrast "
                      f"{r['contrast_out']:.2f})")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("output power / its background")
    ax.set_title("E3: output pulse (input bump contrast 1.69 in "
                 "intensity) at three beam strengths", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ftir_gate_pulse.png"),
                bbox_inches="tight")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("static", "all"):
        print("== E1 static T(gap)")
        static()
    if which in ("knee", "all"):
        print("== E2 knee T(I) at bias gap")
        knee()
    if which in ("pulse", "all"):
        print("== E3 pulse discrimination vs beam strength")
        pulse()
    if which == "all":
        figures()
        print("wrote fig/ftir_gate_knee.png fig/ftir_gate_pulse.png")

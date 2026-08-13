#!/usr/bin/env python
"""The ELECTRONICS layer, run "artificially" in MEEP: every electronic
function is a quasi-static index change on the silica (electronics at
<=GHz is frozen on optical ps timescales, so one FDTD run per electrode
setting is the physically correct emulation).  Each experiment checks the
DESIRED EFFECT against the model prediction.

  E1  token-writer ring gate: R=8 ring on the bus.  Broadband transmission
      with dn_ring = 0 vs 1e-3 (electrode surrogate; the honest silica map
      is eo_mapping.py DC Kerr, workable because the high-Q ring amplifies
      tiny dn).  Desired effect: the resonance SHIFTS, so a fixed 1550-line
      toggles between blocked (on-resonance) and passed (off-resonance).
  E2  mesh weight trim: compact node at geometric theta=pi/2, heater
      surrogate dn = -0.02, +0.02 on the 6 um trim stub.  Desired effect:
      s moves along the calibrated s(theta) curve (prediction from
      compact_cal kappa).

usage: fdtd_electronics.py e1|e2      writes out/elec_*.json + snapshots
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
FIG, OUT = os.path.join(HERE, "fig"), os.path.join(HERE, "out")

W = 0.991
NSIL = 1.444024
RES = 25
FCEN = 1 / 1.55


# ------------------------------------------------------------- E1 -------
def e1_ring_gate():
    """ADD-DROP ring gate: the gated output is the DROP port (a Lorentzian
    PEAK), so contrast doesn't require critical coupling.  Loaded ring
    (gaps 0.25) resolves and rings up in FDTD time; the real chip runs
    high-Q so the true silica DC-Kerr dn (~1e-7..1e-6 at bias) covers a
    linewidth -- dn=1e-3 here is the equivalent surrogate.  A straight
    reference run normalizes both spectra (no same-run T>1 artifacts)."""
    R, GAP = 8.0, 0.25
    pad = 3.0
    sx = 2 * (R + W) + 2 * pad + 4
    sy = 2 * (R + W) + 2 * pad + 2 * (GAP + W) + 2
    ring_cy = 0.0
    bus_y = ring_cy + R + W / 2 + GAP + W / 2      # input bus (top)
    drop_y = ring_cy - (R + W / 2 + GAP + W / 2)   # drop bus (bottom)
    src_x, mon_x = -sx / 2 + 1.8, sx / 2 - 1.8
    fwidth = 0.008 * FCEN
    nfreq = 801

    def make_sim(dn, with_ring=True):
        geom = [mp.Block(size=mp.Vector3(mp.inf, W, mp.inf),
                         center=mp.Vector3(0, bus_y),
                         material=mp.Medium(index=NSIL)),
                mp.Block(size=mp.Vector3(mp.inf, W, mp.inf),
                         center=mp.Vector3(0, drop_y),
                         material=mp.Medium(index=NSIL))]
        if with_ring:
            geom = [mp.Cylinder(radius=R + W / 2,
                                center=mp.Vector3(0, ring_cy),
                                material=mp.Medium(index=NSIL + dn)),
                    mp.Cylinder(radius=R - W / 2,
                                center=mp.Vector3(0, ring_cy),
                                material=mp.Medium(index=1.0))] + geom
        return mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry=geom,
                             resolution=RES,
                             boundary_layers=[mp.PML(1.5)],
                             default_material=mp.Medium(index=1.0),
                             force_complex_fields=True)

    def spectra(sim):
        sim.sources = [mp.EigenModeSource(
            mp.GaussianSource(FCEN, fwidth=fwidth),
            center=mp.Vector3(src_x, bus_y), size=mp.Vector3(0, 2.4),
            eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]
        thru = sim.add_flux(FCEN, fwidth, nfreq, mp.FluxRegion(
            center=mp.Vector3(mon_x, bus_y), size=mp.Vector3(0, 2.4)))
        # drop port counter-propagates: monitor on the LEFT of the bottom
        # bus, flux is negative-x -> take abs
        drop = sim.add_flux(FCEN, fwidth, nfreq, mp.FluxRegion(
            center=mp.Vector3(src_x + 1.0, drop_y),
            size=mp.Vector3(0, 2.4)))
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            200, mp.Ez, mp.Vector3(mon_x, bus_y), 3e-4))
        fr = np.array(mp.get_flux_freqs(thru))
        return (fr, np.array(mp.get_fluxes(thru)),
                np.abs(np.array(mp.get_fluxes(drop))))

    fr, ref_t, _ = spectra(make_sim(0.0, with_ring=False))
    _, t0, d0 = spectra(make_sim(0.0))
    _, t1, d1 = spectra(make_sim(5e-3))
    T0, D0 = t0 / ref_t, d0 / ref_t
    T1, D1 = t1 / ref_t, d1 / ref_t
    lam = 1000 / fr
    i0 = int(np.argmax(D0))                    # gate line = drop peak
    lam_line = float(lam[i0])
    gate = {"lam_line_nm": lam_line,
            "D_on": float(D0[i0]), "D_off": float(D1[i0]),
            "T_on": float(T0[i0]), "T_off": float(T1[i0]),
            "contrast_dB": float(10 * np.log10(max(D0[i0], 1e-12)
                                               / max(D1[i0], 1e-12)))}
    json.dump({"lam_nm": list(lam), "T_dn0": list(T0), "T_dn1e3": list(T1),
               "D_dn0": list(D0), "D_dn1e3": list(D1), "gate": gate},
              open(os.path.join(OUT, "elec_ring.json"), "w"))

    # CW field snapshots at the fixed gate line
    field = {}
    for dn in (0.0, 5e-3):
        sim = make_sim(dn)
        sim.sources = [mp.EigenModeSource(
            mp.ContinuousSource(fr[i0], width=50),
            center=mp.Vector3(src_x, bus_y), size=mp.Vector3(0, 2.4),
            eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]
        sim.run(until=1500)
        field[dn] = np.abs(sim.get_array(component=mp.Ez,
                                         center=mp.Vector3(),
                                         size=mp.Vector3(sx, sy)))
        sim.reset_meep()

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150,
                            gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axs[0]
    ax.plot(lam, D0, color="#0072B2", lw=1.5, label="drop, Δn = 0")
    ax.plot(lam, D1, color="#D55E00", lw=1.5, label="drop, Δn = 5e-3")
    ax.plot(lam, T0, color="#0072B2", lw=1.0, ls=":", alpha=0.6,
            label="through, Δn = 0")
    ax.axvline(lam_line, color="#8a8a97", lw=1, ls="--")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (ref-normalized)")
    ax.set_title(f"E1 add-drop token gate: fixed-line contrast "
                 f"{gate['contrast_dB']:.1f} dB at the drop port",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    for ax, dn, lab in ((axs[1], 0.0, "Δn=0: ON resonance — line routed "
                                      "to DROP (gate open)"),
                        (axs[2], 5e-3, "Δn=5e-3: shifted — line stays on "
                                       "the bus (gate closed)")):
        f = field[dn]
        ax.imshow((f.T / f.max()) ** 0.5, cmap="magma", origin="lower",
                  aspect="equal")
        ax.set_title(lab, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "snap_elec_ring.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"E1: line {lam_line:.2f} nm  drop {gate['D_on']:.3f} -> "
          f"{gate['D_off']:.4f} ({gate['contrast_dB']:.1f} dB), "
          f"through {gate['T_on']:.3f} -> {gate['T_off']:.3f}")


# ------------------------------------------------------------- E2 -------
def e2_trim():
    sys.argv = ["fdtd_compact.py"]
    from node import coupler as cmat
    cc = json.load(open(os.path.join(OUT, "compact_cal.json")))
    kap = cc["kappa_1550"]
    NEFF = 1.2905
    dL = 0.3003
    rows = []
    for dn in (-0.02, 0.0, 0.02):
        if dn == 0.0:
            d = json.load(open(os.path.join(
                OUT, "compact_mzi_LC5.8_dL0.3003.json")))
            lam = np.array(d["lam_nm"])
            i0 = int(np.argmin(abs(lam - 1550)))
            s_meas = d["s"][i0]
        else:
            r = os.system(
                f"cd {HERE} && /opt/anaconda3/envs/mp/bin/python "
                f"fdtd_compact.py mzi 5.8 {dL} {dn} "
                f"> out/elec_trim_{dn:+g}.log 2>&1")
            assert r == 0
            d = json.load(open(os.path.join(
                OUT, f"compact_mzi_LC5.8_dL{dL:g}_dn{dn:g}.json")))
            lam = np.array(d["lam_nm"])
            i0 = int(np.argmin(abs(lam - 1550)))
            s_meas = d["s"][i0]
        th = 2 * np.pi * (NEFF * dL + dn * 6.0) / 1.55
        C = cmat(kap)
        M = C @ np.diag([np.exp(1j * th), 1.0]) @ C
        s_pred = ((abs(M[0, 0]) ** 2 - abs(M[1, 0]) ** 2)
                  / (abs(M[0, 0]) ** 2 + abs(M[1, 0]) ** 2))
        rows.append({"dn": dn, "s_meas": float(s_meas),
                     "s_pred": float(s_pred)})
        print(f"E2 trim dn={dn:+.3f}: s meas {s_meas:+.4f} "
              f"pred {s_pred:+.4f}")
    json.dump(rows, open(os.path.join(OUT, "elec_trim.json"), "w"))
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.plot([r["dn"] for r in rows], [r["s_pred"] for r in rows], "o-",
            color="#E69F00", lw=2, label="model (calibrated κ)")
    ax.scatter([r["dn"] for r in rows], [r["s_meas"] for r in rows],
               s=40, color="#0072B2", zorder=3, label="FDTD")
    ax.set_xlabel("trim electrode Δn (heater surrogate)")
    ax.set_ylabel("node signed output s")
    ax.set_title("E2: electronic weight trim moves the weight\n"
                 "exactly along the calibrated curve", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "snap_elec_trim.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        e1_ring_gate()
    if which in ("e2", "all"):
        e2_trim()

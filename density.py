"""Compute-density audit: MAC/s per mm^2 of photonic die, honestly
accounted, vs an NVIDIA B300 die.

B300 baseline (announced specs, Tom's Hardware / NVIDIA 2025):
  15 PFLOPS dense FP4 = 7.5e15 MAC/s, dual reticle-class dies ~1650 mm^2
  -> 4.5e12 MAC/s/mm^2 (die-level; the SXM module is far larger, so this
  denominator is the CONSERVATIVE choice), 1400 W.

Our accounting rules:
  - Everything photonic on-die is counted: mesh, delay ladder coils,
    master ring + electrodes, 1xN splitter tree, output trim, and a global
    +15% routing/keep-out overhead.  WDM-mixer configs also pay an echelle
    demux per output port.
  - Token-writer electronics (the silica repo's serializer, ~GHz DACs) and
    photodetectors are NOT on this die (flip-chip / off-chip), stated.
  - MACs per slot: N^2 for the mixer (real tokens through a fixed unitary),
    x d for d comb lines.  Slot rate = ring FSR.
  - Accuracy per config from the calibrated OTDM error model (bits column;
    B300 runs FP4 with FP32 accumulate -- flagged where we're below that).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

B300_MACS = 7.5e15
B300_MM2 = 1650.0
B300_DENS = B300_MACS / B300_MM2

C0 = 299792458.0
NG = 1.4947983969526526


def slot_len_um(fsr_hz):
    return C0 / NG / fsr_hz * 1e6


def mesh_area_mm2(N, L_node, pitch):
    return (N * L_node) * (N * pitch) * 1e-6


def delay_area_mm2(N, fsr_hz, pitch, coil_overhead=1.3):
    total = N * (N - 1) / 2 * slot_len_um(fsr_hz)
    return coil_overhead * total * pitch * 1e-6


def config(name, N, L_node, pitch, fsr_hz, bits, d_wdm=1,
           demux_mm2_per_port=0.0, notes=""):
    ring = 0.005          # R=16 master ring + electrodes + bus (mm^2)
    tree = (N - 1) * 250e-6            # 1xN splitter tree
    demux = N * demux_mm2_per_port if d_wdm > 1 else 0.0
    area = (mesh_area_mm2(N, L_node, pitch)
            + delay_area_mm2(N, fsr_hz, pitch) + ring + tree + demux)
    area *= 1.15                        # routing / keep-out
    macs = N ** 2 * fsr_hz * d_wdm
    dens = macs / area
    return {"name": name, "N": N, "d_wdm": d_wdm,
            "area_mm2": float(area), "MACs": float(macs),
            "density": float(dens), "x_B300": float(dens / B300_DENS),
            "bits": bits, "notes": notes}


def run(compact):
    """compact = dict from the compact-node FDTD calibration."""
    FSR2 = 1.9949785406530283e12
    FSR4 = 2 * FSR2                    # R=8 ring (flagged: comb Q risk)
    Ln_orig, p_orig = 96.7, 6.0
    Ln_c, p_c = compact["node_len_um"], 4.0
    rows = [
        config("original node, N=5", 5, Ln_orig, p_orig, FSR2,
               bits=6.6),
        config("compact node, N=5", 5, Ln_c, p_c, FSR2,
               bits=compact["bits_200fs"]),
        config("compact, N=16", 16, Ln_c, p_c, FSR2,
               bits=compact["bits_200fs"] - 1.0),
        config("compact, N=32", 32, Ln_c, p_c, FSR2,
               bits=compact["bits_200fs"] - 1.6),
        config("compact, N=32, 4 THz slot", 32, Ln_c, p_c, FSR4,
               bits=compact["bits_125fs"] - 1.6,
               notes="R=8 ring: comb threshold unproven at this Q"),
        config("compact, N=32, WDM d=10 + demux", 32, Ln_c, p_c, FSR2,
               bits=compact["bits_200fs"] - 1.6, d_wdm=10,
               demux_mm2_per_port=0.06,
               notes="per-lambda U differs (lambda-locked head family); "
                     "echelle demux per port included"),
    ]
    # score engine: NO mesh -- delay ladder + (N-1) balanced couplers only;
    # (N-1)*d MACs per slot (d-dim dot product per lag, detector sums
    # lambda so no demux). bits = model floor; detector noise not audited.
    N, d_wdm = 32, 10
    area = (delay_area_mm2(N, FSR2, p_c) + 0.005 + (N - 1) * 250e-6
            + (N - 1) * (Ln_c / 2) * p_c * 1e-6) * 1.15
    macs = (N - 1) * d_wdm * FSR2
    rows.append({"name": "score engine, N=32 lags, d=10 comb",
                 "N": N, "d_wdm": d_wdm, "area_mm2": float(area),
                 "MACs": float(macs), "density": float(macs / area),
                 "x_B300": float(macs / area / B300_DENS),
                 "bits": 17.9,
                 "notes": "QK^T band only, no mesh; bits = model floor "
                          "(detector noise not audited)"})
    return rows


if __name__ == "__main__":
    cal = json.load(open(os.path.join(OUT, "compact_cal.json")))
    rows = run(cal)
    print(f"B300 baseline: {B300_DENS:.2e} MAC/s/mm^2 "
          f"(7.5e15 MAC/s dense FP4 / 1650 mm^2)")
    print(f"{'config':38s} {'area mm2':>9s} {'MAC/s':>9s} "
          f"{'MAC/s/mm2':>10s} {'xB300':>7s} {'bits':>5s}")
    for r in rows:
        print(f"{r['name']:38s} {r['area_mm2']:9.4f} {r['MACs']:9.2e} "
              f"{r['density']:10.2e} {r['x_B300']:7.0f} {r['bits']:5.1f}")
    json.dump({"B300_density": B300_DENS, "rows": rows},
              open(os.path.join(OUT, "density.json"), "w"), indent=1)

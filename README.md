# Pulse-train MZI mesh: sliding unitary token mixer + photonic attention scores

Feeds the silica-repo MRR pulse train into an N×N (default 5×5) Clements MZI
mesh, with each input port carrying the SAME train delayed by exactly one
slot per port. **ANALYSIS.md answers the "is this attention?" question**
(short: no — it is exactly the Hyena/FNet-class fixed token mixer; but the
photodetector's square law computes the QKᵀ score band natively, and that
part is exact). `report.html` has all results.

Everything is pinned to the silica-eo-converter worktree's FDTD
calibrations, copied verbatim into `calib/` (see Provenance).

## Headline results

| | |
|---|---|
| slot rate | **1.995 THz** = master-MRR FSR (R=16 µm, n_g=1.4948); one-slot delay = 100.53 µm = one ring circumference |
| 5×5 calibrated mesh vs target U | **0.6–0.9 % rms**, net T 0.934 — Clements decomposition run NATIVELY on the measured 45/55 coupler (κ=0.7368), no global optimization |
| token mixing (OTDM) accuracy | **6.6 bits at 200 fs** (full error model) vs **0.8 bits with the 16.4 fs compressed pulse**: compression buys slot rate, NOT mixer accuracy — use the longest pulse the guard band allows |
| ⚠ NEW: per-node phase dispersion | quadrature 2-input FDTD reveals **−0.25°/nm** MZI internal phase slope (≈6 fs differential group delay), absent in the bare coupler (+0.04°/nm); static offset is SET-POINT dependent (−50° at full-cross, −16° at dn=0.01) → calibration LUT must be 2D; the λ-slope is not calibratable and is included in all OTDM numbers |
| identity | peak-sampled outputs ≡ y_{i,t} = Σ_j U_ij·s_{t−j} (causal banded Toeplitz = N orthonormal FIR heads), rms 0.010 at the operating point |
| useful as-is | U=DFT₅ → sliding DFT / 2-THz-frame spectrogram (the FNet mixer), rms 0.0036 |
| attention gap (structural) | best fixed linear fit to vector-token softmax attention R² = 0.54–0.70; to a conv target R² = 1.000000 |
| photonic score band | balanced homodyne of train × k-slot-delayed train (λ/4 offset): s_t·s_{t−k} to **rms 4e-6** after 3-run leak correction (uncorrected 3% — the 45/55 imbalance term, removed exactly) |
| full attention | photonic scores + electronic softmax = float attention to rms 2.5e-6; EO pace (0.1–1 GHz) frames the loop, ~2000 slots |
| FDTD checks | coherent 2-input node: rms 0.0043 @1550 after static phase cal (κ unchanged, LP_eff 23.70→23.88); score head on real pulse trains: token products to **rms 1e-4** |
| scaling | 6.8 bits (N=5) → 4.0 bits (N=32), loss 2.3 dB at N=32, 10-bit phase DACs suffice; N² MACs / 501 fs slot |
| **compute density vs B300** | B300 die = 4.5e12 MAC/s/mm² (15 PFLOPS dense FP4 / 1650 mm²). Compact node (52.6 µm, 4 µm pitch, FDTD-recalibrated, geometric θ, screen only +0.10°/nm): **N=5 → 579×, N=32 → 802×** at 5–7 bits; 4 THz slot → 2184×, WDM d=10 + demux → 1623× (caveats in report §8). Honest accounting: mesh + delay coils + ring + tree + demux, ×1.15 routing; writer DACs/PDs off-die; energy NOT audited |

## Layout

```
ANALYSIS.md        the attention question, answered first
node.py            FDTD-calibrated 2x2 node model (kappa(lam), LP_eff, loss)
clements.py        native-imperfect-coupler Clements decomposition + mesh
otdm.py            pulse trains, one-slot delay ladder, spectral propagation,
                   balanced score head (leak-corrected)
demo_attention.py  demos A-D (Toeplitz identity, sliding DFT, expressivity,
                   score band + closed attention)
scaling.py         N-sweep, loss, DAC quantization
fdtd_node2.py      MEEP: coherent two-input MZI node (DN AMP2 PSI_DEG)
fdtd_coupler2.py   MEEP: single-coupler two-input control (localizes the
                   phase-dispersion screen to the MZI, not the launch)
node2_joint.py     joint recalibration: staircase + both 2-input runs
fdtd_score.py      MEEP: score head on real pulse trains (scale model,
                   150 fs slots, labeled)
fdtd_compact.py    MEEP: compact node (52.6 µm) — coupler sweep, geometric
                   theta, quadrature screen
compact_cal.py     assembles the compact calibration + OTDM accuracy
density.py         compute-density audit vs B300 (out/density.json)
figs.py report.py  figures + self-contained report.html
calib/             copied FDTD calibration JSONs (provenance below)
```

## Reproduce

```bash
PY=/opt/anaconda3/envs/mp/bin/python
$PY node.py && $PY clements.py        # self-tests (exact reconstruction)
$PY demo_attention.py                 # demos A-D  -> out/demo.json
$PY scaling.py                        # N sweep    -> out/scaling.json
$PY fdtd_node2.py 0.01 0.7 90         # MEEP, ~3 min each
$PY fdtd_node2.py 0.01 0.7 0
$PY fdtd_node2.py 0 0.7 90            # set-point dependence control
$PY fdtd_coupler2.py 90               # launch/coupler control
$PY node2_joint.py                    # joint recalibration
$PY fdtd_score.py                     # MEEP, 3 sub-runs
for LC in 5.35 7 8.5; do $PY fdtd_compact.py coupler $LC; done
$PY fdtd_compact.py mzi 5.8 0         # compact cross state
$PY fdtd_compact.py mzi 5.8 0.3003    # geometric theta = pi/2
$PY fdtd_compact.py mzi2 5.8 0        # compact quadrature screen
$PY compact_cal.py && $PY density.py
$PY figs.py && $PY report.py
```

## Provenance of calib/

Copied from `~/.claude/worktrees/silica-eo-converter/silica_eo_converter/out/`
(all MEEP 1.29 2D FDTD on the suspended-silica platform, w=0.991 µm):
`mzi_*_narrow.json` (s(Δn) staircase + spectra, LC=5.35, gap 0.30),
`coupler_design.json`, `dispersion.json` (slab n_g/β₂ sweep),
`pulse_5mm.json` (shortest 5 mm-safe pulse), `ring_q_R16.json` (Q=1.1e7).
The broadband MZI file there has dead monitors (all-zero flux) and is NOT
used; the narrow files span 1470–1630 nm which covers every operating point
except the ±3 dB tails of the 16.4 fs pulse (clamped — see report caveats).

## Caveats

- 2D slab physics; no substrate leakage/roughness (inherited).
- Full-mesh FDTD intractable; instead the exact spectral envelope model is
  pinned by two node-level FDTD checks (§6 of the report).
- Per-slot 2 THz electrical readout is NOT claimed (outputs stay optical,
  are time-binned, or read at electronics rate).
- Token writing at aggregate rate = the silica repo's serializer result,
  taken as given.

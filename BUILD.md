# BUILD.md — making the pulse-train mesh chip

**What you are building:** one suspended-silica photonic layer + one metal
electronics layer above it. A single 1550 nm CW laser enters at one facet;
everything else — the pulse train, the one-slot delays, the unitary mixing,
the score readout — happens on-chip. The optical simulations behind every
number here are 2D MEEP; every electronic function was emulated in MEEP as
a quasi-static Δn and checked for the desired effect (see §7).

```
      LAYER 2 (electronics, on the unreleased mesa + flip-chip):
        Ag electrodes ─ ring lock + token gates (DC-Kerr, GHz)
        NiCr heaters  ─ mesh weight trims (thermo-optic, quasi-static)
        Ge/InGaAs PDs ─ balanced score receivers (flip-chip)
      ───────────────────────────────────────────────────────────
      LAYER 1 (optics): 1.0 µm thermal SiO2, air-clad suspended
        CW in → master MRR → splitter taps → 1-slot delays → mesh
      ───────────────────────────────────────────────────────────
      Si carrier (XeF2-released under all guides)
```

## 1 · Stack and masks (4 levels)

| # | Mask | Field | Layer | Purpose | Min feature |
|---|------|-------|-------|---------|-------------|
| 1 | MZM-WG-01 | dark | optics | waveguide etch (everything optical) | 0.20 µm tap gap |
| 2 | MZM-MET-02 | clear | electronics | Ti 5/Ag 200 nm liftoff: ring electrodes, token-gate pads, rails | 1.0 µm |
| 3 | MZM-HTR-03 | clear | electronics | NiCr 100 nm heaters over mesh trim arms | 2.0 µm |
| 4 | MZM-REL-04 | dark | release | XeF₂ windows | 6 µm |

Process (identical lineage to the silica-repo converter flow):

1. Si ⟨100⟩ wafer; grow **1.0 µm wet thermal oxide** (1100 °C). This oxide
   IS the optical layer; w = 0.991 µm guides, n = 1.444.
2. **Mask 1**: e-beam or DUV. Expose ALL optics at once (§3–§6 geometry).
   RIE full etch (CHF₃/Ar), vertical sidewalls. CD-critical: coupler gaps
   0.30 µm ±20 nm, tap gaps 0.20 µm, node arm-length offsets δL ±25 nm.
3. Sidewall smoothing anneal 1000 °C N₂ 2 h.
4. **Mask 2**: image-reversal liftoff, one evaporation Ti 5 nm/Ag 200 nm.
   Metal only on unreleased mesa; nothing crosses a suspended span.
5. **Mask 3**: NiCr heater strips over (not touching — 1 µm oxide spacer
   ridge) each mesh node's straight sections, ~30 µm long.
6. **Mask 4**: release windows; XeF₂ undercut ~8 µm/side; O₂ plasma strip
   (no liquids after release).
7. Facets: deep trench in mask 1 at the single input facet + inverse taper
   (0.3 µm tip) for the lensed fiber. **One optical input. 1550 nm. CW.**
8. Flip-chip: PD array + CMOS driver die onto mask-2 pads (the whole
   electronics layer is ≤1 GHz — relaxed bumping).

## 2 · The one physics decision per electronic function

Silica has no χ⁽²⁾; the honest electro-optics (silica-repo eo_mapping.py,
DC Kerr χ³ = 2.5e-22 m²/V²) gives only Δn ≈ 2.6e-6 at 100 V/µm bias. That
is enough ONLY where a high-Q resonance amplifies it. So:

| function | mechanism | why it works | verified |
|---|---|---|---|
| ring lock + token gates | DC-Kerr on the master MRR (Q ~ 10⁷: linewidth Δn ~ 1.4e-7 < Kerr 2.6e-6) | resonance shift toggles the fixed 1550 line | E1 (Δn surrogate) |
| mesh weights | **geometric**: arm-length offset δL cut into mask 1 | θ = 2π·n_eff·δL/λ; validated to ~2° | compact_cal |
| mesh fine trim | NiCr heater, dn/dT = 1e-5/K, 30 µm arm, ΔT ≤ 30 K → ±2° | covers fab tolerance + the 2D-LUT static offsets | E2 (Δn surrogate) |
| score readout | balanced PD pairs, 3-measurement leak correction | products exact to 1e-4 (FDTD) | fdtd_score |

**Never** try to hold a mesh weight with a Kerr electrode on a 6 µm stub —
it is 500× too weak. Weights are geometry; electronics only trims and
gates.

## 3 · Master ring (the pulse source — the only light source is 1550 CW)

- R = 16 µm, gap 0.40 µm, w = 0.991. FSR = 1.995 THz = the slot rate.
- Ag electrode pair (3 µm wide, 1 µm from guide edge, inner+outer): DC
  bias 50–100 V/µm + lock servo. Lock the resonance TO the laser (silica
  dn/dT = 1e-5/K → slow drift, kHz servo).
- Kerr comb: FDTD (fdtd_comb_ignite.py, boosted χ³, noise-seeded, labeled)
  shows STIMULATED FWM at ±FSR while the seed is present, but sidebands
  persist only 6–9 dB above the local pedestal once it decays —
  **single-pump self-start is NOT demonstrated in FDTD** (this ring is at
  zero GVD; MI gain is marginal). Real-chip comb operation needs the
  literature Q ≈ 1e8–1e9 silica regime with engineered anomalous
  dispersion (wedge/reflow, LLE dynamics) — the ONE block of this design
  that leans on demonstrated-elsewhere physics.
  Fallback that stays all-silica and 1550-only: run the mesh with the
  16 fs → 200 fs compressed-then-stretched pulses of the silica repo chain.
- Output pulse for the mixer: **200 fs FWHM** (guard 2.5 in the 501 fs
  slot). Do NOT feed the mixer the shortest pulse the comb can make
  (16 fs = 0.8 bits; 200 fs = 6.7 bits — report §3).

## 4 · Splitter + one-slot delays (the OTDM front end)

- Tap-bus ladder (like the N=50 serializer): N−1 taps at gap 0.20 µm,
  lengths set for equal power 1/N, 1/(N−1), …; bus end = last channel.
- **The delay rule: one slot = v_g·T_slot = 100.53 µm of w=0.991 bus =
  exactly one master-ring circumference per tap.** Channel j carries
  j slots.
- Cut tolerance ±λ_g/2 (±0.54 µm = ±2.7 fs); route as raised-cosine
  meanders with bend radius ≥ 8 µm. The end-to-end MEEP cell solves meander
  amplitudes by arc-length bisection and then MEASURES the achieved delays
  at the mesh inputs (fdtd_e2e.py) — copy that solver for layout.
- Residual per-channel timing error budget: 9.6 fs for 6 bits (silica-repo
  OTDM control table).

## 5 · The mesh (N×N, compact node)

- Node (52.6 × 4 µm): 1.5 in / 8 s-bend / 5.8 coupler / 8 / 6 trim stub /
  8 / 5.8 / 8 / 1.5 out. Gap 0.30 µm. Port pitch 4 µm (edge gap 3 µm →
  <1% crosstalk over the array). Node T = 0.975.
- Clements rectangle: N columns × N rows; even columns pair rows (0,1),
  (2,3)…, odd columns (1,2),(3,4)…
- **Weights**: run `clements.decompose(U, kappa=0.7846)` (the decomposition
  is exact for the measured coupler); convert each node's θ to an arm
  offset δL = θ·λ/(2π·n_eff), n_eff = 1.2905, and each φ + output-screen
  phase to input-side stub offsets. Etch the offsets into mask 1.
- Heater trim per node (±2°) absorbs fab error + the static part of the
  quadrature phase offsets (§6).

## 6 · Calibration procedure (all findings from this repo's FDTD)

1. Broadband source, camera on dumps: per-tap ratios, two trim passes.
2. Lock master ring to the laser (DC pad).
3. **Per-node 2D LUT**: the node's quadrature phase offset depends on its
   set point (−50° at full-cross, −16° at mid) — calibrate (set point ×
   offset) per node once, apply via heater. A single number is NOT enough.
4. Input phase screen: absorb into the first column of trims (standard).
5. The λ-slope (+0.10°/nm compact node) is NOT calibratable — it is why
   the operating pulse is 200 fs, not 16 fs.
6. Score receivers: store the two single-arm balanced readings per lag;
   corrected score = both-on − armA − armB (kills the 45/55 leak exactly).

## 7 · What was verified in MEEP (map instruction → evidence)

| claim | file | result |
|---|---|---|
| single 1550 line → ±FSR sidebands | fdtd_comb_ignite.py | stimulated FWM shown; self-start NOT shown (see §3) |
| splitter + physical 1-slot delays + mesh, one cell | fdtd_e2e.py | delays measured at mesh inputs: **0.92 / 1.79 slots** (targets 1/2; residual = tap-transfer ambiguity, the chip cut rule holds ±0.005 slot) |
| token gate = electronic Δn on add-drop ring | fdtd_electronics.py E1 | **13.6 dB** fixed-line contrast (drop 0.60→0.026), field snapshots |
| weight trim = electronic Δn on stub | fdtd_electronics.py E2 | s = −0.50…+0.43 tracks calibrated curve within 0.04 |
| geometric weight δL | fdtd_compact.py mzi 5.8 0.3003 | s = −0.04 vs 0.00 predicted |
| node under coherent 2-input drive | fdtd_node2.py + node2_joint.py | rms 0.0043 @1550 after static cal |
| score product on real pulse trains | fdtd_score.py | rms 1e-4 |
| 2D electronics emulation is legitimate | — | GHz electronics is static on ps optical timescales; one run per setting |

## 8 · Tolerance sheet

| quantity | value | tolerance | sets |
|---|---|---|---|
| coupler gap | 0.30 µm | ±20 nm | κ (±1.5%) |
| node δL (weights) | 0–0.60 µm | ±25 nm | θ ±3° (heater absorbs) |
| delay length | 100.53 µm/slot | ±0.54 µm (λ_g/2) | slot timing ±2.7 fs |
| pulse width at mixer | 200 fs | −50/+50 fs | 6–7 bits |
| heater range | ±2° per node | — | trim + LUT offsets |
| bend radius | ≥ 8 µm | — | <0.2% per bend |
| membrane span | ≤ 24 µm between tethers | — | stress/buckling |

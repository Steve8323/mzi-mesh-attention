# MASKS_PRINT.md — immediately printable 2D patterns

Two aligned layers, same coordinate frame (fdtd_e2e.py layout coords;
sidecar jsons state the frame origin and scale). Everything below is the
EXACT geometry the MEEP end-to-end ('mrr') cell ran — generated from the
same analytic centerlines and verified by re-rasterization against the
simulated MaterialGrid.

## Layer 1 — waveguides (this is mask 1 of BUILD.md)

- **PRIMARY (delivered): `out/litho_optics_bw.png`** — 1-bit, **50 nm/
  pixel exactly** (20 px/µm), WHITE = SiO₂ kept, 100 µm scale bar +
  corner crosses. This bitmap IS the MaterialGrid the MEEP end-to-end
  cell stepped — pattern fidelity is by construction, no polygon
  round-trip involved. Sidecar: `out/litho_optics_50nmpx.json`.
  (Viewing copy at 100 nm/px: `out/litho_optics_view.png`.)
- SECONDARY (scripted, pending): `masks_print.py` also emits vector
  GDSII from the analytic centerlines (25 nm sampling, chunked for the
  GDS record limit) plus a polygon-vs-grid verification; its slow
  rasterization pass kept exceeding the session's job limits, so run it
  offline if you want the vector file — the printable bitmap above does
  not depend on it.

Contents: input bus (left facet) → clock MRR (R = 4.788 µm, coupling gap
0.180 µm — CMT-designed, one-shot verified) → two tap couplers (gap
0.20 µm) → bus bend → one-slot meander delays (arc-length-solved) →
3×3 compact Clements mesh (coupler gaps 0.30 µm) → three outputs (right).

Min features: **0.18 µm** (ring gap), 0.20 µm (tap gaps), 0.30 µm
(mesh coupler gaps) → e-beam or DUV for this layer. Etch: full 1.0 µm
thermal oxide, vertical RIE (BUILD.md steps 2–3).

## Layer 2 — electrodes (see ELECTRODES.md)

- `out/mask_electrodes_1bit.png` + `out/mask_electrodes_scale.json` —
  50 nm/px, WHITE = metal (clear-field liftoff). Single layer, strictly
  planar, no crossings (programmatically verified: PASS). Min feature
  1.5 µm (jog traces) → contact/DLP printable.
- Alignment: both layers share the layout frame; use the corner crosses
  of layer 1 (overlay budget ±0.25 µm is loose — the electrode
  clearances are ≥0.8 µm and nothing on layer 2 is CD-critical).

## Print notes

- The PNGs are exact-scale bitmaps: 1 px = 50 nm. For a DLP writer with
  pixel pitch p µm, resample by nearest-neighbor with integer ratio
  p/0.05 — do NOT anti-alias (grayscale edges print as ragged sidewalls).
- Tone: layer 1 dark-field (white = keep SiO₂), layer 2 clear-field
  (white = metal after liftoff). Invert per your resist polarity.
- The optics layer's 0.18–0.30 µm gaps are beyond contact/DLP resolution;
  layer 1 needs e-beam/DUV (as in BUILD.md). Layer 2 prints anywhere.

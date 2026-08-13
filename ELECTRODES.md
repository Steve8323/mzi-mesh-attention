# ELECTRODES.md — single-layer planar electrode mask (no crossings)

Layer: ONE metal level (Ti/Ag liftoff for pads/traces; the heater strips
are functionally NiCr in BUILD.md's mask 3 — if printed as one layer with
the traces, the strips act as thin-metal resistors; both interpretations
use the same artwork). Frame: fdtd_e2e.py layout coordinates, identical
to the optics mask (out/mask_optics_scale.json). Files:
`out/mask_electrodes_1bit.png` (50 nm/px, WHITE = metal),
`out/mask_electrodes_scale.json`, `fig/mask_electrodes_preview.png`,
generator + verifier: `electrode_mask.py`.

## The 2D constraint and what it forced (user's hard rule: NO overlap)

Unlike waveguides, one metal layer admits no crossings at all. Three
topology decisions follow, all documented in the generator:

1. **The ring's inner electrode is unroutable** — any escape trace crosses
   the ring guide. The ring lock is therefore an OUTER heater arc
   (thermo-optic; one continuous 250°–375° arc kept ≥1 µm off the guide
   and below the bus), terminals routed down the x≈7 and x≈18.2
   corridors. Fast EO on the ring is not available in strict 2D — the
   fast token gate lives on the BUS instead (flanking pair at
   x ∈ [1.0, 4.6], the one place with clear sky above and below).
2. **The output phase screen uses N−1 = 2 trims** (rows 0 and 2; row 1 is
   the phase reference): the middle output row is topologically boxed
   between guides that run to the die edge. Physics unharmed — only
   relative phases matter.
3. **No shared ground rails**: a full-width rail is crossed by every
   signal trace that must reach its pad. Every terminal gets its own pad
   (7 top + 7 bottom, 45 µm pads, ≥8 µm gaps), matched to traces in
   sorted-x order with jog heights deepest-first — planar by
   construction for this staggered cluster pattern, then PROVED by the
   verifier, not assumed.

## Netlist (pad → function)

| edge | pads | net | drives |
|---|---|---|---|
| top | 1 | GATE_S | bus token-gate signal (flank above bus) |
| top | 2, 3 | H1, H1_r | node c0 trim heater (over its 6 µm ph stub) |
| top | 4, 5 | H3, H3_r | node c2 trim heater |
| top | 6, 7 | P1, P1_r | output row-0 phase trim |
| bottom | 1 | GATE_G | bus token-gate ground (flank below bus) |
| bottom | 2, 3 | RING_HTR A/B | ring lock heater arc terminals |
| bottom | 4, 5 | H2, H2_r | node c1 trim heater |
| bottom | 6, 7 | P3, P3_r | output row-2 phase trim |

## Verification (out/electrode_verify.json) — PASS

- distinct-net clearance ≥ 3.0 µm everywhere; the single intentional
  exception is the gate pair across the bus (2.99 µm — that gap IS the
  EO gap, whitelisted at 2.5).
- conductor-to-waveguide-edge clearance ≥ 1.0 µm, whitelisted intentional
  flanks/heaters ≥ 0.8 (measured min 0.87, the ring heater beside the
  ring guide).
- planarity: all conductors are disjoint polygons in one layer with the
  above clearances → zero crossings, of each other or of any waveguide.
- the verifier caught, during development: bottom pads drawn across the
  mesh rows (sign bug), both ring arcs sweeping into the bus, signal
  traces crossing the ground rail, heater terminals flagged against
  their own element, and two corridor pinches — all fixed by
  construction changes, none by relaxing the rules (except the two
  explicit whitelists above).

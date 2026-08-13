# OPTIMIZATION.md — 4-loop campaign with adversarial critic

Question posed: optimize the pulse-train mesh across metrics, with scaling,
and settle whether the "noise" is really noise — or inter-pulse overlap
from dispersion. Method: `opt_harness.py` (metrics: bits, impulse-response
ISI, broadening, walk-off, score rms, density) + `opt_loop.py` (4 loops),
each loop reviewed by an independent critic agent whose findings set the
next loop. All numbers on the FDTD-calibrated compact node.

## The user's noise question — answered

Inter-pulse overlap is REAL but it is not dispersion of the guides:
- Guide β₂/β₃ over the delay lines contribute exactly zero (0.09 fs² of
  GDD — arithmetic, endorsed by critic 1).
- The overlap that matters is (a) slot-tail overlap when the pulse is too
  long for the guard band — it caused a measured CLIFF at 180 fs FWHM in a
  501 fs slot (bits 7.2 → 5.2, neighbor-pulse amplitude 2.1% matches the
  observed rms 2.7%); and (b) per-node phase-screen walk-off (~2.2 fs/node
  bar-vs-cross differential group delay) which matters only for short
  pulses at large N.
- After loop 2 the budget FLIPPED: with realistic control errors
  (phase-set 5 mrad, coupler disorder 12 mrad, j-scaled delay trim 5 mrad,
  1 MHz comb), stochastic control error dominates everything
  dispersion-related at the operating point. "What looks like noise" is,
  in order: coupler fab disorder > phase-set/trim error > slot-overlap
  ISI (only if you overstuff the slot) > node dispersion > guide
  dispersion (nil).

## Loop history (findings → response)

| loop | ran | critic's key hits | response |
|---|---|---|---|
| 1 | baseline attribution, T0/N/slot scaling | split-recommendation untested in implementable form; no stochastic errors; single-seed stats; "91% loss" partly compilable; OAT attribution invalid at 10 fs | loop 2 |
| 2 | chirp-vs-filter split test, 8-seed MC, stochastic terms, 2³ factorial | chirp result = peak-sampling artifact risk (test finite-gate receiver); ring filter tails ~5-bit cap predicted; κ-injection not exact; delay errors must scale ∝j; CEO missing | loop 3 |
| 3 | receiver×arm adjudication, exact per-node κ, sensitivity spec table | gated receiver implementable only as pulsed-LO I/Q homodyne (4 PDs/port, demux, LO skew); CEO modeled wrong (static=trimmable, duplicated RNG row); κ-aware compile would be tautological without characterization error; score head still fictional | loop 4 |
| 4 | κ-aware compile + char error, per-pulse comb walk, score full-error, finalists at spec, honest Pareto | final sign-off + residual risks (below) | report |

## Final numbers (full error model, κ-aware compile, 6–8 seeds)

| config | bits | density (optical-out) | density (electronic readout) |
|---|---|---|---|
| N=5, 501 fs slot | 5.81 ± 0.09 | 579× | 23× |
| N=5, 251 fs | 5.77 ± 0.10 | 1375× | 46× |
| N=8, 501 fs | 5.15 ± 0.30 | 701× | 36× |
| N=8, 251 fs | 5.09 ± 0.28 | 1774× | 73× |
| N=16, 501 fs | 4.19 ± 0.20 | 779× | 70× |
| N=16, 251 fs | 3.97 ± 0.16 | 2079× | 142× |

Electronic-readout density prices the D=40 time-demux receiver bank per
port (the "no 2 THz ADC" elephant, critic 3). Chained optical-in/
optical-out mesh layers keep the 600–2000× numbers; edge layers pay the
readout tax.

## Design rules extracted

1. **Slot stuffing rule**: pulse FWHM ≤ 0.4 × slot (the 180 fs cliff in a
   501 fs slot). Compression buys slot rate, never mixer accuracy.
2. **Receiver/filter rule**: gaussian-class source filter + peak receiver,
   or raw compressed pulses + gated pulsed-LO homodyne — identical bits at
   spec (5.8); NEVER a single/double ring filter (Lorentzian tails: 1.4–3
   bits). Gated homodyne is preferred at large N (walk-off immune) if the
   I/Q + LO-distribution cost is acceptable.
3. **κ-aware compile**: measure each node's coupler to ≤5 mrad in situ and
   re-run the (exact, per-node-κ) Clements decomposition: +1.0 bit at
   spec. This turns the DOMINANT fab error into a calibration step.
4. **Comb spec**: ≤1 MHz linewidth is enough for both mesh and score
   paths (per-pulse phase-walk model).
5. **Score head**: robust at ~1e-5 rms under the full error model at the
   compressed tap; shot-limit line: 8-bit products need 4.2 fJ/slot/port
   (8.4 µW at 2 THz).
6. **Loss endgame** (unresolved, flagged): after κ-aware compile, per-node
   loss (T=0.975) is again the deterministic ceiling (~6.7 bits at N=5).
   Phases can't synthesize gain. Doors: attenuator-trim compile (costs
   overall T), digital post-correction at readout (costs nothing optical,
   only valid at edge layers), or accept the KNOWN measured M as the layer
   weight and train around it (the ML-native answer). The clean floor
   with everything idealized except geometry is ~11.3 bits.

## Critic-4 verdicts on the five conclusions

| conclusion | verdict |
|---|---|
| κ-aware compile +1.0 bit, char ≤3–5 mrad | **ENDORSE** (non-circular; in-situ mrad-class κ characterization procedure asserted, not designed) |
| comb ≤10 MHz not a limiter | **ENDORSE** (real microcombs ≲100 kHz → genuine margin) |
| receiver tie → pick by hardware | ENDORSE-WITH-CAVEAT: it's a TWO-way tie (gauss-filter ≡ TL source — same physical arm, agreement to 1e-5 bits is a consistency check); the gate may be the EASIER receiver (±100 fs timing tolerance vs tens-of-fs for peak) |
| score head robust, shot line 4.2 fJ | ENDORSE-WITH-CAVEAT: robust to modeled coherent errors; absolute 1e-5 rms is a noiseless floor — real floor is detector-set (+3 dB for I/Q, plus losses) |
| honest Pareto, readout collapse | ENDORSE-WITH-CAVEAT: demux mechanism = staggered pulsed-LO I/Q receivers (passive rings can't gate 501 fs; area similar, numbers survive); optical-out column is UNSIMULATED (no chained-layer run); ~4 bits at N=16 is marginal-but-plausible for score use |

## Remaining HIGH caveats (flag, not rerun)

- **H1** No energy/power axis anywhere; per-area density only. Energy/MAC
  is historically where these claims die (cf. 811 pJ/MAC in the AlN
  sister project).
- **H2** The optical-in/optical-out 600–2000× column has no chained-layer
  simulation (error accumulation ~−0.5 bit/doubling of depth, per-layer
  loss 0.88, softmax must detect eventually). Speculative.
- **H3** Electronic-readout demux must be worded as staggered pulsed-LO
  I/Q homodyne banks, not gate rings.
- **H4** The envelope model's COMPOSITION (mesh × pulse train) was never
  checked against full-wave; only node-level FDTD anchors exist. One
  2-node MEEP-vs-propagate() run would close it.
- **H5** Score-head numbers are noiseless floors.
- **H6** The deterministic-loss door (5.8 → ~11.3 bits) was identified,
  never walked: attenuator-trim compile / digital post-correction /
  measured-M-as-weights. Future work.

## Residual-risk statement (critic 4, verbatim)

All accuracy figures in this study are outputs of a frequency-domain
envelope model whose components — coupler κ(λ), node loss, phase screen,
guide dispersion — are individually FDTD-calibrated, but whose composition
(5×5 mesh × pulse train) has never been checked against an independent
full-wave simulation, and whose error budget contains no detector: every
"bits" value assumes noiseless dual-quadrature coherent readout, with shot
noise handled only as an ideal side calculation and ADC/receiver power
entirely unpriced. The stochastic magnitudes that now dominate the budget
(12 mrad κ from a ±20 nm gap spec, 5 mrad phase LUT, j-scaled 5 mrad
trim/thermal) are fab and control assumptions, treated as independent
Gaussians — correlated errors (heater/electrode crosstalk, systematic LUT
bias) would evade the √-cancellation this model enjoys — and the headline
+1.0-bit recovery presumes an in-situ mrad-class κ-characterization
procedure that has not been designed. The token encoder at the 2 THz slot
rate is unmodeled end to end; the electronic-readout densities depend on a
time-demux receiver bank whose mechanism (staggered pulsed-LO homodyne,
not passive gate rings) changes the wording but not, we believe, the
area; the optical-in/optical-out density column has no supporting
chained-layer simulation; and the remaining deterministic loss ceiling
(~6.7 bits, with an identified but unexecuted path to ~11) means the
reported 5.8 bits is neither a floor nor a fundamental limit, but the
state of the calibration engineering as modeled.

## The trajectory

7.2 bits (loop 1: single seed, deterministic, no receiver model) →
5.8 ± 0.1 bits (loop 4: 8 seeds, full stochastic spec, κ-aware compile,
adjudicated receiver). **The number went DOWN every time the model got
more honest — the signature of a critique loop working.**

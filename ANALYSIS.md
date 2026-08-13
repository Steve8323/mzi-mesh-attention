# Is a pulse-train-fed MZI mesh attention?

**Short answer: not attention — it is exactly the *attention-free token mixer*
(Hyena/FNet/S4 class). The delays turn the mesh from a per-vector matrix
multiply into a sliding-window (Toeplitz) mixer over the token sequence.
True attention is impossible for any passive mesh because attention is cubic
in its input and a mesh is linear. BUT the missing bilinearity is exactly
what a square-law photodetector provides: mesh + one-slot delay + balanced
homodyne detection computes the raw attention scores `q_t·k_{t-k}` — the
QKᵀ band — natively at the pulse line rate. Softmax stays electronic.**

## The structure (before any mechanism)

Three objects, one comparison:

1. **What you built.** Input channel `j` of the N×N mesh carries the *same*
   MRR pulse train, delayed by `j−1` slots. At slot `t` the mesh sees the
   window `(s_t, s_{t−1}, …, s_{t−N+1})` and outputs, at port `i`,

       y_i(t) = Σ_j U_ij · s_{t−j+1}

   Each output port is an FIR filter over the token sequence; the mesh is a
   bank of N filters whose kernels are the **rows of a unitary** — N
   orthonormal taps. In matrix language the operator acting on the whole
   sequence is **causal banded Toeplitz**: `A[t, t′] = U_{i, t−t′+1}` for
   `0 ≤ t−t′ < N`, zero otherwise.

2. **What attention is.** `y_t = Σ_{t′} softmax_t′(q_t·k_{t′}/√d) · v_{t′}`.
   The mixing weights *are computed from the data*. Counting powers of the
   input: scores are quadratic, output is cubic (plus softmax).

3. **The comparison.** Your operator has attention's *shape* (a full
   token-mixing matrix `A[t,t′]` applied across the sequence) but with
   `A` fixed and position-only — it depends on `t−t′`, never on content.
   That is precisely the "synthetic / positional attention" family:
   FNet (fixed DFT mixing), Hyena/S4 (long convolutions), Synthesizer
   (learned but input-independent `A`). Those models are the documented
   answer to "is a fixed mixer useful": near-transformer quality at small
   and mid scale, and they are the standard drop-in when you can't afford
   data-dependent mixing.

One-sentence mechanism: *delaying identical trains by one slot per port
pins the mesh's row index to relative position, converting matrix×vector
into convolution — you move the operator along the sequence while pinning
its coefficients.*

## Why no amount of delay/scaling makes it attention

A passive mesh obeys `E_out = T·E_in` — linear, always, in the optical
field. Attention output is *cubic* in the token embeddings. A linear map
cannot equal a cubic one on more than a measure-zero set. So the gap is
structural, not an engineering margin: no mesh topology, delay pattern, or
size N closes it. (Numerically: the best Toeplitz fit to a real softmax
attention layer on random tokens leaves an O(1) residual — `demo_attention.py`
puts a number on it; the same fit to a Hyena-style conv layer is exact.)

## What closes the gap: the photodetector

Photodetection is square-law: `P ∝ |E|²`. Interfere the train with its own
one-slot-delayed copy in the calibrated 50/50 coupler, with a λ/4 static
offset (388 nm of path) on one arm, and balanced-detect:

    P_bar − P_cross = −2·r·t·s_t·s_{t−k}   (homodyne cross term)

— the pairwise product of two *different tokens*. That is the attention
score for relative offset `k`. N−1 taps of the delay ladder + N−1 balanced
pairs compute the whole causal score band `{s_t·s_{t−k}}` in parallel, one
band per slot, at the 2 THz slot rate. This is the O(L·N) part of
attention — the part that costs the FLOPs. Softmax + re-weighting of V is
electronic (and the silica EO port is photon-lifetime-limited to ~0.1–1 GHz,
so a data-dependent second optical pass works on frames of ~2000 slots, not
per-slot).

**Vector tokens for free:** the master MRR is a comb source. Put embedding
dimension `c` on comb line `c` (the mesh couplers are calibrated over
1470–1630 nm); a photodetector sums power over wavelength, so the balanced
pair returns `Σ_c q_{t,c} k_{t−k,c} = q_t·k_{t−k}` — a true d-dimensional
dot product per slot, no extra hardware.

## So what does the fixed mesh compute that is useful?

- `U = DFT_N` → the mesh is a **sliding N-point DFT**: a real-time
  spectrogram of the token stream at 2 THz frame rate (`demo_attention.py`
  runs a chirp through it). FNet is literally this mixer inside a
  transformer.
- `U = any unitary` → an energy-conserving, information-lossless (invertible)
  multi-head FIR token mixer — the Hyena/S4 short-filter primitive. Because
  rows are orthonormal, no token direction is amplified or lost; the layer
  is perfectly conditioned (κ=1) regardless of depth.
- With detection → the QKᵀ score band at line rate (above).

Throughput bookkeeping: N² MACs per slot, slot rate = ring FSR = 1.995 THz.
N=5 → 5.0×10¹³ MAC/s from a passive ~0.5 mm² mesh; N=64 → 8.2×10¹⁵ MAC/s
(before loss/accuracy limits — `scaling.py` quantifies both). Realized
accuracy at the recommended 200 fs operating point: 6.6 bits (5×5, full
FDTD-measured error model including the −0.25°/nm per-node phase
dispersion discovered in the quadrature validation runs — see report §6).

## Honest placement vs prior art

Delayed-replica photonic convolution is Xu et al., Nature 589, 44 (2021)
(11 TOPS photonic convolutional accelerator — wavelength-weighted delay
taps). The two twists here: (1) the tap bank is a *unitary mesh*, so the N
filters form one orthonormal (multi-head, invertible) mixer rather than
independent kernels; (2) the balanced-homodyne reading of the delay ladder
as an attention-score engine. Photonic *full* attention demos exist
(2023–2025, e.g. Nature Comm. photonic attention with EO re-modulation) but
all of them route scores through electronics for the softmax — same
boundary we land on, which is evidence the boundary is real physics
(linearity of passive optics), not a design failure.

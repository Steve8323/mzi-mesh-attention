#!/usr/bin/env python
"""Self-contained HTML gallery of the MEEP geometry + field snapshots."""
import base64
import datetime
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")


def img(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return f"<p style='color:#D55E00'>[missing: {name}]</p>"
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f"<img src='data:image/png;base64,{b}'>"


html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MEEP geometry + field snapshots — compact MZI mesh</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:32px auto;
max-width:1200px;color:#1a1a2e;line-height:1.4;background:#fafafa}}
h1{{font-size:22px}} h2{{font-size:17px;border-bottom:2px solid #0072B2;
padding-bottom:4px;margin-top:30px}}
img{{max-width:100%;border:1px solid #ccc;border-radius:6px;margin:6px 0;
background:#000}}
p.c{{font-size:13.5px;color:#444;margin:4px 0 14px}}
</style></head><body>
<h1>MEEP geometry &amp; field amplitudes — compact silica MZI mesh</h1>
<p class="c">All 2D Ez FDTD, MEEP 1.29, complex fields (|Ez| = envelope).
Node/score scenes at the quantitative resolution (25 px/&mu;m); the full
mesh scene is a res-20 visual run (numbers in the report come from the
res-25 node calibrations). Fields shown as |Ez| with sqrt/0.4 gamma so the
weak arms stay visible.</p>

<h2>1 · Compact node — geometry</h2>
{img('snap_node_eps.png')}
<p class="c">52.6 &mu;m node: 1.5 in &middot; 8 s-bend &middot; 5.8 coupler
&middot; 8 &middot; 6 trim stub &middot; 8 &middot; 5.8 &middot; 8 &middot;
1.5 out, port pitch 4 &mu;m, coupler gap 0.30 &mu;m, w = 0.991 &mu;m
suspended silica in air.</p>

<h2>2 · Compact node — CW steady-state |Ez|</h2>
{img('snap_node_cross.png')}
{img('snap_node_half.png')}
<p class="c">Top: as-drawn (&theta;=0) the node is in its CROSS state —
input on the upper guide exits the lower. Bottom: adding 0.30 &mu;m of
optical path on the trim stub (the GEOMETRIC &theta;=&pi;/2 weight used by
the fixed-weight mesh) splits the output 50/50 — this is the measured
s = &minus;0.04 point from the calibration.</p>

<h2>3 · Score head — geometry</h2>
{img('snap_score_eps.png')}
<p class="c">One calibrated 50/50 coupler. Arm A carries the token train
delayed by one slot; arm B carries the undelayed train with a
quarter-wave static offset.</p>

<h2>4 · Score head — pulse trains interfering, |Ez| frames</h2>
{img('snap_score_frames.png')}
<p class="c">Each frame: token pulses of the two trains meet PAIRWISE in
the coupler (150 fs slots, scale model). The interference splits
asymmetrically depending on the product s_t&middot;s_{{t&minus;1}} — the
balanced detector difference per slot is the attention score. This is the
run that reproduced the products to 1e-4.</p>

<h2>5 · Full compact 5&times;5 Clements mesh — geometry</h2>
{img('snap_mesh_eps.png')}
<p class="c">All 10 nodes in one FDTD cell: 5 columns &times; 52.6 &mu;m,
5 guides at 4 &mu;m pitch (275 &times; 17 &mu;m active). Even columns pair
rows (0,1),(2,3); odd columns pair (1,2),(3,4) — the Clements rectangle.</p>

<h2>6 · Full mesh — pulse train cascading, |Ez| frames</h2>
{img('snap_mesh_frames.png')}
<p class="c">A 3-token train (0.9, &minus;0.7, 0.8) enters the middle port
and cascades through the five columns; as-drawn nodes sit at their cross
state, so power ping-pongs across the array — every coupler event in this
picture is one MAC of the systolic convolution.</p>

<h2>7 · END-TO-END (measured delays: 0.92 / 1.79 slots vs targets 1 / 2): one 1550 input → split → 1-slot delays → mesh</h2>
{img('snap_e2e_eps.png')}
{img('snap_e2e_frames.png')}
<p class="c">The whole optical chain in ONE cell (N=3, res 20, 150 fs
slots — scale model of the 501 fs / 100.53 µm chip rule). One eigenmode
source on the bus; two taps + the bus bend make three channels; the
raised-cosine meanders are SOLVED by arc-length bisection to exactly one
and two slots of extra path; the achieved delays are then MEASURED at the
mesh inputs (title). You can watch the same train arrive at the three
mesh rows staggered by one slot each — the sliding window forming — and
then mix through the three nodes.</p>

<h2>8 · The electronics layer, run "artificially" as Δn</h2>
{img('snap_elec_ring.png')}
<p class="c">E1 — token-writer gate as an ADD-DROP ring: electrode Δn = 0
vs 5e-3 slides the drop resonance 5.5 nm off the fixed 1549.5 nm line —
<b>13.6 dB gate contrast</b> (drop 0.60 → 0.026, through 0.31 → 0.97),
ring visibly lit vs dark. The surrogate Δn matches the real chip's
shift-to-linewidth ratio: at Q=10⁷ the true silica DC-Kerr Δn (~2.6e-6 at
bias) is 18 linewidths — same mechanism, resolvable in FDTD time.</p>
{img('snap_elec_trim.png')}
<p class="c">E2 — weight trim: heater-surrogate Δn = ±0.02 on the 6 µm
trim stub of a node parked at geometric θ=π/2. The signed output s moves
along the calibrated s(θ) curve — the electronic knob does exactly what
the model says it should.</p>

<h2>9 · Comb ignition from ONE line — honest verdict: marginal</h2>
{img('snap_comb.png')}
<p class="c">Single pump on a ring resonance, χ³ boosted (0.05, labeled),
plus a 1e-3 broadband burst standing in for vacuum noise. While the seed
is present, stimulated FWM puts the ±FSR sidebands ~30 dB over the far
floor; in the LATE window (seed gone) they persist only 6–9 dB above the
local pedestal — <b>self-starting ignition is NOT demonstrated in this
FDTD</b>, consistent with this ring sitting at zero GVD where single-pump
MI gain is marginal. The real chip's comb needs the literature silica
regime (Q~1e8, engineered anomalous dispersion, LLE dynamics) — that is
the one block of BUILD.md that leans on demonstrated-elsewhere physics.</p>

<p style="color:#8a8a97">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}
&middot; snapshots.py &middot; local MEEP</p>
</body></html>"""

path = os.path.join(HERE, "snapshots.html")
open(path, "w").write(html)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
copy = os.path.join(HERE, f"snapshots_{stamp}.html")
open(copy, "w").write(html)
print(path)
print(copy)

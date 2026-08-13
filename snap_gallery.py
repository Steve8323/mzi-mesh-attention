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

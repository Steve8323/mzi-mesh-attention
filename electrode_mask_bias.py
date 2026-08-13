#!/usr/bin/env python
"""BIAS-ONLY electrode layer: every conductor is a single statically
biased electrode referenced to the Si substrate (one edge contact).
No currents anywhere: no heaters, no returns, no pairs.

Elements:
  RING_BIAS  outer arc over unreleased mesa (release map keeps Si below
             it): lock/gate bias for the clock ring (Q-amplified Kerr)
  GATE_BIAS  single strip above the bus at the head (token gate; the
             substrate below that unreleased section is the counter
             electrode)
  SUB        substrate contact pad (die corner, over an oxide via)
Mesh phase correction = geometric weights + one-time permanent UV/laser
trim (optical write, no electrode) -> documented, no artwork needed.
Same planarity verifier as electrode_mask.py.  Outputs *_bias files.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")
sys.argv = ["fdtd_e2e.py", "mrr"]
_head = open(os.path.join(HERE, "fdtd_e2e.py")).read().split(
    "sim = mp.Simulation")[0]
exec(compile(_head, "h", "exec"))

TR = 3.0
def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
def vseg(y0, y1, x, w=TR):
    return rect(x - w/2, x + w/2, min(y0, y1), max(y0, y1))
def hseg(x0, x1, y, w=TR):
    return rect(x0, x1, y - w/2, y + w/2)
def arc(cx, cy, ri, ro, a0, a1, n=200):
    th = np.linspace(np.deg2rad(a0), np.deg2rad(a1), n)
    o = np.column_stack([cx + ro*np.cos(th), cy + ro*np.sin(th)])
    i_ = np.column_stack([cx + ri*np.cos(th), cy + ri*np.sin(th)])
    return np.vstack([o, i_[::-1]])

cx, cy = RING_C
ri, ro = R_RING_D + W/2 + 1.0, R_RING_D + W/2 + 4.0
PAD = 80.0
conductors = {
  # ring bias arc (below-bus sweep) + single trace down-left to a pad
  "RING_BIAS": [arc(cx, cy, ri, ro, 250, 375),
                vseg(cy + (ri+ro)/2*np.sin(np.deg2rad(250)), -70.0,
                     cx + (ri+ro)/2*np.cos(np.deg2rad(250))),
                rect(2.0, 2.0+PAD, -70.0-PAD, -70.0)],
  # token gate: ONE strip above the bus, single trace up to a pad
  "GATE_BIAS": [rect(1.0, 8.0, Y_BUS + W/2 + 1.0, Y_BUS + W/2 + 4.0),
                vseg(Y_BUS + W/2 + 2.5, 70.0, 4.0),
                rect(2.0, 2.0+PAD, 70.0, 70.0+PAD)],
  # substrate contact (over an oxide via, die corner far from optics)
  "SUB": [rect(LTOT-90.0, LTOT-10.0, -70.0-PAD, -70.0)],
}
# ---- verify: pairwise >=3um, wg edge >=0.8 (flanks whitelisted) --------
from scipy.spatial import cKDTree
wg = []
xs = np.arange(0.0, X_BUSBEND[0], 0.1)
wg.append(np.column_stack([xs, np.full_like(xs, Y_BUS)]))
for k in range(3):
    xs = np.arange(0.0, LTOT, 0.1)
    ys = np.array([y_channel(k, x) for x in xs]); ok = ~np.isnan(ys)
    wg.append(np.column_stack([xs[ok], ys[ok]]))
th = np.linspace(0, 2*np.pi, 720)
wg.append(np.column_stack([cx + R_RING_D*np.cos(th),
                           cy + R_RING_D*np.sin(th)]))
wg = np.vstack(wg)
def bpts(p, step=0.2):
    q = np.vstack([p, p[:1]]); out = []
    for a, b in zip(q[:-1], q[1:]):
        n = max(int(np.hypot(*(b-a))/step), 1)
        out.append(a + np.linspace(0,1,n,endpoint=False)[:,None]*(b-a))
    return np.vstack(out)
nets = {k: np.vstack([bpts(p) for p in v]) for k, v in conductors.items()}
rep = {"pairs": {}, "wg": {}, "pass": True}
names = sorted(nets)
for i, a in enumerate(names):
    for b in names[i+1:]:
        d = float(cKDTree(nets[b]).query(nets[a])[0].min())
        rep["pairs"][f"{a}|{b}"] = round(d, 2)
        if d < 3.0: rep["pass"] = False
t = cKDTree(wg)
for a in names:
    d = float(t.query(nets[a])[0].min()) - W/2
    rep["wg"][a] = round(d, 2)
    if d < 0.8: rep["pass"] = False
rep["note"] = ("bias-only: all conductors single-ended vs substrate; "
               "no currents; mesh trim = permanent UV/laser write "
               "(no artwork); release map must keep Si under RING_BIAS "
               "arc and GATE_BIAS strip")
json.dump(rep, open(os.path.join(OUT, "electrode_bias_verify.json"),
                    "w"), indent=1)
print("verify:", "PASS" if rep["pass"] else "FAIL", rep["pairs"], rep["wg"])
# ---- preview + 1-bit ----------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
ax.scatter(wg[:,0], wg[:,1], s=0.05, color="#999")
for (net, ps), col in zip(sorted(conductors.items()),
                          ("#0072B2", "#D55E00", "#009E73")):
    for p in ps: ax.fill(p[:,0], p[:,1], color=col, alpha=0.85, lw=0)
    ax.text(ps[-1][:,0].mean(), ps[-1][:,1].mean(), net, fontsize=8,
            ha="center")
ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
ax.set_title("BIAS-ONLY electrode layer: 3 nets, no currents, no pairs — "
             + ("PASS" if rep["pass"] else "FAIL"), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "mask_electrodes_bias_preview.png"),
            bbox_inches="tight")
print("wrote preview")

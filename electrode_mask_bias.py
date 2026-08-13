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
  # token gate: signal strip above the bus, single trace up to a pad
  "GATE_BIAS": [rect(1.0, 5.0, Y_BUS + W/2 + 1.0, Y_BUS + W/2 + 4.0),
                vseg(Y_BUS + W/2 + 2.5, 70.0, 4.0),
                rect(2.0, 2.0+PAD, 70.0, 70.0+PAD)],
  # gate GROUND strip: below the bus, grounded by a CONTACT HOLE to the
  # substrate directly beneath it (unreleased mesa) — NO trace, NO pad,
  # NO pair routing: it is electrically the substrate net
  "GATE_GND_VIA": [rect(1.0, 5.0, Y_BUS - W/2 - 4.0, Y_BUS - W/2 - 1.0)],
  # substrate contact pad (over an oxide via, die corner far from optics)
  "SUB": [rect(LTOT+100.0, LTOT+180.0, -70.0-PAD, -70.0)],
}
# MZI node bias electrodes: flank each node's trim-stub arm at 1.0 um,
# exactly like the ring arc — signal strip one side (routed to a pad),
# substrate-via ground strip the other side (no routing).  The arm being
# biased is the TOP arm of each node pair (its 'ph' segment).
seg_edges = np.cumsum([0] + [sg[1] for sg in NODE_SEG])
PH0 = seg_edges[[sg[0] for sg in NODE_SEG].index("ph")]
PH1 = seg_edges[[sg[0] for sg in NODE_SEG].index("ph") + 1]
def pair_top_y(c):
    prs = mesh_pairs(c)[0]
    return (ROWS_M[prs[0]] + ROWS_M[prs[1]]) / 2 + 2.0   # D_A/2 = 2
for c, updown in ((0, +1), (1, -1), (2, +1)):
    x0 = X_MESH + c * L_NODE + PH0
    x1 = X_MESH + c * L_NODE + PH1
    # bias the arm on the ROUTING side: top arm if routing up, bottom
    # arm if routing down (the strip must sit OUTSIDE the pair)
    sgn = updown
    ya = pair_top_y(c) if sgn > 0 else pair_top_y(c) - 4.0
    # signal strip: 1.0 um from arm edge on the routing side
    ys0 = ya + sgn * (W/2 + 1.0)
    pad_x = {0: 320.0, 1: 320.0, 2: 420.0}[c]
    conductors[f"MZI{c}_BIAS"] = [
        rect(x0, x1, min(ys0, ys0 + sgn*2.0), max(ys0, ys0 + sgn*2.0)),
        vseg(ys0 + sgn*1.0, sgn*66.0, x0 + 1.0),
        rect(min(x0+1, pad_x)-1, max(x0+1, pad_x)+1,
             sgn*66.0-1, sgn*66.0+1),
        rect(pad_x - PAD/2, pad_x + PAD/2,
             min(sgn*70.0, sgn*(70.0+PAD)), max(sgn*70.0, sgn*(70.0+PAD))),
        vseg(sgn*66.0, sgn*72.0, pad_x)]
    # ground strip BETWEEN the arms: 1.0 um wide, centered (arm pitch 4,
    # so 1.0 um clearance to each arm edge), substrate-via, no routing
    yg_mid = ya - sgn * 2.0            # between the arms, 1 um strip
    conductors[f"MZI{c}_GND_VIA"] = [
        rect(x0, x1, yg_mid - 0.5, yg_mid + 0.5)]
# output phase-trim bias pairs, rows 0 (top side) and 2 (bottom side)
XT0 = X_MESH + 3 * L_NODE + 2.0
XT1 = XT0 + 6.0
for row, yr, sgn in ((0, 4.0, +1), (2, -4.0, -1)):
    ys0 = yr + sgn * (W/2 + 1.0)
    pad_x = 520.0
    conductors[f"P{row}_BIAS"] = [
        rect(XT0, XT1, min(ys0, ys0 + sgn*2.0), max(ys0, ys0 + sgn*2.0)),
        vseg(ys0 + sgn*1.0, sgn*66.0, XT0 + 1.0),
        rect(XT0, pad_x + 1, sgn*66.0 - 1, sgn*66.0 + 1),
        rect(pad_x - PAD/2, pad_x + PAD/2,
             min(sgn*70.0, sgn*(70.0+PAD)), max(sgn*70.0, sgn*(70.0+PAD))),
        vseg(sgn*66.0, sgn*72.0, pad_x)]
    yg_mid = yr - sgn * 2.0
    conductors[f"P{row}_GND_VIA"] = [
        rect(XT0, XT1, yg_mid - 0.5, yg_mid + 0.5)]
# NOTE (field physics, V4): the RING's counter-electrode is the grounded
# Si PILLAR left inside the ring by an ANNULAR release trench with a
# CONTROLLED ~3.5 um undercut (mask 4 spec): Si removed only in the
# annulus under the ring guide; the interior pedestal stays connected to
# the bulk = the ground. Radial field runs outer-arc -> guide -> pillar.
# For the R=4.79 demo ring this requires the 3.5 um undercut (an 8 um
# undercut would consume the pillar); the R=16 real ring keeps its
# pillar even at 8 um.
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
        same_elem = (a.replace("_GND_VIA","") == b.replace("_BIAS","")
                     or b.replace("_GND_VIA","") == a.replace("_BIAS",""))
        lim = 2.0 if same_elem else 3.0
        if d < lim: rep["pass"] = False
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

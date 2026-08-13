#!/usr/bin/env python
"""END-TO-END 2D MEEP run: ONE 1550 nm input -> tap-bus splitter ->
PHYSICAL one-slot delay meanders -> 3x3 compact Clements mesh -> outputs.
Everything is silica (n=1.444) in air, one connected layout, one cell.

Scale model, honestly labeled: slots are 150 fs (real chip: 501 fs), so one
slot of delay = v_g*T = 30.1 um of extra waveguide, implemented as
raised-cosine meander bumps whose amplitude is SOLVED by arc-length
bisection so channel j is longer than channel 0 by exactly j slots.
(Real chip rule: 100.53 um per slot, cut to +-lambda_g/2.)  N=3 keeps the
cell tractable at res 20; the 5x5 mesh alone is in snapshots.py.  The tap
transfer point is ambiguous by ~ +-L_tap/2 (~15 fs), so the achieved
delays are MEASURED at the mesh inputs and reported, not assumed.

Outputs: fig/snap_e2e_eps.png, fig/snap_e2e_frames.png,
out/e2e_timing.json (measured per-channel delays + output record).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, OUT = os.path.join(HERE, "fig"), os.path.join(HERE, "out")

W = 0.991
NSIL = 1.444024
RES = 20
GAP_C = 0.30
GAP_TAP = 0.20          # stronger tap -> shorter, less transfer ambiguity
LC = 5.8
LB = 8.0
TRIM = 6.0
D_C = W + GAP_C
D_A = 4.0
NG = 1.4947983969526526
FCEN = 1 / 1.55
T_UNIT_FS = 3.33564

SLOT_FS = 150.0
L_SLOT = 299.792458 / NG * SLOT_FS / 1000.0     # um of guide per slot
SLOT = SLOT_FS / T_UNIT_FS
T0 = 15.0 / T_UNIT_FS
TOK = [0.9, -0.6, 0.75]

# ---------------- layout (absolute x in [0, LTOT], y absolute) ----------
Y_BUS = 56.0
ROWS_D = [44.0, 2.0, -40.0]         # delay-block rows
ROWS_M = [4.0, 0.0, -4.0]           # mesh rows (pitch 4)
X_TAP = [(20.0, 26.0), (38.0, 45.55)]           # tap coupler extents
X_BUSBEND = (50.0, 170.0)      # bus s-bend down = ch2 (gentle, R~30 um)
DROP = {0: 30.0, 1: 80.0}      # per-channel drop widths
X_CONV = (250.0, 310.0)        # converge to mesh rows (R >= 16 um)
NODE_SEG = [("in", 1.5), ("b1", LB), ("c1", LC), ("b2", LB), ("ph", TRIM),
            ("b3", LB), ("c2", LC), ("b4", LB), ("out", 1.5)]
L_NODE = sum(s[1] for s in NODE_SEG)
X_MESH = 310.0
LTOT = X_MESH + 3 * L_NODE + 8.0
BUMP_L = 70.0


def scos(t):
    return (1 - np.cos(np.pi * np.clip(t, 0, 1))) / 2


def node_halfsep(xl):
    acc = 0.0
    for name, ln in NODE_SEG:
        if xl <= acc + ln:
            t = (xl - acc) / ln
            if name in ("in", "ph", "out"):
                return D_A / 2
            if name in ("b1", "b3"):
                return D_A / 2 - (D_A - D_C) / 2 * scos(t)
            if name in ("c1", "c2"):
                return D_C / 2
            return D_C / 2 + (D_A - D_C) / 2 * scos(t)
        acc += ln
    return D_A / 2


def mesh_pairs(c):
    return [(0, 1)] if c % 2 == 0 else [(1, 2)]


def y_mesh_row(i, x):
    """Row i centerline inside the mesh region."""
    c = int((x - X_MESH) // L_NODE)
    c = min(max(c, 0), 2)
    xl = x - X_MESH - c * L_NODE
    for (m, n) in mesh_pairs(c):
        if i in (m, n):
            yc = (ROWS_M[m] + ROWS_M[n]) / 2
            hs = node_halfsep(xl)
            return yc + (hs if i == m else -hs)
    return ROWS_M[i]


# bump amplitudes (filled by solver)
AMP = {1: [None], 2: [None]}
BUMP_X = {1: [(135.0, 205.0)], 2: [(175.0, 245.0)]}


def bump(x, x0, x1, A):
    if x0 <= x <= x1:
        return A / 2 * (1 - np.cos(2 * np.pi * (x - x0) / (x1 - x0)))
    return 0.0


def y_channel(k, x):
    """Centerline of channel k for x >= its tap start (nan before)."""
    if k in (0, 1):
        xs, xe = X_TAP[k]
        y_tap = Y_BUS - (GAP_TAP + W)
        if x < xs:
            return np.nan
        if x <= xe:
            return y_tap
        if x <= xe + DROP[k]:
            return y_tap - (y_tap - ROWS_D[k]) * scos((x - xe) / DROP[k])
    else:
        xs, xe = X_BUSBEND
        if x < xs:
            return Y_BUS if x >= 0 else np.nan
        if x <= xe:
            return Y_BUS - (Y_BUS - ROWS_D[2]) * scos((x - xs) / (xe - xs))
    if x <= X_CONV[0]:
        y = ROWS_D[k]
        for (x0, x1), A in zip(BUMP_X.get(k, []), AMP.get(k, [])):
            if A:
                y += bump(x, x0, x1, A)
        return y
    if x <= X_CONV[1]:
        t = (x - X_CONV[0]) / (X_CONV[1] - X_CONV[0])
        return ROWS_D[k] + (ROWS_M[k] - ROWS_D[k]) * scos(t)
    if x <= X_MESH:
        return ROWS_M[k]
    if x <= X_MESH + 3 * L_NODE:
        return y_mesh_row(k, x)
    return ROWS_M[k]


def path_len(k, x_from, x_to, n=12000):
    xs = np.linspace(x_from, x_to, n)
    ys = np.array([y_channel(k, x) for x in xs])
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def solve_bumps():
    """Solve bump amplitudes so path(ch k) - path(ch 0) = k slots.
    Paths measured source -> mesh entry; tap transfer point = tap center."""
    def total(k):
        if k in (0, 1):
            xc = 0.5 * (X_TAP[k][0] + X_TAP[k][1])
        else:
            xc = X_BUSBEND[0]
        bus = xc - 3.0                       # source at x=3 on the bus
        start = X_TAP[k][0] if k in (0, 1) else X_BUSBEND[0]
        # channel arc length from its start, minus half-tap correction so
        # the reference point is the tap CENTER on the channel too
        ch = path_len(k, start, X_MESH)
        if k in (0, 1):
            ch -= (xc - X_TAP[k][0])
        return bus + ch

    base = total(0)
    for k in (1, 2):
        target = base + k * L_SLOT - (total(k) if True else 0)
        # bisection on a shared amplitude for this channel's bump(s)
        lo, hi = 0.0, 40.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            AMP[k] = [mid] * len(AMP[k])
            if total(k) - base < k * L_SLOT:
                lo = mid
            else:
                hi = mid
        AMP[k] = [0.5 * (lo + hi)] * len(AMP[k])
    return base, [total(k) - base for k in (0, 1, 2)]


base, deltas = solve_bumps()
print("solved bumps:", {k: [round(a, 2) for a in v] for k, v in AMP.items()})
print("path deltas (um):", [round(d, 2) for d in deltas],
      "target:", [0, round(L_SLOT, 2), round(2 * L_SLOT, 2)])
# curvature sanity: R_min = 1 / max|y''| of the bump
for k, v in AMP.items():
    A = v[0]
    rmin = 1.0 / (A / 2 * (2 * np.pi / BUMP_L) ** 2)
    print(f"  ch{k}: bump A={A:.1f} um, min bend radius ~{rmin:.1f} um")


def material(p):
    x = p.x + LTOT / 2
    y = p.y
    if x < 0 or x > LTOT:
        return mp.Medium(index=1.0)
    # bus (up to its bend start; the bend itself is channel 2)
    if x <= X_BUSBEND[0] and abs(y - Y_BUS) <= W / 2:
        return mp.Medium(index=NSIL)
    for k in range(3):
        yk = y_channel(k, x)
        if not np.isnan(yk) and abs(y - yk) <= W / 2:
            return mp.Medium(index=NSIL)
    # other mesh guides inside the mesh region (rows not owned by k loop
    # are already covered: y_channel handles all three rows)
    return mp.Medium(index=1.0)


def train(t):
    T_START = 6 * T0
    e = 0.0j
    for k, a in enumerate(TOK):
        e += a * np.exp(-((t - T_START - k * SLOT) ** 2) / (2 * T0 ** 2))
    return e * np.exp(-2j * np.pi * FCEN * t)


T_END_SRC = 6 * T0 + 3 * SLOT + 6 * T0
sx = LTOT + 6
sy = 116.0
Y_CENTER = 8.0      # layout y in [-40, 57] -> shift down to center in cell

# Rasterize the geometry into a MaterialGrid (numpy-painted, seconds)
# instead of a Python material function (tens of millions of scalar calls
# = an hour of init). Visual/timing run, labeled; quantitative numbers
# stay with the res-25 subpixel-averaged node cells.
NGX, NGY = int(sx * RES), int(sy * RES)
gx = (np.arange(NGX) + 0.5) / RES            # 0..sx
gy = (np.arange(NGY) + 0.5) / RES - sy / 2 + Y_CENTER   # layout frame
X0 = (sx - LTOT) / 2                          # layout x offset in cell
wgt = np.zeros((NGX, NGY), dtype=np.float64)
xl = gx - X0                                  # layout x per column
# bus (up to bend start)
bus_cols = (xl >= 0) & (xl <= X_BUSBEND[0])
wgt[np.ix_(bus_cols, np.abs(gy - Y_BUS) <= W / 2)] = 1.0
# channels
for k in range(3):
    yk = np.array([y_channel(k, x) if 0 <= x <= LTOT else np.nan
                   for x in xl])
    ok = ~np.isnan(yk)
    mask = np.zeros((NGX, NGY), dtype=bool)
    mask[ok] = np.abs(gy[None, :] - yk[ok, None]) <= W / 2
    wgt[mask] = 1.0
mgrid = mp.MaterialGrid(mp.Vector3(NGX, NGY), mp.Medium(index=1.0),
                        mp.Medium(index=NSIL), weights=wgt,
                        do_averaging=False)
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=RES,
                    boundary_layers=[mp.PML(1.5)],
                    geometry=[mp.Block(size=mp.Vector3(sx, sy),
                                       center=mp.Vector3(),
                                       material=mgrid)],
                    default_material=mp.Medium(index=1.0),
                    eps_averaging=False)
sim.sources = [mp.EigenModeSource(
    mp.CustomSource(train, start_time=0, end_time=T_END_SRC,
                    center_frequency=FCEN),
    center=mp.Vector3(-LTOT / 2 + 3.0, Y_BUS - Y_CENTER),
    size=mp.Vector3(0, 2.8),
    eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]

x_in = X_MESH - 2.0 - LTOT / 2
x_out = X_MESH + 3 * L_NODE + 2.0 - LTOT / 2
rec = {"t": [], "in": [[], [], []], "out": [[], [], []]}
ys_probe = np.linspace(-1.0, 1.0, 9)


def grab_ts(sim):
    rec["t"].append(sim.meep_time())
    for i in range(3):
        rec["in"][i].append(sum(abs(sim.get_field_point(
            mp.Ez, mp.Vector3(x_in, ROWS_M[i] - Y_CENTER + y))) ** 2
            for y in ys_probe))
        rec["out"][i].append(sum(abs(sim.get_field_point(
            mp.Ez, mp.Vector3(x_out, ROWS_M[i] - Y_CENTER + y))) ** 2
            for y in ys_probe))


frames, ftimes = [], []


def grab_frame(sim):
    a = np.abs(sim.get_array(component=mp.Ez, center=mp.Vector3(),
                             size=mp.Vector3(sx, sy)))
    frames.append(a.astype(np.float32)[::2, ::2])   # keep memory sane
    ftimes.append(sim.meep_time())


sim.init_sim()
eps = np.real(sim.get_epsilon())
fig, ax = plt.subplots(figsize=(14, 3.6), dpi=150)
ax.imshow(eps.T, cmap="Greys", origin="lower", aspect="equal",
          interpolation="nearest")
ax.set_title("END-TO-END geometry (eps): one 1550 nm bus -> 2 taps + bus "
             "bend -> one-slot meander delays (solved to "
             f"{L_SLOT:.1f} um/slot) -> 3x3 compact mesh", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG, "snap_e2e_eps.png"), bbox_inches="tight")
plt.close(fig)

T_RUN = T_END_SRC + (LTOT * NSIL) + 80
sim.run(mp.at_every(0.5, grab_ts), mp.at_every(55, grab_frame),
        until=T_RUN)

# measured delays: first-token peak time per input channel
t_arr = np.array(rec["t"])
meas = []
for i in range(3):
    tr = np.array(rec["in"][i])
    # first peak = token 0 (0.9): find peak in the first window where the
    # trace first exceeds 30% of its max
    j0 = np.argmax(tr > 0.3 * tr.max())
    jwin = (t_arr > t_arr[j0] - SLOT / 2) & (t_arr < t_arr[j0] + SLOT / 2)
    meas.append(float(t_arr[jwin][np.argmax(tr[jwin])]))
d_slots = [(m - meas[0]) / SLOT for m in meas]
print("measured input delays (slots):", [round(d, 3) for d in d_slots])
json.dump({"slot_fs": SLOT_FS, "L_slot_um": L_SLOT,
           "bump_amps": {str(k): v for k, v in AMP.items()},
           "path_deltas_um": deltas,
           "measured_delays_slots": d_slots,
           "t": rec["t"],
           "inputs": rec["in"], "outputs": rec["out"]},
          open(os.path.join(OUT, "e2e_timing.json"), "w"))

picks = np.linspace(1, len(frames) - 1, 7).astype(int)
vmax = max(frames[k].max() for k in picks)
fig, axs = plt.subplots(len(picks), 1, figsize=(14, 2.1 * len(picks)),
                        dpi=150)
for ax, k in zip(np.atleast_1d(axs), picks):
    ax.imshow((frames[k].T / vmax) ** 0.4, cmap="magma", origin="lower",
              aspect="equal", interpolation="bilinear", vmin=0, vmax=1)
    ax.set_title(f"t = {ftimes[k]*T_UNIT_FS:.0f} fs", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("END-TO-END |Ez|: single 1550 input -> split -> one-slot "
             "meander delays -> 3x3 mesh (delays measured: "
             + ", ".join(f"{d:.2f}" for d in d_slots) + " slots)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "snap_e2e_frames.png"), bbox_inches="tight")
plt.close(fig)
np.savez_compressed(os.path.join(OUT, "snap_e2e_frames.npz"),
                    frames=np.array(frames, dtype=np.float32),
                    times=np.array(ftimes))
print("e2e done")

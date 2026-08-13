#!/usr/bin/env python
"""MEEP geometry renders + field-amplitude snapshot sequences.

Scenes (usage: snapshots.py node|score|mesh):
  node   compact MZI close-up, CW drive, steady state:
         (a) dL=0 (cross state)  (b) dL=0.3003 um (geometric theta=pi/2)
  score  the score head: two interleaved pulse trains (one delayed a slot,
         +90 deg) meeting in the calibrated 50/50 coupler
  mesh   FULL compact 5x5 Clements mesh in one FDTD cell (res 20, visual
         run -- the quantitative numbers stay with the res-25 node runs),
         3-pulse train into the middle port, snapshots as it cascades

All scenes run with complex fields so |Ez| is the clean envelope.
Saves fig/snap_*.png + out/snap_*.npz.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
OUT = os.path.join(HERE, "out")

W = 0.991
NSIL = 1.444024
GAP_C = 0.30
LC = 5.8
LB = 8.0
TRIM = 6.0
D_C = W + GAP_C
D_A = 4.0
LIN = 1.5
NEFF = 1.2905
FCEN = 1 / 1.55

NODE_SEG = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB), ("ph", TRIM),
            ("b3", LB), ("c2", LC), ("b4", LB), ("out", LIN)]
L_NODE = sum(s[1] for s in NODE_SEG)


def node_halfsep(x_local):
    """Half-separation of the two guides of a node at local x (um)."""
    acc = 0.0
    for name, ln in NODE_SEG:
        if x_local <= acc + ln:
            t = (x_local - acc) / ln
            c = (1 - np.cos(np.pi * t)) / 2
            if name in ("in", "ph", "out"):
                return D_A / 2
            if name in ("b1", "b3"):
                return D_A / 2 - (D_A - D_C) / 2 * c
            if name in ("c1", "c2"):
                return D_C / 2
            return D_C / 2 + (D_A - D_C) / 2 * c
        acc += ln
    return D_A / 2


def render_geometry(sim, path, title, cellname):
    eps = np.real(sim.get_epsilon())
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=150)
    ax.imshow(eps.T, cmap="Greys", origin="lower", aspect="equal",
              interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, path), bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(os.path.join(OUT, f"snap_eps_{cellname}.npz"),
                        eps=eps)


def render_frames(frames, times, path, title, per_row=2, gamma=0.5):
    n = len(frames)
    rows = (n + per_row - 1) // per_row
    fig, axs = plt.subplots(rows, per_row, figsize=(12, 1.9 * rows),
                            dpi=150)
    axs = np.atleast_1d(axs).ravel()
    vmax = max(f.max() for f in frames)
    for i, (f, t) in enumerate(zip(frames, times)):
        ax = axs[i]
        ax.imshow((f.T / vmax) ** gamma, cmap="magma", origin="lower",
                  aspect="equal", interpolation="bilinear", vmin=0, vmax=1)
        ax.set_title(f"t = {t*3.33564:.0f} fs", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, path), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ node -------
def scene_node():
    for dL, tag in ((0.0, "cross"), (0.3003, "half")):
        dn_geo = NEFF * dL / TRIM
        seg = NODE_SEG
        edges = np.cumsum([0] + [s[1] for s in seg])
        segx = {nm: (edges[i], edges[i + 1]) for i, (nm, _) in
                enumerate(seg)}
        LTOT = edges[-1]
        sx, sy = LTOT + 8, D_A + W + 8

        def material(p):
            x = min(max(p.x + LTOT / 2, 0.0), LTOT)
            hs = node_halfsep(x)
            if abs(p.y - hs) <= W / 2:
                dn = dn_geo if segx["ph"][0] <= x <= segx["ph"][1] else 0.0
                return mp.Medium(index=NSIL + dn)
            if abs(p.y + hs) <= W / 2:
                return mp.Medium(index=NSIL)
            return mp.Medium(index=1.0)

        sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=25,
                            boundary_layers=[mp.PML(1.5)],
                            material_function=material,
                            extra_materials=[mp.Medium(index=NSIL + dn_geo
                                                       + 1e-4)],
                            default_material=mp.Medium(index=1.0),
                            force_complex_fields=True)
        sim.sources = [mp.EigenModeSource(
            mp.ContinuousSource(FCEN, width=20),
            center=mp.Vector3(-sx / 2 + 2.2, D_A / 2),
            size=mp.Vector3(0, 2.8), eig_band=1, eig_parity=mp.ODD_Z,
            eig_match_freq=True)]
        sim.init_sim()
        if dL == 0.0:
            render_geometry(sim, "snap_node_eps.png",
                            f"COMPACT NODE geometry (eps): {LTOT:.1f} um "
                            f"x {D_A:g} um pitch — s-bends 8 um, couplers "
                            f"LC=5.8, 6 um trim stub", "node")
        sim.run(until=350)
        ez = np.abs(sim.get_array(component=mp.Ez,
                                  center=mp.Vector3(),
                                  size=mp.Vector3(sx, sy)))
        fig, ax = plt.subplots(figsize=(12, 2.6), dpi=150)
        ax.imshow((ez.T / ez.max()) ** 0.5, cmap="magma", origin="lower",
                  aspect="equal", interpolation="bilinear")
        lab = ("theta=0 (as-drawn): CROSS state — all power exits the "
               "other guide" if dL == 0.0 else
               "geometric theta=pi/2 (dL=0.30 um on the trim stub): "
               "50/50 output")
        ax.set_title(f"COMPACT NODE |Ez|, CW steady state — {lab}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"snap_node_{tag}.png"),
                    bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(os.path.join(OUT, f"snap_node_{tag}.npz"),
                            ez=ez)
        sim.reset_meep()
    print("node scene done")


# ----------------------------------------------------------- score -------
def scene_score():
    seg = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB), ("out", 6.0)]
    edges = np.cumsum([0] + [s[1] for s in seg])
    segx = {nm: (edges[i], edges[i + 1]) for i, (nm, _) in enumerate(seg)}
    LTOT = edges[-1]
    sx, sy = LTOT + 8, D_A + W + 8

    def hs_of(x):
        for nm, (a, b) in segx.items():
            if a <= x <= b:
                t = (x - a) / (b - a)
                c = (1 - np.cos(np.pi * t)) / 2
                if nm in ("in", "out"):
                    return D_A / 2
                if nm == "b1":
                    return D_A / 2 - (D_A - D_C) / 2 * c
                if nm == "c1":
                    return D_C / 2
                return D_C / 2 + (D_A - D_C) / 2 * c
        return D_A / 2

    def material(p):
        x = min(max(p.x + LTOT / 2, 0.0), LTOT)
        hs = hs_of(x)
        if abs(p.y - hs) <= W / 2 or abs(p.y + hs) <= W / 2:
            return mp.Medium(index=NSIL)
        return mp.Medium(index=1.0)

    T_UNIT_FS = 3.33564
    SLOT = 150.0 / T_UNIT_FS
    T0 = 15.0 / T_UNIT_FS
    TOK = np.array([0.9, -0.6, 0.75])
    T_START = 6 * T0

    def train(delay, phase):
        def f(t):
            e = 0.0j
            for k, a in enumerate(TOK):
                tk = T_START + (k + delay) * SLOT
                e += a * np.exp(-((t - tk) ** 2) / (2 * T0 ** 2))
            return e * np.exp(-2j * np.pi * FCEN * t) * np.exp(1j * phase)
        return f

    TEND = T_START + 4 * SLOT + 6 * T0
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=25,
                        boundary_layers=[mp.PML(1.5)],
                        material_function=material,
                        default_material=mp.Medium(index=1.0),
                        force_complex_fields=True)
    sim.sources = [
        mp.EigenModeSource(mp.CustomSource(train(1, 0.0), start_time=0,
                                           end_time=TEND,
                                           center_frequency=FCEN),
                           center=mp.Vector3(-sx / 2 + 2.2, D_A / 2),
                           size=mp.Vector3(0, 2.8), eig_band=1,
                           eig_parity=mp.ODD_Z, eig_match_freq=True),
        mp.EigenModeSource(mp.CustomSource(train(0, np.pi / 2),
                                           start_time=0, end_time=TEND,
                                           center_frequency=FCEN),
                           center=mp.Vector3(-sx / 2 + 2.2, -D_A / 2),
                           size=mp.Vector3(0, 2.8), eig_band=1,
                           eig_parity=mp.ODD_Z, eig_match_freq=True)]
    sim.init_sim()
    render_geometry(sim, "snap_score_eps.png",
                    "SCORE HEAD geometry (eps): calibrated 50/50 coupler "
                    "— arm A = train delayed 1 slot, arm B = train + 90 "
                    "deg", "score")
    frames, times = [], []

    def grab(sim):
        frames.append(np.abs(sim.get_array(component=mp.Ez,
                                           center=mp.Vector3(),
                                           size=mp.Vector3(sx, sy))))
        times.append(sim.meep_time())
    sim.run(mp.at_every(18, grab), until=TEND + 60)
    keep = [2, 4, 6, 8, 10, 12]
    keep = [k for k in keep if k < len(frames)]
    render_frames([frames[k] for k in keep], [times[k] for k in keep],
                  "snap_score_frames.png",
                  "SCORE HEAD |Ez|: token pulses of the two trains meet "
                  "pairwise in the coupler — the balanced output "
                  "difference per slot IS s_t*s_{t-1}")
    np.savez_compressed(os.path.join(OUT, "snap_score_frames.npz"),
                        frames=np.array(frames[::2], dtype=np.float32),
                        times=np.array(times[::2]))
    print("score scene done")


# ------------------------------------------------------------ mesh -------
def scene_mesh():
    N = 5
    PITCH = D_A
    ys = [(2 - i) * PITCH for i in range(N)]     # +8 ... -8
    LEAD = 6.0
    cols = [(LEAD + c * L_NODE, LEAD + (c + 1) * L_NODE)
            for c in range(N)]
    LTOT = LEAD * 2 + N * L_NODE
    sx = LTOT + 8
    sy = (N - 1) * PITCH + W + 10

    def pairs(c):
        return [(0, 1), (2, 3)] if c % 2 == 0 else [(1, 2), (3, 4)]

    def material(p):
        x = min(max(p.x + LTOT / 2, 0.0), LTOT)
        # which column?
        for c, (a, b) in enumerate(cols):
            if a <= x <= b:
                xl = x - a
                in_pair = {}
                for (m, n) in pairs(c):
                    in_pair[m] = ("top", (m, n))
                    in_pair[n] = ("bot", (m, n))
                for i, yc in enumerate(ys):
                    if i in in_pair:
                        role, (m, n) = in_pair[i]
                        yc_pair = (ys[m] + ys[n]) / 2
                        hs = node_halfsep(xl)
                        yg = yc_pair + (hs if role == "top" else -hs)
                    else:
                        yg = yc
                    if abs(p.y - yg) <= W / 2:
                        return mp.Medium(index=NSIL)
                return mp.Medium(index=1.0)
        for yc in ys:
            if abs(p.y - yc) <= W / 2:
                return mp.Medium(index=NSIL)
        return mp.Medium(index=1.0)

    T_UNIT_FS = 3.33564
    SLOT = 150.0 / T_UNIT_FS
    T0 = 15.0 / T_UNIT_FS
    TOK = [0.9, -0.7, 0.8]
    T_START = 6 * T0

    def train(t):
        e = 0.0j
        for k, a in enumerate(TOK):
            e += a * np.exp(-((t - T_START - k * SLOT) ** 2)
                            / (2 * T0 ** 2))
        return e * np.exp(-2j * np.pi * FCEN * t)

    TEND = T_START + 3 * SLOT + 6 * T0
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=20,
                        boundary_layers=[mp.PML(1.5)],
                        material_function=material,
                        default_material=mp.Medium(index=1.0),
                        force_complex_fields=True)
    sim.sources = [mp.EigenModeSource(
        mp.CustomSource(train, start_time=0, end_time=TEND,
                        center_frequency=FCEN),
        center=mp.Vector3(-sx / 2 + 2.0, ys[2]), size=mp.Vector3(0, 2.8),
        eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]
    sim.init_sim()
    render_geometry(sim, "snap_mesh_eps.png",
                    f"FULL COMPACT 5x5 CLEMENTS MESH geometry (eps): "
                    f"{LTOT:.0f} um x {sy - 10 + W:.0f} um active — 10 "
                    f"nodes, 5 columns x {L_NODE:.1f} um", "mesh")
    frames, times = [], []

    def grab(sim):
        frames.append(np.abs(sim.get_array(component=mp.Ez,
                                           center=mp.Vector3(),
                                           size=mp.Vector3(sx, sy))))
        times.append(sim.meep_time())
    sim.run(mp.at_every(60, grab),
            until=TEND + (LTOT * NSIL) / 1.0 + 60)
    picks = np.linspace(2, len(frames) - 1, 6).astype(int)
    render_frames([frames[k] for k in picks], [times[k] for k in picks],
                  "snap_mesh_frames.png",
                  "FULL 5x5 MESH |Ez|: 3-token pulse train (into port 2) "
                  "cascading through all 5 columns — as-drawn nodes are "
                  "cross-state so power ping-pongs across the array",
                  per_row=1, gamma=0.4)
    np.savez_compressed(os.path.join(OUT, "snap_mesh_frames.npz"),
                        frames=np.array(frames[::2], dtype=np.float32),
                        times=np.array(times[::2]))
    print("mesh scene done")


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "all"
    if scene in ("node", "all"):
        scene_node()
    if scene in ("score", "all"):
        scene_score()
    if scene in ("mesh", "all"):
        scene_mesh()

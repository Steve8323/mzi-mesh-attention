#!/usr/bin/env python
"""FDTD demonstration of the photonic attention score: two pulse TRAINS
(one the one-slot-delayed copy of the other, quarter-wave offset) interfere
in the calibrated 50/50 coupler; the balanced per-slot energy difference is
the token product a_k * a_{k-1}.

Scale model, honestly labeled: slots here are 150 fs (45 um of bus) so the
train fits an FDTD cell; the physics (pulse interference in one coupler) is
slot-spacing-independent as long as pulses don't overlap, which they don't
at either spacing.  Three runs implement the leak-corrected balanced
receiver of otdm.score_band: (both arms) - (arm A alone) - (arm B alone).

usage: fdtd_score.py [resolution]     writes out/fdtd_score.json
"""
import json
import os
import sys

import numpy as np
import meep as mp

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 25

W = 0.991
NSIL = 1.444024
GAP_C = 0.30
LC = 5.35
LB = 14.0
D_C = W + GAP_C
D_A = 6.0
PADX = 4.0
LIN = 3.0
LOUT = 8.0

seg = [("in", LIN), ("b1", LB), ("c1", LC), ("b2", LB), ("out", LOUT)]
edges = np.cumsum([0] + [s[1] for s in seg])
LTOT = edges[-1]
segx = {name: (edges[i], edges[i + 1]) for i, (name, _) in enumerate(seg)}


def y_top(x):
    for name, (a, b) in segx.items():
        if a <= x <= b:
            t = (x - a) / (b - a)
            c = (1 - np.cos(np.pi * t)) / 2
            if name in ("in", "out"):
                return D_A / 2 if name == "in" else D_A / 2
            if name == "b1":
                return D_A / 2 - (D_A - D_C) / 2 * c
            if name == "c1":
                return D_C / 2
            return D_C / 2 + (D_A - D_C) / 2 * c   # b2
    return D_A / 2


def material(p):
    x = min(max(p.x + LTOT / 2, 0.0), LTOT)
    yc = y_top(x)
    if abs(p.y - yc) <= W / 2 or abs(p.y + yc) <= W / 2:
        return mp.Medium(index=NSIL)
    return mp.Medium(index=1.0)


fcen = 1 / 1.55
FS = 1.0 / 3.33564                 # 1 fs in meep time units... (inverse)
T_FS = 1 / 299.792458 * 1000       # meep time units per fs = 0.29979... no:
T_UNIT_FS = 3.33564                # 1 meep time unit = 3.33564 fs
SLOT_FS = 150.0
SLOT = SLOT_FS / T_UNIT_FS         # 44.97 t.u.
T0 = 15.0 / T_UNIT_FS              # 15 fs Gaussian T0
TOKENS = np.array([0.9, -0.6, 0.75, 0.45])
NTOK = len(TOKENS)
T_START = 6 * T0


def train_func(tokens, delay_slots, phase):
    def f(t):
        env = 0.0j
        for k, a in enumerate(tokens):
            tk = T_START + (k + delay_slots) * SLOT
            env += a * np.exp(-((t - tk) ** 2) / (2 * T0 ** 2))
        return env * np.exp(-2j * np.pi * fcen * t) * np.exp(1j * phase)
    return f


T_END_SRC = T_START + (NTOK + 1) * SLOT + 6 * T0
sx = LTOT + 2 * PADX
sy = D_A + W + 2 * 4.0
src_x = -sx / 2 + PADX * 0.55
mon_x = sx / 2 - PADX * 0.55
T_RUN = T_END_SRC + (sx * NSIL) + 40


def run_case(armA, armB):
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=RES,
                        boundary_layers=[mp.PML(1.5)],
                        material_function=material,
                        default_material=mp.Medium(index=1.0),
                        force_complex_fields=True)
    sources = []
    if armA:
        sources.append(mp.EigenModeSource(
            mp.CustomSource(train_func(TOKENS, 1, 0.0), start_time=0,
                            end_time=T_END_SRC, center_frequency=fcen),
            center=mp.Vector3(src_x, D_A / 2), size=mp.Vector3(0, 3.6),
            eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True))
    if armB:
        sources.append(mp.EigenModeSource(
            mp.CustomSource(train_func(TOKENS, 0, np.pi / 2), start_time=0,
                            end_time=T_END_SRC, center_frequency=fcen),
            center=mp.Vector3(src_x, -D_A / 2), size=mp.Vector3(0, 3.6),
            eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True))
    sim.sources = sources
    ys = np.linspace(-1.2, 1.2, 13)
    rec = {"t": [], "top": [], "bot": []}

    def grab(sim):
        rec["t"].append(sim.meep_time())
        rec["top"].append(sum(abs(sim.get_field_point(
            mp.Ez, mp.Vector3(mon_x, D_A / 2 + y))) ** 2 for y in ys))
        rec["bot"].append(sum(abs(sim.get_field_point(
            mp.Ez, mp.Vector3(mon_x, -D_A / 2 + y))) ** 2 for y in ys))
    sim.run(mp.at_every(0.5, grab), until=T_RUN)
    return (np.array(rec["t"]), np.array(rec["top"]), np.array(rec["bot"]))


print("run 1/3: both arms")
t, topAB, botAB = run_case(True, True)
print("run 2/3: arm A alone")
_, topA, botA = run_case(True, False)
print("run 3/3: arm B alone")
_, topB, botB = run_case(False, True)

# slot windows: first pulse of arm B (slot index 0) arrives at the monitors
# at T_START + travel; detect from the single-arm-B trace (token 0 = 0.9)
power_ref = topB + botB
i_pk = int(np.argmax(power_ref))
t_pk = t[i_pk]                      # arrival of arm-B token argmax
k_max = int(np.argmax(np.abs(TOKENS)))
t0_arrival = t_pk - k_max * SLOT    # arrival time of slot 0


def slot_energy(tr, k):
    w = (t > t0_arrival + k * SLOT - SLOT / 2) & \
        (t <= t0_arrival + k * SLOT + SLOT / 2)
    return float(np.sum(tr[w]))


n_out = NTOK + 1
diff = []
for k in range(n_out):
    dAB = slot_energy(topAB, k) - slot_energy(botAB, k)
    dA = slot_energy(topA, k) - slot_energy(botA, k)
    dB = slot_energy(topB, k) - slot_energy(botB, k)
    diff.append(dAB - dA - dB)
diff = np.array(diff)
expected = np.zeros(n_out)
for k in range(n_out):
    if 1 <= k < NTOK:
        expected[k] = TOKENS[k] * TOKENS[k - 1]
g = np.dot(diff, expected) / np.dot(expected, expected)
resid = diff / g - expected
rms = float(np.sqrt(np.mean(resid[1:NTOK] ** 2))
            / np.sqrt(np.mean(expected[1:NTOK] ** 2)))
json.dump(dict(tokens=TOKENS.tolist(), slot_fs=SLOT_FS,
               diff=diff.tolist(), expected=expected.tolist(),
               gain=float(g), rms_rel=rms,
               t=t.tolist(), topAB=topAB.tolist(), botAB=botAB.tolist()),
          open("out/fdtd_score.json", "w"))
print("balanced diff / g :", np.round(diff / g, 4))
print("expected products :", np.round(expected, 4))
print(f"relative rms error: {rms:.4f}")

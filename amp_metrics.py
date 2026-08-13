#!/usr/bin/env python
"""How perfect is it, per amplitude — the ADC-style characterization.

A linear photonic system has two error classes: MULTIPLICATIVE (coupler/
phase/loss errors — relative error independent of amplitude) and ADDITIVE
(crosstalk from other slots, ISI tails, leak residuals — a fixed floor
that eats small amplitudes).  These measurements separate them:

  A  amplitude-transfer linearity: one token, amplitude swept 0.05..0.97,
     recovered/ideal ratio (deviation = compression/nonlinearity; the
     model is linear so deviation exposes amplitude-dependent MODEL error)
  B  bits vs token-distribution scale A (tokens ~ U(-A, A)), N = 5/8/16:
     multiplicative errors -> flat; a droop at small A exposes the floor
  C  per-|token| binned relative error at full scale (N=5): the
     'effective bits per amplitude level' curve — the honest answer to
     'how perfect', level by level
  D  score head: product error binned by |s_t * s_{t-1}| (products span
     4 decades when amplitudes span 2)
  E  hardware-level: per-amplitude residuals of the REAL FDTD score run
     (tokens 0.9/-0.6/0.75/0.45 from fdtd_score.py)

All model rows use the full loop-4 error spec + kappa-aware compile
(char 3 mrad).  Writes out/amp_metrics.json + fig/fig10_amplitude.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opt_harness import run_config, score_full

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")
SPEC = dict(phase_set_rms=5e-3, kappa_sigma=12e-3, delay_phase_rms=5e-3,
            comb_linewidth_hz=1e6, kappa_aware=True, kappa_char_rms=3e-3)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "purple": "#CC79A7", "grey": "#8a8a97"}

res = {}

# ---- A: transfer linearity (folded into C) ------------------------------
res["A_note"] = ("Maxwell + this model are strictly linear in field "
                 "amplitude, so a lone-token gain sweep is exactly flat "
                 "by construction; the meaningful amplitude dependence is "
                 "the per-level binned measurement in C (which includes "
                 "the additive crosstalk floor) and the |E|^2 score path "
                 "in D/E.")

# ---- B: bits vs amplitude scale ----------------------------------------
print("== B: bits vs token amplitude scale, N=5/8/16")
rows = []
for N in (5, 8, 16):
    for A in (0.1, 0.2, 0.4, 0.7, 0.97):
        bits = [run_config(N=N, T0=120e-15, seed=s, want_score=False,
                           amp_scale=A, **SPEC)["bits"]
                for s in range(4)]
        rows.append({"N": N, "A": A, "bits_mean": float(np.mean(bits)),
                     "bits_std": float(np.std(bits))})
        print(f"  N={N:3d} A={A:4.2f}  bits {np.mean(bits):.2f}"
              f"±{np.std(bits):.2f}")
res["B_scale"] = rows

# ---- C: per-|token| binned effective bits (N=5, full scale) ------------
print("== C: per-amplitude-level effective bits")
allS, allref, alltok = [], [], []
for s in range(6):
    r = run_config(N=5, T0=120e-15, seed=s, want_score=False,
                   return_arrays=True, n_tokens=256, **SPEC)
    n_tok = len(r["tokens"])
    # port-0 output at slot k estimates sum U_0j s_{k-j}; to bin BY TOKEN
    # use the deconvolved per-token residual: compare recovered vs ideal
    # contribution is entangled -- instead bin the SLOT error by the rms
    # of the window tokens it depends on
    S, ref = r["S"], r["ref"]
    allS.append(S)
    allref.append(ref)
    alltok.append(r["tokens"])
bins = np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.97])
binC = []
for lo, hi in zip(bins[:-1], bins[1:]):
    es, sig = [], []
    for S, ref, tok in zip(allS, allref, alltok):
        n_tok = len(tok)
        for k in range(5 - 1, n_tok):          # full windows only
            w = np.abs(tok[k - 4:k + 1])
            if lo <= w.mean() < hi:
                es.append(np.mean(np.abs(S[:, k] - ref[:, k]) ** 2))
                sig.append(np.mean(np.abs(ref[:, k]) ** 2))
    if es:
        rel = np.sqrt(np.mean(es) / np.mean(sig))
        binC.append({"amp_lo": float(lo), "amp_hi": float(hi),
                     "rel_rms": float(rel),
                     "bits": float(-np.log2(rel)), "n": len(es)})
        print(f"  |s|∈[{lo:.2f},{hi:.2f})  rel rms {rel:.4f} "
              f"({-np.log2(rel):.1f} bits, n={len(es)})")
res["C_levels"] = binC

# ---- D: score product error vs product magnitude -----------------------
print("== D: score error vs |product|")
rng = np.random.default_rng(11)
prods, errs = [], []
for s in range(6):
    tk = rng.uniform(-0.97, 0.97, 48)
    # reuse score_full but get arrays: quick local recompute
    from opt_harness import score_full as _sf
    # measure rms per run and record mean |product| per run scale
    for A in (0.3, 0.6, 0.97):
        tks = tk * (A / 0.97)
        e = _sf(tks, 9.86e-15, lag=1, dkap=12e-3 * (rng.standard_normal()),
                comb_linewidth_hz=1e6, seed_err=s)
        prods.append(float(np.mean(np.abs(tks[1:] * tks[:-1]))))
        errs.append(float(e))
res["D_score"] = sorted(
    [{"mean_abs_product": p, "rel_rms": e} for p, e in zip(prods, errs)],
    key=lambda r: r["mean_abs_product"])
for r0 in res["D_score"][::4]:
    print(f"  <|prod|>={r0['mean_abs_product']:.3f} rel rms "
          f"{r0['rel_rms']:.2e}")

# ---- E: FDTD per-amplitude residuals (hardware) ------------------------
d = json.load(open(os.path.join(OUT, "fdtd_score.json")))
diffg = np.array(d["diff"]) / d["gain"]
expd = np.array(d["expected"])
okk = expd != 0
res["E_fdtd"] = [{"product": float(e), "measured": float(m),
                  "residual": float(m - e)}
                 for e, m in zip(expd[okk], diffg[okk])]
print("== E: FDTD residuals per product:",
      [f"{r['product']:+.3f}->{r['residual']:+.5f}" for r in res["E_fdtd"]])

json.dump(res, open(os.path.join(OUT, "amp_metrics.json"), "w"), indent=1)

# ---- figure -------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=150)
ax = axs[0]
for N, col in ((5, C["blue"]), (8, C["orange"]), (16, C["verm"])):
    rr = [r for r in res["B_scale"] if r["N"] == N]
    ax.errorbar([r["A"] for r in rr], [r["bits_mean"] for r in rr],
                yerr=[r["bits_std"] for r in rr], fmt="o-", color=col,
                lw=2, capsize=3, label=f"N={N}")
ax.set_xlabel("token amplitude scale A  (tokens ~ U(−A, A))")
ax.set_ylabel("bits (full error spec)")
ax.set_title("relative accuracy is amplitude-FLAT:\nerrors are "
             "multiplicative, no additive droop", fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax = axs[1]
bc = res["C_levels"]
x = [(r["amp_lo"] + r["amp_hi"]) / 2 for r in bc]
ax.plot(x, [r["bits"] for r in bc], "o-", color=C["green"], lw=2)
ax.set_xlabel("mean |token| in the window")
ax.set_ylabel("effective bits of that slot")
ax.set_title("per-level 'how perfect': small-amplitude windows\n"
             "lose bits to the additive crosstalk floor", fontsize=9)
ax = axs[2]
pr = res["D_score"]
ax.loglog([r["mean_abs_product"] for r in pr],
          [r["rel_rms"] for r in pr], "o", color=C["purple"], alpha=0.8,
          label="model, full errors")
ef = res["E_fdtd"]
ax.loglog([abs(r["product"]) for r in ef],
          [abs(r["residual"]) / abs(r["product"]) for r in ef], "s",
          color=C["verm"], ms=8, label="FDTD (fdtd_score.py)")
ax.set_xlabel("|token product|")
ax.set_ylabel("relative error of the score")
ax.set_title("score head vs product magnitude —\nFDTD points sit on the "
             "model floor", fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig10_amplitude.png"), bbox_inches="tight")
print("wrote out/amp_metrics.json + fig/fig10_amplitude.png")

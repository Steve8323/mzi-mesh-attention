#!/usr/bin/env python
"""All report figures. Reads out/*.json, writes fig/*.png (150 dpi).
Skips FDTD panels gracefully if those runs haven't finished."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")

# Okabe-Ito (colorblind-safe), fixed assignment order
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
     "ink": "#1a1a2e", "grey": "#8a8a97"}
plt.rcParams.update({"font.size": 9.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------- fig 1 -----
def fig_schematic():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=150)
    ax.set_xlim(0, 105)
    ax.set_ylim(-2, 30)
    ax.axis("off")
    ax.set_title("SCHEMATIC — MRR pulse train  →  one-slot delay ladder  →  "
                 "5×5 Clements mesh (+ balanced score head)", fontsize=10)

    # master MRR + bus
    ax.add_patch(Circle((7, 24), 3.1, fc="none", ec=C["blue"], lw=2))
    ax.plot([2, 20], [20, 20], color=C["blue"], lw=2)
    ax.annotate("master MRR R=16 µm\nFSR 1.995 THz, Q 1.1e7", (7, 24),
                textcoords="offset points", xytext=(0, 26), ha="center",
                fontsize=8)
    ax.text(2.0, 18.4, "CW in", fontsize=8)
    # pulse train symbol
    for k in range(5):
        x = 12.5 + 1.6 * k
        ax.plot([x, x], [20.15, 21.4], color=C["ink"], lw=1.4)
    ax.text(15.6, 22.1, "tokens sₜ (EO-written)", fontsize=8, ha="center")

    # delay ladder
    ways = np.linspace(17.5, 2.5, 5)
    ax.plot([20, 24], [20, 20], color=C["blue"], lw=2)
    ax.text(22, 20.7, "×5 split", fontsize=7.5, ha="center")
    for j, y in enumerate(ways):
        ax.plot([24, 30], [20, y], color=C["blue"], lw=1.4)
        # delay loops
        for k in range(j):
            ax.add_patch(Circle((32.4 + 2.4 * k, y), 1.0, fc="none",
                                ec=C["orange"], lw=1.6))
        ax.plot([30, 44], [y, y], color=C["blue"], lw=1.4)
        ax.text(44.6, y - 0.35, f"s(t−{j})", fontsize=8, color=C["ink"])
    ax.text(30, -0.7, "one-slot delay = 100.53 µm/tap\n"
                      "(= one ring circumference)", fontsize=8,
            color=C["orange"], ha="center")

    # mesh: Clements 5x5 grid of MZI nodes
    x0, y0, dx, dy = 52, 2.5, 7.5, 3.75
    for col in range(5):
        for row in range(4):
            if (row + col) % 2 == 0:
                yc = y0 + (3 - row) * dy + dy / 2 + 7.5 - 3.75 / 2
                xc = x0 + col * dx
                ax.add_patch(Rectangle((xc - 2.1, yc - 1.05), 4.2, 2.1,
                                       fc="#eef3fa", ec=C["blue"], lw=1.2))
                ax.plot([xc - 0.8, xc + 0.8], [yc, yc], color=C["verm"],
                        lw=1.6)
    for j, y in enumerate(ways):
        ax.plot([48, 52 + 4 * dx + 3], [y, y], color=C["blue"], lw=1.0,
                zorder=0)
    ax.text(x0 + 2 * dx, 21.5, "10 MZI nodes (θ, φ electrodes)\n"
            "coupler κ FDTD-calibrated 45/55", ha="center", fontsize=8)
    for j, y in enumerate(ways):
        ax.annotate("", (93.5, y), (52 + 4 * dx + 3, y),
                    arrowprops=dict(arrowstyle="->", color=C["blue"],
                                    lw=1.2))
        ax.text(94.2, y - 0.35, f"yₜ[{j}]", fontsize=8)
    ax.text(75, -0.7, "yₜ = U·(sₜ … sₜ₋₄): sliding unitary token mixer",
            fontsize=8.5, color=C["ink"], ha="center")

    # balanced score head
    ax.add_patch(Rectangle((84, 23), 13.5, 5.6, fc="#fdf3e7",
                           ec=C["orange"], lw=1.2))
    ax.text(90.75, 27.3, "score head (per lag k)", fontsize=8, ha="center")
    ax.text(90.75, 24.0,
            "50/50 + λ/4, balanced PD\n→ sₜ·sₜ₋ₖ  (QKᵀ band)",
            fontsize=7.5, ha="center")
    fig.savefig(os.path.join(FIG, "fig1_schematic.png"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 2 -----
def fig_node_cal():
    d = jload("node_model.json")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), dpi=150)
    ax = axs[0]
    ax.plot(d["dn_tab"], d["s_model_tab"], color=C["blue"], lw=2,
            label="node model", zorder=2)
    ax.scatter(d["dn_tab"], d["s_tab"], s=28, color=C["verm"], zorder=3,
               label="FDTD (mzi.py)")
    ax.set_xlabel("Δn (phase electrode)")
    ax.set_ylabel("signed output s")
    ax.set_title(f"s(Δn) @1550 — model rms {d['s_dn_model_rms']:.4f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    lam = np.array(d["lam_grid_um"]) * 1000
    ax.plot(lam, np.array(d["kappa_grid"]) / (np.pi / 4), color=C["blue"],
            lw=2)
    ax.axhline(1.0, color=C["grey"], lw=1, ls="--")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("κ / (π/4)")
    ax.set_title("coupler angle vs λ (from s₀(λ))", fontsize=9)
    ax = axs[2]
    ax.plot(lam, np.array(d["amp_grid"]) ** 2, color=C["green"], lw=2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("node power transmission")
    ax.set_title(f"per-node loss — T(1550)={d['node_power_T_1550']:.3f}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_node_cal.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 3 -----
def fig_toeplitz():
    A = jload("demo.json")["A_toeplitz"]
    S = np.array(A["S_real"]) + 1j * np.array(A["S_imag"])
    R = np.array(A["ref_real"]) + 1j * np.array(A["ref_imag"])
    g = np.vdot(R, S) / np.vdot(R, R)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2), dpi=150)
    ax = axs[0]
    k = np.arange(S.shape[1])
    ax.plot(k, np.real(R[0]), color=C["blue"], lw=2,
            label="ideal Toeplitz  Σⱼ U₀ⱼ sₜ₋ⱼ")
    ax.plot(k, np.real(S[0] / g), color=C["verm"], lw=1.1, ls="--",
            label="mesh output (peak-sampled)")
    ax.set_xlabel("slot t")
    ax.set_ylabel("Re y₀(t)")
    ax.set_title(f"output port 0 — rms {A['otdm_vs_ideal_toeplitz_rms']:.4f}"
                 f" ({A['trace_fwhm_fs']:.0f} fs pulse)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    sw = A["pulse_sweep"]
    fw = [r["fwhm_fs"] for r in sw]
    ax.plot(fw, [r["bits_vs_ideal"] for r in sw], "o-", color=C["blue"],
            lw=2, label="incl. measured node phase\ndispersion (−0.25°/nm)")
    ax.plot(fw, [-np.log2(max(r["err_vs_ideal_noscreen"], 1e-12))
                 for r in sw], "s-", color=C["orange"], lw=2,
            label="κ(λ) + loss only")
    ax.set_xlabel("pulse FWHM (fs)   [slot = 501 fs]")
    ax.set_ylabel("accuracy (bits, vs ideal U)")
    ax.set_title("the mesh wants LONG pulses — compression is for the\n"
                 "serializer, not the mixer", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_toeplitz.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 4 -----
def fig_stft():
    B = jload("demo.json")["B_stft"]
    P = np.array(B["spectrogram"])
    Rf = np.array(B["reference"])
    tok = np.array(B["tokens"])
    fig = plt.figure(figsize=(10.5, 4.2), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tok, color=C["ink"], lw=1.2)
    ax.set_ylabel("token sₜ")
    ax.set_title("chirped token sequence through the DFT₅ mesh = "
                 "sliding 5-point DFT at 2 THz frame rate", fontsize=9.5)
    for i, (dat, ttl) in enumerate([(P, "mesh |y|² (photonic)"),
                                    (Rf, "exact sliding DFT")]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(dat / dat.max(), aspect="auto", cmap="magma",
                  origin="lower", interpolation="nearest")
        ax.set_xlabel("slot t")
        ax.set_ylabel("DFT bin")
        ax.set_title(ttl + (f"   (rms {B['rms_err_norm']:.3f})"
                            if i == 0 else ""), fontsize=9)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_stft.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 5 -----
def fig_attention():
    d = jload("demo.json")
    Cd, D = d["C_expressivity"], d["D_scores"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), dpi=150)
    ax = axs[0]
    names = [("conv (Hyena-style)", Cd["R2_toeplitz_to_conv"], C["blue"]),
             ("attn scalar β=1", Cd["R2_toeplitz_to_attention_b1"],
              C["orange"]),
             ("attn scalar β=4", Cd["R2_toeplitz_to_attention_b4"],
              C["orange"]),
             ("attn vec d=4 β=1", Cd["R2_toeplitz_to_vec_attention_b1"],
              C["verm"]),
             ("attn vec d=4 β=4", Cd["R2_toeplitz_to_vec_attention_b4"],
              C["verm"])]
    ys = np.arange(len(names))[::-1]
    for y, (nm, r2, col) in zip(ys, names):
        ax.barh(y, r2, height=0.62, color=col, alpha=0.85)
        ax.text(r2 - 0.02, y, f"{r2:.3f}", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _ in names], fontsize=8)
    ax.set_xlabel("R² of best fixed linear (Toeplitz) fit")
    ax.set_xlim(0, 1.05)
    ax.set_title("what the mesh CAN be (R²=1)\nvs attention (R²<1, "
                 "structural)", fontsize=9)
    ax = axs[1]
    b1 = np.array([np.nan if v is None else v for v in D["band1_meas"]])
    tok = np.array(D["tokens"])
    truth = tok * np.roll(tok, 1)
    ok = ~np.isnan(b1)
    ax.scatter(truth[ok], b1[ok], s=16, color=C["blue"], alpha=0.8)
    lim = [-1, 1]
    ax.plot(lim, lim, color=C["grey"], lw=1, ls="--")
    ax.set_xlabel("sₜ·sₜ₋₁ (exact)")
    ax.set_ylabel("balanced-homodyne readout")
    e1 = D["band_rms_err"]["1"]
    ax.set_title(f"photonic score, lag 1 — rms {e1:.1e}\n(leak-corrected; "
                 f"uncorrected {D['band_rms_err_uncorrected']['1']:.3f})",
                 fontsize=9)
    ax = axs[2]
    yph, yex = np.array(D["y_photonic"]), np.array(D["y_exact"])
    ax.plot(yex, color=C["blue"], lw=2, label="softmax attention (float)")
    ax.plot(yph, color=C["verm"], lw=1.1, ls="--",
            label="photonic scores + e⁻ softmax")
    ax.set_xlabel("slot t")
    ax.set_ylabel("attention output")
    ax.set_title(f"window-5 attention closed through the chip — "
                 f"rms {D['attention_pipeline_rms']:.1e}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_attention.png"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 6 -----
def fig_fdtd():
    sc = jload("fdtd_score.json")
    jt = jload("node2_joint.json")
    if not (sc and jt):
        print("  fdtd figs skipped (runs not finished)")
        return False
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), dpi=150)
    ax = axs[0]
    labels, meas, pred = [], [], []
    for row in jt["rows"]:
        psi = f"{row['psi_deg']:.0f}°"
        labels += [f"ψ={psi}\nbar", f"ψ={psi}\ncross"]
        meas += [row["meas_bar_1550"], row["meas_cross_1550"]]
        pred += [row["pred_bar_fit"], row["pred_cross_fit"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, meas, 0.36, color=C["blue"], label="FDTD (2 inputs)")
    ax.bar(x + 0.19, pred, 0.36, color=C["orange"],
           label="node model + static\nphase cal (κ unchanged)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("output power fraction")
    err = float(np.max(np.abs(np.array(meas) - np.array(pred))))
    ax.set_title(f"coherent 2-input MZI node @1550 — "
                 f"max dev {err:.3f}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    t = np.array(sc["t"]) * 3.33564 / 1000  # ps
    ax.plot(t, np.array(sc["topAB"]) / max(sc["topAB"]), color=C["blue"],
            lw=1, label="port bar")
    ax.plot(t, -np.array(sc["botAB"]) / max(sc["topAB"]), color=C["verm"],
            lw=1, label="− port cross")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("|E|² (norm)")
    ax.set_title("FDTD: two interleaved trains in the coupler", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    diffg = np.array(sc["diff"]) / sc["gain"]
    expd = np.array(sc["expected"])
    x = np.arange(len(diffg))
    ax.bar(x - 0.19, diffg, 0.36, color=C["blue"],
           label="balanced diff / g (FDTD)")
    ax.bar(x + 0.19, expd, 0.36, color=C["orange"], label="aₖ·aₖ₋₁ exact")
    ax.set_xlabel("slot k")
    ax.set_ylabel("token product")
    ax.set_title(f"score head in real FDTD — rms {sc['rms_rel']:.3f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_fdtd.png"), bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------- fig 7 -----
def fig_scaling():
    d = jload("scaling.json")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), dpi=150)
    rows = d["scaling"]
    N = [r["N"] for r in rows]
    ax = axs[0]
    ax.plot(N, [r["bits"] for r in rows], "o-", color=C["blue"], lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels(N)
    ax.set_xlabel("mesh size N")
    ax.set_ylabel("accuracy (bits)")
    ax.set_title("token-mixer accuracy vs N\n(45/55 coupler + 1.7%/node "
                 "loss)", fontsize=9)
    ax = axs[1]
    ax.plot(N, [r["loss_dB"] for r in rows], "o-", color=C["orange"], lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels(N)
    ax.set_xlabel("mesh size N")
    ax.set_ylabel("net insertion loss (dB)")
    ax.set_title("loss vs N — 2.3 dB at N=32", fontsize=9)
    ax = axs[2]
    q = d["quantization"]
    bits = [r["bits_dac"] if r["bits_dac"] else 14 for r in q]
    ax.plot(bits, [r["rms_err"] for r in q], "o-", color=C["green"], lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("phase-DAC resolution (bits, 14≡∞)")
    ax.set_ylabel("5×5 rms element error")
    ax.set_title("10-bit phase DACs suffice", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_scaling.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 8 -----
def fig_density():
    d = jload("density.json")
    cc = jload("compact_cal.json")
    if not (d and cc):
        print("  density fig skipped")
        return False
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=150,
                            gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    rows = d["rows"]
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(100, 1000, color=C["green"], alpha=0.12)
    ax.text(310, len(rows) - 0.45, "target band", fontsize=8,
            color=C["green"], ha="center")
    for y, r in zip(ys, rows):
        col = C["blue"] if r["d_wdm"] == 1 else C["purple"]
        ax.barh(y, r["x_B300"], height=0.62, color=col, alpha=0.9)
        ax.text(r["x_B300"] * 1.12, y,
                f"{r['x_B300']:.0f}×  ({r['bits']:.1f} b)",
                va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(30, 3e4)
    ax.set_xlabel("compute density vs B300 die (MAC/s/mm², die-level)")
    ax.set_title("B300 = 7.5e15 MAC/s dense-FP4 ÷ 1650 mm² = "
                 "4.5e12 MAC/s/mm²", fontsize=9)
    ax = axs[1]
    labels = ["original\n96.7 µm", f"compact\n{cc['node_len_um']:.1f} µm"]
    Ls = [96.7 * 6, cc["node_len_um"] * 4]
    ax.bar([0, 1], Ls, 0.55, color=[C["grey"], C["blue"]])
    for i, v in enumerate(Ls):
        ax.text(i, v * 1.02, f"{v:.0f} µm²/node", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("node footprint (µm²)")
    ax.set_title(f"node shrink {96.7*6/(cc['node_len_um']*4):.1f}× — "
                 f"T={cc['node_T_1550']:.3f}/node,\n"
                 f"geometric θ validated (s {cc['geo_phase_s_meas']:+.3f} "
                 f"vs {cc['geo_phase_s_pred']:+.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_density.png"), bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------- fig 9 -----
def fig_opt():
    l1, l3, l4 = jload("opt_loop1.json"), jload("opt_loop3.json"), \
        jload("opt_loop4.json")
    if not (l1 and l3 and l4):
        print("  opt fig skipped")
        return False
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), dpi=150)
    ax = axs[0, 0]
    sw = l1["T0_sweep"]
    fw = [r["fwhm_fs"] for r in sw]
    ax.plot(fw, [r["bits"] for r in sw], "o-", color=C["blue"], lw=2,
            label="mesh bits")
    ax2c = [r["score_rms"]["1"] for r in sw]
    ax.set_xlabel("pulse FWHM (fs)  [slot 501 fs]")
    ax.set_ylabel("mesh accuracy (bits)")
    ax.axvline(0.4 * 501, color=C["verm"], lw=1, ls="--")
    ax.text(0.4 * 501 + 6, 3.2, "slot-stuffing cliff\n(overlap ISI)",
            fontsize=8, color=C["verm"])
    ax.set_title("loop 1: the overlap-'noise' cliff — pulse ≤ 0.4·slot",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    for term, col in (("kappa_sigma", C["verm"]),
                      ("phase_set_rms", C["blue"]),
                      ("delay_phase_rms", C["orange"])):
        rows = l3["sensitivity"][term]
        ax.plot([r["mrad"] for r in rows],
                [r["bits_mean"] for r in rows], "o-", color=col, lw=2,
                label=term.replace("_rms", "").replace("_", " "))
    ax.set_xlabel("control error (mrad rms)")
    ax.set_ylabel("bits (N=5, 120 fs)")
    ax.set_title("loop 3: coupler disorder dominates the budget",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    ka = l4["kappa_aware"]
    xs = np.arange(len(ka))
    ax.bar(xs, [r["bits_mean"] for r in ka], 0.6,
           yerr=[r["bits_std"] for r in ka],
           color=[C["grey"]] + [C["green"]] * (len(ka) - 1), capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["tag"].replace("aware_char", "χ ")
                        .replace("mrad", "") for r in ka], fontsize=8)
    ax.set_ylabel("bits at full spec")
    ax.set_title("loop 4: κ-aware compile (+1.0 bit) vs\n"
                 "characterization error", fontsize=9)
    ax = axs[1, 1]
    par = l4["pareto"]
    for key, col, lab in (("dens_optical_out", C["blue"],
                           "optical-in/optical-out"),
                          ("dens_electronic_readout", C["verm"],
                           "full electronic readout")):
        ax.scatter([r[key] for r in par], [r["bits"] for r in par],
                   s=42, color=col, label=lab)
        for r in par:
            ax.annotate(f"N{r['N']}/{r['slot_fs']:.0f}",
                        (r[key], r["bits"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5)
    ax.axvspan(100, 1000, color=C["green"], alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel("compute density (×B300 die)")
    ax.set_ylabel("bits at full spec")
    ax.set_title("loop 4: the honest Pareto — readout is the tax",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_opt.png"), bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    fig_schematic()
    fig_node_cal()
    fig_toeplitz()
    fig_stft()
    fig_attention()
    ok = fig_fdtd()
    fig_scaling()
    ok2 = fig_density()
    ok3 = fig_opt()
    print("figs written", "(some pending)" if not (ok and ok2 and ok3)
          else "")

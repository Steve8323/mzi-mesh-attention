#!/usr/bin/env python
"""Optimization loop driver.  usage: opt_loop.py K  (K = loop number)
Each loop runs a sweep set defined below, writes out/opt_loopK.json.
The loop CONTENTS evolve between runs based on the critic's findings —
see OPTIMIZATION.md for the log."""
import json
import os
import sys

import numpy as np

from opt_harness import run_config, attribution

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def loop1():
    res = {"loop": 1, "purpose": "baseline + error attribution + scaling"}
    print("== attribution, mesh operating point (T0=120 fs)")
    res["attribution_120fs"] = attribution(N=5, T0=120e-15)
    print("== attribution, compressed pulse (T0=9.86 fs)")
    res["attribution_10fs"] = attribution(N=5, T0=9.86e-15)
    print("== T0 sweep")
    res["T0_sweep"] = [run_config(N=5, T0=T0 * 1e-15)
                       for T0 in (9.86, 30, 60, 90, 120, 150, 180)]
    for r in res["T0_sweep"]:
        print(f"  T0={r['T0_fs']:6.1f} fs bits={r['bits']:.1f} "
              f"isi_win={r['isi_leak_win']:.2e} "
              f"score1={r['score_rms'][1]:.2e} "
              f"score4={r['score_rms'][4]:.2e}")
    print("== N sweep at T0=120 fs")
    res["N_sweep"] = [run_config(N=N, T0=120e-15, want_score=False)
                      for N in (3, 5, 8, 16)]
    for r in res["N_sweep"]:
        print(f"  N={r['N']:3d} bits={r['bits']:.1f} "
              f"isi_win={r['isi_leak_win']:.2e} "
              f"walkoff={r['walkoff_fs']:.1f} fs "
              f"dens={r['density_xB300']:.0f}x")
    print("== slot-rate check (4 THz, T0=60 fs)")
    res["fast_slot"] = run_config(N=5, T0=60e-15, slot_fs=250.63)
    r = res["fast_slot"]
    print(f"  slot=251 fs bits={r['bits']:.1f} "
          f"isi_win={r['isi_leak_win']:.2e} dens={r['density_xB300']:.0f}x")
    return res


def _mc(seeds=8, **kw):
    rows = [run_config(seed=s, **kw) for s in range(seeds)]
    b = np.array([r["bits"] for r in rows])
    br = np.array([r["bits_rowgain"] for r in rows])
    isi = np.array([r["isi_leak"] for r in rows])
    return {"bits_mean": float(b.mean()), "bits_std": float(b.std()),
            "bits_rowgain_mean": float(br.mean()),
            "bits_rowgain_std": float(br.std()),
            "isi_mean": float(isi.mean()), "n_seeds": seeds,
            "kw": {k: (v if not isinstance(v, float) else v)
                   for k, v in kw.items() if k != "want_score"}}


# stochastic error magnitudes (provenance in OPTIMIZATION.md):
#  phase_set 5 mrad  = staircase LUT accuracy (node.py model_rms 0.0038 s)
#  kappa    12 mrad  = gap +-20 nm via coupler_design Lpi(gap) slope
#  delay ph 10 mrad  = 0.55 deg trim spec (silica-repo control table)
STOCH = dict(phase_set_rms=5e-3, kappa_sigma=12e-3, delay_phase_rms=10e-3)


def loop2():
    """Critic-1 driven: (HIGH-1) test the split recommendation in its
    implementable forms; (HIGH-2/3) stochastic errors + 8-seed stats;
    (HIGH-4) per-row gain refit; (MED-5) factorial attribution."""
    res = {"loop": 2}
    print("== HIGH-1: stretcher vs bandwidth-filter (the real split test)")
    a = run_config(N=5, T0=9.86e-15, prechirp_fs2=1180.0, want_score=False)
    b = run_config(N=5, T0=9.86e-15, want_score=False)
    c = run_config(N=5, T0=120e-15, want_score=False)
    res["split_test"] = {"chirp_stretched_10fs": a, "raw_10fs": b,
                         "filtered_TL_120fs": c}
    print(f"  chirped 10fs->~120fs env: bits={a['bits']:.2f} "
          f"(broad={a['broadening']:.2f})")
    print(f"  raw 10 fs               : bits={b['bits']:.2f}")
    print(f"  TL 120 fs (= tap+filter): bits={c['bits']:.2f}")
    print("== HIGH-2/3: Monte Carlo with stochastic errors, 8 seeds")
    res["mc_ideal_set"] = _mc(N=5, T0=120e-15, want_score=False)
    res["mc_stoch"] = _mc(N=5, T0=120e-15, want_score=False, **STOCH)
    for tag in ("mc_ideal_set", "mc_stoch"):
        m = res[tag]
        print(f"  {tag}: bits {m['bits_mean']:.2f}+-{m['bits_std']:.2f}  "
              f"rowgain {m['bits_rowgain_mean']:.2f}"
              f"+-{m['bits_rowgain_std']:.2f}")
    print("== N sweep, 8 seeds, stochastic errors on")
    res["N_sweep_mc"] = [dict(N=N, **_mc(N=N, T0=120e-15,
                                         want_score=False, **STOCH))
                         for N in (3, 5, 8, 16)]
    for m in res["N_sweep_mc"]:
        print(f"  N={m['N']:3d} bits {m['bits_mean']:.2f}"
              f"+-{m['bits_std']:.2f}  rowgain "
              f"{m['bits_rowgain_mean']:.2f}+-{m['bits_rowgain_std']:.2f}")
    print("== MED-5: 2^3 factorial attribution (screen x kappa x loss)")
    fact = {}
    for s in (0, 1):
        for k in (0, 1):
            for lo in (0, 1):
                r = run_config(N=5, T0=120e-15, want_score=False,
                               use_screen=bool(s), use_kappa_disp=bool(k),
                               use_loss=bool(lo))
                fact[f"s{s}k{k}l{lo}"] = {"rms": r["rms_err"],
                                          "rms_row": r["rms_err_rowgain"]}
    res["factorial"] = fact
    print("   " + "  ".join(f"{key}:{v['rms']:.4f}"
                            for key, v in fact.items()))
    return res


STOCH3 = dict(phase_set_rms=5e-3, kappa_sigma=12e-3,
              delay_phase_rms=5e-3, ceo_jitter_rms=5e-3)   # j-scaled x2


def loop3():
    """Critic-2 driven: (1) receiver-realism re-adjudication of the split
    recommendation with real filter lineshapes; (2) control-error
    sensitivity spec sweep with exact per-node kappa and j-scaled
    delay/CEO terms, paired seeds."""
    res = {"loop": 3}
    print("== A: split re-adjudication x receiver realism (4 seeds)")
    arms = {
        "raw_10fs": dict(T0=9.86e-15),
        "chirp_10fs": dict(T0=9.86e-15, prechirp_fs2=1180.0),
        "filt_gauss": dict(T0=9.86e-15, filter_kind="gauss"),
        "filt_ring1": dict(T0=9.86e-15, filter_kind="ring1"),
        "filt_ring2": dict(T0=9.86e-15, filter_kind="ring2"),
        "TL_120fs": dict(T0=120e-15),
    }
    recv = {"peak": dict(receiver="peak"),
            "peak+50fs": dict(receiver="peak", sample_offset_fs=50.0),
            "gate250": dict(receiver="gate", gate_fs=250.0)}
    adjud = {}
    for an, akw in arms.items():
        adjud[an] = {}
        for rn, rkw in recv.items():
            bits = [run_config(N=5, want_score=False, seed=s,
                               **akw, **rkw)["bits"] for s in range(4)]
            adjud[an][rn] = {"bits_mean": float(np.mean(bits)),
                             "bits_std": float(np.std(bits))}
        print(f"  {an:11s} " + "  ".join(
            f"{rn}:{adjud[an][rn]['bits_mean']:.2f}"
            f"±{adjud[an][rn]['bits_std']:.2f}" for rn in recv))
    res["adjudication"] = adjud
    print("== B: control-error sensitivity spec sweep (paired seeds)")
    sens = {}
    for term in ("phase_set_rms", "kappa_sigma", "delay_phase_rms",
                 "ceo_jitter_rms"):
        rows = []
        for scale in (0.0, 0.5, 1.0, 2.0, 4.0):
            bits = []
            for s in range(4):
                kw = {k: 0.0 for k in STOCH3}
                kw[term] = STOCH3[term] * scale
                bits.append(run_config(N=5, T0=120e-15, want_score=False,
                                       seed=s, seed_err=777 + s,
                                       **kw)["bits"])
            rows.append({"scale": scale,
                         "mrad": STOCH3[term] * scale * 1e3,
                         "bits_mean": float(np.mean(bits)),
                         "bits_std": float(np.std(bits))})
        sens[term] = rows
        print(f"  {term:16s} " + "  ".join(
            f"x{r['scale']:g}:{r['bits_mean']:.2f}" for r in rows))
    res["sensitivity"] = sens
    print("== C: all terms together at spec (reference point)")
    res["all_at_spec"] = _mc(seeds=8, N=5, T0=120e-15, want_score=False,
                             **STOCH3)
    m = res["all_at_spec"]
    print(f"  bits {m['bits_mean']:.2f}±{m['bits_std']:.2f}  "
          f"rowgain {m['bits_rowgain_mean']:.2f}")
    return res


SPEC4 = dict(phase_set_rms=5e-3, kappa_sigma=12e-3, delay_phase_rms=5e-3,
             comb_linewidth_hz=1e6)     # 1 MHz comb line; CEO static term
#                                         deleted (critic-3 HIGH-2)


def loop4():
    """Critic-3 driven synthesis: kappa-aware compile WITH
    characterization error, per-pulse comb phase noise, score head under
    the full error model, finalists at spec, honest Pareto."""
    from opt_harness import score_full
    res = {"loop": 4}
    print("== A: kappa-aware compile vs characterization error (8 seeds)")
    rows = []
    for tag, kw in [("naive", dict(kappa_aware=False))] + [
            (f"aware_char{c*1e3:g}mrad",
             dict(kappa_aware=True, kappa_char_rms=c))
            for c in (0.0, 2e-3, 5e-3, 10e-3)]:
        m = _mc(seeds=8, N=5, T0=120e-15, want_score=False, **SPEC4, **kw)
        rows.append({"tag": tag, **m})
        print(f"  {tag:18s} bits {m['bits_mean']:.2f}±{m['bits_std']:.2f}")
    res["kappa_aware"] = rows
    print("== B: comb linewidth sweep (mesh bits + score rms, 4 seeds)")
    rows = []
    rng = np.random.default_rng(5)
    for lw in (0.0, 1e5, 1e6, 1e7):
        kw = dict(SPEC4)
        kw["comb_linewidth_hz"] = lw
        m = _mc(seeds=4, N=5, T0=120e-15, want_score=False,
                kappa_aware=True, kappa_char_rms=3e-3, **kw)
        sc = np.mean([score_full(rng.uniform(-0.97, 0.97, 32), 9.86e-15,
                                 lag=4, comb_linewidth_hz=lw, seed_err=s)
                      for s in range(4)])
        rows.append({"linewidth_hz": lw, "bits": m["bits_mean"],
                     "score_rms_lag4": float(sc)})
        print(f"  lw={lw:8.0e} Hz  bits {m['bits_mean']:.2f}  "
              f"score4 rms {sc:.2e}")
    res["comb"] = rows
    print("== C: finalists at full spec, kappa-aware char 3 mrad, 8 seeds")
    fin = {}
    for tag, kw in (("TL120_peak", dict(T0=120e-15, receiver="peak")),
                    ("raw10_gate", dict(T0=9.86e-15, receiver="gate",
                                        gate_fs=250.0)),
                    ("gaussfilt_peak", dict(T0=9.86e-15,
                                            filter_kind="gauss",
                                            receiver="peak"))):
        m = _mc(seeds=8, N=5, want_score=False, kappa_aware=True,
                kappa_char_rms=3e-3, **SPEC4, **kw)
        fin[tag] = m
        print(f"  {tag:15s} bits {m['bits_mean']:.2f}±{m['bits_std']:.2f}")
    res["finalists"] = fin
    print("== D: score head, full error model at the compressed tap")
    rng = np.random.default_rng(9)
    rows = []
    for dkap, lw in ((0.0, 0.0), (12e-3, 0.0), (12e-3, 1e6)):
        sc = {lag: float(np.mean([score_full(
            rng.uniform(-0.97, 0.97, 32), 9.86e-15, lag=lag, dkap=dkap,
            comb_linewidth_hz=lw, seed_err=s) for s in range(4)]))
            for lag in (1, 4)}
        rows.append({"dkap": dkap, "lw": lw, "score_rms": sc})
        print(f"  dkap={dkap:5.3f} lw={lw:8.0e}  "
              f"lag1 {sc[1]:.2e}  lag4 {sc[4]:.2e}")
    res["score_full"] = rows
    # analytic shot-noise line: homodyne amplitude SNR = sqrt(2 Nph);
    # 8-bit products need Nph ~ 3.3e4/slot = 4.2 fJ at 1550 nm
    res["shot_note"] = ("homodyne shot limit: bits ~ log2(sqrt(2*Nph)); "
                        "8 bits -> 3.3e4 photons/slot = 4.2 fJ/slot/port "
                        "= 8.4 uW/port at 2 THz")
    print("== E: final Pareto (kappa-aware, full spec, best receiver)")
    par = []
    for N in (5, 8, 16):
        for slot in (501.26, 250.63):
            T0 = 120e-15 if slot > 400 else 60e-15
            m = _mc(seeds=6, N=N, T0=T0, slot_fs=slot, want_score=False,
                    kappa_aware=True, kappa_char_rms=3e-3, **SPEC4)
            r1 = run_config(N=N, T0=T0, slot_fs=slot, want_score=False)
            dens_opt = r1["density_xB300"]
            # electronic-readout density (critic-3 HIGH-1 cost 2): time-
            # demux D=40 (50 GHz receivers) per port: 40 gate rings
            # (~0.002 mm2 each) + I/Q pads, x1.15 routing
            from opt_harness import B300_DENS as _B
            import otdm as _ot
            macs = N ** 2 * (1.0 / (slot * 1e-15))
            area_ph = macs / (dens_opt * _B)
            area_rx = N * 40 * 0.002 * 1.15
            dens_el = macs / ((area_ph + area_rx) * _B)
            par.append({"N": N, "slot_fs": slot,
                        "bits": m["bits_mean"], "bits_std": m["bits_std"],
                        "dens_optical_out": dens_opt,
                        "dens_electronic_readout": float(dens_el)})
            print(f"  N={N:3d} slot={slot:5.0f} bits "
                  f"{m['bits_mean']:.2f}±{m['bits_std']:.2f} "
                  f"dens(opt) {dens_opt:.0f}x  dens(e-out) {dens_el:.0f}x")
    res["pareto"] = par
    return res


res = {1: loop1, 2: loop2, 3: loop3, 4: loop4}[K]()
json.dump(res, open(os.path.join(OUT, f"opt_loop{K}.json"), "w"), indent=1)
print(f"wrote out/opt_loop{K}.json")

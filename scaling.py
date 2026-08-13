"""How the token mixer scales with N: mesh fidelity, net transmission,
phase-set resolution, and throughput.

Error sources included (all pinned to FDTD calibration):
  - kappa = 0.7368 (45/55 coupler) at 1550, decomposition run natively on it
  - 1.7% power loss per MZI node (differential across paths -> the floor)
  - optional phase-DAC quantization of every theta/phi (electrode set once)

Throughput: N^2 MACs per slot at the ring FSR (1.995 THz).
"""
import json
import os

import numpy as np

from node import NodeCal, LAM0
from clements import decompose, mesh_matrix, fidelity, haar
from otdm import FSR

HERE = os.path.dirname(os.path.abspath(__file__))
cal = NodeCal()
KAPPA0 = float(cal.kappa(LAM0))
rng = np.random.default_rng(3)


def quantize(nodes, d, nbits):
    if nbits is None:
        return nodes, d
    qth = np.pi / (2 ** nbits - 1)          # theta in [0, pi]
    qph = 2 * np.pi / (2 ** nbits - 1)      # phi in [-pi, pi)
    nn = [{"m": n["m"], "theta": np.round(n["theta"] / qth) * qth,
           "phi": np.round(n["phi"] / qph) * qph} for n in nodes]
    dq = np.exp(1j * np.round(np.angle(d) / qph) * qph)
    return nn, dq


def run():
    rows = []
    for N in (2, 5, 8, 16, 32):
        errs, Ts = [], []
        for _ in range(6):
            U = haar(N, rng)
            nodes, d, _ = decompose(U, KAPPA0)
            M = mesh_matrix(nodes, d, N, cal=cal)
            e, T = fidelity(M, U)
            errs.append(e)
            Ts.append(T)
        rows.append({
            "N": N, "nodes": N * (N - 1) // 2,
            "rms_err": float(np.mean(errs)),
            "bits": float(-np.log2(np.mean(errs))),
            "power_T": float(np.mean(Ts)),
            "loss_dB": float(-10 * np.log10(np.mean(Ts))),
            "MACs_per_s": float(N ** 2 * FSR),
        })
        print(f"N={N:3d}  nodes={N*(N-1)//2:4d}  rms={np.mean(errs):.4f} "
              f"({-np.log2(np.mean(errs)):.1f} bits)  "
              f"T={np.mean(Ts):.3f} ({-10*np.log10(np.mean(Ts)):.2f} dB)  "
              f"{N**2*FSR:.2e} MAC/s")
    qrows = []
    N = 5
    U = haar(N, rng)
    nodes, d, _ = decompose(U, KAPPA0)
    for nb in (6, 8, 10, 12, None):
        nn, dq = quantize(nodes, d, nb)
        M = mesh_matrix(nn, dq, N, cal=cal)
        e, _ = fidelity(M, U)
        qrows.append({"bits_dac": nb if nb else 0, "rms_err": float(e)})
        print(f"  5x5 phase DAC {nb if nb else 'inf'} bits -> rms {e:.4f}")
    json.dump({"scaling": rows, "quantization": qrows},
              open(os.path.join(HERE, "out", "scaling.json"), "w"), indent=1)


if __name__ == "__main__":
    run()

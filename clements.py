"""Clements-style decomposition of an NxN unitary directly onto the PHYSICAL
node of node.py — including its FDTD-measured imperfect coupler kappa.

Node (kappa from calibration, phase on the top input):
  M(th, ph) = C(k) . diag(e^{i th}, 1) . C(k) . diag(e^{i ph}, 1)
            = [[A e^{i ph}, B], [B e^{i ph}, Dd]]
  A = e^{i th} c^2 - s^2,  B = i s c (e^{i th} + 1),  Dd = c^2 - s^2 e^{i th}
  (c = cos k, s = sin k;  M is symmetric, unitary for unit node amplitude).

Because the nulling equations only need the RATIO |B|/|A| (monotone in th,
range [0, tan 2k]), the decomposition runs natively with the imperfect
coupler: residuals appear only when a null demands a ratio > tan(2k),
i.e. a nearly perfect cross state.  The diagonal-sweep identity

  M(th,ph)^H diag(d1,d2) = diag(dB*(d2/B)e^{-i ph}·B ... ) M(th, ph')

holds for ANY kappa thanks to unitarity (arg A + arg Dd = 2 arg B + pi);
it is verified numerically in __main__ rather than trusted.

Result:  U = diag(D) @ prod_k M(nodes[k]),  K = N(N-1)/2 nodes,
D = output phase screen (N output phase shifters).
"""
import numpy as np

from node import node_matrix, NodeCal, LAM0


def _ABD(th, k):
    c, s = np.cos(k), np.sin(k)
    A = np.exp(1j * th) * c ** 2 - s ** 2
    B = 1j * s * c * (np.exp(1j * th) + 1.0)
    Dd = c ** 2 - s ** 2 * np.exp(1j * th)
    return A, B, Dd


def _solve_theta(ratio, k):
    """th in [0, pi] with |B(th)|/|A(th)| = ratio (clamped to reachable)."""
    lo, hi = 0.0, np.pi
    A0, B0, _ = _ABD(0.0, k)
    rmax = abs(B0) / abs(A0)
    if ratio >= rmax:
        return 0.0, ratio - rmax          # unreachable residual
    f = lambda th: abs(_ABD(th, k)[1]) - ratio * abs(_ABD(th, k)[0])
    for _ in range(80):                    # bisection, |B|/|A| monotone dec.
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), 0.0


def _embed(N, m, M2):
    T = np.eye(N, dtype=complex)
    T[m:m + 2, m:m + 2] = M2
    return T


def _final_index(N):
    """Map (creation order) -> final node-list index, for per-node kappa.
    Final list = [left creations in order] + [right creations reversed]."""
    p = sum(i + 1 for i in range(N - 1) if i % 2 == 1)   # left count
    q = sum(i + 1 for i in range(N - 1) if i % 2 == 0)   # right count
    return p, q


def decompose(U, kappa=np.pi / 4):
    """U = diag(d) @ prod_k M(nodes[k]) with couplers kappa (exact when
    every null is reachable).  kappa may be a SCALAR or an ARRAY of
    per-node coupler angles indexed by final node-list position — the
    kappa-aware compile: each nulling step uses the measured kappa of the
    physical node it configures.  Returns (nodes, d, resid)."""
    U = np.array(U, dtype=complex)
    N = U.shape[0]
    p, q = _final_index(N)
    kap_arr = (np.full(p + q, float(kappa))
               if np.isscalar(kappa) else np.asarray(kappa, dtype=float))

    def kap_of(is_right, created):
        # right creation #c -> final index p + (q-1-c); left #c -> c
        return kap_arr[p + (q - 1 - created) if is_right else created]

    V = U.copy()
    right, left = [], []
    resid = 0.0
    for i in range(N - 1):
        for j in range(i + 1):
            if i % 2 == 0:
                kap = kap_of(True, len(right))
                r, m = N - 1 - j, i - j
                u1, u2 = V[r, m], V[r, m + 1]
                # null (V M^H)[r,m]:  u1*conj(A)e^{-iph} = -u2*conj(B)
                th, miss = _solve_theta(
                    abs(u1) / max(abs(u2), 1e-300), kap)
                resid += miss
                A, B, _ = _ABD(th, kap)
                ph = (np.angle(u1 * np.conj(A))
                      - np.angle(-u2 * np.conj(B))) if abs(u2) > 0 else 0.0
                V = V @ _embed(N, m, node_matrix(th, ph, kap)).conj().T
                right.append({"m": m, "theta": th, "phi": ph,
                              "kappa": kap})
            else:
                kap = kap_of(False, len(left))
                r, c = N - 1 - i + j, j
                m = r - 1
                u1, u2 = V[m, c], V[m + 1, c]
                # null (M V)[m+1,c]:  B e^{i ph} u1 = -Dd u2
                th, miss = _solve_theta(
                    abs(u2) / max(abs(u1), 1e-300), kap)
                resid += miss
                _, B, Dd = _ABD(th, kap)
                ph = (np.angle(-Dd * u2)
                      - np.angle(B * u1)) if abs(u1) > 0 else 0.0
                V = _embed(N, m, node_matrix(th, ph, kap)) @ V
                left.append({"m": m, "theta": th, "phi": ph, "kappa": kap})
    d = np.diag(V).copy()
    offdiag = np.max(np.abs(V - np.diag(d)))
    # sweep D through the daggered left factors:
    # M(th,ph)^H diag(d1,d2) = diag(delta1, delta2) M(th, ph')
    chain = []
    for nd in reversed(left):
        m, th, ph, kap = nd["m"], nd["theta"], nd["phi"], nd["kappa"]
        A, B, Dd = _ABD(th, kap)
        d1, d2 = d[m], d[m + 1]
        ph_new = np.angle((np.conj(A) / A) * (B / np.conj(B)) * (d1 / d2))
        delta1 = (np.conj(B) / B) * np.exp(-1j * ph) * d2
        delta2 = (np.conj(Dd) / Dd) * d2
        d[m], d[m + 1] = delta1, delta2
        chain.insert(0, {"m": m, "theta": th, "phi": ph_new,
                         "kappa": kap})
    nodes = chain + list(reversed(right))
    return nodes, d, float(resid + offdiag)


def mesh_matrix(nodes, d, N, lam=LAM0, cal=None, ideal=False,
                out_phases=True):
    """Physical mesh transfer at wavelength lam [um].

    ideal=True -> kappa=pi/4, lossless. cal=NodeCal -> kappa(lam), per-node
    amplitude a(lam), all electrode phases scale as lam0/lam (Kerr dn is
    dispersionless, phase = 2 pi dn L / lam)."""
    if ideal:
        kap, amp, scale = np.pi / 4, 1.0, 1.0
    else:
        kap, amp, scale = cal.kappa(lam), cal.amp(lam), LAM0 / lam
    M = np.eye(N, dtype=complex)
    for nd in nodes:
        M = M @ _embed(N, nd["m"],
                       node_matrix(nd["theta"] * scale, nd["phi"] * scale,
                                   kap, amp))
    if out_phases:
        M = np.diag(np.exp(1j * np.angle(d) * scale)) @ M
    return M


def fidelity(M, U):
    """Global-phase/loss-invariant relative rms element error, net power."""
    g = np.sqrt(np.mean(np.abs(M) ** 2) / np.mean(np.abs(U) ** 2))
    a = np.angle(np.trace(U.conj().T @ M))
    Mn = M * np.exp(-1j * a) / g
    err = np.sqrt(np.mean(np.abs(Mn - U) ** 2) / np.mean(np.abs(U) ** 2))
    return float(err), float(g ** 2)


def haar(N, rng):
    z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def dft(N):
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, k) / N) / np.sqrt(N)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    cal = NodeCal()
    kap = float(cal.kappa(LAM0))
    # 1) sweep identity, imperfect coupler
    th, ph = 1.234, -0.777
    d1, d2 = np.exp(1j * 0.9), np.exp(-1j * 1.7)
    A, B, Dd = _ABD(th, kap)
    L = node_matrix(th, ph, kap).conj().T @ np.diag([d1, d2])
    R = np.diag([(np.conj(B) / B) * np.exp(-1j * ph) * d2,
                 (np.conj(Dd) / Dd) * d2]) @ node_matrix(
        th, np.angle((np.conj(A) / A) * (B / np.conj(B)) * (d1 / d2)), kap)
    print("sweep identity err (kappa=%.4f): %.2e" % (kap, np.max(np.abs(L - R))))
    # 2) exact reconstruction with ideal and with calibrated coupler
    for N in (2, 3, 5, 8, 16):
        U = haar(N, rng)
        for tag, k in (("ideal ", np.pi / 4), ("calib ", kap)):
            nodes, d, res = decompose(U, k)
            M = mesh_matrix(nodes, d, N, ideal=(tag == "ideal "),
                            cal=cal)
            if tag == "calib ":  # compare unitary part: divide out amp
                e, T = fidelity(M, U)
                print(f"N={N:3d} {tag} nodes={len(nodes):4d} resid={res:.1e} "
                      f"rms err={e:.2e}  power T={T:.3f}")
            else:
                print(f"N={N:3d} {tag} nodes={len(nodes):4d} resid={res:.1e} "
                      f"recon err={np.max(np.abs(M - U)):.2e}")
    # 3) DFT_5 (structured target, has exact cross states -> check resid)
    U = dft(5)
    nodes, d, res = decompose(U, kap)
    M = mesh_matrix(nodes, d, 5, cal=cal)
    e, T = fidelity(M, U)
    print(f"DFT5 calib: resid={res:.2e}  rms err={e:.2e}  power T={T:.3f}")

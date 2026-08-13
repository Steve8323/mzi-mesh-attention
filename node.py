"""FDTD-calibrated 2x2 MZI node model for the silica mesh.

Every number here is pinned to a MEEP measurement from the silica-eo-converter
worktree (calib/, copied verbatim; see calib provenance in README):

  - kappa(lambda): effective per-coupler splitting angle INCLUDING the s-bend
    approach coupling, extracted from the full-MZI dn=0 spectrum via
    s0(lambda) = cos(4 kappa)  (two identical couplers, zero internal phase).
  - LP_eff: effective phase-section length (electrode overlap included),
    fit to the measured s(dn) staircase at 1550 nm.
  - a(lambda): per-node amplitude transmission sqrt((P_bar+P_cross)/P_in).

Node convention (directional coupler C, phase on the TOP arm):
  C(k)   = [[cos k, i sin k], [i sin k, cos k]]
  M(th)  = C . diag(e^{i th}, 1) . C          (internal phase th)
  node   = M(th) . diag(e^{i ph}, 1)          (external phase ph at input)

For ideal k = pi/4:  M(th) = i e^{i th/2} [[sin(th/2), cos(th/2)],
                                           [cos(th/2), -sin(th/2)]].
Phases are Kerr-electrode set: dispersionless dn => th(lambda) = th0*lam0/lam.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "calib")
LAM0 = 1.550  # um


def _load_narrow():
    rows = []
    for f in sorted(glob.glob(os.path.join(CAL, "mzi_*_narrow.json"))):
        d = json.load(open(f))
        rows.append(d)
    rows.sort(key=lambda d: d["dn"])
    return rows


class NodeCal:
    """Calibration container; build once, reuse everywhere."""

    def __init__(self):
        rows = _load_narrow()
        base = [r for r in rows if r["dn"] == 0.0][0]
        lam = np.array(base["lam_nm"]) / 1000.0  # um
        order = np.argsort(lam)
        self.lam_grid = lam[order]
        s0 = np.array(base["s"])[order]
        pb = np.array(base["P_bar"])[order]
        pc = np.array(base["P_cross"])[order]
        pin = np.array(base["P_in"])[order]
        # kappa(lambda) from s0 = cos(4 kappa)
        self.kappa_grid = np.arccos(np.clip(s0, -1, 1)) / 4.0
        # amplitude transmission per node
        T = np.clip((pb + pc) / pin, 1e-6, None)
        self.amp_grid = np.sqrt(T)
        # LP_eff from the s(dn) staircase at 1550
        self.dn_tab, self.s_tab = [], []
        for r in rows:
            l = np.array(r["lam_nm"]) / 1000.0
            i = np.argmin(abs(l - LAM0))
            self.dn_tab.append(r["dn"])
            self.s_tab.append(r["s"][i])
        self.dn_tab = np.array(self.dn_tab)
        self.s_tab = np.array(self.s_tab)
        self.LP_eff = self._fit_LP()
        self.model_rms = self._model_rms()

    # -- internals --------------------------------------------------------
    def kappa(self, lam):
        return np.interp(lam, self.lam_grid, self.kappa_grid)

    def amp(self, lam):
        return np.interp(lam, self.lam_grid, self.amp_grid)

    def theta_of_dn(self, dn, lam=LAM0):
        return 2 * np.pi * dn * self.LP_eff / lam

    def dn_of_theta(self, th):
        return th * LAM0 / (2 * np.pi * self.LP_eff)

    def _s_model(self, dn, LP, lam=LAM0):
        th = 2 * np.pi * dn * LP / lam
        M = node_matrix(th, 0.0, self.kappa(lam), 1.0)
        pb, pc = abs(M[0, 0]) ** 2, abs(M[1, 0]) ** 2
        return (pb - pc) / (pb + pc)

    def _fit_LP(self):
        LPs = np.linspace(18.0, 28.0, 2001)
        errs = [
            np.sqrt(np.mean([(self._s_model(d, LP) - s) ** 2
                             for d, s in zip(self.dn_tab, self.s_tab)]))
            for LP in LPs
        ]
        return float(LPs[int(np.argmin(errs))])

    def _model_rms(self):
        e = [self._s_model(d, self.LP_eff) - s
             for d, s in zip(self.dn_tab, self.s_tab)]
        return float(np.sqrt(np.mean(np.square(e))))


def coupler(k):
    return np.array([[np.cos(k), 1j * np.sin(k)],
                     [1j * np.sin(k), np.cos(k)]])


def node_matrix(theta, phi, kappa=np.pi / 4, amp=1.0):
    """Full physical node: amp * C(k).diag(e^{i th},1).C(k).diag(e^{i ph},1)."""
    C = coupler(kappa)
    M = C @ np.diag([np.exp(1j * theta), 1.0]) @ C @ np.diag(
        [np.exp(1j * phi), 1.0])
    return amp * M


def save_model(cal, path):
    out = {
        "LP_eff_um": cal.LP_eff,
        "s_dn_model_rms": cal.model_rms,
        "kappa_1550": float(cal.kappa(LAM0)),
        "power_split_1550": float(np.sin(cal.kappa(LAM0)) ** 2),
        "amp_1550": float(cal.amp(LAM0)),
        "node_power_T_1550": float(cal.amp(LAM0) ** 2),
        "lam_grid_um": cal.lam_grid.tolist(),
        "kappa_grid": cal.kappa_grid.tolist(),
        "amp_grid": cal.amp_grid.tolist(),
        "dn_tab": cal.dn_tab.tolist(),
        "s_tab": cal.s_tab.tolist(),
        "s_model_tab": [cal._s_model(d, cal.LP_eff) for d in cal.dn_tab],
    }
    json.dump(out, open(path, "w"), indent=1)
    return out


if __name__ == "__main__":
    cal = NodeCal()
    out = save_model(cal, os.path.join(HERE, "out", "node_model.json"))
    print(f"LP_eff = {out['LP_eff_um']:.3f} um (nominal 24.0)")
    print(f"s(dn) model rms = {out['s_dn_model_rms']:.4f}")
    print(f"kappa(1550) = {out['kappa_1550']:.4f} rad "
          f"(ideal pi/4 = {np.pi/4:.4f}), power split "
          f"{out['power_split_1550']*100:.1f}%/{ (1-out['power_split_1550'])*100:.1f}%")
    print(f"node power transmission = {out['node_power_T_1550']:.4f}")

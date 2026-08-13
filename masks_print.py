#!/usr/bin/env python
"""Fabrication print pattern — the EXACT geometry MEEP ran (fdtd_e2e.py
'mrr' end-to-end cell), exported as vector artwork + 1-bit print bitmap.

Vector construction: the analytic centerlines y_channel(k, x) (with the
solved meander amplitudes), the bus, and the clock ring are offset by
+-W/2 into polygons sampled at 25 nm steps; the ring is an annulus with
1440 vertices.  This is the same math the MaterialGrid rasterizer used,
so the verification diff (re-raster of the polygons vs the MEEP weights)
must agree to edge pixels only.

Outputs:
  out/mask_optics.gds            GDSII, layer 1 (waveguide etch, dark
                                 field: polygons = SiO2 kept)
  out/mask_optics_1bit.png       1-bit, EXACTLY 50 nm/px (20 px/um),
                                 WHITE = guide; +margin strip with 100 um
                                 scale bar and 10 um corner crosses
  out/mask_optics_scale.json     sidecar: scale, die size, frame origin
  out/mask_verify.json           re-raster vs MEEP MaterialGrid diff
  fig/mask_print_preview.png     dimensioned preview
usage: masks_print.py
"""
import json
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")

# import the e2e geometry with mrr mode WITHOUT running the simulation:
# execute the file head up to the Simulation construction
sys.argv = ["fdtd_e2e.py", "mrr"]
_src = open(os.path.join(HERE, "fdtd_e2e.py")).read()
_head = _src.split("sim = mp.Simulation")[0]
exec(compile(_head, "fdtd_e2e_head", "exec"))
# names now available: W, Y_BUS, X_BUSBEND, y_channel, LTOT, RING_C,
# R_RING_D, wgt (the MEEP raster), RES, L_SLOT, AMP, ...

DX = 0.025          # centerline sampling step (um)
PXUM = 20           # print bitmap: 20 px/um = 50 nm/px


def strip_polygon(xs, ys, width):
    """Constant-width strip around a polyline centerline (normal offset)."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    dx, dy = np.gradient(xs), np.gradient(ys)
    n = np.hypot(dx, dy)
    n[n == 0] = 1.0
    nx, ny = -dy / n, dx / n
    up = np.column_stack([xs + nx * width / 2, ys + ny * width / 2])
    dn = np.column_stack([xs - nx * width / 2, ys - ny * width / 2])
    return np.vstack([up, dn[::-1]])


def annulus_polygons(cx, cy, R, width, nv=1440):
    """Ring as one polygon with a keyhole slit (GDS-safe annulus)."""
    th = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    ro, ri = R + width / 2, R - width / 2
    outer = np.column_stack([cx + ro * np.cos(th), cy + ro * np.sin(th)])
    inner = np.column_stack([cx + ri * np.cos(th), cy + ri * np.sin(th)])
    return np.vstack([outer, outer[:1], inner[:1], inner[::-1], inner[-1:]])


def chunked_strips(xs, ys, width, nmax=3000, seg_um=75.0):
    """Split a long centerline into abutting strip polygons (GDS XY
    records max ~8k vertices; strips have 2x centerline points).
    Chunks share one overlapping sample so the strips join seamlessly."""
    out = []
    i = 0
    while i < len(xs) - 1:
        step = xs[1] - xs[0] if len(xs) > 1 else 0.025
        nseg = min(nmax, max(int(seg_um / max(step, 1e-6)), 50))
        j = min(i + nseg, len(xs))
        out.append(strip_polygon(xs[i:j], ys[i:j], width))
        i = j - 2                      # 2-sample overlap
    return out


def build_polys(dx):
    P = []
    xs = np.arange(0.0, X_BUSBEND[0] + dx, dx)
    P += chunked_strips(xs, np.full_like(xs, Y_BUS), W)
    for k in range(3):
        xs = np.arange(0.0, LTOT + dx, dx)
        ys = np.array([y_channel(k, x) for x in xs])
        ok = ~np.isnan(ys)
        P += chunked_strips(xs[ok], ys[ok], W)
    P.append(annulus_polygons(RING_C[0], RING_C[1], R_RING_D, W,
                              nv=1440 if dx < 0.1 else 720))
    return P


polys = []
# bus
xs = np.arange(0.0, X_BUSBEND[0] + DX, DX)
polys += chunked_strips(xs, np.full_like(xs, Y_BUS), W)
# three channels
for k in range(3):
    xs = np.arange(0.0, LTOT + DX, DX)
    ys = np.array([y_channel(k, x) for x in xs])
    ok = ~np.isnan(ys)
    polys += chunked_strips(xs[ok], ys[ok], W)
# clock ring
polys.append(annulus_polygons(RING_C[0], RING_C[1], R_RING_D, W))

# ---- GDSII (hand-written, minimal, um database with 1 nm precision) ----
def gds_write(path, polys, layer=1):
    def rec(tag, data=b""):
        return struct.pack(">HH", 4 + len(data), tag) + data

    def i16(*v):
        return struct.pack(">%dh" % len(v), *v)

    def i32(*v):
        return struct.pack(">%dl" % len(v), *v)

    def f64(x):
        # GDS 8-byte real
        if x == 0:
            return b"\x00" * 8
        s = 0
        if x < 0:
            s, x = 0x8000, -x
        e = 0
        while x >= 1:
            x /= 16.0
            e += 1
        while x < 1 / 16.0:
            x *= 16.0
            e -= 1
        m = int(x * (1 << 56))
        return struct.pack(">HxxL", s | (e + 64) << 8 | 0, 0)[:2] + \
            m.to_bytes(7, "big") if False else \
            struct.pack(">B", (s >> 8) | (e + 64)) + m.to_bytes(7, "big")

    name = b"MZIMESH\x00"
    out = b""
    out += rec(0x0002, i16(600))                        # HEADER v6
    out += rec(0x0102, i16(*( [2026, 8, 13, 0, 0, 0] * 2 )))  # BGNLIB
    out += rec(0x0206, b"MZI_MRR_E2E\x00")              # LIBNAME
    out += rec(0x0305, f64(1e-3) + f64(1e-9))           # UNITS um/1nm
    out += rec(0x0502, i16(*([2026, 8, 13, 0, 0, 0] * 2)))    # BGNSTR
    out += rec(0x0606, b"TOP\x00")                      # STRNAME
    for p in polys:
        pts = np.asarray(p)
        xy = []
        for x, y in pts:
            xy += [int(round(x * 1000)), int(round(y * 1000))]
        xy += xy[:2]                                    # close
        out += rec(0x0800)                              # BOUNDARY
        out += rec(0x0D02, i16(layer))                  # LAYER
        out += rec(0x0E02, i16(0))                      # DATATYPE
        out += rec(0x1003, i32(*xy))                    # XY
        out += rec(0x1100)                              # ENDEL
    out += rec(0x0700)                                  # ENDSTR
    out += rec(0x0400)                                  # ENDLIB
    open(path, "wb").write(out)


gds_write(os.path.join(OUT, "mask_optics.gds"), polys)

# ---- 1-bit print bitmap at 50 nm/px ------------------------------------
from matplotlib.path import Path as MplPath
MARGIN = 12.0                       # um margin strip for marks
x0, x1 = -MARGIN, LTOT + MARGIN
y0, y1 = -46.0 - MARGIN, 62.0 + MARGIN
nx, ny = int(round((x1 - x0) * PXUM)), int(round((y1 - y0) * PXUM))
gx = x0 + (np.arange(nx) + 0.5) / PXUM
gy = y0 + (np.arange(ny) + 0.5) / PXUM
img = np.zeros((nx, ny), dtype=bool)
def paint(img, polys, gxa, gya):
    for p in polys:
        ia = max(int(np.searchsorted(gxa, p[:, 0].min() - 0.1)) - 1, 0)
        ib = min(int(np.searchsorted(gxa, p[:, 0].max() + 0.1)) + 1,
                 len(gxa))
        ja = max(int(np.searchsorted(gya, p[:, 1].min() - 0.1)) - 1, 0)
        jb = min(int(np.searchsorted(gya, p[:, 1].max() + 0.1)) + 1,
                 len(gya))
        if ib <= ia or jb <= ja:
            continue
        XX, YY = np.meshgrid(gxa[ia:ib], gya[ja:jb], indexing="ij")
        m = MplPath(p).contains_points(
            np.column_stack([XX.ravel(), YY.ravel()]))
        img[ia:ib, ja:jb] |= m.reshape(ib - ia, jb - ja)
polys_paint = build_polys(0.2)   # 0.5 nm chord error,
# far below the 50 nm pixel — same print quality, ~40x faster
paint(img, polys_paint, gx, gy)


def paint_rect(img, xa, xb, ya, yb):
    ia, ib = int((xa - x0) * PXUM), int((xb - x0) * PXUM)
    ja, jb = int((ya - y0) * PXUM), int((yb - y0) * PXUM)
    img[max(ia, 0):ib, max(ja, 0):jb] = True


# 100 um scale bar in the bottom margin; 10 um corner crosses
paint_rect(img, 10, 110, y0 + 2.0, y0 + 4.0)
for cx_, cy_ in ((x0 + 6, y1 - 6), (x1 - 6, y1 - 6),
                 (x0 + 6, y0 + 8), (x1 - 6, y0 + 8)):
    paint_rect(img, cx_ - 5, cx_ + 5, cy_ - 0.5, cy_ + 0.5)
    paint_rect(img, cx_ - 0.5, cx_ + 0.5, cy_ - 5, cy_ + 5)

plt.imsave(os.path.join(OUT, "mask_optics_1bit.png"),
           img.T[::-1], cmap="gray", vmin=0, vmax=1)
json.dump({"px_per_um": PXUM, "nm_per_px": 1000.0 / PXUM,
           "width_um": x1 - x0, "height_um": y1 - y0,
           "frame": "fdtd_e2e layout coords, origin at (%.1f, %.1f) um"
                    % (x0, y0),
           "tone": "WHITE = SiO2 guide kept (dark-field etch mask)",
           "scale_bar_um": 100.0,
           "min_features_um": {"ring_gap": GAP_RING, "tap_gap": GAP_TAP,
                               "coupler_gap": GAP_C}},
          open(os.path.join(OUT, "mask_optics_scale.json"), "w"), indent=1)

# ---- verification: re-raster polygons vs the MEEP MaterialGrid ----------
# rebuild polygon raster on the MEEP grid (RES px/um over sx x sy cell)
mx = (np.arange(wgt.shape[0]) + 0.5) / RES - (0)   # gx of fdtd raster
# fdtd raster coords: gx in 0..sx with layout offset X0; reuse its arrays
img2 = np.zeros_like(wgt, dtype=bool)
GX = (np.arange(wgt.shape[0]) + 0.5) / RES - X0     # layout x
GY = (np.arange(wgt.shape[1]) + 0.5) / RES - sy / 2 + Y_CENTER
paint(img2, polys_paint, GX, GY)
meep = wgt > 0.5
diff = np.logical_xor(img2, meep)
mismatch = diff.sum() / max(meep.sum(), 1)
json.dump({"mismatch_fraction_of_guide_px": float(mismatch),
           "guide_px": int(meep.sum()), "diff_px": int(diff.sum()),
           "pass": bool(mismatch < 0.02)},
          open(os.path.join(OUT, "mask_verify.json"), "w"), indent=1)
print(f"verification: {diff.sum()} / {meep.sum()} px differ "
      f"({100*mismatch:.2f}%) -> {'PASS' if mismatch < 0.02 else 'FAIL'}")

# ---- preview -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5.2), dpi=170)
ax.imshow(img.T[::-1], cmap="gray", origin="upper", aspect="equal",
          extent=[x0, x1, y0, y1])
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title(f"PRINT MASK — waveguide layer (mask 1 of BUILD.md), "
             f"die {x1-x0:.0f} × {y1-y0:.0f} µm, 50 nm/px | min features: "
             f"ring gap {GAP_RING} µm, tap gap {GAP_TAP} µm, coupler gap "
             f"{GAP_C} µm | verify diff {100*mismatch:.2f}%", fontsize=9)
ax.annotate("clock ring R=4.79", (RING_C[0], RING_C[1] - 7),
            color="#E69F00", fontsize=8, ha="center")
ax.annotate("100 µm scale bar", (60, y0 + 6), color="#E69F00",
            fontsize=8, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "mask_print_preview.png"),
            bbox_inches="tight")
print("wrote out/mask_optics.gds, out/mask_optics_1bit.png, "
      "out/mask_optics_scale.json, fig/mask_print_preview.png")

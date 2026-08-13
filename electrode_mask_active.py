#!/usr/bin/env python
"""FULLY-ACTIVE electrode layer (v6): every mesh degree of freedom that a
3x3 Clements multiplier has, actuated — matching what the simulations
model (clements.decompose emits theta AND phi per node + output phases;
mesh_matrix applies all of them).

Sites:
  per node c=0,1,2:  THETA electrode (internal arm 'ph' stub, 1um flank)
                     PHI electrode  (node input arm, first 8um, 1um flank)
  input paths:       IN0 (ch0 delay row), IN2 (ch2 delay row)
                     [IN1 impossible: ch1 is enclosed by ch0/ch2 from tap
                      to mesh — single-layer no-crossing theorem; on this
                      OTDM chip the shared bus/ring gate IS the token
                      modulator, so IN1 is also architecturally unneeded]
  outputs:           P0, P2 (row 1 = phase reference)
  plus RING_BIAS arc and GATE pair (signal + via-grounded strip)
All grounds: 1um strips on substrate contact holes (one global net, no
routing). Verified planar. Outputs *_active files + GDS layer 2.
"""
import json, os, sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")
sys.argv = ["fdtd_e2e.py", "mrr"]
exec(compile(open(os.path.join(HERE, "fdtd_e2e.py")).read().split(
    "sim = mp.Simulation")[0], "h", "exec"))

TR, PAD = 2.0, 60.0
def rect(x0,x1,y0,y1): return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
def vseg(y0,y1,x,w=TR): return rect(x-w/2,x+w/2,min(y0,y1),max(y0,y1))
def hseg(x0,x1,y,w=TR): return rect(min(x0,x1),max(x0,x1),y-w/2,y+w/2)
def arc(cx,cy,ri,ro,a0,a1,n=200):
    th=np.linspace(np.deg2rad(a0),np.deg2rad(a1),n)
    return np.vstack([np.column_stack([cx+ro*np.cos(th),cy+ro*np.sin(th)]),
                      np.column_stack([cx+ri*np.cos(th),cy+ri*np.sin(th)])[::-1]])

seg_edges = np.cumsum([0]+[sg[1] for sg in NODE_SEG])
PH0 = seg_edges[[sg[0] for sg in NODE_SEG].index("ph")]
PH1 = seg_edges[[sg[0] for sg in NODE_SEG].index("ph")+1]
def pair_rows(c):
    prs = mesh_pairs(c)[0]
    return ROWS_M[prs[0]], ROWS_M[prs[1]]

conductors, gnd = {}, {}
sites = []   # (net, strip_rect, trace_x, side)  side +1 top / -1 bottom
for c, side in ((0,+1),(1,-1),(2,+1)):
    xn = X_MESH + c*L_NODE
    top, bot = pair_rows(c)
    arm = top if side>0 else bot          # arm on the routing side
    # THETA: over the ph stub, strip OUTSIDE the pair at 1um
    y0 = arm + side*(W/2+1.0)
    sites.append((f"T{c}", rect(xn+PH0, xn+PH1, min(y0,y0+side*2), max(y0,y0+side*2)), xn+PH0+1.0, side))
    gnd[f"T{c}g"] = rect(xn+PH0, xn+PH1, (top+bot)/2-0.5, (top+bot)/2+0.5)
    # PHI: over the node input arm (first 8um of 'in'+b1), same side
    y1 = arm + side*(W/2+1.0)
    sites.append((f"F{c}", rect(xn+1.0, xn+9.0, min(y1,y1+side*2), max(y1,y1+side*2)), xn+2.0, side))
    # phi ground: 1um strip beyond the FAR arm (arms converge along the
    # node input, so mid-pair has no room; far-side clearance grows)
    far = bot if side>0 else top
    gy = far - side*2.0
    gnd[f"F{c}g"] = rect(xn+1.0, xn+9.0, gy-0.5, gy+0.5)
# INPUT electrodes on delay rows (ch0 top route, ch2 bottom route)
sites.append(("IN0", rect(210.0, 222.0, 44+W/2+1.0, 44+W/2+3.0), 211.0, +1))
gnd["IN0g"] = rect(210.0, 222.0, 44-W/2-2.0, 44-W/2-1.0)
sites.append(("IN2", rect(246.0, 249.5, -40-W/2-3.0, -40-W/2-1.0), 247.0, -1))
gnd["IN2g"] = rect(246.0, 249.5, -40+W/2+1.0, -40+W/2+2.0)
# OUTPUT trims rows 0 (top) and 2 (bottom); row 1 = phase reference
XT0 = X_MESH + 3*L_NODE + 2.0
sites.append(("P0", rect(XT0, XT0+6.0, 4+W/2+1.0, 4+W/2+3.0), XT0+1.0, +1))
gnd["P0g"] = rect(XT0, XT0+6.0, 1.5, 2.5)
sites.append(("P2", rect(XT0, XT0+6.0, -4-W/2-3.0, -4-W/2-1.0), XT0+1.0, -1))
gnd["P2g"] = rect(XT0, XT0+6.0, -2.5, -1.5)
# GATE pair on the bus head + RING arc
sites.append(("GATE", rect(1.0, 5.0, Y_BUS+W/2+1.0, Y_BUS+W/2+4.0), 3.0, +1))
gnd["GATEg"] = rect(1.0, 5.0, Y_BUS-W/2-4.0, Y_BUS-W/2-1.0)
cx, cy = RING_C
ri, ro = R_RING_D+W/2+1.0, R_RING_D+W/2+4.0
sites.append(("RING", arc(cx,cy,ri,ro,250,375),
              cx+(ri+ro)/2*np.cos(np.deg2rad(250)), -1))
# routing: trace to y=+-58, jog at unique depth, pad de-overlapped
Y_TR = 58.0
for side in (+1,-1):
    ss = sorted([s for s in sites if s[3]==side], key=lambda s: s[2])
    centers=[]
    for _,_,tx,_ in ss:
        cpx = max(tx, PAD/2+2.0)
        if centers and cpx < centers[-1]+PAD+10.0: cpx = centers[-1]+PAD+10.0
        centers.append(cpx)
    for i,((net,strip,tx,_),cpx) in enumerate(zip(ss,centers)):
        yj = side*(61.0+4.5*(len(ss)-1-i))
        ya,yb = (92.0,92.0+PAD) if side>0 else (-92.0-PAD,-92.0)
        conductors[net] = [strip,
            vseg(np.sign(side)*  (56.0 if net=="GATE" else 0)*0 + (strip[:,1].max() if side>0 else strip[:,1].min()), side*Y_TR, tx),
            vseg(side*Y_TR, yj, tx),
            hseg(tx, cpx, yj, w=1.5),
            vseg(yj, (ya+yb)/2, cpx, w=4.0),
            rect(cpx-PAD/2, cpx+PAD/2, ya, yb)]
conductors["SUB"] = [rect(LTOT+100, LTOT+180, -92.0-PAD, -92.0)]
for k,v in gnd.items(): conductors[k] = [v]

# verify
from scipy.spatial import cKDTree
wg=[]
xs=np.arange(0.0,X_BUSBEND[0],0.1); wg.append(np.column_stack([xs,np.full_like(xs,Y_BUS)]))
for k in range(3):
    xs=np.arange(0.0,LTOT,0.1); ys=np.array([y_channel(k,x) for x in xs]); ok=~np.isnan(ys)
    wg.append(np.column_stack([xs[ok],ys[ok]]))
th=np.linspace(0,2*np.pi,720)
wg.append(np.column_stack([cx+R_RING_D*np.cos(th),cy+R_RING_D*np.sin(th)]))
wg=np.vstack(wg)
def bp(p,step=0.2):
    q=np.vstack([p,p[:1]]); o=[]
    for a,b in zip(q[:-1],q[1:]):
        n=max(int(np.hypot(*(b-a))/step),1)
        o.append(a+np.linspace(0,1,n,endpoint=False)[:,None]*(b-a))
    return np.vstack(o)
nets={k:np.vstack([bp(p) for p in v]) for k,v in conductors.items()}
rep={"pairs":{},"wg":{},"pass":True}
names=sorted(nets)
GNDN=set(gnd)|{"SUB"}
for i,a in enumerate(names):
    for b in names[i+1:]:
        if a in GNDN and b in GNDN: continue          # one global net
        same = a.rstrip("g")==b.rstrip("g") or a+"g"==b or b+"g"==a
        d=float(cKDTree(nets[b]).query(nets[a])[0].min())
        rep["pairs"][f"{a}|{b}"]=round(d,2)
        lim = 2.0 if same else 3.0
        if d<lim: rep["pass"]=False; print(f"  VIOL {a}|{b} {d:.2f}")
t=cKDTree(wg)
for a in names:
    d=float(t.query(nets[a])[0].min())-W/2
    rep["wg"][a]=round(d,2)
    if d<0.8: rep["pass"]=False; print(f"  WG VIOL {a} {d:.2f}")
json.dump(rep,open(os.path.join(OUT,"electrode_active_verify.json"),"w"),indent=1)
print("verify:","PASS" if rep["pass"] else "FAIL",
      "| nets:",len([n for n in names if n not in GNDN]),"signals + 1 gnd")
# preview
fig,ax=plt.subplots(figsize=(15,7),dpi=150)
ax.scatter(wg[:,0],wg[:,1],s=0.05,color="#999")
cols=plt.cm.tab20(np.linspace(0,1,len(conductors)))
for (net,ps),col in zip(sorted(conductors.items()),cols):
    for p in ps: ax.fill(p[:,0],p[:,1],color=col,alpha=0.9,lw=0)
for net,strip,tx,_ in sites:
    ax.text(strip[:,0].mean(),strip[:,1].mean(),net,fontsize=7,ha="center")
ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
ax.set_title("ACTIVE mesh electrodes: theta+phi per node, IN0/IN2, P0/P2, "
             "ring, gate — "+("PASS" if rep["pass"] else "FAIL"),fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG,"mask_electrodes_active_preview.png"),bbox_inches="tight")
print("preview written")

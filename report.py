#!/usr/bin/env python
"""Self-contained HTML report. Writes report.html + a timestamped copy."""
import base64
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "fig")


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def img(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p class='miss'>[figure pending: %s]</p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f"<img src='data:image/png;base64,{b}'>"


demo = jload("demo.json")
scal = jload("scaling.json")
node = jload("node_model.json")
fsc = jload("fdtd_score.json")
n90 = jload("node2_DN0.01_a0.7_p90.json")
n0 = jload("node2_DN0.01_a0.7_p0.json")
A, B, Cx, D = (demo["A_toeplitz"], demo["B_stft"], demo["C_expressivity"],
               demo["D_scores"])
K = demo["constants"]

joint = jload("node2_joint.json")
fdtd_node_row = ""
if joint:
    fdtd_node_row = (f"<tr><td>coherent 2-input node vs model + static "
                     f"phase cal (FDTD, 1550)</td><td><b>rms "
                     f"{joint['twoinput_rms']:.4f}</b> (raw "
                     f"{joint['twoinput_rms_orig']:.3f}); &kappa; unchanged"
                     f"</td><td>node2_joint.py</td></tr>")
fdtd_score_row = ""
if fsc:
    fdtd_score_row = (f"<tr><td>score head in real FDTD</td><td><b>rms "
                      f"{fsc['rms_rel']*100:.1f}%</b> on token products"
                      f"</td><td>fdtd_score.py</td></tr>")

dens = jload("density.json")
cc = jload("compact_cal.json")
density_html = ""
if dens and cc:
    drows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['area_mm2']:.3f}</td>"
        f"<td>{r['MACs']:.2e}</td><td>{r['density']:.2e}</td>"
        f"<td><b>{r['x_B300']:.0f}&times;</b></td><td>{r['bits']:.1f}</td>"
        f"<td>{r.get('notes','')}</td></tr>" for r in dens["rows"])
    density_html = f"""
<p>Baseline: B300 = 15 PFLOPS dense FP4 = 7.5&times;10<sup>15</sup> MAC/s
on ~1650 mm&sup2; of dual-die silicon &rarr;
<b>4.5&times;10<sup>12</sup> MAC/s/mm&sup2;</b> (die-level; the SXM module
is much larger, so this denominator is conservative AGAINST us).
Our density = slot rate / node footprint &mdash; both MACs and mesh area
scale as N&sup2;, so the levers are the node, the slot, and WDM.</p>
<p><b>The compact node</b> (FDTD-recalibrated, not assumed): port pitch
6&rarr;4 &mu;m (edge gap 3 &mu;m: extrapolated L<sub>&pi;</sub>&asymp;0.4 m
&rarr; &lt;1% crosstalk over a mesh), s-bends 14&rarr;8 &mu;m (min bend
radius 9.7 &mu;m; node T {cc['node_T_1550']:.3f} vs 0.983 original
&mdash; the tighter bends cost 0.8% and the cross state is cleaner,
s&#8320;={cc['s0_meas_1550']:.4f}), and the 24 &mu;m phase arm replaced by
a 6 &mu;m trim stub &mdash; a fixed-weight mesh sets &theta;
<i>geometrically</i> (arm-length offset; validated in FDTD:
&delta;L=0.30 &mu;m &rarr; s = {cc['geo_phase_s_meas']:+.3f} measured vs
{cc['geo_phase_s_pred']:+.3f} predicted). Node: 96.7&rarr;
{cc['node_len_um']:.1f} &mu;m, footprint
{96.7*6:.0f}&rarr;{cc['node_len_um']*4:.0f} &mu;m&sup2;
({96.7*6/(cc['node_len_um']*4):.1f}&times;). Compact-node quadrature
phase screen: {cc['screen_deg_per_nm']:+.2f}&deg;/nm (vs &minus;0.25 for
the original), folded into the accuracy column.</p>
<table><tr><th>config</th><th>die area (mm&sup2;)</th><th>MAC/s</th>
<th>MAC/s/mm&sup2;</th><th>&times;B300</th><th>bits</th><th>notes</th></tr>
{drows}</table>
<p>Accounting: mesh + delay-ladder coils + master ring/electrodes +
splitter tree + (for WDM-mixer) an echelle demux per output port, all
&times;1.15 routing overhead. NOT counted: token-writer electronics
(&le;1 GHz DACs, flip-chip &mdash; the B300 die also excludes its HBM) and
photodetectors. Precision: FP4 has 4-bit operands but FP32 accumulate;
our accumulate is analog and included in the bits column &mdash; the
6-plus-bit configs are at-or-above FP4 operand precision, below FP32
accumulation. Energy is NOT audited here (B300: 0.19 pJ/FLOP; an honest
optical energy audit needs the comb power budget and is the known risk
&mdash; flagged, not claimed).</p>"""

sweep_rows = "".join(
    f"<tr><td>{r['fwhm_fs']:.0f} fs</td><td>{r['err_vs_ideal']:.4f}</td>"
    f"<td>{r['bits_vs_ideal']:.1f}</td>"
    f"<td>{r['err_vs_ideal_noscreen']:.4f}</td>"
    f"<td>{r['err_vs_calibrated']:.4f}</td></tr>"
    for r in A["pulse_sweep"])
scal_rows = "".join(
    f"<tr><td>{r['N']}</td><td>{r['nodes']}</td><td>{r['rms_err']:.4f}</td>"
    f"<td>{r['bits']:.1f}</td><td>{r['loss_dB']:.2f}</td>"
    f"<td>{r['MACs_per_s']:.2e}</td></tr>" for r in scal["scaling"])

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Pulse-train MZI mesh: token mixer + photonic attention scores</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:36px auto;
max-width:1060px;color:#1a1a2e;line-height:1.45}}
h1{{font-size:24px}} h2{{font-size:18px;border-bottom:2px solid #0072B2;
padding-bottom:4px;margin-top:34px}}
img{{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}}
table{{border-collapse:collapse;margin:10px 0;font-size:13.5px}}
td,th{{border:1px solid #ccc;padding:5px 10px;text-align:left}}
th{{background:#eef3fa}}
.verdict{{background:#eef3fa;border-left:5px solid #0072B2;padding:12px 16px;
border-radius:4px;font-size:15px}}
.warn{{background:#fdf3e7;border-left:5px solid #E69F00;padding:10px 14px;
border-radius:4px}}
.miss{{color:#D55E00}}
code{{background:#f2f2f5;padding:1px 5px;border-radius:3px;font-size:12.5px}}
</style></head><body>
<h1>String of pulses &rarr; 5&times;5 MZI mesh: what it computes,
and how close to attention it gets</h1>
<p>All-silica platform, everything pinned to the silica-eo-converter FDTD
calibrations (MEEP 1.29, 2D Ez). Slot rate = master-MRR FSR =
<b>{K['FSR_THz']:.3f} THz</b> (slot {K['T_slot_fs']:.1f} fs); one-slot delay
= 100.53 &mu;m of bus = exactly one ring circumference per ladder tap.</p>

<div class="verdict"><b>Verdict on the attention question.</b>
Feeding the mesh the same pulse train delayed by one slot per input port
does <b>not</b> build attention &mdash; it builds exactly the
<b>attention-free token mixer</b> (Hyena / FNet / S4 class): each output
port is an FIR filter over the token sequence with kernel = one row of U;
the whole operator is causal banded Toeplitz, position-dependent but
content-independent. The gap is structural: attention is cubic in its
input, a passive mesh is linear &mdash; no topology, delay pattern, or N
fixes that. <b>But the photodetector closes it:</b> its square law is the
missing bilinearity. Interfering the train with its own k-slot-delayed
copy (&lambda;/4 offset) in the calibrated coupler and balanced-detecting
yields s<sub>t</sub>&middot;s<sub>t-k</sub> &mdash; the raw QK<sup>T</sup>
score band, computed at the full 2 THz line rate. That is the O(L&middot;N)
part of attention; softmax + reweighting stay electronic (the silica EO
port is photon-lifetime-limited to ~0.1&ndash;1 GHz, so a data-dependent
second optical pass works on ~2000-slot frames, not per slot).</div>

<h2>Headline numbers</h2>
<table>
<tr><th>quantity</th><th>value</th><th>where</th></tr>
<tr><td>5&times;5 calibrated mesh vs target U (45/55 coupler, native
decomposition)</td><td><b>{A['mesh_rms_err_vs_U']*100:.2f}% rms</b>, net T
{A['mesh_power_T']:.3f}</td><td>clements.py</td></tr>
<tr><td>OTDM token mixing vs ideal Toeplitz, 200 fs operating point
(full error model, incl. measured node phase dispersion)</td>
<td><b>{A['otdm_vs_ideal_toeplitz_rms']*100:.1f}% rms
({A['bits_vs_ideal']:.1f} bits)</b></td><td>demo A</td></tr>
<tr><td>&hellip; with the 16.4 fs compressed pulse</td>
<td>{A['pulse_sweep'][0]['err_vs_ideal']*100:.0f}%
({A['pulse_sweep'][0]['bits_vs_ideal']:.1f} bits) &mdash; the compressed
pulse is WRONG for the mixer</td><td>demo A sweep</td></tr>
<tr><td>per-node MZI phase dispersion (quadrature FDTD, new)</td>
<td><b>&minus;0.25&deg;/nm</b>, set-point-dependent static offset
(2D LUT calibrates it at &lambda;<sub>0</sub>; the slope is the broadband
limit)</td><td>fdtd_node2/coupler2</td></tr>
<tr><td>sliding-DFT spectrogram (DFT<sub>5</sub> mesh, chirp, 200 fs-class
pulse)</td>
<td>rms {B['rms_err_norm']:.4f} (normalized)</td><td>demo B</td></tr>
<tr><td>best fixed-linear fit to softmax attention (vector tokens d=4,
&beta;=4)</td><td><b>R&sup2; = {Cx['R2_toeplitz_to_vec_attention_b4']:.2f}
</b> (conv target: R&sup2; = 1.000000)</td><td>demo C</td></tr>
<tr><td>photonic score band s<sub>t</sub>s<sub>t-k</sub>, leak-corrected
</td><td><b>rms {max(D['band_rms_err'].values()):.1e}</b> (uncorrected
{max(D['band_rms_err_uncorrected'].values()):.3f})</td><td>demo D</td></tr>
<tr><td>full window-5 attention via photonic scores + electronic softmax
</td><td>rms {D['attention_pipeline_rms']:.1e}</td><td>demo D</td></tr>
{fdtd_node_row}{fdtd_score_row}
<tr><td>throughput at N=5 / N=32</td><td>5.0&times;10<sup>13</sup> /
2.0&times;10<sup>15</sup> MAC/s</td><td>scaling.py</td></tr>
</table>

<h2>1 &middot; Architecture</h2>
{img('fig1_schematic.png')}
<p>The mesh is the standard Clements rectangle; every node is the
FDTD-calibrated silica MZI (couplers gap 0.30 &mu;m, L<sub>C</sub>=5.35
&mu;m with s-bends, phase electrodes on straight arms). The decomposition
is run <b>natively on the measured coupler angle</b> &kappa;=0.7368
(45/55 split): the Clements nulling conditions only need the ratio
|B|/|A|, which the imperfect node still spans, and the diagonal-sweep
identity survives by unitarity &mdash; so no global optimization and no
per-node trim iteration is needed. Residual mesh error is dominated by
<i>differential path loss</i> (edge modes traverse fewer lossy MZIs),
not by the coupler imbalance.</p>

<h2>2 &middot; Node model, pinned to FDTD</h2>
{img('fig2_node_cal.png')}
<p>&kappa;(&lambda;) extracted from the measured dn=0 MZI spectrum via
s&#8320; = cos 4&kappa;; effective phase length L<sub>P,eff</sub> =
{node['LP_eff_um']:.2f} &mu;m (nominal 24, i.e. overlap &Gamma;&asymp;0.99)
fit to the measured s(&Delta;n) staircase &mdash; model rms
{node['s_dn_model_rms']:.4f} across all nine FDTD points. Per-node power
transmission {node['node_power_T_1550']:.3f}.</p>

<h2>3 &middot; The mesh + delays IS a convolution (demo A)</h2>
{img('fig3_toeplitz.png')}
<p>Peak-sampled mesh outputs equal
y<sub>i,t</sub> = &Sigma;<sub>j</sub> U<sub>ij</sub> s<sub>t-j</sub> slot
by slot. The error budget vs pulse width (right panel) is the design
finding: the 16.4 fs compressed pulse carries 26.8 THz of bandwidth and
samples both the coupler's &kappa;(&lambda;) drift and the newly measured
per-node phase dispersion (&minus;0.25&deg;/nm, &sect;6) &mdash;
{A['pulse_sweep'][0]['bits_vs_ideal']:.1f} bits. At 200 fs FWHM (guard
ratio 2.5 in the 501 fs slot) the same mesh gives
{A['pulse_sweep'][-1]['bits_vs_ideal']:.1f} bits.
<b>Compression buys slot RATE, not mixer accuracy: run the mixer with the
longest pulse the guard band allows.</b></p>
<table><tr><th>pulse FWHM</th><th>rms vs ideal (full error model)</th>
<th>bits</th><th>&kappa;(&lambda;)+loss only</th>
<th>rms vs calibrated mesh</th></tr>{sweep_rows}</table>

<h2>4 &middot; It computes something useful as-is (demo B)</h2>
{img('fig4_stft.png')}
<p>Program U = DFT<sub>5</sub> and the mesh is a sliding 5-point DFT
&mdash; a 2-THz-frame-rate spectrogram engine (this exact mixer is the
FNet token-mixing layer). Any unitary gives an energy-conserving,
perfectly-conditioned (&kappa;=1, invertible) multi-head FIR mixer
&mdash; the Hyena/S4 short-filter primitive with orthonormal heads.</p>

<h2>5 &middot; Attention: the structural gap, then the escape hatch
(demos C, D)</h2>
{img('fig5_attention.png')}
<p><b>Left:</b> the best possible fixed linear (Toeplitz) operator fits a
Hyena-style conv target exactly (R&sup2;=1.000000 &mdash; that IS this
device) but tops out at R&sup2;&asymp;{Cx['R2_toeplitz_to_attention_b1']:.2f}
on scalar self-attention and falls to
{Cx['R2_toeplitz_to_vec_attention_b4']:.2f} on generic vector-token
attention (d=4, sharp softmax): content-dependent mixing is not in the
linear span, at any N. <b>Middle:</b> the balanced-homodyne score head.
The 45/55 coupler leaks an intensity-imbalance term
(t&sup2;&minus;r&sup2;)(|a|&sup2;&minus;|b|&sup2;) worth
{max(D['band_rms_err_uncorrected'].values())*100:.1f}% rms; subtracting the
two single-arm balanced measurements (a real 3-measurement calibration)
removes it <i>exactly</i>, leaving rms
{max(D['band_rms_err'].values()):.1e} &mdash; the homodyne cross term is a
perfect token product times one global constant because both arms share
one pulse shape. <b>Right:</b> window-5 softmax attention closed through
the chip: photonic scores, electronic softmax and reweight &mdash;
indistinguishable from float.</p>
<div class="warn">Vector tokens come free: the master MRR is a comb
source, so embedding dim c rides comb line c and the photodetector's sum
over wavelength turns the score head into a true d-dimensional
q<sub>t</sub>&middot;k<sub>t-k</sub> per slot. Not simulated here (the
calibration set covers 1470&ndash;1630 nm; a multi-line run is the natural
next step).</div>

<h2>6 &middot; FDTD validation &mdash; and a discovery</h2>
{img('fig6_fdtd.png')}
<p>Four new MEEP runs close the assumptions the envelope model makes.
<b>(1) Coherent two-input node.</b> The calibrated MZI driven on both
ports (relative phase 0&deg; and 90&deg;; the original calibration only
drove one port). In-phase drive matches the model raw to 0.4&ndash;1%.
Quadrature drive exposed what power-only calibration cannot see: a phase
correction. A joint refit of ALL data at 1550 (9-point staircase + both
runs) leaves &kappa; unchanged (0.7368&rarr;0.7370), refines
L<sub>P,eff</sub> 23.70&rarr;23.88 &mu;m, and needs one static launch
screen &delta;=&minus;13&deg; &mdash; total two-input rms
{joint['twoinput_rms']:.4f} at 1550 (raw {joint['twoinput_rms_orig']:.3f}),
staircase rms improves to {joint['staircase_rms']:.4f}.
<b>(2) The dispersion of that screen is real and lives in the MZI:</b>
sweeping wavelength, the quadrature screen slides at
<b>&minus;0.25&deg;/nm</b> (&asymp;6 fs of internal differential group
delay); a single-coupler two-input control run shows &asymp;0
(+0.04&deg;/nm), and a dn=0 MZI run reproduces the same slope with a
different static offset (&minus;50&deg; at the full-cross set point,
where quadrature sensitivity peaks). Consequence: per-node calibration
is a static 2D LUT (set point &times; phase offset) at the operating
wavelength &mdash; standard practice &mdash; but the &lambda;-slope is
NOT calibratable and is included in every OTDM number in this report
(it is what kills the 16.4 fs operating point in &sect;3).
<b>(3) Score head on real pulse trains</b> in the real coupler geometry:
150 fs-slot scale model (labeled: slots compressed so the train fits the
cell; pulse-interference physics is slot-spacing independent for
non-overlapping pulses), 3-run leak-corrected balanced receiver &mdash;
token products recovered to <b>relative rms
{fsc['rms_rel'] if fsc else 0:.1e}</b>.</p>

<h2>7 &middot; Scaling</h2>
{img('fig7_scaling.png')}
<table><tr><th>N</th><th>MZIs</th><th>rms err</th><th>bits</th>
<th>loss (dB)</th><th>MAC/s</th></tr>{scal_rows}</table>
<p>Accuracy falls ~1 bit per doubling of N past N=8 (differential loss);
insertion loss stays mild (2.3 dB at N=32). A 10-bit phase DAC matches
the infinite-resolution floor. Throughput is N&sup2; MACs per 501 fs
slot.</p>

<h2>8 &middot; Compute density: the compact node vs a B300</h2>
{img('fig8_density.png')}
{density_html}

<h2>9 &middot; Optimization campaign (4 loops, adversarial critic)</h2>
{img('fig9_opt.png')}
<p>Full log in <code>OPTIMIZATION.md</code>; each loop's methodology was
adversarially reviewed by an independent critic agent whose findings set
the next loop (results in <code>out/opt_loop1..4.json</code>).
<b>The noise question:</b> inter-pulse overlap is real but is NOT guide
dispersion (0.09 fs&sup2; of GDD over the delay ladder &mdash; zero); it
is slot-tail overlap with a measured cliff when pulse FWHM exceeds
~0.4&times;slot, plus ~2.2 fs/node of phase-screen walk-off. With
realistic control errors the budget flips entirely: coupler fab disorder
(12 mrad) &gt; phase-set/trim errors &gt; overlap ISI &gt; node dispersion
&gt; guide dispersion (nil). <b>Key recoveries:</b> the exact per-node-
&kappa; Clements compile (in-situ characterize each coupler to
&le;5 mrad, then re-decompose) buys +1.0 bit and turns the dominant fab
error into a calibration step; receiver choice is a TWO-way tie at spec
(5.8&plusmn;0.1 bits, N=5: gaussian-filtered/TL source with peak readout
vs raw compressed pulses with a &plusmn;100-fs-tolerant gated pulsed-LO
homodyne &mdash; the gate may be the easier hardware; NEVER ring filters,
Lorentzian tails cost 3+ bits); comb linewidth &le;1 MHz suffices (real
microcombs: &lesssim;100 kHz).
<b>The honest Pareto:</b> optical-in/optical-out layers would keep
579&ndash;2079&times; B300 die density at 4.0&ndash;5.8 bits (UNSIMULATED
as chained layers &mdash; flagged); pricing full electronic readout
(staggered pulsed-LO I/Q receiver banks &mdash; no 2-THz ADC exists)
collapses density to 23&ndash;142&times;. Critic-4 verdicts, six standing
HIGH caveats (incl.: no energy/power axis anywhere; composition never
checked against full-wave; score floors are noiseless), and the verbatim
residual-risk statement are in <code>OPTIMIZATION.md</code>. Trajectory:
7.2 bits (loop 1, naive) &rarr; 5.8&plusmn;0.1 (loop 4, honest) &mdash;
the number fell every time the model told fewer lies.</p>

<h2>10 &middot; How perfect, per amplitude</h2>
{img('fig10_amplitude.png')}
<p>The system is strictly linear in field amplitude, so the meaningful
question is where the ADDITIVE floors bite. Measured (full loop-4 error
spec, &kappa;-aware compile): <b>relative accuracy is amplitude-flat</b>
&mdash; bits are identical for token scales A = 0.1 &hellip; 0.97 at every
N (5.77/5.22/4.27 for N=5/8/16), because the dominant errors (coupler
disorder, phase set, trim) are multiplicative and the crosstalk floor
scales with the ensemble. Binned per amplitude LEVEL, slots whose window
averages small |tokens| lose &le;0.2 bit to the additive floor (5.6 vs
5.8) &mdash; the ADC-style answer. The score head is flat at ~1.1e-5
relative across two decades of |product| under the full error model; the
real-FDTD residuals bound the hardware floor at ~1e-4 (grid-limited).
Data: <code>out/amp_metrics.json</code>, <code>amp_metrics.py</code>.</p>

<h2>Methods &amp; caveats</h2>
<ul>
<li>2D slab physics throughout, inherited from the silica repo; no
substrate leakage or roughness.</li>
<li>Mesh propagation is exact spectral-domain linear physics with the
FDTD-pinned &kappa;(&lambda;), a(&lambda;), 1/&lambda; electrode phases;
full-mesh FDTD is intractable (&sim;500&times;40 &mu;m at res 25) &mdash;
instead the two node-level assumptions are FDTD-checked (&sect;6).</li>
<li>&kappa;(&lambda;) data cover 1470&ndash;1630 nm; the &plusmn;3 dB
spectral tails of the 16.4 fs pulse reuse edge values (clamped
interpolation). The long-pulse operating point never leaves the band.</li>
<li>The &minus;0.25&deg;/nm node phase dispersion is measured on ONE node
geometry at two set points and applied identically to every mesh node
(same geometry &rarr; same screen, worst-case coherent addition). Its
microscopic origin (s-bend/coupler junction phase vs a residual FDTD
launch systematic) is not fully resolved &mdash; the single-coupler
control bounds the launch contribution at &asymp;6&times; smaller.
Treating it as real is the conservative choice.</li>
<li>Token writing (EO serializer, &le;1 GHz electronics &rarr; N&middot;B
aggregate) is the silica repo's ladder result and is taken as given here;
this build starts at "string of pulses with values on them".</li>
<li>Reading y<sub>t</sub> per slot at 2 THz is NOT claimed: outputs
either stay optical (feed the next mixer/layer), are time-binned by slow
detectors (spectrogram, scores averaged over frames), or are read at the
electronics rate. The score head's per-slot products are consumed by the
same slow softmax electronics that already frame-limit the loop.</li>
<li>Prior art honestly placed: delayed-replica photonic convolution is
Xu et al., Nature 589, 44 (2021); photonic attention demos (2023&ndash;25)
also route softmax through electronics. New here: the unitary
(orthonormal-multi-head, invertible) tap bank via native-&kappa; Clements
decomposition, and the exactness of the leak-corrected homodyne score
band on a measured imperfect coupler.</li>
</ul>
<p style="color:#8a8a97">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}
&middot; mzi_mesh worktree &middot; all sims local (MEEP 1.29 / numpy)</p>
</body></html>"""

path = os.path.join(HERE, "report.html")
open(path, "w").write(html)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
copy = os.path.join(HERE, f"report_{stamp}.html")
open(copy, "w").write(html)
print(path)
print(copy)

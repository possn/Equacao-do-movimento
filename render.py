import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ============================================================
# OUTPUT
# ============================================================
OUT = "Equacao_do_Movimento_Spontanea_Simples_60s.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60
W_IN, H_IN = 12.8, 7.2
DPI = 120

# ============================================================
# CYCLE (3 fases, loop)
# CRF (pausa) -> Insp -> Pausa fim-insp -> Exp -> Pausa fim-exp
# ============================================================
T_HOLD_EE  = 1.0
T_INSP     = 2.0
T_HOLD_EI  = 1.0
T_EXP      = 2.0
T_HOLD_EE2 = 1.0
T_CYCLE = T_HOLD_EE + T_INSP + T_HOLD_EI + T_EXP + T_HOLD_EE2  # 7 s

# ============================================================
# "Fisiologia didáctica" (espontânea)
# Convenção simples (ensino):
#   Paw (boca) = 0 cmH2O
#   Equação do movimento: Paw = R*Flow + E*V - Pmus
#   -> como Paw=0:  Pmus = R*Flow + E*V
#
# R em cmH2O/(L/s), C em L/cmH2O, E = 1/C
# ============================================================
R = 5.0
C = 0.12
E = 1.0 / C

# esforço muscular (amplitude)
PMUS_PEAK = 6.0  # cmH2O (didáctico)

# volume alvo só para estética (não mexe na dinâmica de sinais)
VT_TARGET = 0.5  # L

# ============================================================
# Helpers
# ============================================================
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def phase_in_cycle(tau):
    a = T_HOLD_EE
    b = a + T_INSP
    c = b + T_HOLD_EI
    d = c + T_EXP
    if tau < a:
        return "hold_ee", tau / max(T_HOLD_EE, 1e-6)
    if tau < b:
        return "insp", (tau - a) / max(T_INSP, 1e-6)
    if tau < c:
        return "hold_ei", (tau - b) / max(T_HOLD_EI, 1e-6)
    if tau < d:
        return "exp", (tau - c) / max(T_EXP, 1e-6)
    return "hold_ee2", (tau - d) / max(T_HOLD_EE2, 1e-6)

def pmus_of_tau(tau):
    """
    Pmus >0 representa "força inspiratória" (queda pleural).
    0 em CRF, sobe na inspiração, mantém na pausa, e desce na expiração.
    """
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ee2"):
        return 0.0
    if ph == "insp":
        return PMUS_PEAK * smoothstep(x)
    if ph == "hold_ei":
        return PMUS_PEAK
    # expiração: relaxamento
    return PMUS_PEAK * (1.0 - smoothstep(x))

def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def history(t, window, fps):
    t0 = max(0.0, t - window)
    n = int(max(160, min(int(window*fps), int((t - t0)*fps + 1))))
    return np.linspace(t0, t, n)

# ============================================================
# Precompute one-cycle RC dynamics
# Dynamics: 0 = R*Flow + E*V - Pmus
# -> Flow = (Pmus - E*V)/R
# ============================================================
N_PRE = 5000
tau_grid = np.linspace(0.0, T_CYCLE, N_PRE)
dt = tau_grid[1] - tau_grid[0]

pmus_grid = np.array([pmus_of_tau(tau) for tau in tau_grid])

V = np.zeros_like(tau_grid)      # L above CRF
Flow = np.zeros_like(tau_grid)   # L/s

for k in range(1, len(tau_grid)):
    pm = pmus_grid[k-1]
    flow = (pm - E*V[k-1]) / R
    V[k] = V[k-1] + flow * dt
    Flow[k] = flow

# scale volume to VT_TARGET for nicer lung inflation (keeps shapes)
Vmin, Vmax = float(np.min(V)), float(np.max(V))
if (Vmax - Vmin) > 1e-9:
    V_scaled = (V - Vmin) * (VT_TARGET / (Vmax - Vmin))
else:
    V_scaled = V.copy()

Flow_scaled = np.gradient(V_scaled, dt)

# compute terms (cmH2O)
Pres = R * Flow_scaled          # resistivo
Pel  = E * V_scaled             # elástico
Pmus_scaled = Pres + Pel        # because Paw=0

def interp(arr, tau):
    tau = tau % T_CYCLE
    return float(np.interp(tau, tau_grid, arr))

def v_of_tau(tau): return interp(V_scaled, tau)
def f_of_tau(tau): return interp(Flow_scaled, tau)
def pres_of_tau(tau): return interp(Pres, tau)
def pel_of_tau(tau): return interp(Pel, tau)
def pmus_term_of_tau(tau): return interp(Pmus_scaled, tau)

# ============================================================
# Simple lung drawing (clean, big, no clutter)
# ============================================================
def draw_lung(ax, inflate=0.0):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # thorax frame
    ax.add_patch(Rectangle((0.08, 0.08), 0.84, 0.84, fill=False, lw=3, edgecolor="#111827", alpha=0.75))
    ax.text(0.50, 0.915, "Ventilação espontânea", ha="center", va="center",
            fontsize=14, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.90))

    # lung shapes (2 ellipses) – inflate changes size slightly
    s = 1.0 + 0.18*np.clip(inflate, 0.0, 1.0)
    fill = (0.97, 0.78, 0.82)
    edge = (0.60, 0.20, 0.30)

    ax.add_patch(Ellipse((0.43, 0.58), 0.28*s, 0.45*s, angle=8,
                         facecolor=fill, edgecolor=edge, lw=4))
    ax.add_patch(Ellipse((0.57, 0.58), 0.28*s, 0.45*s, angle=-8,
                         facecolor=fill, edgecolor=edge, lw=4))

    # trachea
    ax.add_patch(FancyBboxPatch((0.485, 0.75), 0.03, 0.10,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="#111827", edgecolor="#111827"))

    # diaphragm (moves down with inspiration)
    dia = 0.20 - 0.10*np.clip(inflate, 0.0, 1.0)
    xs = np.linspace(0.14, 0.86, 260)
    arch = dia + 0.06*np.sin(np.pi*(xs-0.14)/(0.86-0.14))
    ax.plot(xs, arch, lw=7, color="#111827")
    ax.text(0.86, dia+0.02, "Diafragma", ha="right", va="center",
            fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    # airflow arrow (depends on flow sign)
    # red for inspiratory, green for expiratory, grey for zero
    return

# ============================================================
# RENDER
# ============================================================
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "24"]
)

HIST = 10.0
total_frames = int(DURATION_S * FPS)

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE
    ph, _ = phase_in_cycle(tau)

    Vnow = v_of_tau(tau)
    Fnow = f_of_tau(tau)
    Pres_now = pres_of_tau(tau)
    Pel_now  = pel_of_tau(tau)
    Pmus_now = pmus_term_of_tau(tau)

    # time history
    th = history(t, HIST, FPS)
    tau_h = np.array([tt % T_CYCLE for tt in th])
    Vh = np.array([v_of_tau(x) for x in tau_h])
    Fh = np.array([f_of_tau(x) for x in tau_h]) * 60.0  # L/min

    Pres_h = np.array([pres_of_tau(x) for x in tau_h])
    Pel_h  = np.array([pel_of_tau(x) for x in tau_h])
    Pmus_h = Pres_h + Pel_h

    # layout: 2 columns
    fig.clf()
    gs = fig.add_gridspec(
        2, 2,
        left=0.04, right=0.985, top=0.92, bottom=0.08,
        wspace=0.18, hspace=0.28,
        width_ratios=[1.05, 1.00],
        height_ratios=[1.00, 1.00]
    )

    ax_lung  = fig.add_subplot(gs[:, 0])     # big left
    ax_terms = fig.add_subplot(gs[0, 1])     # top-right: equation + bars
    ax_wave  = fig.add_subplot(gs[1, 1])     # bottom-right: small waveforms

    # -------------------------
    # LEFT: lung + diaphragm
    # -------------------------
    inflate = float(np.clip(Vnow / max(VT_TARGET, 1e-6), 0.0, 1.0))
    draw_lung(ax_lung, inflate=inflate)

    # arrow + phase badge
    if Fnow > 1e-4:
        col = "#dc2626"
        ax_lung.annotate("", xy=(0.50, 0.58), xytext=(0.92, 0.58),
                         arrowprops=dict(arrowstyle="->", lw=5, color=col))
        ax_lung.text(0.92, 0.63, "Ar entra", ha="right", fontsize=12, weight="bold",
                     color=col, bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88))
    elif Fnow < -1e-4:
        col = "#16a34a"
        ax_lung.annotate("", xy=(0.92, 0.58), xytext=(0.50, 0.58),
                         arrowprops=dict(arrowstyle="->", lw=5, color=col))
        ax_lung.text(0.92, 0.63, "Ar sai", ha="right", fontsize=12, weight="bold",
                     color=col, bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88))
    else:
        ax_lung.text(0.92, 0.61, "Fluxo = 0", ha="right", fontsize=12, weight="bold",
                     color="#6b7280", bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88))

    phase_txt = {
        "hold_ee":  "CRF (pausa)",
        "insp":     "INSPIRAÇÃO",
        "hold_ei":  "Fim INSP (pausa)",
        "exp":      "EXPIRAÇÃO",
        "hold_ee2": "CRF (pausa)",
    }[ph]
    ax_lung.text(0.10, 0.12, phase_txt,
                 fontsize=14, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#111827", edgecolor="none", alpha=0.90),
                 color="white")

    ax_lung.text(0.10, 0.06, f"V = {Vnow:.2f} L   |   Fluxo = {Fnow*60:.0f} L/min",
                 fontsize=11, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
                 color="#111827")

    # -------------------------
    # TOP-RIGHT: equation + bars
    # -------------------------
    ax_terms.set_xlim(0, 1)
    ax_terms.set_ylim(0, 1)
    ax_terms.axis("off")

    # equation (big, clean)
    ax_terms.text(0.02, 0.92, "Equação do movimento (espontânea)", fontsize=14, weight="bold", color="#111827")
    ax_terms.text(0.02, 0.78, "Paw = R·Fluxo + E·V − Pmus", fontsize=16, weight="bold", color="#111827")
    ax_terms.text(0.02, 0.66, "Como Paw = 0 (boca):   Pmus = R·Fluxo + E·V", fontsize=14, weight="bold", color="#111827")

    # numeric values
    ax_terms.text(0.02, 0.55, f"R·Fluxo  (resistivo) = {Pres_now:+.1f} cmH₂O", fontsize=12, weight="bold", color="#dc2626")
    ax_terms.text(0.02, 0.48, f"E·V      (elástico)  = {Pel_now:+.1f} cmH₂O", fontsize=12, weight="bold", color="#7c3aed")
    ax_terms.text(0.02, 0.41, f"Pmus     (esforço)   = {Pmus_now:+.1f} cmH₂O", fontsize=12, weight="bold", color="#111827")

    # bar chart (stacked) – occupies big area, no text overlap
    # Scale for consistent look
    max_scale = max(8.0, float(np.max(Pmus_h))*1.15)

    # base frame
    bx0, by0, bw, bh = 0.62, 0.18, 0.34, 0.64
    ax_terms.add_patch(Rectangle((bx0, by0), bw, bh, fill=False, lw=2.2, edgecolor="#111827", alpha=0.85))
    ax_terms.text(bx0 + bw/2, by0 + bh + 0.03, "Contributos (cmH₂O)", ha="center", fontsize=11, weight="bold", color="#111827")

    # map value to height
    def h_of(val):
        return (val / max_scale) * (bh*0.92)

    # stacked bar up from baseline
    base = by0 + 0.04
    pres_h = max(0.0, h_of(Pres_now))
    pel_h  = max(0.0, h_of(Pel_now))
    total_h = pres_h + pel_h

    ax_terms.add_patch(Rectangle((bx0+0.08, base), 0.10, pres_h, facecolor="#fecaca", edgecolor="#dc2626", lw=1.8))
    ax_terms.add_patch(Rectangle((bx0+0.08, base+pres_h), 0.10, pel_h, facecolor="#ede9fe", edgecolor="#7c3aed", lw=1.8))

    # total marker line
    ax_terms.plot([bx0+0.06, bx0+0.24], [base+total_h, base+total_h], color="#111827", lw=2.2)
    ax_terms.text(bx0+0.26, base+total_h, f"{Pmus_now:.1f}", va="center", fontsize=11, weight="bold", color="#111827")

    # legend
    ax_terms.text(bx0+0.08, by0+0.05, "resistivo", fontsize=10, color="#dc2626", weight="bold")
    ax_terms.text(bx0+0.18, by0+0.05, "elástico",  fontsize=10, color="#7c3aed", weight="bold")

    # one-line humour (minimal, not cringe)
    ax_terms.text(0.02, 0.16, "Ideia-chave: no início ΔP é maior → fluxo acelera; depois desacelera até 0.", fontsize=11, weight="bold", color="#374151")

    # -------------------------
    # BOTTOM-RIGHT: waveforms (simple, 2 curves)
    # -------------------------
    ax_wave.set_title("Fluxo e Volume (janela ~10 s)", fontsize=12, weight="bold")
    ax_wave.plot(th, Fh, lw=2.4, label="Fluxo (L/min)")
    ax_wave.plot(th, Vh, lw=2.4, label="Volume (L)")
    ax_wave.axhline(0, color="#9ca3af", lw=1.1)
    ax_wave.grid(True, alpha=0.20)
    ax_wave.set_xlabel("Tempo (s)")
    ax_wave.legend(loc="upper right", fontsize=10, frameon=True)

    fig.suptitle("Equação do Movimento — Ventilação Espontânea (visual simples)", fontsize=16, weight="bold", y=0.985)
    fig.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)

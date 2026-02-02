import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Rectangle, FancyBboxPatch

# ============================================================
# OUTPUT
# ============================================================
OUT = "Equacao_do_Movimento_AulaBasica_ICU.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 24
DURATION_S = 60

# ============================================================
# Breath model (volume-control, inspiratory flow decelerating)
# Use a simple flow waveform: constant flow during insp then 0 hold plateau, then passive exp.
# This produces a clear peak (R*flow) and plateau (E*V + PEEP).
# ============================================================
RR = 12.0                 # breaths/min
Ttot = 60.0 / RR          # s
Ti = 1.0                  # inspiratory time (s)
Thold = 0.5               # inspiratory hold (plateau) (s)
Te = max(0.5, Ttot - Ti - Thold)

VT = 0.50                 # L
PEEP = 5.0                # cmH2O

# Flow during insp: constant (square) for didactic peak/plateau separation
FLOW_INSP = VT / Ti       # L/s (so V reaches VT at end insp)

# Passive expiration time constant (for V decay)
TAU_EXP = 1.2

# ============================================================
# Scenarios: Normal, Obstructive (R↑), Restrictive/ARDS (E↑)
# R in cmH2O/(L/s), E in cmH2O/L
# ============================================================
SCENARIOS = [
    dict(name="Normal",            R=8.0,  E=14.0, color="#2563eb"),
    dict(name="Obstrutivo (R↑)",   R=20.0, E=14.0, color="#dc2626"),
    dict(name="Restritivo/ARDS (E↑)", R=8.0,  E=28.0, color="#7c3aed"),
]

# Segment timing: each scenario runs for 20s (3 x 20 = 60)
SEG_DUR = DURATION_S / 3.0

# ============================================================
# Helpers
# ============================================================
def phase_in_breath(tau):
    # tau in [0, Ttot)
    if tau < Ti:
        return "insp", tau / Ti
    if tau < Ti + Thold:
        return "hold", (tau - Ti) / Thold
    return "exp", (tau - Ti - Thold) / max(Te, 1e-6)

def flow_wave(tau):
    ph, _ = phase_in_breath(tau)
    if ph == "insp":
        return FLOW_INSP
    return 0.0

def volume_wave(tau):
    ph, x = phase_in_breath(tau)
    if ph == "insp":
        return VT * (tau / Ti)  # linear ramp
    if ph == "hold":
        return VT
    # exp: passive decay from VT toward 0 (relative to end-exp baseline)
    # V(t) = VT * exp(-t/TAU)
    texp = tau - (Ti + Thold)
    return VT * np.exp(-texp / TAU_EXP)

def make_history(t, window, fps):
    t0 = max(0.0, t - window)
    n = int(max(120, min(int(window * fps), int((t - t0) * fps + 1))))
    return np.linspace(t0, t, n)

def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# ============================================================
# Plot helpers
# ============================================================
def draw_sum_bars(ax, PEEP, Pres, Pel, Paw, maxP):
    """
    Stacked vertical bars: PEEP base + resistive + elastic = total Paw.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, maxP)
    ax.axis("off")

    x = 0.20
    w = 0.20

    # Base PEEP
    ax.add_patch(Rectangle((x, 0), w, PEEP, facecolor="#9ca3af", alpha=0.85, edgecolor="none"))
    # Elastic
    ax.add_patch(Rectangle((x, PEEP), w, max(Pel,0), facecolor="#7c3aed", alpha=0.55, edgecolor="none"))
    # Resistive (can be 0 during hold)
    ax.add_patch(Rectangle((x, PEEP + max(Pel,0)), w, max(Pres,0), facecolor="#dc2626", alpha=0.55, edgecolor="none"))

    # Total indicator
    ax.plot([x-0.08, x+w+0.08], [Paw, Paw], lw=2.5, color="#111827")
    ax.text(x + w/2, Paw + 0.6, "Paw", ha="center", fontsize=10, weight="bold", color="#111827")

    # Labels
    ax.text(0.02, maxP*0.92, "Barras (soma)", fontsize=11, weight="bold", color="#111827")
    ax.text(0.02, maxP*0.80, "Paw = PEEP + E·V + R·V̇", fontsize=10.5, weight="bold", color="#111827")

    ax.text(0.46, PEEP + max(Pel,0) + max(Pres,0) - 0.5, f"{Paw:.1f}", fontsize=10, color="#111827", va="top")
    ax.text(0.46, PEEP + max(Pel,0) - 0.5, f"{(PEEP+Pel):.1f}", fontsize=10, color="#7c3aed", va="top")
    ax.text(0.46, PEEP - 0.3, f"{PEEP:.1f}", fontsize=10, color="#6b7280", va="top")

    # Legend chips
    ax.add_patch(Rectangle((0.02, maxP*0.12), 0.03, maxP*0.04, facecolor="#9ca3af", alpha=0.85, edgecolor="none"))
    ax.text(0.06, maxP*0.14, "PEEP", fontsize=9.5, color="#111827", va="center")

    ax.add_patch(Rectangle((0.02, maxP*0.07), 0.03, maxP*0.04, facecolor="#7c3aed", alpha=0.55, edgecolor="none"))
    ax.text(0.06, maxP*0.09, "Elástica (E·V)", fontsize=9.5, color="#111827", va="center")

    ax.add_patch(Rectangle((0.02, maxP*0.02), 0.03, maxP*0.04, facecolor="#dc2626", alpha=0.55, edgecolor="none"))
    ax.text(0.06, maxP*0.04, "Resistiva (R·V̇)", fontsize=9.5, color="#111827", va="center")

def scenario_of_time(t):
    idx = int(t // SEG_DUR)
    idx = max(0, min(2, idx))
    return idx, SCENARIOS[idx]

# ============================================================
# RENDER
# ============================================================
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "22"]
)

HIST = 10.0
total_frames = int(DURATION_S * FPS)

# Precompute max pressure for stable y-limits
# choose conservative envelope
MAX_PAW = 45.0

for i in range(total_frames):
    t = i / FPS
    scen_idx, scen = scenario_of_time(t)
    R = scen["R"]
    E = scen["E"]

    # time within breath
    tau = t % Ttot
    ph, _ = phase_in_breath(tau)

    V = volume_wave(tau)         # L
    Vdot = flow_wave(tau)        # L/s
    Pres = R * Vdot              # cmH2O
    Pel = E * V                  # cmH2O
    Paw = PEEP + Pel + Pres      # cmH2O

    # History
    th = make_history(t, HIST, FPS)
    tau_h = np.array([tt % Ttot for tt in th])
    V_h = np.array([volume_wave(ta) for ta in tau_h])
    Vdot_h = np.array([flow_wave(ta) for ta in tau_h])
    Pres_h = R * Vdot_h
    Pel_h = E * V_h
    Paw_h = PEEP + Pres_h + Pel_h

    # Cycle boundaries for shading
    t_breath_start = t - (t % Ttot)
    t_insp_end = t_breath_start + Ti
    t_hold_end = t_insp_end + Thold
    t_exp_end = t_breath_start + Ttot

    fig.clf()
    gs = fig.add_gridspec(
        2, 3,
        left=0.04, right=0.985, top=0.92, bottom=0.08,
        wspace=0.28, hspace=0.42,
        width_ratios=[1.25, 1.55, 1.10],
        height_ratios=[1.35, 1.05]
    )

    ax_eq   = fig.add_subplot(gs[:, 0])
    ax_paw  = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[1, 1])
    ax_bars = fig.add_subplot(gs[0, 2])
    ax_txt  = fig.add_subplot(gs[1, 2])

    # ------------------------------------------------------------
    # Left: equation + sliders concept (R and E)
    # ------------------------------------------------------------
    ax_eq.set_xlim(0, 1)
    ax_eq.set_ylim(0, 1)
    ax_eq.axis("off")

    ax_eq.text(0.02, 0.94, "Equação do Movimento", fontsize=15, weight="bold", color="#111827")
    ax_eq.text(0.02, 0.86, "Paw(t) = R·V̇(t) + E·V(t) + PEEP", fontsize=14, weight="bold", color="#111827")

    # Chips
    ax_eq.text(0.02, 0.78, "Interpretação clínica (ultra simples):", fontsize=11.5, weight="bold", color="#111827")
    ax_eq.text(0.03, 0.71, "• R·V̇  → 'tubo' (vias aéreas / resistência)", fontsize=11, color="#111827")
    ax_eq.text(0.03, 0.65, "• E·V  → 'balão' (rigidez / elastância)", fontsize=11, color="#111827")
    ax_eq.text(0.03, 0.59, "• PEEP → 'base' (pulmão já aberto)", fontsize=11, color="#111827")

    # Draw R slider
    ax_eq.text(0.02, 0.49, f"R = {R:.0f} cmH₂O/(L/s)", fontsize=12, weight="bold", color="#dc2626")
    ax_eq.add_patch(Rectangle((0.02, 0.45), 0.80, 0.03, facecolor="#fee2e2", edgecolor="#fecaca"))
    r_frac = np.clip((R - 6.0) / (24.0 - 6.0), 0, 1)
    ax_eq.add_patch(Rectangle((0.02, 0.45), 0.80*r_frac, 0.03, facecolor="#dc2626", alpha=0.75, edgecolor="none"))

    # Draw E slider
    ax_eq.text(0.02, 0.38, f"E = {E:.0f} cmH₂O/L", fontsize=12, weight="bold", color="#7c3aed")
    ax_eq.add_patch(Rectangle((0.02, 0.34), 0.80, 0.03, facecolor="#ede9fe", edgecolor="#ddd6fe"))
    e_frac = np.clip((E - 10.0) / (32.0 - 10.0), 0, 1)
    ax_eq.add_patch(Rectangle((0.02, 0.34), 0.80*e_frac, 0.03, facecolor="#7c3aed", alpha=0.65, edgecolor="none"))

    # PEEP
    ax_eq.text(0.02, 0.27, f"PEEP = {PEEP:.0f} cmH₂O", fontsize=12, weight="bold", color="#6b7280")

    # Scenario badge
    badge = FancyBboxPatch((0.02, 0.12), 0.80, 0.10,
                           boxstyle="round,pad=0.02,rounding_size=0.03",
                           facecolor="#f3f4f6", edgecolor="#e5e7eb")
    ax_eq.add_patch(badge)
    ax_eq.text(0.04, 0.17, f"Cenário: {scen['name']}", fontsize=12.5, weight="bold", color="#111827", va="center")
    ax_eq.text(0.04, 0.09, "Humor clínico: se o pico sobe → olha para R.\nSe o plateau sobe → culpa o E.",
               fontsize=9.8, color="#111827")

    # ------------------------------------------------------------
    # Top middle: Paw curve with peak/plateau concept
    # ------------------------------------------------------------
    ax_paw.set_title("Paw(t) (cmH₂O) — pico vs plateau", fontsize=11.5, weight="bold")
    ax_paw.plot(th, Paw_h, lw=3.0, color=scen["color"], label="Paw")
    ax_paw.plot(th, PEEP + Pel_h, lw=2.0, color="#7c3aed", alpha=0.55, label="PEEP+E·V (plateau base)")
    ax_paw.axhline(PEEP, color="#9ca3af", lw=1.2, alpha=0.8)
    ax_paw.set_ylim(0, MAX_PAW)
    ax_paw.grid(True, alpha=0.25)
    ax_paw.set_xlabel("Tempo (s)")
    ax_paw.set_ylabel("cmH₂O")
    ax_paw.legend(loc="upper right", fontsize=9, frameon=True)

    # Shade phases
    ax_paw.axvspan(t_breath_start, t_insp_end, color="#fecaca", alpha=0.12)   # insp
    ax_paw.axvspan(t_insp_end, t_hold_end, color="#fef3c7", alpha=0.14)      # hold
    ax_paw.axvspan(t_hold_end, t_exp_end, color="#bbf7d0", alpha=0.08)       # exp

    # markers
    ax_paw.scatter([t], [Paw], s=55, color="#111827", zorder=5)

    # annotate peak vs plateau during hold
    if ph == "hold":
        ax_paw.text(t, Paw + 2.0, "Plateau (fluxo=0)\n≈ PEEP + E·V",
                    fontsize=9.5, color="#111827",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="#e5e7eb"))
    elif ph == "insp":
        ax_paw.text(t, Paw + 2.0, "Pico = Plateau + R·V̇",
                    fontsize=9.5, color="#111827",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="#e5e7eb"))

    # ------------------------------------------------------------
    # Bottom middle: flow and volume overlays
    # ------------------------------------------------------------
    ax_flow.set_title("V̇(t) (L/s) e V(t) (L)", fontsize=11.0, weight="bold")
    ax_flow.plot(th, Vdot_h, lw=2.6, color="#dc2626", label="V̇ (fluxo)")
    ax_flow.plot(th, V_h, lw=2.6, color="#2563eb", label="V (volume)")
    ax_flow.axhline(0, color="#9ca3af", lw=1.1)
    ax_flow.set_ylim(-0.2, max(FLOW_INSP*1.5, 0.9))
    ax_flow.grid(True, alpha=0.25)
    ax_flow.set_xlabel("Tempo (s)")
    ax_flow.legend(loc="upper right", fontsize=9, frameon=True)

    ax_flow.axvspan(t_breath_start, t_insp_end, color="#fecaca", alpha=0.12)
    ax_flow.axvspan(t_insp_end, t_hold_end, color="#fef3c7", alpha=0.14)
    ax_flow.axvspan(t_hold_end, t_exp_end, color="#bbf7d0", alpha=0.08)

    # ------------------------------------------------------------
    # Top right: stacked bars (visual sum)
    # ------------------------------------------------------------
    draw_sum_bars(ax_bars, PEEP=PEEP, Pres=Pres, Pel=Pel, Paw=Paw, maxP=MAX_PAW)

    # ------------------------------------------------------------
    # Bottom right: live numbers + “teaching punchline”
    # ------------------------------------------------------------
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")

    ax_txt.text(0.02, 0.92, "Leituras (agora)", fontsize=12.5, weight="bold", color="#111827")
    ax_txt.text(0.02, 0.82, f"V̇ = {Vdot:.2f} L/s", fontsize=11.5, weight="bold", color="#dc2626")
    ax_txt.text(0.02, 0.74, f"V  = {V:.2f} L", fontsize=11.5, weight="bold", color="#2563eb")
    ax_txt.text(0.02, 0.66, f"R·V̇ = {Pres:.1f} cmH₂O", fontsize=11.5, weight="bold", color="#dc2626")
    ax_txt.text(0.02, 0.58, f"E·V  = {Pel:.1f} cmH₂O", fontsize=11.5, weight="bold", color="#7c3aed")
    ax_txt.text(0.02, 0.50, f"PEEP = {PEEP:.1f} cmH₂O", fontsize=11.5, weight="bold", color="#6b7280")
    ax_txt.text(0.02, 0.40, f"Paw  = {Paw:.1f} cmH₂O", fontsize=12.5, weight="bold", color="#111827")

    # Quick rules (peak vs plateau)
    ax_txt.text(
        0.02, 0.22,
        "Regras rápidas:\n"
        "• Se o pico ↑ e o plateau ~igual → resistência (R↑)\n"
        "• Se o plateau ↑ → elastância (E↑) / pulmão rígido\n"
        "• PEEP desloca tudo para cima (base)",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f3f4f6", edgecolor="#e5e7eb", alpha=0.95),
        color="#111827"
    )

    fig.suptitle("Equação do Movimento — vídeo didático (ICU)", fontsize=15, weight="bold", y=0.985)
    plt.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)

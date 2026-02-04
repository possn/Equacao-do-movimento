import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch

OUT = "Equacao_Movimento_Respiracao_Espontanea_VNI.mp4"

FPS = 20
DURATION = 60
W, H = 1920, 1080
DPI = 100

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "22"],
)

t_all = np.linspace(0, DURATION, int(DURATION * FPS), endpoint=False)

def canvas_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

def box(ax, x, y, w, h, text, fs=18, fc="#ffffff", ec="#111827", lw=2, tc="#111827", bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.98
        )
    )
    ax.text(
        x + w * 0.03, y + h * 0.62,
        text,
        ha="left", va="center",
        fontsize=fs,
        color=tc,
        weight="bold" if bold else "normal",
        linespacing=1.15
    )

def draw_lung(ax, inflate, label="Pulmão"):
    cx, cy = 0.22, 0.52
    w = 0.22 * (1 + 0.14 * inflate)
    h = 0.36 * (1 + 0.14 * inflate)

    lung = Ellipse(
        (cx, cy), w, h,
        facecolor="#f4a3a8", edgecolor="#7f1d1d", lw=4, alpha=0.98
    )
    ax.add_patch(lung)

    # traqueia
    ax.add_patch(Rectangle((cx - 0.014, cy + 0.18), 0.028, 0.10, facecolor="#111827"))

    ax.text(cx, cy - 0.30, label, ha="center", fontsize=14, weight="bold", color="#111827")

def draw_bars(ax, pmus, paw=0.0, show_paw=False):
    # base area for bars (fixed; no overlap)
    x0 = 0.48
    y0 = 0.58

    ax.text(x0, y0 + 0.11, "Contribuições (instantâneo)", fontsize=16, weight="bold")

    # Pmus bar
    ax.text(x0, y0 + 0.055, "Pmus", fontsize=14, weight="bold", color="#2563eb")
    ax.add_patch(Rectangle((x0, y0 + 0.018), 0.38, 0.028, facecolor="#e5e7eb", edgecolor="none"))
    ax.add_patch(Rectangle((x0, y0 + 0.018), 0.38 * pmus, 0.028, facecolor="#2563eb", edgecolor="none"))

    if show_paw:
        ax.text(x0, y0 - 0.01, "Paw (VNI: PS/CPAP)", fontsize=14, weight="bold", color="#7c3aed")
        ax.add_patch(Rectangle((x0, y0 - 0.048), 0.38, 0.028, facecolor="#e5e7eb", edgecolor="none"))
        ax.add_patch(Rectangle((x0, y0 - 0.048), 0.38 * paw, 0.028, facecolor="#7c3aed", edgecolor="none"))

def phase(t):
    if t < 15:
        return 1
    if t < 30:
        return 2
    if t < 45:
        return 3
    return 4

# simple oscillator for visuals
def smooth_cycle(t, period=6.0):
    x = 0.5 + 0.5 * np.sin(2 * np.pi * t / period)
    return float(np.clip(x, 0.0, 1.0))

for t in t_all:
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ph = phase(t)

    # spontaneous effort (Pmus) – smooth
    pmus = smooth_cycle(t, period=6.0)

    # VNI assistance in phase 4
    # idea: part of total "driving pressure" is provided by Paw (pressure support / CPAP)
    paw = 0.55 * smooth_cycle(t, period=6.0) if ph == 4 else 0.0

    # lung inflation: in spontaneous phases follows Pmus; in VNI phase follows (Pmus + Paw)
    drive = pmus if ph < 4 else np.clip(0.55 * pmus + 0.85 * paw, 0, 1)
    draw_lung(ax, inflate=0.85 * drive, label="Pulmão (estilizado)")

    # Title
    ax.text(
        0.5, 0.94,
        "Equação do Movimento — Respiração Espontânea → Ponte para VNI",
        ha="center", fontsize=22, weight="bold", color="#111827"
    )

    # ===== Phase-specific content (fixed positions, no collisions) =====
    if ph == 1:
        box(
            ax, 0.44, 0.28, 0.52, 0.22,
            "1) O motor é o músculo:\n"
            "   Pmus cria a 'força' para mover o ar.",
            fs=18, fc="#ffffff", ec="#111827", bold=False
        )
        draw_bars(ax, pmus, show_paw=False)

    elif ph == 2:
        box(
            ax, 0.44, 0.26, 0.52, 0.26,
            "2) Para onde vai a pressão?\n"
            "   • Parte vence a Resistência → gera Fluxo\n"
            "   • Parte distende o pulmão → gera Volume",
            fs=18, fc="#ffffff", ec="#111827"
        )

        # simple split visual: two labelled bars
        ax.text(0.50, 0.47, "R · Fluxo (resistência)", fontsize=16, weight="bold", color="#dc2626")
        ax.add_patch(Rectangle((0.50, 0.43), 0.40, 0.028, facecolor="#fee2e2", edgecolor="none"))
        ax.add_patch(Rectangle((0.50, 0.43), 0.40 * (0.65 * pmus), 0.028, facecolor="#dc2626", edgecolor="none"))

        ax.text(0.50, 0.38, "Volume / C (elástico)", fontsize=16, weight="bold", color="#2563eb")
        ax.add_patch(Rectangle((0.50, 0.34), 0.40, 0.028, facecolor="#dbeafe", edgecolor="none"))
        ax.add_patch(Rectangle((0.50, 0.34), 0.40 * (0.85 * pmus), 0.028, facecolor="#2563eb", edgecolor="none"))

    elif ph == 3:
        # big equation + immediate reading
        box(
            ax, 0.40, 0.52, 0.58, 0.16,
            "Pmus = R · Fluxo  +  Volume / C",
            fs=26, fc="#f8fafc", ec="#111827", lw=2, bold=True
        )
        box(
            ax, 0.40, 0.28, 0.58, 0.20,
            "Leitura básica:\n"
            "• Se R↑ (asma/DPOC) → precisa mais Pmus para o mesmo Fluxo\n"
            "• Se C↓ (pulmão rígido) → precisa mais Pmus para o mesmo Volume\n"
            "• Se Pmus falha (fadiga) → Fluxo e Volume caem",
            fs=16, fc="#ffffff", ec="#111827"
        )
        draw_bars(ax, pmus, show_paw=False)

    else:
        # VNI bridge: add Paw term. Keep it simple and correct.
        box(
            ax, 0.38, 0.58, 0.60, 0.16,
            "Com VNI (PS/CPAP):\nPmus + Paw ≈ R · Fluxo + Volume / C",
            fs=20, fc="#ede9fe", ec="#6d28d9", lw=2, tc="#111827", bold=True
        )
        box(
            ax, 0.38, 0.28, 0.60, 0.26,
            "Ponte clínica:\n"
            "• PS (pressão de suporte) fornece Paw na inspiração → reduz Pmus (descarrega trabalho)\n"
            "• CPAP aumenta volume de fim-expiração (recruta/estabiliza) → melhora C e reduz esforço\n"
            "Resultado: menor dispneia/fadiga, melhor ventilação com menos custo muscular.",
            fs=15, fc="#ffffff", ec="#111827"
        )
        draw_bars(ax, pmus, paw=paw, show_paw=True)

    writer.append_data(canvas_rgb(fig))

writer.close()
print("OK:", OUT)

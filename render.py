import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Ellipse, Rectangle

OUT = "Equacao_Movimento_Respiracao_Espontanea.mp4"

FPS = 20
DURATION = 60

W, H = 1920, 1080
DPI = 100

fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset","ultrafast","-crf","22"]
)

# timeline
t_all = np.linspace(0, DURATION, int(DURATION*FPS))

def draw_lung(ax, inflate):
    cx, cy = 0.23, 0.52
    w = 0.22*(1+0.12*inflate)
    h = 0.36*(1+0.12*inflate)

    lung = Ellipse((cx,cy), w, h,
                   facecolor="#f4a3a8",
                   edgecolor="#7f1d1d",
                   lw=4)

    ax.add_patch(lung)

    # trachea
    ax.add_patch(Rectangle((cx-0.015, cy+0.18),0.03,0.10,
                           facecolor="#111827"))

    ax.text(cx, cy-0.28,"Pulmão",
            ha="center",fontsize=14,weight="bold")

def canvas_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:,:,:3]

for t in t_all:

    fig.clf()
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis("off")

    # phases
    phase1 = t < 20
    phase2 = 20 <= t < 40
    phase3 = t >= 40

    # muscle drive
    pmus = 0.5 + 0.5*np.sin(2*np.pi*t/6)
    pmus = np.clip(pmus,0,1)

    # lung inflation
    inflate = pmus*0.8 if phase1 or phase2 else 0.7

    draw_lung(ax, inflate)

    # title
    ax.text(0.5,0.94,
        "Equação do Movimento — Respiração Espontânea",
        ha="center",fontsize=22,weight="bold")

    # =======================
    # PHASE 1 — Motor
    # =======================
    if phase1:

        ax.text(0.55,0.62,"Pmus (força muscular)",
                fontsize=18,weight="bold")

        ax.add_patch(Rectangle((0.55,0.55),0.35,0.06,
                     facecolor="#e5e7eb"))

        ax.add_patch(Rectangle((0.55,0.55),0.35*pmus,0.06,
                     facecolor="#2563eb"))

    # =======================
    # PHASE 2 — Split
    # =======================
    if phase2:

        ax.text(0.55,0.70,"Pmus",
                fontsize=18,weight="bold")

        # flow bar
        ax.text(0.50,0.48,"Fluxo (R)",
                fontsize=15,weight="bold",color="#dc2626")

        ax.add_patch(Rectangle((0.48,0.42),0.30,0.05,
                     facecolor="#fecaca"))

        ax.add_patch(Rectangle((0.48,0.42),0.30*pmus,0.05,
                     facecolor="#dc2626"))

        # volume bar
        ax.text(0.78,0.48,"Volume (C)",
                fontsize=15,weight="bold",color="#2563eb")

        ax.add_patch(Rectangle((0.76,0.42),0.18,0.05,
                     facecolor="#bfdbfe"))

        ax.add_patch(Rectangle((0.76,0.42),0.18*pmus,0.05,
                     facecolor="#2563eb"))

    # =======================
    # PHASE 3 — Equation
    # =======================
    if phase3:

        ax.text(0.5,0.55,
            "Pmus = R · Fluxo  +  Volume / C",
            ha="center",
            fontsize=26,
            weight="bold")

        ax.text(0.5,0.48,
            "Resistência das vias aéreas + Elasticidade pulmonar",
            ha="center",
            fontsize=16)

    writer.append_data(canvas_rgb(fig))

writer.close()
print("OK:", OUT)

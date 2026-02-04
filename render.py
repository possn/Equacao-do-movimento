import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = "Equacao_Movimento_Respiracao_Espontanea.mp4"

FPS = 20
DURATION = 60
FRAMES = FPS * DURATION

figsize = (12.8, 7.2)

def draw_bar(ax, x, y, w, h, frac, label, color):
    ax.add_patch(plt.Rectangle((x,y), w, h, fill=False, lw=2))
    ax.add_patch(plt.Rectangle((x,y), w*frac, h, color=color, alpha=0.8))
    ax.text(x+w/2, y+h+0.03, label, ha="center", fontsize=11, weight="bold")

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "23"]
)

for i in range(FRAMES):

    t = i / FPS

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis("off")

    # ===== PHASES =====
    if t < 10:
        phase = "base"
    elif t < 25:
        phase = "resistive"
    elif t < 40:
        phase = "elastic"
    else:
        phase = "vni"

    # ===== TITLE =====
    ax.text(0.5,0.94,"Equação do Movimento — Respiração Espontânea",
            ha="center", fontsize=18, weight="bold")

    # ===== EQUATIONS =====
    if phase == "base":
        eq = "Pmus = R · Fluxo + V / C"
    elif phase == "resistive":
        eq = "Pmus =  R · Fluxo  + V / C"
    elif phase == "elastic":
        eq = "Pmus = R · Fluxo +  V / C"
    else:
        eq = "Pmus + Paw = R · Fluxo + V / C"

    ax.text(0.5,0.75,eq,ha="center",fontsize=26,weight="bold")

    # ===== SIDE TEXT =====
    if phase == "resistive":
        txt = "Resistência vias aéreas\n↑R → ↑ esforço\nEx: asma"
    elif phase == "elastic":
        txt = "Elastância pulmonar\nPulmão rígido → mais trabalho\nEx: ARDS"
    elif phase == "vni":
        txt = "VNI ajuda:\nPS ↓ Pmus\nCPAP melhora complacência"
    else:
        txt = "Equação base\nPressão muscular = carga mecânica"

    ax.text(0.08,0.55,txt,fontsize=13)

    # ===== BARS =====
    if phase == "base":
        effort = 0.4
        paw = 0.0
    elif phase == "resistive":
        effort = 0.75
        paw = 0.0
    elif phase == "elastic":
        effort = 0.75
        paw = 0.0
    else:
        effort = 0.35
        paw = 0.45

    draw_bar(ax,0.55,0.42,0.32,0.06,effort,"Pmus (esforço)","#ef4444")

    if phase == "vni":
        draw_bar(ax,0.55,0.30,0.32,0.06,paw,"Paw (VNI)","#22c55e")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
    writer.append_data(frame)
    plt.close(fig)

writer.close()

print("OK ->", OUT)

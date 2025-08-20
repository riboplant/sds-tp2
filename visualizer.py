from __future__ import annotations
import os
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import vicsek
from datetime import datetime

def animate_vicsek(xy_list, theta_list, va_hist, L: float, color_by_angle: bool = False):
    T = len(xy_list)
    fig, (ax_anim, ax_va) = plt.subplots(1, 2, figsize=(12, 6))
    ax_anim.set_xlim(0, L)
    ax_anim.set_ylim(0, L)
    ax_anim.set_aspect('equal')
    ax_anim.set_title("Simulación Vicsek")

    if color_by_angle:
        initial_colors = theta_list[0]
        cmap = plt.cm.hsv
    else:
        initial_colors = 'blue'
        cmap = None

    # Inicializar quiver
    print(xy_list[0])
    scat = ax_anim.quiver(
        xy_list[0][:, 0], xy_list[0][:, 1],
        np.cos(theta_list[0]), np.sin(theta_list[0]),
        angles='xy', scale_units='xy', scale=1.0, width=0.005,
        color=initial_colors if not color_by_angle else cmap(initial_colors / (2*np.pi))
    )

    # Configuración subplot de parámetro de orden
    ax_va.set_title("Evolución del parámetro de orden")
    ax_va.set_xlabel("Iteraciones")
    ax_va.set_ylabel("Parámetro de orden $v_a$")
    ax_va.set_xlim(0, N)
    ax_va.set_ylim(0, 1)
    line_va, = ax_va.plot([], [], lw=1.5, color='blue')

    def update(frame):
        xy = xy_list[frame]
        theta = theta_list[frame]
        scat.set_offsets(xy)
        scat.set_UVC(np.cos(theta), np.sin(theta))
        if color_by_angle:
            scat.set_color(cmap((theta + np.pi) / (2 * np.pi)))
        line_va.set_data(np.arange(frame + 1), va_hist[:frame + 1])
        return scat, line_va

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

def key_name(name: str):
    base = name[:-4].lower()  # quitar ".txt"
    if base == "static":
        return (1, float("inf"))
    return (0, int(base))

timestamp = input("Ingrese la timestamp de la simulacion: ")
directory = f"data/{timestamp}"
static_file = os.path.join(directory, "static.txt")
N, L, v, r = 0, 0, 0, 0
with open(static_file, "r") as f:
    N = int(f.readline())
    L = float(f.readline())
    v = float(f.readline())
    r = float(f.readline())

xy = []
theta = []
va_hist = []
for root, _, files in os.walk(directory):
    files = sorted(files, key=key_name)
    for name in files:
        if name != "static.txt":
            dynamic_file = os.path.join(directory, name)
            print(f"Procesando {name}")
            xy_d = []
            theta_d = []
            with open(dynamic_file, "r") as f:
                for line in f:
                    vals = line.strip().split(' ')
                    xy_d.append([float(vals[0]), float(vals[1])])
                    theta_d.append(float(vals[2]))
            xy.append(np.asarray(xy_d))
            theta.append(np.asarray(theta_d))
            va_hist.append(vicsek.order_parameter(theta_d))

animate_vicsek(xy, theta, va_hist, L, False)

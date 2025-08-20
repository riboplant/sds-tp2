from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import vicsek

def animate_vicsek(xy_list, theta_list, va_hist, v: float, L: float, T: int, color_by_angle: bool = False):
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
    scat = ax_anim.quiver(
        xy_list[0][:, 0], xy_list[0][:, 1],
        np.cos(theta_list[0]), np.sin(theta_list[0]),
        angles='xy', scale_units='xy', scale=1/(v*10), width=0.005,
        color=initial_colors if not color_by_angle else cmap(initial_colors / (2*np.pi))
    )

    # Configuración subplot de parámetro de orden
    ax_va.set_title("Evolución del parámetro de orden")
    ax_va.set_xlabel("Iteraciones")
    ax_va.set_ylabel("Parámetro de orden $v_a$")
    ax_va.set_xlim(0, T)
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

timestamp = input("Ingrese la timestamp de la simulacion: ")
N, L, v, r, eta, T, xy, theta, va_hist = vicsek.process_simulation(timestamp)
animate_vicsek(xy, theta, va_hist, v, L, T, True)

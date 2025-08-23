import matplotlib.pyplot as plt
import vicsek
import numpy as np
import os

def seasonal_evolution_by_noise(graph_name):
    file = os.path.join(f"data/graphs/seasonal_evolution_by_noise/{graph_name}.txt")
    with open(file, 'r') as f:
        for line in f:
            vals = line.split(' ')
            N, eta, _ = vals[:3]
            va_hist = [float(x) for x in vals[3:]]
            plt.plot(va_hist, label=f"N={N}, \u03B7={eta}")
    plt.title("Evolución temporal para distintos valores de ruido \u03B7")
    plt.xlabel("Tiempo")
    plt.ylabel("Parámetro de orden $v_a$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    

def va_evolution_by_noise(axes: plt.Axes, N_values: list[int], L: float, r: float, v: float, eta_min: float, eta_max: float, eta_step: float, T):
    for N in N_values:
        va_hist_N = []
        for eta in range(eta_min, eta_max + eta_step):
            params = vicsek.VicsekParams(N=N, L=L, r=r, v=v, eta=eta, seed=42)
            _, _, va_hist = vicsek.simulate(params, T=T)
            va_hist_N.append(np.mean(va_hist))
        axes.plot(va_hist, label=f"N={N}")
    axes.set_title("Evolución de v_a en funcion de \u03B7")
    axes.set_xlabel("\u03B7")
    axes.set_ylabel("$v_a$")
    axes.legend()
    axes.grid(True, alpha=0.3)

plt.figure(figsize=(12,12))
graph_name = input("Ingrese el nombre del grafico: ")
seasonal_evolution_by_noise(graph_name)
plt.tight_layout()
plt.show()

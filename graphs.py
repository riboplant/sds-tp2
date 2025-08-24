import matplotlib.pyplot as plt
import numpy as np
import os

def seasonal_evolution_by_noise(graph_name: str):
    file = os.path.join(f"data/graphs/seasonal_evolution_by_noise/{graph_name}.txt")
    with open(file, 'r') as f:
        for line in f:
            vals = line.split(' ')
            N, eta, _ = vals[:3]
            va_hist = [float(x) for x in vals[3:]]
            plt.plot(va_hist, label=f"N={N}, \u03B7={eta}")
    plt.title("Evolución temporal para distintos valores de ruido \u03B7")
    plt.xlabel("t")
    plt.ylabel("$v_a$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    

def fixed_density(graph_name: str):
    file = os.path.join(f"data/graphs/fixed_density", f"{graph_name}.txt")
    with open(file, 'r') as f:
        N, L, q = 0, 0, 0
        etas = []
        va_avgs = []
        for line in f:
            vals = line.split(' ')
            if q > 0:
                eta, va_avg = vals
                etas.append(float(eta))
                va_avgs.append(float(va_avg))
                q -= 1
            else:
                if len(va_avgs) > 0:
                    plt.plot(etas, va_avgs, marker="D", linestyle="None", label=f"N={N}, L={L}")
                    etas = []
                    va_avgs = []
                N, L, q = vals[:3]
                q = int(q)
        if len(va_avgs) > 0:
            plt.plot(etas, va_avgs, marker="D", linestyle="None", label=f"N={N}, L={L}")

    plt.title("Evolucion del $v_a$ promedio para distintos valores de \u03B7 manteniendo fija la densidad \u03C1")
    plt.xlabel("\u03B7")
    plt.ylabel("$v_a$ promedio")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(12,12))
graph_name = input("Ingrese el nombre del grafico: ")
#seasonal_evolution_by_noise(graph_name)
fixed_density(graph_name)
plt.tight_layout()
plt.show()

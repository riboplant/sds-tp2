import matplotlib.pyplot as plt
import os
import numpy as np

def seasonal_evolution_by_noise(graph_name: str):
    file = os.path.join(f"data/graphs/seasonal_evolution_by_noise/{graph_name}.txt")
    T = 0
    with open(file, 'r') as f:
        for line in f:
            vals = line.split(' ')
            N, L, eta, T = vals[:4]
            va_hist = [float(x) for x in vals[4:]]
            plt.plot(va_hist, label=f"N={N}, L={L}, \u03B7={eta}")
    t0 = 0.6*len(va_hist)
    plt.axvline(t0, linestyle="--", linewidth=1.0, alpha=0.7)
    plt.text(t0, plt.ylim()[1]*0.05, "t0", rotation=90, va="bottom", ha="right", fontsize=8)
    plt.xticks(np.arange(0, int(T)+1, 100))
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
        va_errs = []
        for line in f:
            vals = line.split(' ')
            if q > 0:
                eta, va_avg, err = vals
                etas.append(float(eta))
                va_avgs.append(float(va_avg))
                va_errs.append(float(err))
                q -= 1
            else:
                if len(va_avgs) > 0:
                    plt.errorbar(etas, va_avgs, yerr=va_errs, fmt="D-", capsize=4, label=f"N={N}, L={L}")
                    etas = []
                    va_avgs = []
                    va_errs = []
                N, L, q = vals[:3]
                q = int(q)
        if len(va_avgs) > 0:
            plt.errorbar(etas, va_avgs, yerr=va_errs, fmt="D-", capsize=4, label=f"N={N}, L={L}")
    plt.title("Evolucion del $v_a$ promedio para distintos valores de \u03B7 manteniendo fija la densidad \u03C1")
    plt.xlabel("\u03B7")
    plt.ylabel("$v_a$ promedio")
    plt.legend()
    plt.grid(True, alpha=0.3)

def varied_density_eta_fixed(graph_name: str):
    file = os.path.join(f"data/graphs/varied_density_eta_fixed", f"{graph_name}.txt")
    L, eta = 0, 0
    density = []
    v_a = []
    errs = []
    with open(file, 'r') as f:
        for line in f:
            vals = line.split(' ')
            if L == 0:
                L = float(vals[0])
                eta = float(vals[1])
            else:
                density.append(float(vals[0]))
                v_a.append(float(vals[1]))
                errs.append(float(vals[2]))
    
    plt.errorbar(density, v_a, yerr=errs, fmt="D-", capsize=4, label=f"L={L}, eta={eta}")
    plt.title("Evolucion del $v_a$ promedio para distintas \u03C1 manteniendo fijo \u03B7")
    plt.ylim(0,1)
    plt.xlabel('\u03C1')
    plt.ylabel("$v_a$ promedio")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(12,12))
graph_name = input("Ingrese el nombre del grafico: ")
#seasonal_evolution_by_noise(graph_name)
#fixed_density(graph_name)
varied_density_eta_fixed(graph_name)
plt.tight_layout()
plt.show()

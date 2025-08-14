from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# Math Utilities
# =============================

def wrap_periodic(xy: np.ndarray, L: float) -> np.ndarray:
    return np.mod(xy, L)

def minimum_image(dx: np.ndarray, L: float) -> np.ndarray:
    return dx - L * np.round(dx / L)

def pairwise_periodic_deltas(xy: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dx = minimum_image(dx, L)
    dy = minimum_image(dy, L)
    return dx, dy

# =============================
# Modelo de Vicsek
# =============================

@dataclass
class VicsekParams:
    N: int = 300
    L: float = 7.0
    r: float = 1.0
    v: float = 0.03
    eta: float = 0.1
    seed: Optional[int] = 0

@dataclass
class VicsekState:
    xy: np.ndarray
    theta: np.ndarray

def initialize(params: VicsekParams) -> VicsekState:
    rng = np.random.default_rng(params.seed)
    xy = rng.uniform(0.0, params.L, size=(params.N, 2))
    theta = rng.uniform(-math.pi, math.pi, size=params.N)
    return VicsekState(xy=xy, theta=theta)

def order_parameter(theta: np.ndarray) -> float:
    vx = np.cos(theta).mean()
    vy = np.sin(theta).mean()
    return float(np.hypot(vx, vy))

def step(state: VicsekState, params: VicsekParams, rng: np.random.Generator) -> VicsekState:
    N, L, r, v, eta = params.N, params.L, params.r, params.v, params.eta
    dx, dy = pairwise_periodic_deltas(state.xy, L)
    dist2 = dx*dx + dy*dy
    mask = dist2 <= (r * r)
    sin_th = np.sin(state.theta)[None, :]
    cos_th = np.cos(state.theta)[None, :]
    sum_sin = (mask * sin_th).sum(axis=1)
    sum_cos = (mask * cos_th).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1)
    mean_sin = sum_sin / counts
    mean_cos = sum_cos / counts
    mean_angle = np.arctan2(mean_sin, mean_cos)
    noise = rng.uniform(-eta/2.0, eta/2.0, size=N)
    new_theta = mean_angle + noise
    vx = v * np.cos(new_theta)
    vy = v * np.sin(new_theta)
    new_xy = wrap_periodic(state.xy + np.stack([vx, vy], axis=1), L)
    return VicsekState(xy=new_xy, theta=new_theta)

# =============================
# Simulaciones
# =============================

def simulate(params: VicsekParams, T: int = 1000) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed)
    state = initialize(params)
    va_hist = []
    for t in range(T):
        va_hist.append(order_parameter(state.theta))
        state = step(state, params, rng)
    return {"va": np.array(va_hist)}

def seasonal_evolution_by_noise(N: int, L: float, r: float, v: float, eta_values: list, T: int = 1000):
    plt.figure(figsize=(8,6))
    for eta in eta_values:
        params = VicsekParams(N=N, L=L, r=r, v=v, eta=eta, seed=42)
        results = simulate(params, T=T)
        plt.plot(results["va"], label=f"eta={eta}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Parámetro de orden $v_a$")
    plt.title("Evolución temporal para distintos valores de ruido $\eta$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def sweep_eta_time_series(params: VicsekParams, etas, T: int = 1000, record_every: int = 5,
                           R: int = 5, T_trans: int = 0, save_path: Optional[str] = None,
                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Promedia la evolución temporal del orden v_a(t) para varios valores de ruido η.
    - etas: lista/array de valores de η.
    - R: número de realizaciones independientes por η.
    - T_trans: iteraciones iniciales a descartar en el promedio temporal (si se desea comparar régimen estacionario temprano).
    Devuelve un dict con 'etas', 't', 'va_mean' (len(etas) x len(t)) y 'va_std'."""
    if plt is None:
        raise RuntimeError("matplotlib no disponible para graficar")

    etas = np.asarray(list(etas), dtype=float)
    rng_master = np.random.default_rng(params.seed if seed is None else seed)

    # Eje temporal que se registrará
    t_idx = np.arange(0, T, record_every)
    va_mean = np.zeros((len(etas), len(t_idx)))
    va_std  = np.zeros_like(va_mean)

    for j, eta in enumerate(etas):
        runs = []
        for r_id in range(R):
            # Semilla distinta por realización, reproducible
            subseed = int(rng_master.integers(0, 2**31-1))
            local = VicsekParams(**{**params.__dict__, 'eta': float(eta), 'seed': subseed})
            rng = np.random.default_rng(local.seed)
            state = initialize(local)
            series = []
            for t in range(T):
                if t % record_every == 0:
                    series.append(order_parameter(state.theta))
                state = step(state, local, rng)
            series = np.asarray(series)
            runs.append(series)
        runs = np.stack(runs, axis=0)  # (R, len(t_idx))
        if T_trans > 0:
            # si se desea descartar transitorio en el promedio por tiempo
            trans_idx = max(0, T_trans // record_every)
            va_mean[j, :] = runs[:, trans_idx:].mean(axis=0, keepdims=False, dtype=float).mean() * 0 + runs[:, :].mean(axis=0)
        else:
            va_mean[j, :] = runs.mean(axis=0)
        va_std[j, :]  = runs.std(axis=0, ddof=1) if R > 1 else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, eta in enumerate(etas):
        ax.plot(t_idx, va_mean[j], label=f"$\eta$={eta}")
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Parámetro de orden $v_a$')
    ax.set_title('Evolución temporal promedio de $v_a$ para distintos $\eta$')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Ruido')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    return {"etas": etas, "t": t_idx, "va_mean": va_mean, "va_std": va_std}

def sweep_eta_stationary(params: VicsekParams, etas, T: int = 1200, T_trans: int = 200,
                          R: int = 5, save_path: Optional[str] = None,
                          seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Promedia el valor estacionario de v_a para varios η.
    Calcula <v_a> como promedio temporal sobre [T_trans, T) y luego promedia sobre R realizaciones.
    Devuelve dict con 'etas', 'va_mean', 'va_std'."""
    if plt is None:
        raise RuntimeError("matplotlib no disponible para graficar")

    etas = np.asarray(list(etas), dtype=float)
    rng_master = np.random.default_rng(params.seed if seed is None else seed)

    va_means = []
    for eta in etas:
        vals = []
        for r_id in range(R):
            subseed = int(rng_master.integers(0, 2**31-1))
            local = VicsekParams(**{**params.__dict__, 'eta': float(eta), 'seed': subseed})
            rng = np.random.default_rng(local.seed)
            state = initialize(local)
            series = []
            for t in range(T):
                if t >= T_trans:
                    series.append(order_parameter(state.theta))
                state = step(state, local, rng)
            vals.append(np.mean(series) if len(series) else np.nan)
        vals = np.asarray(vals, dtype=float)
        va_means.append((np.nanmean(vals), np.nanstd(vals, ddof=1) if R > 1 else 0.0))

    va_means = np.asarray(va_means)  # (len(etas), 2)
    va_mean = va_means[:, 0]
    va_std  = va_means[:, 1]

    # Plot con barras de error
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(etas, va_mean, yerr=va_std, fmt='o-', capsize=4, linewidth=1.5)
    ax.set_xlabel('Ruido $\eta$')
    ax.set_ylabel('Orden estacionario $\langle v_a \rangle$')
    ax.set_title('Orden estacionario vs. ruido $\eta$ (promedio sobre R realizaciones)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    return {"etas": etas, "va_mean": va_mean, "va_std": va_std}

def animate_vicsek(params, T=200):
    rng = np.random.default_rng(params.seed)
    state = initialize(params)

    fig, (ax_anim, ax_va) = plt.subplots(1, 2, figsize=(12, 6))
    ax_anim.set_xlim(0, params.L)
    ax_anim.set_ylim(0, params.L)
    ax_anim.set_aspect('equal')
    scat = ax_anim.quiver(state.xy[:, 0], state.xy[:, 1],
                          np.cos(state.theta), np.sin(state.theta),
                          angles='xy', scale_units='xy', scale=1.0, width=0.005)
    ax_anim.set_title("Simulación Vicsek")

    va_values = []
    t_values = []
    ax_va.set_title("Evolución del orden")
    ax_va.set_xlabel("Iteraciones")
    ax_va.set_ylabel("Parámetro de orden $v_a$")
    line_va, = ax_va.plot([], [], lw=1.5)
    ax_va.set_xlim(0, T)
    ax_va.set_ylim(0, 1)

    def update(frame):
        nonlocal state
        state = step(state, params, rng)
        scat.set_offsets(state.xy)
        scat.set_UVC(np.cos(state.theta), np.sin(state.theta))
        va = order_parameter(state.theta)
        va_values.append(va)
        t_values.append(frame)
        line_va.set_data(t_values, va_values)
        return scat, line_va

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

def order_parameter(theta: np.ndarray) -> float:
    """Parámetro de orden v_a = |<v>| con |v|=1 (el módulo se cancela)."""
    vx = np.cos(theta).mean()
    vy = np.sin(theta).mean()
    return float(np.hypot(vx, vy))

def run_and_plot_order(params: VicsekParams, T: int = 1500, record_every: int = 5,
                       save_path: Optional[str] = None) -> dict:
    """Simula T pasos y grafica v_a vs. tiempo. Devuelve diccionario con arrays."""
    rng = np.random.default_rng(params.seed)
    state = initialize(params)
    times, vah = [], []
    for t in range(T):
        if t % record_every == 0:
            times.append(t)
            vah.append(order_parameter(state.theta))
        state = step(state, params, rng)
    times = np.array(times)
    vah = np.array(vah)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, vah, lw=1.5)
    ax.set_title('Evolución del orden — Modelo de Vicsek')
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Parámetro de orden $v_a$')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return {"t": times, "va": vah}

def sweep_density_stationary(
    params: VicsekParams,
    rhos,
    T: int = 1500,
    T_trans: int = 300,
    R: int = 5,
    mode: str = 'vary_N',         # 'vary_N' fija L y ajusta N; 'vary_L' fija N y ajusta L
    save_path: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    if plt is None:
        raise RuntimeError("matplotlib no disponible para graficar")

    rhos = np.asarray(list(rhos), dtype=float)
    rng_master = np.random.default_rng(params.seed if seed is None else seed)

    va_means = []
    for rho in rhos:
        vals = []
        for r_id in range(R):
            subseed = int(rng_master.integers(0, 2**31-1))
            if mode == 'vary_N':
                L = params.L
                N = max(1, int(round(float(rho) * (L**2))))
            elif mode == 'vary_L':
                N = params.N
                L = float(np.sqrt(N / float(rho)))
            else:
                raise ValueError("mode debe ser 'vary_N' o 'vary_L'")

            local_dict = {**params.__dict__, 'N': N, 'L': L, 'seed': subseed}
            local = VicsekParams(**local_dict)

            rng = np.random.default_rng(local.seed)
            state = initialize(local)
            vals_t = []
            for t in range(T):
                if t >= T_trans:
                    vals_t.append(order_parameter(state.theta))
                state = step(state, local, rng)
            vals.append(np.mean(vals_t) if len(vals_t) else np.nan)

        vals = np.asarray(vals, dtype=float)
        va_means.append((np.nanmean(vals), np.nanstd(vals, ddof=1) if R > 1 else 0.0))

    va_means = np.asarray(va_means)
    va_mean = va_means[:, 0]
    va_std  = va_means[:, 1]

    # Plot con barras de error
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(rhos, va_mean, yerr=va_std, fmt='o-', capsize=4, linewidth=1.5)
    ax.set_xlabel('Densidad $\\rho$')
    ax.set_ylabel('Orden estacionario $\\langle v_a \\rangle$')
    ax.set_title('Orden estacionario vs. densidad $\\rho$ (promedio sobre R realizaciones)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    return {"rhos": rhos, "va_mean": va_mean, "va_std": va_std}



if __name__ == "__main__":
    seasonal_evolution_by_noise(N=300, L=7.0, r=1.0, v=0.03, eta_values=[0.1, 0.5, 1.0, 2.0], T=500)
    params = VicsekParams(N=300, L=7.0, r=1.0, v=0.03, eta=0.5, seed=1)
    animate_vicsek(params, T=300)
    rhos = [0.2, 0.5, 1.0, 2.0]
    res_rho = sweep_density_stationary(
        params,
        rhos=rhos,
        T=1500,
        T_trans=300,
        R=5,
        mode='vary_N',
    )

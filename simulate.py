from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import vicsek
from datetime import datetime

# def seasonal_evolution_by_noise(N: int, L: float, r: float, v: float, eta_values: list, T: int = 1000):
#     plt.figure(figsize=(8,6))
#     for eta in eta_values:
#         params = vicsek.VicsekParams(N=N, L=L, r=r, v=v, eta=eta, seed=42)
#         results = vicsek.simulate(params, T=T)
#         plt.plot(results["va"], label=f"eta={eta}")
#     plt.xlabel("Iteraciones")
#     plt.ylabel("Parámetro de orden $v_a$")
#     plt.title("Evolución temporal para distintos valores de ruido $\eta$")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# def sweep_eta_time_series(params: vicsek.VicsekParams, etas, T: int = 1000, record_every: int = 5,
#                            R: int = 5, T_trans: int = 0, save_path: Optional[str] = None,
#                            seed: Optional[int] = None) -> Dict[str, np.ndarray]:
#     """Promedia la evolución temporal del orden v_a(t) para varios valores de ruido η.
#     - etas: lista/array de valores de η.
#     - R: número de realizaciones independientes por η.
#     - T_trans: iteraciones iniciales a descartar en el promedio temporal (si se desea comparar régimen estacionario temprano).
#     Devuelve un dict con 'etas', 't', 'va_mean' (len(etas) x len(t)) y 'va_std'."""
#     if plt is None:
#         raise RuntimeError("matplotlib no disponible para graficar")

#     etas = np.asarray(list(etas), dtype=float)
#     rng_master = np.random.default_rng(params.seed if seed is None else seed)

#     # Eje temporal que se registrará
#     t_idx = np.arange(0, T, record_every)
#     va_mean = np.zeros((len(etas), len(t_idx)))
#     va_std  = np.zeros_like(va_mean)

#     for j, eta in enumerate(etas):
#         runs = []
#         for r_id in range(R):
#             # Semilla distinta por realización, reproducible
#             subseed = int(rng_master.integers(0, 2**31-1))
#             local = vicsek.VicsekParams(**{**params.__dict__, 'eta': float(eta), 'seed': subseed})
#             rng = np.random.default_rng(local.seed)
#             state = vicsek.initialize(local)
#             series = []
#             for t in range(T):
#                 if t % record_every == 0:
#                     series.append(vicsek.order_parameter(state.theta))
#                 state = vicsek.step(state, local, rng)
#             series = np.asarray(series)
#             runs.append(series)
#         runs = np.stack(runs, axis=0)  # (R, len(t_idx))
#         if T_trans > 0:
#             # si se desea descartar transitorio en el promedio por tiempo
#             trans_idx = max(0, T_trans // record_every)
#             va_mean[j, :] = runs[:, trans_idx:].mean(axis=0, keepdims=False, dtype=float).mean() * 0 + runs[:, :].mean(axis=0)
#         else:
#             va_mean[j, :] = runs.mean(axis=0)
#         va_std[j, :]  = runs.std(axis=0, ddof=1) if R > 1 else 0.0

#     # Plot
#     fig, ax = plt.subplots(figsize=(7.5, 4.5))
#     for j, eta in enumerate(etas):
#         ax.plot(t_idx, va_mean[j], label=f"$\eta$={eta}")
#     ax.set_xlabel('Iteraciones')
#     ax.set_ylabel('Parámetro de orden $v_a$')
#     ax.set_title('Evolución temporal promedio de $v_a$ para distintos $\eta$')
#     ax.grid(True, alpha=0.3)
#     ax.legend(title='Ruido')
#     fig.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=150)
#     plt.show()

#     return {"etas": etas, "t": t_idx, "va_mean": va_mean, "va_std": va_std}

# def sweep_eta_stationary(params: vicsek.VicsekParams, etas, T: int = 1200, T_trans: int = 200,
#                           R: int = 5, save_path: Optional[str] = None,
#                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
#     """Promedia el valor estacionario de v_a para varios η.
#     Calcula <v_a> como promedio temporal sobre [T_trans, T) y luego promedia sobre R realizaciones.
#     Devuelve dict con 'etas', 'va_mean', 'va_std'."""
#     if plt is None:
#         raise RuntimeError("matplotlib no disponible para graficar")

#     etas = np.asarray(list(etas), dtype=float)
#     rng_master = np.random.default_rng(params.seed if seed is None else seed)

#     va_means = []
#     for eta in etas:
#         vals = []
#         for r_id in range(R):
#             subseed = int(rng_master.integers(0, 2**31-1))
#             local = vicsek.VicsekParams(**{**params.__dict__, 'eta': float(eta), 'seed': subseed})
#             rng = np.random.default_rng(local.seed)
#             state = vicsek.initialize(local)
#             series = []
#             for t in range(T):
#                 if t >= T_trans:
#                     series.append(vicsek.order_parameter(state.theta))
#                 state = vicsek.step(state, local, rng)
#             vals.append(np.mean(series) if len(series) else np.nan)
#         vals = np.asarray(vals, dtype=float)
#         va_means.append((np.nanmean(vals), np.nanstd(vals, ddof=1) if R > 1 else 0.0))

#     va_means = np.asarray(va_means)  # (len(etas), 2)
#     va_mean = va_means[:, 0]
#     va_std  = va_means[:, 1]

#     # Plot con barras de error
#     fig, ax = plt.subplots(figsize=(7.0, 4.2))
#     ax.errorbar(etas, va_mean, yerr=va_std, fmt='o-', capsize=4, linewidth=1.5)
#     ax.set_ylabel(r'Orden estacionario $\left< v_a \right>$')
#     ax.set_title(r'Orden estacionario vs. ruido $\eta$ (promedio sobre R realizaciones)')
#     ax.set_title('Orden estacionario vs. ruido $\eta$ (promedio sobre R realizaciones)')
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=150)
#     plt.show()

#     return {"etas": etas, "va_mean": va_mean, "va_std": va_std}

# def animate_vicsek(params, T=200):
#     rng = np.random.default_rng(params.seed)
#     state = vicsek.initialize(params)

#     fig, (ax_anim, ax_va) = plt.subplots(1, 2, figsize=(12, 6))
#     ax_anim.set_xlim(0, params.L)
#     ax_anim.set_ylim(0, params.L)
#     ax_anim.set_aspect('equal')
#     scat = ax_anim.quiver(state.xy[:, 0], state.xy[:, 1],
#                           np.cos(state.theta), np.sin(state.theta),
#                           angles='xy', scale_units='xy', scale=1.0, width=0.005)
#     ax_anim.set_title("Simulación Vicsek")

#     va_values = []
#     t_values = []
#     ax_va.set_title("Evolución del orden")
#     ax_va.set_xlabel("Iteraciones")
#     ax_va.set_ylabel("Parámetro de orden $v_a$")
#     line_va, = ax_va.plot([], [], lw=1.5)
#     ax_va.set_xlim(0, T)
#     ax_va.set_ylim(0, 1)

#     def update(frame):
#         nonlocal state
#         state = vicsek.step(state, params, rng)
#         scat.set_offsets(state.xy)
#         scat.set_UVC(np.cos(state.theta), np.sin(state.theta))
#         va = order_parameter(state.theta)
#         va_values.append(va)
#         t_values.append(frame)
#         line_va.set_data(t_values, va_values)
#         return scat, line_va

#     ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
#     plt.tight_layout()
#     plt.show()

# def order_parameter(theta: np.ndarray) -> float:
#     """Parámetro de orden v_a = |<v>| con |v|=1 (el módulo se cancela)."""
#     vx = np.cos(theta).mean()
#     vy = np.sin(theta).mean()
#     return float(np.hypot(vx, vy))

# def run_and_plot_order(params: vicsek.VicsekParams, T: int = 1500, record_every: int = 5,
#                        save_path: Optional[str] = None) -> dict:
#     """Simula T pasos y grafica v_a vs. tiempo. Devuelve diccionario con arrays."""
#     rng = np.random.default_rng(params.seed)
#     state = vicsek.initialize(params)
#     times, vah = [], []
#     for t in range(T):
#         if t % record_every == 0:
#             times.append(t)
#             vah.append(vicsek.order_parameter(state.theta))
#         state = vicsek.step(state, params, rng)
#     times = np.array(times)
#     vah = np.array(vah)

#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(times, vah, lw=1.5)
#     ax.set_title('Evolución del orden — Modelo de Vicsek')
#     ax.set_xlabel('Iteraciones')
#     ax.set_ylabel('Parámetro de orden $v_a$')
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=150)
#     plt.show()
#     return {"t": times, "va": vah}

# def sweep_density_stationary(
#     params: vicsek.VicsekParams,
#     rhos,
#     T: int = 1500,
#     T_trans: int = 300,
#     R: int = 5,
#     mode: str = 'vary_N',         # 'vary_N' fija L y ajusta N; 'vary_L' fija N y ajusta L
#     save_path: Optional[str] = None,
#     seed: Optional[int] = None
# ) -> Dict[str, np.ndarray]:
#     if plt is None:
#         raise RuntimeError("matplotlib no disponible para graficar")

#     rhos = np.asarray(list(rhos), dtype=float)
#     rng_master = np.random.default_rng(params.seed if seed is None else seed)

#     va_means = []
#     for rho in rhos:
#         vals = []
#         for r_id in range(R):
#             subseed = int(rng_master.integers(0, 2**31-1))
#             if mode == 'vary_N':
#                 L = params.L
#                 N = max(1, int(round(float(rho) * (L**2))))
#             elif mode == 'vary_L':
#                 N = params.N
#                 L = float(np.sqrt(N / float(rho)))
#             else:
#                 raise ValueError("mode debe ser 'vary_N' o 'vary_L'")

#             local_dict = {**params.__dict__, 'N': N, 'L': L, 'seed': subseed}
#             local = vicsek.VicsekParams(**local_dict)

#             rng = np.random.default_rng(local.seed)
#             state = vicsek.initialize(local)
#             vals_t = []
#             for t in range(T):
#                 if t >= T_trans:
#                     vals_t.append(vicsek.order_parameter(state.theta))
#                 state = vicsek.step(state, local, rng)
#             vals.append(np.mean(vals_t) if len(vals_t) else np.nan)

#         vals = np.asarray(vals, dtype=float)
#         va_means.append((np.nanmean(vals), np.nanstd(vals, ddof=1) if R > 1 else 0.0))

#     va_means = np.asarray(va_means)
#     va_mean = va_means[:, 0]
#     va_std  = va_means[:, 1]

#     # Plot con barras de error
#     fig, ax = plt.subplots(figsize=(7.0, 4.2))
#     ax.errorbar(rhos, va_mean, yerr=va_std, fmt='o-', capsize=4, linewidth=1.5)
#     ax.set_xlabel('Densidad $\\rho$')
#     ax.set_ylabel('Orden estacionario $\\langle v_a \\rangle$')
#     ax.set_title('Orden estacionario vs. densidad $\\rho$ (promedio sobre R realizaciones)')
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=150)
#     plt.show()

#     return {"rhos": rhos, "va_mean": va_mean, "va_std": va_std}



# if __name__ == "__main__":
#     seasonal_evolution_by_noise(N=300, L=7.0, r=1.0, v=0.03, eta_values=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0], T=500)
#     params = VicsekParams(N=300, L=7.0, r=1.0, v=0.03, eta=0.5, seed=1)
#     params2 = VicsekParams(N=50, L=10, r=1.0, v=0.03, eta=2, seed=1)
#     animate_vicsek(params, T=300)
#     animate_vicsek(params2, T=300)
#     rhos = [0.1, 0.25, 0.5, 1.0, 2.0]
#     res_rho = sweep_density_stationary(
#         params,
#         rhos=rhos,
#         T=1500,
#         T_trans=300,
#         R=5,
#         mode='vary_L',
#     )
#     sweep_density_stationary(
#         params,
#         rhos=rhos,
#         T=1500,
#         T_trans=300,
#         R=5,
#         mode='vary_N',
#     )
#     etas = [0.2, 0.5, 1.0, 2.0, 4.0]
#     res_eta = sweep_eta_stationary(
#         params,
#         etas=etas,
#         T=1500,
#         T_trans=300,
#         R=5,
#     )

def animate_vicsek(filename: str, L: float, color_by_angle: bool = False):
    data = np.load(filename, allow_pickle=True)
    xy_list = data["xy"]
    theta_list = data["theta"]
    va_hist = data["va"]
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

timestamp = int(datetime.now().timestamp())
params = vicsek.VicsekParams(seed=timestamp)
N = input(f'Ingrese la cantidad de particulas N (default {params.N}): ')
L = input(f'Ingrese la longitud de la grilla L (default {params.L}): ')
r = input(f'Ingrese el radio de interaccion entre particulas r (default {params.r}): ')
v = input(f'Ingrese el modulo de la velocidad de las particulas v (default {params.v}): ')
eta = input(f'Ingrese \u03B7 (default {params.eta}): ')
T = input('Ingrese la cantidad de frames T (default 300): ')
params.N = params.N if N == "" else int(N)
params.L = params.L if L == "" else float(L)
params.r = params.r if r == "" else float(r)
params.v = params.v if v == "" else float(v)
params.eta = params.eta if eta == "" else float(eta)
T = 300 if T == "" else int(T)

states_xy, states_theta, _ = vicsek.simulate(params, T=T)
vicsek.save_simulation(params, states_xy, states_theta)
print("Nueva simulacion 'Off - Lattice' disponible")
print(f"N = {params.N}, L = {params.L}, v = {params.v}, r = {params.r}, eta = {params.eta}")
print(f"Timestamp de la simulacion: {timestamp}")

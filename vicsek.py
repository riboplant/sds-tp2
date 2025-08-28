import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

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

def dist_periodic(xy1, xy2, L):
        x1, y1 = xy1
        x2, y2 = xy2
        dx = x1 - x2
        dy = y1 - y2
        dx -= L * round(dx / L)
        dy -= L * round(dy / L)
        return math.sqrt(dx**2 + dy**2)

# =============================
# Modelo de Vicsek
# =============================

@dataclass
class VicsekParams:
    N: int = 300
    L: float = 5.0
    r: float = 1.0
    v: float = 0.03
    eta: float = 0.1
    voter: str = 'NO'
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

def get_indexes_to_check(i, j):
    res = []
    res.append([i,j])
    res.append([i-1, j+1])
    res.append([i, j+1])
    res.append([i+1, j+1])
    res.append([i+1, j])
    return res

def cell_index_method(xy, N, L, r_c):
    M = max(1, int(np.floor(L / r_c)))   # tamaño de grilla
    l = L / M

    # índices de celda (j=columna=x, i=fila=y)  [consistentes]
    ij = np.floor(xy / l).astype(int) % M
    j = ij[:, 0]; i = ij[:, 1]

    # mapa celda -> lista de índices
    cells = [[[] for _ in range(M)] for _ in range(M)]
    for k in range(N):
        cells[i[k]][j[k]].append(k)

    res = [[] for _ in range(N)]
    r2 = r_c * r_c

    # vecinos para evitar doble contaje (misma celda y “semiplano”)
    neigh = [(0,0), (0,1), (1,1), (1,0), (1,-1)]

    for ci in range(M):
        for cj in range(M):
            A = cells[ci][cj]
            if not A: 
                continue
            Ax = xy[A, 0][:, None]
            Ay = xy[A, 1][:, None]

            for di, dj in neigh:
                ni = (ci + di) % M
                nj = (cj + dj) % M
                B = cells[ni][nj]
                if not B:
                    continue

                Bx = xy[B, 0][None, :]
                By = xy[B, 1][None, :]

                # imagen mínima vectorizada
                dx = Ax - Bx
                dy = Ay - By
                dx -= L * np.round(dx / L)
                dy -= L * np.round(dy / L)
                d2 = dx*dx + dy*dy

                mask = d2 <= r2

                if di == 0 and dj == 0:
                    # misma celda: quedarnos con la parte superior de la matriz para no duplicar
                    iu = np.triu_indices(len(A), k=1)
                    keep = np.zeros_like(mask, dtype=bool)
                    keep[iu] = True
                    mask &= keep

                # volcar pares
                I, J = np.where(mask)
                for ii, jj in zip(I, J):
                    a = A[ii]; b = B[jj]
                    res[a].append(b)
                    res[b].append(a)

    return res

def step(state: VicsekState, params: VicsekParams, rng: np.random.Generator) -> VicsekState:
    N, L, r_c, v, eta, voter = params.N, params.L, params.r, params.v, params.eta, params.voter
    noise = rng.uniform(-eta/2.0, eta/2.0, size=N)
    in_range = cell_index_method(state.xy, N, L, r_c)
    angles = np.empty(N, dtype=float)
    for i in range(N):
        inds = [i] + in_range[i]
        if voter == 'SI':
            angles[i] = state.theta[inds[rng.integers(0, len(inds))]]
        else:
            theta = state.theta[inds]
            s = np.sin(theta).sum()
            c = np.cos(theta).sum()
            angles[i] = np.arctan2(s, c)
    new_theta = angles + noise
    vx = v * np.cos(new_theta)
    vy = v * np.sin(new_theta)
    new_xy = wrap_periodic(state.xy + np.stack([vx, vy], axis=1), L)
    return VicsekState(xy=new_xy, theta=new_theta)

# =============================
# Simulaciones
# =============================

def generate_dynamic_file(state: VicsekState, params: VicsekParams, directory: str, t: int):
    N, v = params.N, params.v
    xy, theta = state.xy, state.theta
    file = os.path.join(directory, f"{t}.txt")
    with open(file, 'w') as f:
        for i in range(N):
            f.write(f"{xy[i][0]} {xy[i][1]} {theta[i]}\n")

def simulate(params: VicsekParams, T: int = 300):
    rng = np.random.default_rng(params.seed)
    state = initialize(params)
    states_xy = [state.xy.copy()]
    states_theta = [state.theta.copy()]
    va_hist = [order_parameter(state.theta)]
    
    for _ in range(1, T):
        state = step(state, params, rng)
        states_xy.append(state.xy.copy())
        states_theta.append(state.theta.copy())
        va_hist.append(order_parameter(state.theta))

    return states_xy, states_theta, va_hist

def save_simulation(simulation_name, params, states_xy, states_theta):
    T = len(states_xy)
    directory = f"data/simulations/{simulation_name}"
    os.makedirs(directory, exist_ok=True)

    static_file = os.path.join(directory, "static.txt")
    with open(static_file, 'w') as f:
        f.write(f"{params.N}\n{params.L}\n{params.v}\n{params.r}\n{params.eta}\n{T}\n")
    
    for t in range(T):
        file = os.path.join(directory, f"{t}.txt")
        xy = states_xy[t]
        theta = states_theta[t]
        with open(file, 'w') as f:
            for i in range(params.N):
                f.write(f"{xy[i][0]} {xy[i][1]} {theta[i]}\n")

def key_name(name: str):
    base = name[:-4].lower()  # quitar ".txt"
    if base == "static":
        return (1, float("inf"))
    return (0, int(base))

def process_simulation(simulation_name: str):
    directory = f"data/simulations/{simulation_name}"
    static_file = os.path.join(directory, "static.txt")
    N, L, v, r, eta, T = 0, 0, 0, 0, 0, 0
    with open(static_file, "r") as f:
        N = int(f.readline())
        L = float(f.readline())
        v = float(f.readline())
        r = float(f.readline())
        eta = float(f.readline())
        T = int(f.readline())

    xy = []
    theta = []
    va_hist = []
    for _, _, files in os.walk(directory):
        files = sorted(files, key=key_name)
        for name in files:
            if name != "static.txt":
                dynamic_file = os.path.join(directory, name)
                xy_d = []
                theta_d = []
                with open(dynamic_file, "r") as f:
                    for line in f:
                        vals = line.strip().split(' ')
                        xy_d.append([float(vals[0]), float(vals[1])])
                        theta_d.append(float(vals[2]))
                xy.append(np.asarray(xy_d))
                theta.append(np.asarray(theta_d))
                va_hist.append(order_parameter(theta_d))    
    return N, L, v, r, eta, T, xy, theta, va_hist

import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime

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

def generate_dynamic_file(state: VicsekState, params: VicsekParams, directory: str, t: int):
    N, v = params.N, params.v
    xy, theta = state.xy, state.theta
    file = os.path.join(directory, f"{t}.txt")
    with open(file, 'w') as f:
        for i in range(N):
            f.write(f"{xy[i][0]} {xy[i][1]} {v * np.cos(theta[i])} {v * np.sin(theta[i])}\n")


def simulate(params: VicsekParams, T: int = 1000) -> str:
    N, L, v = params.N, params.L, params.v
    directory = f"data/{params.seed}"
    os.makedirs(directory, exist_ok=True)

    rng = np.random.default_rng(params.seed)
    state = initialize(params)
    states_xy = [state.xy.copy()]
    states_theta = [state.theta.copy()]
    va_hist = [order_parameter(state.theta)]

    static_file = os.path.join(directory, "static.txt")
    with open(static_file, 'w') as f:
        f.write(f"{N}\n{L}\n")
        for i in range(0, N):
            f.write("properties\n") # TODO: ver que propiedades ponemos aca para cada particula

    generate_dynamic_file(state, params, directory, 0)
    
    for t in range(1, T):
        state = step(state, params, rng)
        generate_dynamic_file(state, params, directory, t)
        states_xy.append(state.xy.copy())
        states_theta.append(state.theta.copy())
        va_hist.append(order_parameter(state.theta))

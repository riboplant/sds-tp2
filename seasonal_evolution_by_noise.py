import matplotlib.pyplot as plt
import vicsek

def seasonal_evolution_by_noise(N: int, L: float, r: float, v: float, eta_values: float, T: int = 1000):
    plt.figure(figsize=(8,6))
    for eta in eta_values:
        params = vicsek.VicsekParams(N=N, L=L, r=r, v=v, eta=eta, seed=42)
        _, _, va_hist = vicsek.simulate(params, T=T)
        plt.plot(va_hist, label=f"eta={eta}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Parámetro de orden $v_a$")
    plt.title("Evolución temporal para distintos valores de ruido \u03B7")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

N = int(input(f'Ingrese la cantidad de particulas N: '))
L = float(input(f'Ingrese la longitud de la grilla L: '))
r = float(input(f'Ingrese el radio de interaccion entre particulas r: '))
v = float(input(f'Ingrese el modulo de la velocidad de las particulas v: '))
T = int(input('Ingrese la cantidad de frames T: '))
seasonal_evolution_by_noise(N, L, r, v, [0.1, 0.25, 0.5, 1.0, 2.0, 4.0], T)

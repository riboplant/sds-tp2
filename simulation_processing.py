import os
import vicsek
import numpy as np

def seasonal_evolution_by_noise(graph_name: str, simulations: list[str]):
    graph_dir = f"data/graphs/seasonal_evolution_by_noise"
    os.makedirs(graph_dir, exist_ok=True)
    graph_name = f"{graph_name}.txt"
    graph_file = os.path.join(graph_dir, graph_name)

    with open(graph_file, 'w') as graph_f:
        for simulation in simulations:
            sim_directory = f"data/simulations/{simulation}"
            static_file = os.path.join(sim_directory, "static.txt")
            N, L, eta, T = 0, 0, 0, 0
            with open(static_file, 'r') as f:
                N = int(f.readline())
                L = float(f.readline())
                next(f)
                next(f)
                eta = float(f.readline())
                T = int(f.readline())

            graph_f.write(f"{N} {L} {eta} {T}")

            for _, _, files in os.walk(sim_directory):
                files = sorted(files, key=vicsek.key_name)
                for name in files:
                    if name != "static.txt":
                        dynamic_file = os.path.join(sim_directory, name)
                        theta_d = []
                        with open(dynamic_file, "r") as f:
                            for line in f:
                                vals = line.strip().split(' ')
                                theta_d.append(float(vals[2]))
                        graph_f.write(f" {vicsek.order_parameter(theta_d)}")
            graph_f.write("\n")

def fixed_density(graph_name: str, simulations: list[list[str]]):
    graph_dir = f"data/graphs/fixed_density"
    os.makedirs(graph_dir, exist_ok=True)
    graph_name = f"{graph_name}.txt"
    graph_file = os.path.join(graph_dir, graph_name)

    with open(graph_file, 'w') as graph_f:
        for sim_group in simulations:
            first = True
            N, L = 0, 0
            for simulation in sim_group:
                sim_directory = f"data/simulations/{simulation}"
                static_file = os.path.join(sim_directory, "static.txt")
                eta = 0
                with open(static_file, 'r') as f:
                    N = int(f.readline())
                    L = float(f.readline())
                    next(f)
                    next(f)
                    eta = float(f.readline())
                if first:
                    first = False
                    graph_f.write(f"{N} {L} {len(sim_group)}\n")

                va_hist = []
                for _, _, files in os.walk(sim_directory):
                    files = sorted(files, key=vicsek.key_name)
                    for name in files:
                        if name != "static.txt":
                            dynamic_file = os.path.join(sim_directory, name)
                            theta_d = []
                            with open(dynamic_file, "r") as f:
                                for line in f:
                                    vals = line.strip().split(' ')
                                    theta_d.append(float(vals[2]))
                            va_hist.append(vicsek.order_parameter(theta_d))
                graph_f.write(f"{eta} {np.mean(va_hist)}\n")

def varied_density_eta_fixed(graph_name: str, simulations: list[str]):
    graph_dir = f"data/graphs/varied_density_eta_fixed"
    os.makedirs(graph_dir, exist_ok=True)
    graph_name = f"{graph_name}.txt"
    graph_file = os.path.join(graph_dir, graph_name)

    with open(graph_file, 'w') as graph_f:
        N, L, eta = 0, 0, 0
        first = True
        for simulation in simulations:
            sim_directory = f"data/simulations/{simulation}"
            static_file = os.path.join(sim_directory, "static.txt")
            with open(static_file, 'r') as f:
                N = int(f.readline())
                L = float(f.readline())
                next(f)
                next(f)
                eta = float(f.readline())
            if first:
                first = False
                graph_f.write(f"{L} {eta}\n")
            va_hist = []
            for _, _, files in os.walk(sim_directory):
                files = sorted(files, key=vicsek.key_name)
                for name in files:
                    if name != "static.txt":
                        dynamic_file = os.path.join(sim_directory, name)
                        theta_d = []
                        with open(dynamic_file, "r") as f:
                            for line in f:
                                vals = line.strip().split(' ')
                                theta_d.append(float(vals[2]))
                        va_hist.append(vicsek.order_parameter(theta_d))
            graph_f.write(f"{N/(L**2)} {np.mean(va_hist)}\n")

def generate_seasonal_evolution_by_noise():
    graph_name = input("Ingrese el nombre del grafico a generar: ")
    simulations = input("Ingrese, separados por coma y sin espacios, los nombres de cada una de las simulaciones a incluir: ")
    simulations = simulations.split(',')
    seasonal_evolution_by_noise(graph_name, simulations)

def generate_fixed_density():
    graph_name = input("Ingrese el nombre del grafico a generar: ")
    sim_group = input("Ingrese, separados por coma y sin espacios, los nombres de cada una de las simulaciones a incluir en este grupo. Las mismas deben tener mismo N y L, y deben estar escritas segun orden creciente: ")
    simulations = []
    while(sim_group != ""):
        simulations.append(sim_group.split(','))
        sim_group = input("Ingrese, separados por coma y sin espacios, los nombres de cada una de las simulaciones a incluir en este grupo. Las mismas deben tener mismo N y L, y deben estar escritas segun orden creciente (vacio para terminar): ")
    fixed_density(graph_name, simulations)

def generate_varied_density_eta_fixed():
    graph_name = input("Ingrese el nombre del grafico a generar: ")
    simulations = input("Ingrese, separados por coma y sin espacios, los nombres de cada una de las simulaciones a incluir: ")
    simulations = simulations.split(',')
    varied_density_eta_fixed(graph_name, simulations)

generate_seasonal_evolution_by_noise()
#generate_fixed_density()
#generate_varied_density_eta_fixed()

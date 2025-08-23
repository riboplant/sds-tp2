import os
import vicsek

def seasonal_evolution_by_noise(graph_name, simulations: list[str]):
    graph_dir = f"data/graphs/seasonal_evolution_by_noise"
    os.makedirs(graph_dir, exist_ok=True)
    graph_name = f"{graph_name}.txt"
    graph_file = os.path.join(graph_dir, graph_name)

    with open(graph_file, 'w') as graph_f:
        for simulation in simulations:
            sim_directory = f"data/simulations/{simulation}"
            static_file = os.path.join(sim_directory, "static.txt")
            N, eta, T = 0, 0, 0
            with open(static_file, 'r') as f:
                N = int(f.readline())
                next(f)
                next(f)
                next(f)
                eta = float(f.readline())
                T = int(f.readline())

            graph_f.write(f"{N} {eta} {T}")

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

graph_name = input("Ingrese el nombre del grafico a generar: ")
simulations = input("Ingrese, separados por coma y sin espacios, los nombres de cada una de las simulaciones a incluir: ")
simulations = simulations.split(',')
seasonal_evolution_by_noise(graph_name, simulations)

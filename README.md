# TP2 - Simulación de Sistemas

El presente trabajo implementa y analiza el modelo de Vicsek "off-lattice" para estudiar la transición orden-desorden en sistemas de autopropulsión, desarrollado en el marco de la materia Simulación de Sistemas del Instituto Tecnológico de Buenos Aires. El objetivo es caracterizar el parámetro de orden bajo diferentes configuraciones, comparar el comportamiento clásico con una variante tipo votante e identificar el rol del ruido y la densidad en la dinámica colectiva.

Las funcionalidades incluidas son las siguientes:

- <b>Simulador Vicsek/Voter</b>: Evoluciona partículas autopropulsadas con condiciones periódicas de contorno y permite alternar entre la dinámica tradicional y un modo de votante.
- <b>Cell Index Method</b>: Utiliza una grilla dinámica para acelerar la detección de vecinos, reduciendo el costo computacional por paso temporal.
- <b>Persistencia de Corridas</b>: Guarda cada simulación bajo `data/simulations/` en formato legible (estado estático y archivos dinámicos por iteración).
- <b>Procesamiento Estadístico</b>: Resume series temporales para comparar el orden para distintos valores de ruido o densidad y calcula medias/desvíos.
- <b>Generación de Gráficos</b>: Produce figuras a partir de los resúmenes generados, facilitando el análisis del orden estacionario.
- <b>Visualización</b>: Anima las trayectorias y orientaciones de las partículas, coloreando por ángulo para resaltar estructuras colectivas.

<details>
  <summary>Contenidos</summary>
  <ol>
    <li><a href="#instalación">Instalación</a></li>
    <li><a href="#instrucciones">Instrucciones</a></li>
    <li><a href="#manual-de-usuario">Manual de Usuario</a></li>
    <li><a href="#integrantes">Integrantes</a></li>
  </ol>
</details>

## Instalación

Clonar el repositorio:

- HTTPS:
  ```sh
  git clone https://github.com/riboplant/sds-tp2.git
  ```
- SSH:
  ```sh
  git clone git@github.com:riboplant/sds-tp2.git
  ```

Desde la raíz del proyecto puede crearse un entorno virtual y activar las dependencias necesarias:

```sh
python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install numpy matplotlib
```

> **Requisito**: Python 3.10 (o superior) con `pip` disponible.

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

## Instrucciones

Todos los comandos deben ejecutarse desde la raíz del repositorio con el entorno virtual (si se creó) ya activado. Los scripts principales son:

- `simulate.py`: genera simulaciones del modelo Vicsek (y variante votante) y persiste los estados bajo `data/simulations/<nombre>/`.
- `visualizer.py`: anima una simulación previamente guardada, mostrando posiciones y evolución del parámetro de orden.
- `simulation_processing.py`: a partir de múltiples simulaciones, produce resúmenes estadísticos para ruido o densidad.
- `graphs.py`: lee los resúmenes del punto anterior y construye las figuras finales.

La carpeta `data/` se crea en la primera ejecución y contendrá dos subdirectorios:

```text
data/
|-- simulations/        # corridas individuales 
\-- graphs/             # resúmenes utilizados para graficar
```

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

## Manual de Usuario

A continuación se detallan los comandos y parámetros de cada script. En todos los casos se asume que el comando se ejecuta desde el directorio raíz del proyecto.

### Simulación Vicsek (Off-Lattice)

```sh
python3 simulate.py
```

El script solicita interactivamente los parámetros de la corrida. Presionando Enter se aceptan los valores por defecto (`N=300`, `L=5.0`, `r=1.0`, `v=0.03`, `η=0.1`, `T=300`).

- `simulation_name`: nombre de la carpeta donde se guarda la corrida (`data/simulations/<simulation_name>/`).
- `N`: cantidad de partículas.
- `L`: longitud del lado del dominio cuadrado con contorno periódico.
- `r`: radio de interacción (también funciona como `r_c`).
- `v`: módulo de la velocidad autopropulsada.
- `η`: amplitud del ruido uniforme.
- `voter`: escribir `SI` para activar la regla de votante (el ángulo se copia de un vecino al azar); cualquier otra entrada mantiene la dinámica Vicsek estándar.
- `T`: cantidad de iteraciones que se registran.

Cada corrida genera:

- `static.txt`: contiene `N`, `L`, `v`, `r`, `η` y `T`.
- `0.txt`, `1.txt`, ... `T-1.txt`: posición `(x, y)` y ángulo `θ` de cada partícula en cada frame.

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

### Visualización de simulaciones

```sh
python3 visualizer.py
```

Ingresar el `simulation_name` guardado previamente. El visualizador reconstruye los estados con `vicsek.process_simulation`, anima el sistema y muestra en paralelo la evolución del parámetro de orden `v_a(t)`. La opción `color_by_angle` está activada por defecto para evidenciar dominios con distintas orientaciones.

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

### Procesamiento estadístico

```sh
python3 simulation_processing.py
```

El archivo define tres rutinas que leen varias corridas y generan resúmenes en `data/graphs/`:

1. `seasonal_evolution_by_noise`: agrupa simulaciones con parámetros idénticos salvo `η` y escribe, por cada corrida, la serie temporal de `v_a`.
2. `fixed_density`: asume conjuntos con densidad `ρ = N/L²` fija y múltiples valores de `η`, guardando la media y el desvío estándar de `v_a` tras descartar transitorios (`30%` inicial).
3. `varied_density_eta_fixed`: mantiene fijo `η` y registra cómo varía el orden con la densidad (se descarta el `60%` inicial de cada corrida).

Para ejecutar una rutina, puede descomentarse la llamada correspondiente al final del archivo o invocarla manualmente desde un intérprete Python. Cada función pedirá por consola los nombres de las simulaciones involucradas y el identificador del gráfico resultante.

Los archivos generados (`*.txt`) pueden reutilizarse en posteriores ejecuciones sin repetir las simulaciones.

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

### Gráficos a partir de resúmenes

```sh
python3 graphs.py
```

Seleccionar el tipo de gráfico descomentando la función deseada al final del script (las otras permanecen disponibles):

- `seasonal_evolution_by_noise(graph_name)`
- `fixed_density(graph_name)`
- `varied_density_eta_fixed(graph_name)`

El programa leerá el archivo `data/graphs/<carpeta>/<graph_name>.txt`, levantará los datos y mostrará la figura con Matplotlib (`plt.show()`).

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

## Integrantes

Martín Alejandro Barnatán (64463) - mbarnatan@itba.edu.ar

Ignacio Martín Maruottolo Quiroga (64611) - imaruottoloquiroga@itba.edu.ar

Ignacio Pedemonte Berthoud (64908) - ipedemonteberthoud@itba.edu.ar

<p align="right">(<a href="#tp2---simulación-de-sistemas">Volver</a>)</p>

#SIMULACIÓN 3 MODOS 

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import hermite
import scipy.special as sp
from itertools import combinations
import random
from skimage.io import imsave

# Crear carpeta de salida
output_folder = "superposiciones"
os.makedirs(output_folder, exist_ok=True)

# Parámetros
lambda_ = 632.8e-9
k = 2 * np.pi / lambda_
W0 = 1e-3
z_R = np.pi * W0**2 / lambda_

# Funciones auxiliares
def A_lm(l, m, W_z):
    return np.sqrt(2 / (np.pi * W_z**2) * (1 / (2**(l + m) 
                                                * sp.factorial(l) 
                                                * sp.factorial(m))))

def W(z):
    return W0 * np.sqrt(1 + (z / z_R)**2)

def R(z):
    return z * (1 + (z_R / z)**2) if z != 0 else np.inf

def xi(z):
    return np.arctan(z / z_R)

def hermite_gaussian(x, y, z, l, m):
    H_l = hermite(l)
    H_m = hermite(m)
    Wz = W(z)
    Alm = A_lm(l, m, Wz)
    G_l = H_l(np.sqrt(2) * x / Wz)
    G_m = H_m(np.sqrt(2) * y / Wz)
    fase = np.exp(-1j * k * z - 1j * k * (x**2 + y**2) 
                  / (2 * R(z)) + 1j * (l + m + 1) * xi(z))
    amplitud = Alm * G_l * G_m * np.exp(-(x**2 + y**2) / Wz**2)
    return amplitud * fase

def intensidad(hermite_gaussian, W_0, W_z):
    factor_W = (W_0 / W_z)**2
    return factor_W * (abs(hermite_gaussian))**2

# Crear malla de puntos
gridsize = 128
x_vals = np.linspace(-2e-3, 2e-3, gridsize)
y_vals = np.linspace(-2e-3, 2e-3, gridsize)
X, Y = np.meshgrid(x_vals, y_vals)
z = 0

# 1. Precomputar todos los modos HG
modos = {}

for l in range(6):
    for m in range(6):
        modo = np.zeros(X.shape, dtype=complex)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                modo[i, j] = hermite_gaussian(X[i, j], Y[i, j], z, l, m)
        modos[(l, m)] = modo

# 2. Calcular intensidad máxima global
print("Calculando intensidad máxima global...")
intensidad_maxima_global = 0

for (l, m), modo in modos.items():
    I = intensidad(modo, W0, W(z))
    intensidad_maxima_global = max(intensidad_maxima_global, np.max(I))

for (l1, m1), (l2, m2), (l3, m3) in combinations(modos.keys(), 3):
    for pesos in [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [1/3, 1/3, 1/3]]:
        w1, w2, w3 = pesos
        superposicion = w1 * modos[(l1, m1)] + w2 * modos[(l2, m2)] 
        + w3 * modos[(l3, m3)]
        I = intensidad(superposicion, W0, W(z))
        intensidad_maxima_global = max(intensidad_maxima_global, np.max(I))

print(f"Intensidad máxima global: {intensidad_maxima_global}")

# 3. Generar imágenes
total_imagenes = 10000
np.random.seed(42)

# Combinaciones únicas de 3 modos (sin importar orden)
combinaciones_modos = list(set(
    tuple(sorted([a, b, c]))
    for a, b, c in combinations(modos.keys(), 3)
))
random.shuffle(combinaciones_modos)

# Selección aleatoria de 36 posiciones para insertar modos puros
posiciones_modos_puros = list(np.random.choice(total_imagenes, size=36,
                                               replace=False))
random.shuffle(posiciones_modos_puros)  # Esto rompe la correlación ordenada

# Selección aleatoria de 36 modos puros distintos
modos_puros_disponibles = sorted(modos.keys())
modos_puros_seleccionados = random.sample(modos_puros_disponibles, 36)

# Mapeo de índice a modo puro (l, m)
mapa_modos_puros = {
    idx: modos_puros_seleccionados[i]
    for i, idx in enumerate(posiciones_modos_puros)
}

for idx in range(total_imagenes):
    if idx in mapa_modos_puros:
        (l1, m1) = mapa_modos_puros[idx]
        w1 = 1.0

        if (l1, m1) == (0, 0):
            (l2, m2), w2 = (0, 2), 0.0
            (l3, m3), w3 = (0, 1), 0.0
            
        elif (l1, m1) == (0, 1):
            (l2, m2), w2 = (0, 0), 0.0
            (l3, m3), w3 = (0, 2), 0.0
        else:
            (l2, m2), w2 = (0, 0), 0.0
            (l3, m3), w3 = (0, 1), 0.0
    else:
        combo_idx = (idx - len(mapa_modos_puros)) % len(combinaciones_modos)
        (l1, m1), (l2, m2), (l3, m3) = combinaciones_modos[combo_idx]
        w1, w2, w3 = np.random.dirichlet([1, 1, 1])

    # Ordenar por (l, m)
    modos_y_pesos = sorted(
        [((l1, m1), w1), ((l2, m2), w2), ((l3, m3), w3)],
        key=lambda x: x[0]
    )
    (l1, m1), w1 = modos_y_pesos[0]
    (l2, m2), w2 = modos_y_pesos[1]
    (l3, m3), w3 = modos_y_pesos[2]

    # Crear superposición
    superposicion = (
        w1 * modos[(l1, m1)] +
        w2 * modos[(l2, m2)] +
        w3 * modos[(l3, m3)]
    )
    I = intensidad(superposicion, W0, W(z)) / intensidad_maxima_global

    # Guardar imagen
    filename = f"{idx:04d}_{l1}_{m1}_{w1:.4f}_{l2}_{m2}_{w2:.4f}_{l3}_{m3}_{w3:.4f}.png"
    filepath = os.path.join(output_folder, filename)

    plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.imshow(I, cmap='inferno', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Imagen {idx + 1}/{total_imagenes} guardada: {filename}")

import numpy as np
from skimage import color, io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def procesar_imagen(nombre_imagen):
    # Leer la imagen RGB
    imagen_rgb = io.imread(nombre_imagen)

    # Verificar que sea una imagen RGB
    if len(imagen_rgb.shape) != 3 or imagen_rgb.shape[2] != 3:
        raise ValueError("La imagen de entrada debe ser una imagen RGB.")

    # Convertir la imagen RGB al espacio de color XYZ
    imagen_xyz = color.rgb2xyz(imagen_rgb / 255.0)  # Escalar a rango [0, 1]

    # Extraer el canal X
    canal_x = imagen_xyz[:, :, 0]

    # Calcular el umbral con el método de Otsu
    umbral = threshold_otsu(canal_x)

    # Binarizar la imagen
    f_plantilla = (canal_x > umbral).astype(np.uint8)

    # Multiplicar el canal X por la máscara binaria
    imagen_procesada = ((canal_x * 255).astype(np.uint8) * f_plantilla)

    # Mostrar la imagen final procesada
    plt.figure(figsize=(10, 10))
    plt.imshow(imagen_procesada, cmap='gray')
    plt.title('Imagen Final Procesada')
    plt.axis('off')
    plt.show()

    return imagen_procesada


procesar_imagen('images/hueso.jpg')

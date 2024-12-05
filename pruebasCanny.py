import cv2

# Cargar la imagen
img = cv2.imread('image.png')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de la media a la imagen
#blurred = cv2.blur(gray, (5, 5))

# Detectar bordes usando el algoritmo de Canny
edges = cv2.Canny(gray, 60, 70)

# Encontrar contornos en la imagen
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen
cv2.imshow('Imagen con contornos', img)


# Esperar a que el usuario cierre la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()
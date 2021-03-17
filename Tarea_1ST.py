import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv_helper(fragment, kernel):
    """ multiplica 2 matrices y devuelve su suma"""

    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row, col] * kernel[row, col]
    return result


def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col = image.shape  # asigna alto y ancho de la imagen
    kernel_row, kernel_col = kernel.shape  # asigna alto y ancho del filtro

    output = np.zeros(image.shape)  # matriz donde guardo el resultado

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = conv_helper(
                image[row:row + kernel_row,
                col:col + kernel_col], kernel)

    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()

    return output


"Main"
path = r"C:\Users\EDGAR CRUZ\Pictures\IMG_5966.JPG"
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
cv2.imshow('image', gray)
kernel = np.ones((3,3))  # Se usa el filtro estandar
resultado = convolution(gray, kernel)
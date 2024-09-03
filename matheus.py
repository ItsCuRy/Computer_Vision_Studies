import cv2
import numpy as np
from scipy.interpolate import griddata


image = cv2.imread("Photos/paisagem.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1. Smooth the image replacing every pixel by the average gray-level values
# of some small neighborhood of it. This serves a few purposes. It is easier to separate
# the objects from the background when the image is smoother, see [1, 2, 5].
# Averaging also makes the samples we take less vulnerable to errors both in
# magnitude and location, as slopes become more moderate. 

smoothed_image = cv2.blur(gray_image, (5, 5))

# Step 2. Derive the gray-level gradient magnitude image from the smoothed
# original image. 

sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(sobel_x, sobel_y)
magnitude = magnitude.astype(np.uint8)

# Step 3. Apply thresholding and a local-maxima-directed thinning processes to
# the gradient, obtaining paths and points in the image plane having local gradient
# maxima. These point at pixels at which the original image gray levels are good
# candidates for local threshold. 

_, threshold_image = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Direção do gradiente em radianos
direction = np.arctan2(sobel_y, sobel_x)

# Converter a direção do gradiente para graus e normalizar entre 0 e 180
direction = np.degrees(direction)
direction[direction < 0] += 180

non_max_suppression = np.zeros_like(magnitude)

for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        # Obter a direção do pixel atual
        angle = direction[i, j]

        # Comparar o pixel atual com seus vizinhos na direção do gradiente
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            before_pixel = magnitude[i, j - 1]
            after_pixel = magnitude[i, j + 1]
        elif (22.5 <= angle < 67.5):
            before_pixel = magnitude[i - 1, j + 1]
            after_pixel = magnitude[i + 1, j - 1]
        elif (67.5 <= angle < 112.5):
            before_pixel = magnitude[i - 1, j]
            after_pixel = magnitude[i + 1, j]
        else:
            before_pixel = magnitude[i - 1, j - 1]
            after_pixel = magnitude[i + 1, j + 1]

        # Manter o pixel atual se for um máximo local
        if (magnitude[i, j] >= before_pixel) and (magnitude[i, j] >= after_pixel):
            non_max_suppression[i, j] = magnitude[i, j]
        else:
            non_max_suppression[i, j] = 0

# Normalizar a imagem final para visualização
non_max_suppression = (non_max_suppression / non_max_suppression.max()) * 255
non_max_suppression = non_max_suppression.astype(np.uint8)


# Step 4. Sample the smoothed image at the places at which the maximal
# gradient-mask points. The gray-level values of the so-determined set of points are to
# be interpolated over the image. The interpolation process that is described in the
# sequel requires that all values on the border of the image also be supplied. We could
# either set the values at the border to the original image gray levels or supply
# somewhat higher values, making sure that the interpolation surface will be above the
# image surface near the border, so that no false objects will be detected there.
# However, it may happen that objects close to the image border are cut by the image frame. 
# Thus, supplying high border values would lead to total or partial loss of
# those objects. The solution chosen was to derive a one-dimensional gradient along
# the border line (like we do in two dimensions), seek for object boundaries there, and
# if such boundaries are found, sample the boundary values and 1-dimensionally and
# linearly interpolate them along the frame fine. Thus the border values are set as a
# result of a 1-dimensional segmentation process. 


sampled_points = []
sampled_values = []
for i in range(0, gray_image.shape[0]):
    for j in range(0, gray_image.shape[1]):
        if non_max_suppression[i, j] != 0:
           sampled_points.append([i,j])
           sampled_values.append(non_max_suppression[i, j])


grid_x, grid_y = np.mgrid[0:gray_image.shape[0], 0:gray_image.shape[1]]

threshold_surface = griddata(sampled_points,sampled_values, (grid_x, grid_y), method='cubic', fill_value=0)

# Normalizar para visualização
threshold_surface = np.clip(threshold_surface, 0, 255).astype(np.uint8)


for i in range(0, gray_image.shape[0]):
    for j in range(0, gray_image.shape[1]):
        if gray_image[i, j] >= threshold_surface[i, j]:
            gray_image[i, j] = 255
        else:
            gray_image[i, j] = 0

cv2.imshow("Threshold Surface", threshold_surface)
cv2.imshow("Original Image", gray_image)
cv2.waitKey(0)
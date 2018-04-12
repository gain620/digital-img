import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def limit(input_val):
    if input_val > 255:
        input_val = 255
    elif input_val < 0:
        input_val = 0

    return input_val


def mean_filter(input_img, kernel):
    output_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            if i == 0 or i == input_img.shape[0] - 1:
                output_img[i][j] = input_img[i][j]
            elif j == 0 or j == input_img.shape[1] - 1:
                output_img[i][j] = input_img[i][j]
            else:
                mean_sum = 0
                for a in range(0, kernel.shape[0]):
                    for b in range(0, kernel.shape[1]):
                        mean_sum = mean_sum + kernel[a][b] * input_img[i - 1 + a][j - 1 + b]

                output_img[i][j] = limit(int(mean_sum))

    return output_img


# laplacian_mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # laplacian mask
unsharp_mask_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # unsharp mask1
# unsharp_mask_2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # unsharp mask2

read = misc.imread('sample.jpg')
img = np.array(read, dtype=np.int32)
new_img = mean_filter(img, unsharp_mask_1)

misc.imsave('unsharp_mask_filter.jpg', new_img)
plt.imshow(new_img, cmap='gray')
plt.show()

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#laplacian_mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # laplacian mask
unsharp_mask_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # unsharp mask1
#unsharp_mask_2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # unsharp mask2


def limit(input_val):
    if input_val > 255:
        input_val = 255
    elif input_val < 0:
        input_val = 0

    return input_val


read = misc.imread('sample.jpg')
img = np.array(read, dtype=np.int32)

output = np.zeros(img.shape, dtype=np.uint8)

for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        if i == 0 or i == img.shape[0]-1:
            output[i][j] = img[i][j]
        elif j == 0 or j == img.shape[1]-1:
            output[i][j] = img[i][j]
        else:
            mean_sum = 0
            for a in range(0, unsharp_mask_1.shape[0]):
                for b in range(0, unsharp_mask_1.shape[1]):
                    mean_sum = mean_sum + unsharp_mask_1[a][b] * img[i-1+a][j-1+b]

            output[i][j] = limit(mean_sum)

misc.imsave('unsharp_mask_filter.jpg', output)
plt.imshow(output, cmap='gray')
plt.show()

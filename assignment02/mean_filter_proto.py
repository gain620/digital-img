import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # mean filter

divide = mask.size

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
            for a in range(0, mask.shape[0]):
                for b in range(0, mask.shape[1]):
                    mean_sum = mean_sum + mask[a][b] * img[i-1+a][j-1+b]

            output[i][j] = int(mean_sum / divide)

misc.imsave('mean_filter.jpg', output)
plt.imshow(output, cmap='gray')
plt.show()

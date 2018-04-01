import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

# alpha 값이 0~1 사이면 명암비가 커지고, -1~0 이면 명암비가 작아진다.
def contrast(input_img,alpha):
    output_img = np.zeros(input_img.shape, dtype=np.int16)

    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            output_img[i][j] = img[i][j] + (img[i][j] - 128) * alpha
            # limit pixels to greyscale from 0 ~ 255
            if output_img[i][j] > 255:
                output_img[i][j] = 255
            elif output_img[i][j] < 0:
                output_img[i][j] = 0

    return output_img


img = misc.imread('./low_contrast.jpg')
new_img = contrast(img, 0.5)
plt.imshow(new_img, cmap = 'gray')
plt.show()
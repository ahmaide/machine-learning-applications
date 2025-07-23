import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def abstraction(updated_y, updated_x, filename):

    map1_load = Image.open(filename).convert("L")

    map1 = np.array(map1_load)

    plt.imshow(map1, cmap='gray')
    plt.show()

    print("size: ",map1.shape)

    map_values = (map1 > 128).astype(np.uint8)

    y, x = map_values.shape

    my = y / updated_y
    mx = x / updated_x

    a_map = np.zeros((updated_y, updated_x), dtype=np.uint8)

    for i in range(updated_y):
        for j in range(updated_x):
            y_s = int(round(i * my))
            y_e = int(round((i + 1) * my))
            x_s = int(round(j * mx))
            x_e = int(round((j + 1) * mx))
            b = map_values[y_s:y_e, x_s:x_e]
            a_map[i, j] = 0 if np.all(b == 0) else 1

    plt.imshow(a_map, cmap='gray')
    plt.show()

    print("new size: ",a_map.shape)

    return a_map


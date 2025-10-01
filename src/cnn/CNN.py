import numpy as np
from PIL import Image
from scipy.signal import correlate2d 
import matplotlib.pyplot as plt


class Convolution:
    def __init__(self, input_shape,  n_kernels, kernel_size):
        self.input_shape = input_shape # (channel, height, width)
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        # Example: 2 kernels each having chnnel amount of filters, and each kernel has kernel size
        self.kernels = np.random.rand(self.n_kernels, input_shape[0], kernel_size, kernel_size)

        print(f"[INIT] Input shape: {self.input_shape}")
        print(f"[INIT] Kernel shape: {self.kernels.shape}")
    
    def forward(self, X):
        C, H, W = self.input_shape
        X = X.astype(np.float32)
        
        print(C, H, W)

        # Here we need to do the cross corelation.
        out_maps = []
        # Loop over the kernels
        for k in range(self.n_kernels):

            # Create feature map for each kernel, Size of feature map depends on the number of channels
            feature_map = np.zeros((H - self.kernel_size + 1, W - self.kernel_size + 1))

            # Loope over all the channels
            for c in range(C):
                feature_map += correlate2d(X[c], self.kernels[k][c], mode="valid")
            
            feature_map = np.maximum(0, feature_map)

            out_maps.append(feature_map)

        return out_maps

    def backward(self):
        pass

class MaxPolling:
    def __init__(self):
        pass

class Dense:
    def __init__(self):
        pass

img = Image.open("image.png")

gray_img = img.convert("L")

gray_array = np.array(gray_img)

plt.imshow(gray_array, cmap="gray")

plt.show()

gray_array = np.expand_dims(gray_array, axis=0)

# print(gray_array.shape)


cnn = Convolution(gray_array.shape, 2, 50)

maps = cnn.forward(gray_array)

plt.imshow(maps[0], cmap="gray")

plt.axis("off")

plt.show()

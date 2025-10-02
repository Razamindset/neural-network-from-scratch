import numpy as np
from PIL import Image
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt


class Convolution:
    def __init__(self, input_shape,  n_kernels, kernel_size):
        self.input_shape = input_shape # (channel, height, width)
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        # Example: 2 kernels each having chnnel amount of filters, and each kernel has kernel size
        self.kernels = np.random.rand(self.n_kernels, input_shape[0], kernel_size, kernel_size)

        # 1 Bias for each kernel. Note Each kernel can have multiple features
        self.kernels_bias = np.random.rand(self.n_kernels)

        print(f"[INIT] Input shape: {self.input_shape}")
        print(f"[INIT] Kernel shape: {self.kernels.shape}")
    
    def forward(self, X):
        C, H, W = self.input_shape

        X = X.astype(np.float32)

        self.last_input = X        

        # Here we need to do the cross corelation.
        out_maps = []
        # Loop over the kernels
        for k in range(self.n_kernels):

            # Create feature map for each kernel, Size of feature map depends on the number of channels
            feature_map = np.zeros((H - self.kernel_size + 1, W - self.kernel_size + 1))

            # Loope over all the channels
            for c in range(C):
                feature_map += correlate2d(X[c], self.kernels[k][c], mode="valid")


            feature_map += self.kernels_bias[k]

            out_maps.append(feature_map)
        
        out_maps = np.array(out_maps)

        self.last_output = out_maps.copy()

        # RELU Activation 
        out_maps = np.maximum(0, out_maps)

        return out_maps, self.last_output

    def backward(self,output_gradient, learning_rate):
        """Calculate and update the kernel gradient and bias"""

        # Output gradient is the part=diff of error at the output layer dE / dY  

        # we need t calculate dL / dK -> Kernel gradient.
        # dL / dB bias gradient for the kernel.
        # For now we are not using any bias gradient.

        # dE / dK = X -> cross-corr -> Error at output
        # dE / sB = error at output
        # dE / dX = sum(error at each output ->cross corelate -> each kernel)

        relu_mask = (self.last_output > 0).astype(float)
        output_gradient = output_gradient * relu_mask 

        C, H, W = self.input_shape

        kernel_gradients = np.zeros_like(self.kernels) # = dE / dK

        input_gradients = np.zeros(self.input_shape) # = dE/ dX

        bias_gradients = np.zeros_like(self.kernels_bias)

        # For each kernel
        for k in range(self.n_kernels):
            for c in range(C):
                # For kth kernel and cth channel or depth update its gradients
                kernel_gradients[k, c] += correlate2d(self.last_input[c], output_gradient[k], mode="valid")

                input_gradients[c] += convolve2d(output_gradient[k], self.kernels[k, c], mode="full")

            bias_gradients[k] = np.sum(output_gradient[k])

        # Gradient always points uphil so we negate to go downhill 
        self.kernels -= learning_rate * kernel_gradients

        self.kernels_bias -= learning_rate * bias_gradients

        return input_gradients

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


cnn = Convolution(gray_array.shape, 5, 25)

maps, _ = cnn.forward(gray_array)

plt.imshow(maps[0], cmap="gray")

plt.axis("off")

plt.show()

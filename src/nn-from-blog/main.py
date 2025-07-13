import numpy as np

# I have added the corresponding network image for clear understanding as network.png calc1.png and calc2.png
# All the attribuition is to this great blog https://victorzhou.com/blog/intro-to-neural-networks/ for code and images
# All the code is the same but i have added a lot of comments in my language that make me understnd things easier
# It must be noted that this is not the optimal way but its perfect for understanding
# Uses: stochastic gradient descent - SGD

def sigmoid(x):
    """Squishify the ouput values btw 1 and 0"""
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    # Differentiate the above function Used in backprop
    # As the sigmoid function is used in feed forward. When back propogating we take derivaties
    # As the drivative is applied for each sum and sigmoid activation. The sigmoid activation derviate is needed
    # That is calculated here. See the calc images for understanding
    # As the weight is constant is remains intact while the function takes the differentiation
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    """MSE is the mean of sum of all the losses for all examples"""
    # MSE = 1/n(sigma((y_true - y_pred)**2))
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''
    
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()

        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
    
    def feedforward(self, x):
        # X is a numpy array with two elements one for weight and one for height
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1) # see the image for clarification
        h2 = sigmoid(self.w3 * x[0] + self.w4 + x[1] + self.b2)
        
        # Output layer
        o1 = sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)
        
        # When dealing with unknown amount of params we will use np.dot() which will multiply and add for weighted sum
        return o1
    
    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000
        
        for epoch in range(epochs):
            # For each iteration we will go over all the data one by one
            for x, y_true in zip(data, all_y_trues):
                # Feed forward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 + x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = h1 * self.w5 + h2 * self.w6 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Now we have the prediction. Based in which we will calculate the loss and update the weigts.
                # The simplest way to understand is this
                # The idea is we want to see how changing a weight or bias changes the loss function
                # So we are looking for such a chnage that minimizes the loss
                # For that we do the math to see the relation.
                # The rate change of loss is splitted into several parts for calculation
                # Now we can use this dependence to update the weight -> w1 = w1 - learning_rate * dL/dw1
                
                # Lets calculate partial derivatives
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Neuron o1
                # For this layer output is the prediction so for d_h/d_w it is d_pred/dw = input * sigmoid_derivative
                # For say neuron h1 we will use d_h1/d_w1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1) 
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1) 
                d_ypred_d_b3 = derivative_sigmoid(sum_o1) 

                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h1)
                
                # Neuron h2
                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)

                # Update the  weights and biases
                # Neuron 1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # neuron 2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3 
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4 
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                # neuron 3
                
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5 
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6 
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                # the middle term is missing as it is the last layer as the y_pred/y_output = 1
                
                # Lets dcalcute the loss and log information
                if epoch % 10 == 0:
                    y_pred = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_pred)
                    print("Epoch %d loss: %.3f" % (epoch, loss))

if __name__ == "__main__":
    # Define dataset
    data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
    ])
    all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
    ])
    # Train our neural network!
    network = NeuralNetwork()
    network.train(data, all_y_trues)
    
    # Make some predictions
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
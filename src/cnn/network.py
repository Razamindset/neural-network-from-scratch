def predict(network, input):
    """Perfrom the forward pass for all the layers in the network"""

    # We will loop through all the layer passes output of one as input to the other
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, X_train, y_train, learning_rate=0.001, epochs=1000, verbose=True):
    for epoch in range(epochs):

        error = 0

        for x, y in zip(X_train, y_train):

            output = predict(network, x)

            error += loss(y, output)

            # Backward
            gradient = loss_prime(y, output)

            for layer in reversed(network):
                layer.backward(gradient, learning_rate)
        
        error /= len(X_train)
        if verbose:
            print(f"{epoch + 1}/{epochs}, error={error}")
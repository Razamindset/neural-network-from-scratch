def predict(network, input):
    """Perfrom the forward pass for all the layers in the network"""

    # We will loop through all the layer passes output of one as input to the other
    output = input

    # print(f"[DEBUG] Input to this layer: {input.reshape(-1, 1)}")
    for i, layer in enumerate(network):

        # if i == 1:
        #     print(f"[DEBUG] Shape from {i+1}th layer: {output.shape}")
        #     print(f"[DEBUG] output from {i+1}th layer: {output}")

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
                gradient = layer.backward(gradient, learning_rate)
        
        error /= len(X_train)
        if verbose:
            if epoch % (epochs // 10) == 0:
                print(f"{epoch + 1}/{epochs}, error={error}")
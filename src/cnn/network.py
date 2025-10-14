import numpy as np

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

def train(network, loss, loss_prime, X_train, y_train, learning_rate=0.001, epochs=1000, batch_size=32, verbose=True):
    num_samples = len(X_train)

    for epoch in range(epochs):

        # Shuffle data for each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        X_train = X_train[indices]
        y_train = y_train[indices]

        error = 0

        # Loop over samples in batch
        for start in range(0, num_samples, batch_size):
            end  = start + batch_size

            batch_X = X_train[start:end]
            batch_y = y_train[start:end]

            # The main difference with batching is that gradients are updated for each batch
            #  instead of full samples. Which makes this process better and faster

            batch_error = 0

            for x, y in zip(batch_X, batch_y):

                output = predict(network, x)

                batch_error += loss(y, output)

                # Backward
                gradient = loss_prime(y, output)

                for layer in reversed(network):
                    gradient = layer.backward(gradient, learning_rate)

            # Average batch error
            error += batch_error / len(batch_X)

        # Error over all samples
        error /= num_samples

        if verbose:
            # if epoch % 10 == 0:
            print(f"{epoch + 1}/{epochs}, error={error}")
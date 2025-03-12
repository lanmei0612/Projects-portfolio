import numpy as np


fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """ Initialize weights & biases.
        Weights should be initialized with values drawn from a normal
        distribution scaled by 0.01.
        Biases are initialized to 0.0.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0 * np.zeros(n_neurons)
        self.inputs = None
        self.z = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        """ A forward pass through the layer to give z.
        Compute it using np.dot(...) and then add the biases.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dz):
        """
        Backward pass
        """
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.weights.T)


class ReLu:
    """
    ReLu activation
    """

    def __init__(self):
        self.dz = None
        self.activity = None
        self.z = None

    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        self.activity = z.copy()
        self.activity[self.z <= 0] = 0.0

    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0


class Softmax:
    def __init__(self):
        self.dz = None
        self.probs = None

    def forward(self, z):
        """
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dprobs):
        """
        """
        self.dz = np.empty_like(dprobs)

        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            prob = prob.reshape(-1, 1)
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)


class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)

    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        batch_sz, n_class = probs.shape
        self.dprobs = -oh_y_true / probs
        self.dprobs = self.dprobs / batch_sz


class SGD:
    """

    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights = layer.weights - layer.dweights * self.learning_rate
        layer.biases = layer.biases - layer.dbiases * self.learning_rate


# Helper functions
# Convert probabilities to predictions
def predictions(probs):
    """
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds.astype(int)


# Accuracy
def accuracy(y_preds, y_true):
    """
    """
    return np.mean(y_preds == y_true)


# Training
# A single forward pass through the entire network.
def forward_pass(X, y_true, oh_y_true):
    """
    """
    hidden1.forward(X)
    activation1.forward(hidden1.z)
    hidden2.forward(activation1.activity)
    activation2.forward(hidden2.z)
    output.forward(activation2.activity)
    probs = output_activation.forward(output.z)

    return probs


# A single backward pass through the entire network.
def backward_pass(probs, y_true, oh_y_true):
    """
    """
    crossentropy.backward(probs, oh_y_true)
    output_activation.backward(crossentropy.dprobs)
    output.backward(output_activation.dz)
    activation2.backward(output.dinputs)
    hidden2.backward(activation2.dz)
    activation1.backward(hidden2.dinputs)
    hidden1.backward(activation1.dz)


# Initialize the network and set hyperparameters For example, number of epochs to train, batch size, number of neurons, etc.
hidden1 = DenseLayer(3, 4)
activation1 = ReLu()
hidden2 = DenseLayer(4, 8)
activation2 = ReLu()
output = DenseLayer(8, 3)
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD(0.1)
epochs = 500
n_batch = 20
batch_sz = len(X_train) // n_batch
n_class = 3

# Training loop
for epoch in range(epochs):
    print('epoch:', epoch)
    for batch_i in range(n_batch):
        x_batch = X_train[batch_i*batch_sz:(batch_i+1)*batch_sz]
        y_batch = y_train[batch_i*batch_sz:(batch_i+1)*batch_sz]
        oh_y_true = np.eye(n_class)[y_batch]
        probs = forward_pass(x_batch, y_batch, oh_y_true)
        loss = crossentropy.forward(probs, oh_y_true)
        y_perdict = predictions(probs)
        acc = accuracy(y_perdict, y_batch)
        print('loss:', loss, ', accuracy:', acc)
        backward_pass(probs, y_batch, oh_y_true)
        optimizer.update_params(hidden1)
        optimizer.update_params(hidden2)
        optimizer.update_params(output)

oh_y_test_true = np.eye(n_class)[y_test]
test_result = forward_pass(X_test, y_test, oh_y_test_true)
loss = crossentropy.forward(test_result, oh_y_test_true)
y_perdict = predictions(test_result)
acc = accuracy(y_perdict, y_test)
print('Test loss:', loss, ', Test accuracy:', acc)
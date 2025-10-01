import numpy as np

def sigmoid(x):
    """Activation: f(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def feedforward(self, inputs):
        total = float(np.dot(self.weights, np.asarray(inputs, dtype=float)) + self.bias)
        return sigmoid(total)

class OurNeuralNetwork:
    """
    2-input -> [h1,h2] -> o1
    All neurons use sigmoid. This mirrors the structure in the blog before training.
    h1,h2,o1 start with weights=[0,1], bias=0 to match the example.
    """
    def __init__(self):
        weights = np.array([0.0, 1.0])
        bias = 0.0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        # o1 takes [out_h1, out_h2] as its inputs
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return float(out_o1)

if __name__ == "__main__":
    # Demo values from the tutorial
    weights = np.array([0.0, 1.0])
    bias = 4.0
    neuron = Neuron(weights, bias)
    x = np.array([2.0, 3.0])
    print("Single neuron:", neuron.feedforward(x))  # ≈ 0.99909

    net = OurNeuralNetwork()
    print("2–2–1 net:", net.feedforward(x))  # ≈ 0.72163256

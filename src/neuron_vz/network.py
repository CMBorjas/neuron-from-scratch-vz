import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    Maps any real value into the range [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
    """
    Derivative of the sigmoid function.
    f'(x) = f(x) * (1 - f(x))
    Used during backpropagation to calculate gradients.
    """
    fx = sigmoid(x)
    return fx * (1.0 - fx)

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) Loss function.
    Calculates the average squared difference between true and predicted values.
    y_true and y_pred must be numpy arrays of the same length.
    """
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    """
    A single neuron that applies weights, adds a bias, 
    and passes the result through a sigmoid activation function.
    """
    def __init__(self, weights, bias):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def feedforward_float(self, inputs):
        """
        Calculates the neuron's output for a given input.
        Formula: sigmoid(dot(weights, inputs) + bias)
        """
        total = float(np.dot(self.weights, np.asarray(inputs, dtype=float)) + self.bias)
        return sigmoid(total)

class OurNeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    All neurons use the sigmoid activation function.
    """
    def __init__(self):
        # Initial weights for the feedforward_float example (matching tutorial)
        initial_weights = np.array([0.0, 1.0])
        initial_bias = 0.0
        
        # Neurons used in the initial feedforward_float method
        self.h1 = Neuron(initial_weights, initial_bias)
        self.h2 = Neuron(initial_weights, initial_bias)
        self.o1 = Neuron(initial_weights, initial_bias)

        # Weights for the train/feedforward methods initialized with random normal distribution
        # Weights from inputs to hidden layer neuron 1 (h1)
        self.w_in1_h1 = np.random.normal() # originally w1
        self.w_in2_h1 = np.random.normal() # originally w2
        
        # Weights from inputs to hidden layer neuron 2 (h2)
        self.w_in1_h2 = np.random.normal() # originally w3
        self.w_in2_h2 = np.random.normal() # originally w4
        
        # Weights from hidden layer to output layer neuron 1 (o1)
        self.w_h1_o1 = np.random.normal()  # originally w5
        self.w_h2_o1 = np.random.normal()  # originally w6

        # Biases for each neuron
        self.b_h1 = np.random.normal() # originally b1
        self.b_h2 = np.random.normal() # originally b2
        self.b_o1 = np.random.normal() # originally b3

    def feedforward_float(self, x):
        """
        Original feedforward method using the Neuron class objects.
        This mirrors the deterministic example in the blog post.
        """
        out_h1 = self.h1.feedforward_float(x)
        out_h2 = self.h2.feedforward_float(x)
        # o1 takes outputs from h1 and h2 as inputs
        out_o1 = self.o1.feedforward_float(np.array([out_h1, out_h2]))
        return float(out_o1)

    def feedforward(self, x):
        """
        Calculates the network output by manually applying weights and biases.
        Uses the internal random weights instead of Neuron objects.
        """
        # Hidden layer outputs
        h1 = sigmoid(self.w_in1_h1 * x[0] + self.w_in2_h1 * x[1] + self.b_h1)
        h2 = sigmoid(self.w_in1_h2 * x[0] + self.w_in2_h2 * x[1] + self.b_h2)
        
        # Output layer
        o1 = sigmoid(self.w_h1_o1 * h1 + self.w_h2_o1 * h2 + self.b_o1)
        return o1

    def train(self, data, all_y_trues):
        """
        Trains the neural network using stochastic gradient descent.
        - data: (n x 2) numpy array of input features (n samples, 2 features).
        - all_y_trues: numpy array of true labels (n elements).
        """
        learn_rate = 0.1
        epochs = 1000 # Number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- 1. Feedforward (Forward Pass) ---
                
                # Hidden Neuron 1
                sum_h1 = self.w_in1_h1 * x[0] + self.w_in2_h1 * x[1] + self.b_h1
                h1 = sigmoid(sum_h1)

                # Hidden Neuron 2
                sum_h2 = self.w_in1_h2 * x[0] + self.w_in2_h2 * x[1] + self.b_h2
                h2 = sigmoid(sum_h2)

                # Output Neuron 1
                sum_o1 = self.w_h1_o1 * h1 + self.w_h2_o1 * h2 + self.b_o1
                o1 = sigmoid(sum_o1)
                
                y_pred = o1

                # --- 2. Calculate Partial Derivatives (Backpropagation) ---
                
                # Derivative of Loss with respect to Prediction (y_pred)
                d_loss_d_ypred = -2.0 * (y_true - y_pred)

                # Output Neuron o1 derivatives
                d_ypred_d_w_h1_o1 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w_h2_o1 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b_o1 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w_h1_o1 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w_h2_o1 * deriv_sigmoid(sum_o1)

                # Hidden Neuron h1 derivatives
                d_h1_d_w_in1_h1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w_in2_h1 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b_h1 = deriv_sigmoid(sum_h1)

                # Hidden Neuron h2 derivatives
                d_h2_d_w_in1_h2 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w_in2_h2 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b_h2 = deriv_sigmoid(sum_h2)

                # --- 3. Update Weights and Biases (Gradient Descent) ---
                
                # Update Hidden Neuron h1 weights and bias
                self.w_in1_h1 -= learn_rate * d_loss_d_ypred * d_ypred_d_h1 * d_h1_d_w_in1_h1
                self.w_in2_h1 -= learn_rate * d_loss_d_ypred * d_ypred_d_h1 * d_h1_d_w_in2_h1
                self.b_h1 -= learn_rate * d_loss_d_ypred * d_ypred_d_h1 * d_h1_d_b_h1

                # Update Hidden Neuron h2 weights and bias
                self.w_in1_h2 -= learn_rate * d_loss_d_ypred * d_ypred_d_h2 * d_h2_d_w_in1_h2
                self.w_in2_h2 -= learn_rate * d_loss_d_ypred * d_ypred_d_h2 * d_h2_d_w_in2_h2
                self.b_h2 -= learn_rate * d_loss_d_ypred * d_ypred_d_h2 * d_h2_d_b_h2

                # Update Output Neuron o1 weights and bias
                self.w_h1_o1 -= learn_rate * d_loss_d_ypred * d_ypred_d_w_h1_o1
                self.w_h2_o1 -= learn_rate * d_loss_d_ypred * d_ypred_d_w_h2_o1
                self.b_o1 -= learn_rate * d_loss_d_ypred * d_ypred_d_b_o1

        # --- Calculate total loss at the end of every 10th epoch
        if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

if __name__ == "__main__":
    # Demo values from the tutorial for single neuron
    weights = np.array([0.0, 1.0])
    bias = 4.0
    neuron = Neuron(weights, bias)
    x = np.array([2.0, 3.0])
    print("Single neuron:", neuron.feedforward_float(x))  # ≈ 0.99909

    # Demo for 2-2-1 network feedforward before training
    net = OurNeuralNetwork()
    print("2–2–1 net:", net.feedforward_float(x))  # ≈ 0.72163256

    # Part 3: Calculate MSE Loss
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    print("MSE Loss:", mse_loss(y_true, y_pred))

    # Part 4: Training data (Weight - 135, Height - 66)
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],   # Bob
        [17, 4],   # Charlie
        [-15, -6]  # Diana
    ])

    # Labels: 1 for Female, 0 for Male
    all_y_trues = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    # Train our neural network
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # Make predictions for new samples
    # Input data needs to be preprocessed similarly (Weight - 135, Height - 66)
    # Emily: 128 pounds, 63 inches -> [-7, -3]
    # Frank: 155 pounds, 68 inches -> [20, 2]
    emily = np.array([-7, -3]) 
    frank = np.array([20, 2])  
    
    print("Emily (Predicted probability of being Female): %.3f" % network.feedforward(emily)) # Expected ~0.951
    print("Frank (Predicted probability of being Female): %.3f" % network.feedforward(frank)) # Expected ~0.039
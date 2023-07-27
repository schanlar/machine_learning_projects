import numpy as np

class NeuralNetwork:
    """
        A simple neural network consisting of two layers:
            1. a hidden layer
            2. an output layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        """ Initialize the size of layers alogn with the weight and biases """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.zeros(1, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.zeros(1, self.output_size)

    def sigmoid(self, x):
        """
            A sigmoid activation function in order to introduce non-linearity
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
            The derivative of a sigmoid function
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
            The input data is passed through the neural net to get the predicted output.
            In forward pass, first we calculate the output of the hidden layer:
                hidden_output = X\dot{W1} + b1
            Then we apply the sigmoid activation function to the hidden output
                output = sigmoid(X\dot{W1} + b1)
        """
        self.hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)

        self.output = self.sigmoid(np.dot(self.hidden_output, self.W2) + self.b2)
        return self.output
    
    def backward(self, X, y, learning_rate):
        """
            First we compute the gradients of the output layer.
                loss = y - output
                loss_gradient = (y - output) * sigmoid_derivative(output)

            Next we calculate the gradient of the loss function with respect to W2 (d_W2)
                d_W2 = hidden_output.T \dot{loss_gradient}
            Similary, we calculate the gradient of the loss function with respect to W1 (d_W1),
            with respect to b2 (bias of neuron in input layer), and b1 (bias of neuron in hidden layer)
            Finally, we update the weights/biases. 
            
            Here learning rate is the hyper parameter. A low learning rate can cause the model to get caught
            in local optima, while a high learning rate can cause the model to overshoot the general solution
        """
        d_output  = (y - self.output) * self.sigmoid_derivative(self.output)
        d_W2 = np.dot(self.hidden_output.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.hidden_output)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        self.W2 += learning_rate * d_W2
        self.b2 += learning_rate * d_b2
        self.W1 += learning_rate * d_W1
        self.b1 += learning_rate * d_b1

    def train(self, X, y, epochs, learning_rate):
        """
            A method to train the neural net using both the forward and backward passes.
            The function will run for a specified number of epochs, calculating:
                1. the forward pass,
                2. the backward pass, 
                3. updating the weights
        """
        for epoch in range(epochs):
            output = self.forward(X)

            self.backward(X, y, learning_rate)
            loss = np.mean((y - output)**2)

    def predict(self, X):
        """
            To predict on any new data all we need to do is a single
            forward pass through the neural net
        """
        return self.forward(X)
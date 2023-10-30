import numpy as np

class LinearRegression():
	def __init__(self, learning_rate=0.01, max_iter=1000):
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.weights = None
		self.bias = None 


	def fit(self, X, y):
		num_samples, num_features = X.shape

		# Initialize the weights and bias to zeros
		self.weights = np.zeros(num_features)
		self.bias = 0

		# Gradient descent
		for _ in range(self.max_iter):
			# Predict the output using the current weishts
			# and bias
			y_pred = self.predict(X)

		# Calculate the gradients:
		dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
		db = (1 / num_samples) * np.sum(y_pred - y)

		# Update the weights and bias
		self.weights -= self.learning_rate * dw
		self.bias -= self.learning_rate * db


	def predict(self, X){
		return np.dot(X, self.weights) + self.bias

	

import numpy as np

class Neural_Network:
	def __init__(self):
		self.W1 = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1]]).T
		self.W2 = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]]).T

	def sigmoid(self, z, diff=False):
		if diff : return self.sigmoid(z)*(1-self.sigmoid(z))
		return 1/(1+np.exp(-(z)))
 

	def mean_square_error(self, X, Y):
		self.Y_hat = self.forward(X)
		return 0.5*sum((Y-self.Y_hat)**2)

	def generate_partial_derivates(self, X, Y):
		self.Y_hat = self.forward(X)

		delta2 = np.multiply(-(Y-self.Y_hat), self.sigmoid(self.S2, diff=True))
		dEdW2 = np.dot(self.H, delta2)
		print(self.W2 - 0.1*dEdW2)
		#delta1 = np.dot(delta2, self.W2)
		#dEdW1 = np.dot(X, delta1)

		#return dEdW1, dEdW2

	def update_weights(self, X, Y):
		self.Y_hat = self.forward(X)
		dEdW1, dEdW2 = self.generate_partial_derivates(X,Y)
		self.W1 -= 0.1*dEdW1
		self.W2 -= 0.1*dEdW2

	def forward(self, X):
		self.S1 = np.dot(self.W1.T, X)
		self.H = self.sigmoid(self.S1)
		self.H = np.vstack((np.array([1,1]), self.H))
		self.S2 = np.dot(self.W2.T, self.H)
		self.Y_hat = self.sigmoid(self.S2)
		return self.Y_hat

cnn = Neural_Network()
X = np.array([[1, 0.1, 0.1], [1, 0.1, 0.2]]).T
Y = np.array([[1,0],[0,1]]).T
print(cnn.forward(X))
#print(cnn.mean_square_error(X,Y))
print(cnn.S2)#np.array([[0.206,0.207],[0.259]])
print(cnn.sigmoid(cnn.S2,diff=True))
cnn.update_weights(X,Y)
print(cnn.W2)
# dEdW1,dEdW2 = cnn.generate_partial_derivates(X,Y)
# print(dEdW2)
# print(dEdW1)
# print(cnn.mean_square_error(X,Y))
# for i in range(50):
# 	cnn.update_weights(X,Y)
# 	print(cnn.mean_square_error(X,Y))
#cnn.update_weights(X,Y)

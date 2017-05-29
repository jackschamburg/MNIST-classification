import numpy as np
import matplotlib.pyplot as plt 
class Neural_Network:
	def __init__(self):
		self.W1 = np.random.standard_normal(size=(Ninput, Nhidden))
		self.W2 = np.random.standard_normal(size=(Nhidden, Noutput))
		self.b1 = np.random.standard_normal(size=(Nhidden, 1))
		self.b2 = np.random.standard_normal(size=(Nhidden, 1))

	def sigmoid(self, z, diff=False):
		if diff: return np.multiply(self.sigmoid(z),(1-self.sigmoid(z)))
		return 1/(1+np.exp(-(z)))

	def mean_square_error(self, X, Y):
		self.forward(X)
		e = Y - self.Y_hat
		return 0.5*np.sum(np.square(e))

	def update_weights(self, X, Y):
		self.forward(X)
		
		delta3 = np.multiply((self.Y_hat - Y), self.sigmoid(self.S2, diff=True))
		delta2 = np.multiply((np.dot(self.W2.T, delta3)),self.sigmoid(self.S1, diff=True))
				
		dEdW2 = np.dot(delta3, (self.H).T)
		dEdW1 = np.dot(delta2, X.T)
		self.W1 = self.W1.T - 0.1*dEdW1
		self.W2 = self.W2.T - 0.1*dEdW2
		self.b2 = np.mean(self.b2 - 0.1*delta3,axis=1)
		self.b1 = np.mean(self.b1 - 0.1*delta2,axis=1)
		
	def forward(self, X):
		self.S1 = np.dot(self.W1.T, X) + self.b1
		self.H = self.sigmoid(self.S1)
		self.S2 = np.dot(self.W2.T, self.H) + self.b2
		self.Y_hat = self.sigmoid(self.S2)

Ninput = 784
Nhidden = 30
Noutput = 10
Nepoch = 30
minibatchsize = 20
learningrate = 3

digit = Neural_Network()
xfile = "TrainDigitX.csv.gz"
yfile = "TrainDigitY.csv.gz"
Y = (np.loadtxt(yfile,delimiter=',')).T
print(Y.shape)
exit(0)
X = (np.loadtxt(xfile,delimiter=',')).T
print("loaded X")
Y = (np.loadtxt(yfile,delimiter=','))
print("loaded Y")

for i in range(Nepoch):	
	for j in range(1, 2501):
		minibatch = X[:Ninput, :minibatchsize*j]
		minibatchlabels = Y[1, :minibatchsize*j]

		digit.forward(minibatch)
		digit.update_weights(minibatch, minibatchlabels)
		print(digit.mean_square_error(minibatch, minibatchlabels))




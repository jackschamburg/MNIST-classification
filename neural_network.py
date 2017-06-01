import numpy as np
import matplotlib.pyplot as plt 
import pickle

# 3 Layer Neural Network
class Neural_Network:
	def __init__(self):
		self.W1 = np.random.standard_normal(size=(Ninput, Nhidden))
		self.W2 = np.random.standard_normal(size=(Nhidden, Noutput))
		self.b1 = np.random.standard_normal(size=(Nhidden, 1))
		self.b2 = np.random.standard_normal(size=(Noutput, 1))

	# Given a digit returns vector
	def vectorize(self, D):
		vectors = []
		for digit in D:
			v = [0] * 10
			v[int(digit)] = 1
			vectors.append(v)
		return np.array(vectors).T
	
	def sigmoid(self, z, diff=False):
		if diff: return np.multiply(self.sigmoid(z),(1-self.sigmoid(z)))
		return 1/(1+np.exp(-(z)))
	
	# Given a data set of inputs and labels
	# returns the quadratic error 
	def quadratic_error(self, X, Y):
		self.forward(X)
		diff = self.Y_hat - self.vectorize(Y)
		return 1/(2*len(Y))*np.sum(np.square(diff))
	
	# Computes the probabilities of the classes
	def softmax(self, Z, diff=False):
		if diff: return np.multiply(self.softmax(Z),(1-self.softmax(Z)))
		e_Z = np.exp(Z - np.max(Z))
		probs = e_Z / np.sum(e_Z, axis=0)
		return probs

	# Update the weights using stochastic
	# Gradient descent
	def update_weights(self, X, Y):
		self.forward(X)
		
		delta3 = np.multiply((self.Y_hat - self.vectorize(Y)), self.softmax(self.S2, diff=True))
		delta2 = np.multiply((np.dot(self.W2, delta3)), self.sigmoid(self.S1, diff=True))
				
		dEdW2 = np.dot(self.H, delta3.T)
		dEdW1 = np.dot(X, delta2.T)

		self.W1 = self.W1 - learningrate*(dEdW1/minibatchsize)
		self.W2 = self.W2 - learningrate*(dEdW2/minibatchsize)
		self.b2 = self.b2 - learningrate*(np.mean(delta3,axis=1)).reshape((Noutput,1))
		self.b1 = self.b1 - learningrate*(np.mean(delta2,axis=1)).reshape((Nhidden,1))

	# Forward propogate a data set through the network
	def forward(self, X):
		self.S1 = (np.dot(self.W1.T, X)) + self.b1
		self.H = self.sigmoid(self.S1)
		self.S2 = (np.dot(self.W2.T, self.H)) + self.b2
		self.Y_hat = self.softmax(self.S2)

		digits = [0,1,2,3,4,5,6,7,8,9]
		pred = []
		for column in self.Y_hat.T:
			pred.append((np.random.choice(digits, p=column)))
		
		self.predictions = np.array(pred)

if __name__ == "__main__":
	# Parameters and hyper-parameters
	Nsample = 50000
	Ninput = 784
	Nhidden = 35
	Noutput = 10
	Nepoch = 30
	minibatchsize = 20
	learningrate = 3
	digit = Neural_Network()
	
	# Files to work with
	xtrain = "TrainDigitX.csv.gz"
	ytrain = "TrainDigitY.csv.gz"
	xtest = "TestDigitX.csv.gz"
	ytest = "TestDigitY.csv.gz"
	xpredict = "TestDigitX2.csv.gz"
	ypredict = "PredictTestY2.csv.gz"

	# Load in files
	print("Loading Training X")
	X = (np.loadtxt(xtrain,delimiter=',')).T
	print("Loading Training Y")
	Y = (np.loadtxt(ytrain,delimiter=','))
	print("Loading Testing X")
	XT = (np.loadtxt(xtest,delimiter=',')).T
	print("Loading Testing Y")
	YT = (np.loadtxt(ytest,delimiter=','))
	#print("Loading Predict X")
	#XP = (np.loadtxt(xpredict,delimiter=',')).T

	digit = Neural_Network()

	accuracy_per_epoch = []
	for i in range(Nepoch):	
		for j in range(int(Nsample/minibatchsize)):

			minibatch = X[:Ninput, j*minibatchsize:(j+1)*minibatchsize]
			minibatchlabels = Y[j*minibatchsize:(j+1)*minibatchsize]	
			digit.update_weights(minibatch, minibatchlabels)

		# TEST CURRENT EPOCH #
		digit.forward(XT)
		PV = digit.predictions
	
		wrong, correct,acc = 0,0,0
		for k in range(len(YT)):
			if PV[k]==int(YT[k]): correct+= 1
			else: wrong += 1

		acc = correct/(correct+wrong)
		err = wrong/(correct+wrong)
		accuracy_per_epoch.append(err)
		print("------------------------------")
		print("Epoch: ", i)
		print("correct: ", correct)
		print("wrong: ", wrong)
		print("accuracy: ", acc)
		print("error", err)

		# SHUFFLE #
		Z = np.vstack((X,Y))
		np.random.shuffle(Z.T)
		X = Z[:Ninput,:]
		Y = Z[Ninput,:]

	# PREDICT #
	#digit.forward(XP)
	#blindpredictions = digit.predictions
	#np.savetxt(ypredict, blindpredictions)

	print("------------------------------")
	plt.plot(accuracy_per_epoch)
	plt.grid(1)
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.show()


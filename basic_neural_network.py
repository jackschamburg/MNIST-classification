import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:
	def __init__(self):
		self.W1 = np.array([[0.1, 0.2], [0.1, 0.1]])
		self.W2 = np.array([[0.1, 0.1], [0.1, 0.2]])
		self.b1 = np.array([[0.1], [0.1]])
		self.b2 = np.array([[0.1], [0.1]])

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
		delta2 = np.multiply((np.dot(self.W2, delta3)),self.sigmoid(self.S1, diff=True))
		dEdW2 = np.dot(self.H, delta3.T)
		dEdW1 = np.dot(X, delta2.T)
		self.W1 = self.W1 - 0.1*(dEdW1/2)
		self.W2 = self.W2 - 0.1*(dEdW2/2)
		self.b2 = self.b2 - 0.1*np.mean(delta3,axis=1)
		self.b1 = self.b1 - 0.1*np.mean(delta2,axis=1)
		print("W1",self.W1)
		print("W2",self.W2)
		print("b1",self.b1)
		print("b2",self.b2)		
		
		print("dEdW1", dEdW1)
		print("dEdW2", dEdW2)
		print("dEdb1", np.mean(delta3,axis=1))
		print("dEdb2", np.mean(delta3,axis=1))
 

	def forward(self, X):
		self.S1 = np.dot(self.W1.T, X) + self.b1
		self.H = self.sigmoid(self.S1)
		self.S2 = np.dot(self.W2.T, self.H) + self.b2
		self.Y_hat = self.sigmoid(self.S2)

cnn = Neural_Network()
X = np.matrix([[0.1, 0.1], [0.1, 0.2]])
Y = np.matrix([[1,0],[0,1]])
cnn.forward(X)
print("X")
print(X)
print("Y")
print(Y)
print("W1")
print(cnn.W1)
print("W2")
print(cnn.W2)
print("b1")
print(cnn.b1)
print("b2")
print(cnn.b2)
print("S1")
print(cnn.S1)
print("H")
print(cnn.H)
print("S2")
print(cnn.S2)
print("Y_hat")
print(cnn.Y_hat)
#cnn.update_weights(X,Y)
#print("W1")
#print(cnn.W1)
#print("W2")
#print(cnn.W2)
#print("b1")
#print(cnn.b1)
#print("b2")
#print(cnn.b2)
#exit(0)
#print("new error {}".format(cnn.mean_square_error(X,Y)))
#cnn.update_weights(X,Y)
#E = cnn.mean_square_error(X,Y)
#A = [E]
for i in range(3):
	#cnn.forward(X)
	print("EPOCH:",i+1)
	cnn.update_weights(X,Y)
	#A.append(cnn.mean_square_error(X,Y))
#print("W1")
#print(cnn.W1)
#print("W2")
#print(cnn.W2)
#print("b1")
#print(cnn.b1)
#print("b2")
#print(cnn.b2)
#print(cnn.Y_hat)
#print(cnn.mean_square_error(X,Y))
#plt.plot(A)
#plt.grid(1)
#plt.xlabel('Epoch')
#plt.ylabel('Error')
#plt.show()

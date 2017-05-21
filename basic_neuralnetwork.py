import numpy as np

class Neural_Network:
    def __init__(self):
        self.W1 = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1]]).T
        self.W2 = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]]).T

    def sigmoid(self, z,diff=False):
        if not diff: return 1/(1+np.exp(-z))

    def forward(self, X):
        self.S1 = np.dot(self.W1.T, X)
        self.H = self.sigmoid(self.S1)
        self.H = np.vstack((np.array([1,1]), self.H))
        self.S2 = np.dot(self.W2.T, self.H)
        self.Y = self.sigmoid(self.S2)
        return self.Y

cnn = Neural_Network()
X = np.array([[1, 0.1, 0.1], [1, 0.1, 0.2]]).T
cnn.forward(X)
print(cnn.Y)

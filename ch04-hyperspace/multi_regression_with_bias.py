import numpy as np

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    y_hat = predict(X, w)
    error = y_hat - Y
    return np.average(error ** 2)

def gradient(X, Y, w):
    y_hat = predict(X, w)
    error = y_hat - Y
    return 2 * np.matmul(X.T, error) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=100000, lr=0.001)

print("\nWeights: %s" % w.T)
print("\nA few predictionsL")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

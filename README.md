# AI_1
Building a Perceptron Using Sigmoid Function
import numpy as np
class Perceptron:
    def __init__(self, eta=0.01, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                output = self.sigmoid(self.net_input(xi))
                update = self.eta * (target - output)
                self.w_[1:] += update * xi
                self.w_[0] += update
# Manually entered dataset (Iris-like values)
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Iris-setosa
    [4.9, 3.0, 1.4, 0.2],   # Iris-setosa
    [6.3, 3.3, 6.0, 2.5],   # Not setosa
    [5.8, 2.7, 5.1, 1.9]    # Not setosa
])
model = Perceptron(eta=0.05, n_iter=50)
model.fit(X_train, y_train)
print("Enter flower measurements:")

sepal_length = float(input("Sepal Length: "))
sepal_width  = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width  = float(input("Petal Width: "))

manual_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

prediction = model.predict(manual_input)

if prediction[0] == 1:
    print("\nPrediction: Iris-setosa ")
else:
    print("\nPrediction: Not Iris-setosa ")

# Labels: 1 = Iris-setosa, 0 = Not Iris-setosa
y_train = np.array([1, 1, 0, 0])

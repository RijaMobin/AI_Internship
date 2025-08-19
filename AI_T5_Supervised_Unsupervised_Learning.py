# -------------------------------------------------
# Supervised & Unsupervised Learning (Scratch + Library) with Visuals
# -------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans

# =================================================
# SUPERVISED LEARNING
# =================================================

# -----------------------
# Linear Regression (Scratch)
# -----------------------
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        self.m = 0
        self.b = 0
        n = len(X)
        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            dm = (-2/n) * sum(X * (y - y_pred))
            db = (-2/n) * sum(y - y_pred)
            self.m -= self.lr * dm
            self.b -= self.lr * db
    
    def predict(self, X):
        return self.m * X + self.b

# Test Linear Regression
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=1)
X = X.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr_scratch = LinearRegressionScratch()
lr_scratch.fit(X_train, y_train)
preds_scratch = lr_scratch.predict(X_test)
print("[Supervised] Linear Regression (Scratch) MSE:", mean_squared_error(y_test, preds_scratch))

lr_lib = LinearRegression()
lr_lib.fit(X_train.reshape(-1,1), y_train)
preds_lib = lr_lib.predict(X_test.reshape(-1,1))
print("[Supervised] Linear Regression (Library) MSE:", mean_squared_error(y_test, preds_lib))

# Plot Linear Regression results
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, preds_scratch, color="red", label="Scratch Prediction")
plt.plot(X_test, preds_lib, color="green", linestyle="dashed", label="Library Prediction")
plt.title("Linear Regression (Supervised)")
plt.legend()
plt.show()


# -----------------------
# Logistic Regression (Scratch)
# -----------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

# Test Logistic Regression
X, y = make_classification(
    n_samples=300, n_features=2, n_classes=2, 
    n_informative=2, n_redundant=0, n_repeated=0,
    random_state=1
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_scratch = LogisticRegressionScratch()
log_scratch.fit(X_train, y_train)
preds_scratch = log_scratch.predict(X_test)
print("\n[Supervised] Logistic Regression (Scratch) Accuracy:", accuracy_score(y_test, preds_scratch))

log_lib = LogisticRegression()
log_lib.fit(X_train, y_train)
preds_lib = log_lib.predict(X_test)
print("[Supervised] Logistic Regression (Library) Accuracy:", accuracy_score(y_test, preds_lib))

# Plot Logistic Regression Decision Boundary
plt.figure(figsize=(6,5))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', alpha=0.7, label="Actual")

# Decision boundary for library model
x_min, x_max = X_test[:,0].min()-1, X_test[:,0].max()+1
y_min, y_max = X_test[:,1].min()-1, X_test[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = log_lib.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap="bwr")

plt.title("Logistic Regression (Supervised)")
plt.show()


# =================================================
# UNSUPERVISED LEARNING
# =================================================

# -----------------------
# KMeans Clustering (Scratch)
# -----------------------
class KMeansScratch:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idx]
        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(self.k)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.argmin(distances, axis=0)

# Test KMeans
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans_scratch = KMeansScratch(k=3)
kmeans_scratch.fit(X)
labels_scratch = kmeans_scratch.predict(X)

kmeans_lib = KMeans(n_clusters=3, n_init=10)
labels_lib = kmeans_lib.fit_predict(X)

print("\n[Unsupervised] KMeans (Scratch) Centroids:\n", kmeans_scratch.centroids)
print("[Unsupervised] KMeans (Library) Centroids:\n", kmeans_lib.cluster_centers_)

# Plot Clustering Results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=labels_scratch, cmap="viridis", alpha=0.7)
plt.scatter(kmeans_scratch.centroids[:,0], kmeans_scratch.centroids[:,1], c="red", marker="X", s=200)
plt.title("KMeans Scratch (Unsupervised)")

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=labels_lib, cmap="viridis", alpha=0.7)
plt.scatter(kmeans_lib.cluster_centers_[:,0], kmeans_lib.cluster_centers_[:,1], c="red", marker="X", s=200)
plt.title("KMeans Library (Unsupervised)")

plt.show()

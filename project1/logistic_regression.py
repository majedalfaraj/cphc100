import numpy as np
import tqdm
import pandas as pd

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.train_loss = []
        self.val_loss = []
        self.bias = 0.0

        self.weights = None

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        self.weights = np.zeros(X.shape[1]) # 1 weight per feature
        for _ in range(self.num_epochs):
            shuffle = np.random.permutation((len(X)))
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
            for i in np.arange(0, len(X), self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                Y_batch = Y_shuffled[i : i + self.batch_size].reshape(-1)
                # (f"X_b {X_batch.shape}, Y_b {Y_batch.shape}")
                self.weights -= self.learning_rate * self.gradient(X_batch, Y_batch)
                self.bias -= self.learning_rate * np.mean(self.predict_proba(X_batch) - Y_batch)

            self.train_loss.append(self.CELoss(Y, self.predict_proba(X)))
            if X_val is not None and Y_val is not None:
                self.val_loss.append(self.CELoss(Y_val, self.predict_proba(X_val)))

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        P = self.predict_proba(X)
        # print(f"P {P.shape} X.T {X.T.shape} grad {(X.T @ (P - Y)).shape}")
        return (1/len(Y)) * X.T @ (P - Y) + 2 * self.regularization_lambda * self.weights

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        # print(f"Wx + b {(X@self.weights).shape}")
        return sigmoid(X @ self.weights + self.bias)


    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def CELoss(self, target, prob):
        eps = 1e-9
        return -np.mean(target*np.log(prob + eps) + (1 - target)*np.log(1-prob + eps))

    def get_loss_csv(self):
        epochs = 1 + np.arange(self.num_epochs)
        loss_df = pd.DataFrame({"epoch": epochs, "train_loss": self.train_loss, "val_loss":self.val_loss})
        loss_df.to_csv("losses.csv")
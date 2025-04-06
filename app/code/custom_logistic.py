import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import KFold


class LogisticRegression:
    def __init__(self, k, n, method, run_id, alpha=0.001, max_iter=5000, lambda_=0.0, use_l2=False):
        self.k = k
        self.n = n
        self.method = method
        self.run_id = run_id
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.use_l2 = use_l2

    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []

        if self.method == "batch":
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W -= self.alpha * grad

        elif self.method == "minibatch":
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0] - batch_size + 1)
                X_batch = X[ix:ix + batch_size]
                Y_batch = Y[ix:ix + batch_size]
                loss, grad = self.gradient(X_batch, Y_batch)
                self.losses.append(loss)
                self.W -= self.alpha * grad

        elif self.method == "sto":
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx].reshape(1, -1)
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W -= self.alpha * grad
        else:
            raise ValueError("Invalid method")

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(Y * np.log(h)) / m
        if self.use_l2:
            loss += self.lambda_ * np.sum(self.W ** 2) / 2
        error = h - Y
        grad = X.T @ error / m
        if self.use_l2:
            grad += self.lambda_ * self.W
        return loss, grad

    def h_theta(self, X, W):
        return np.exp(X @ W) / np.sum(np.exp(X @ W), axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.h_theta(X, self.W), axis=1)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def macro_f1(self, y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='macro')

    def cross_validate(self, X, Y, k_folds=5):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            with mlflow.start_run(run_name=f"Run_{self.run_id}_Fold_{fold}"):
                mlflow.log_param("method", self.method)
                mlflow.log_param("alpha", self.alpha)
                mlflow.log_param("lambda", self.lambda_)
                mlflow.log_param("use_l2", self.use_l2)
                mlflow.log_param("fold", fold)
                mlflow.log_param("max_iter", self.max_iter)

                # Train
                self.fit(X_train, Y_train)

                # Predict
                y_pred = self.predict(X_val)
                y_true = np.argmax(Y_val, axis=1)

                # Metrics
                acc = self.accuracy(y_true, y_pred)
                f1 = self.macro_f1(y_true, y_pred)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("macro_f1", f1)

                # Signature & log model
                input_example = np.array([X_val[0]])
                signature = infer_signature(X_val, y_pred)
                mlflow.sklearn.log_model(self, f"model_fold_{fold}", input_example=input_example, signature=signature)
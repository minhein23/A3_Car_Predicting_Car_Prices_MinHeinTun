from sklearn.model_selection import KFold  # For cross-validation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting feature importance
import math  # For mathematical operations like square root

class LinearRegression(object):
    """
    A custom implementation of Linear Regression with support for regularization, momentum, and different optimization methods.
    """
    
    kfold = KFold(n_splits=3)  # Default 3-fold cross-validation
            
    def __init__(self, regularization, lr, method, theta_init, momentum, num_epochs=500, batch_size=50, cv=kfold):
        """
        Initializes the Linear Regression model.

        Args:
            regularization: Regularization method (e.g., Lasso, Ridge, ElasticNet).
            lr (float): Learning rate for gradient descent.
            method (str): Optimization method ('sto' for stochastic, 'mini' for mini-batch, else full-batch).
            theta_init (str): Method to initialize weights ('zeros' or 'xavier').
            momentum (float or str): Momentum term for gradient descent ('without' for no momentum).
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for mini-batch gradient descent.
            cv: Cross-validation method (default is 3-fold KFold).
        """
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.theta_init = theta_init
        self.momentum = momentum
        self.cv = cv
        self.regularization = regularization

    def mse(self, ytrue, ypred):
        """
        Computes the Mean Squared Error (MSE) between true and predicted values.

        Args:
            ytrue (np.array): True target values.
            ypred (np.array): Predicted target values.

        Returns:
            float: Mean Squared Error.
        """
        return ((ytrue - ypred) ** 2).sum() / ypred.shape[0]
    
    def r2(self, ytrue, ypred):
        """
        Computes the R-squared (coefficient of determination) metric.

        Args:
            ytrue (np.array): True target values.
            ypred (np.array): Predicted target values.

        Returns:
            float: R-squared value.
        """
        return 1 - ((((ytrue - ypred) ** 2).sum()) / (((ytrue - ytrue.mean()) ** 2).sum()))
    
    def fit(self, X_train, y_train):
        """
        Fits the Linear Regression model using the specified optimization method and cross-validation.

        Args:
            X_train (np.array): Training feature data.
            y_train (np.array): Training target data.
        """
        self.kfold_scores = list()  # Stores validation scores for each fold
        self.val_loss_old = np.infty  # Tracks the previous validation loss
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]  # Training data for the current fold
            y_cross_train = y_train[train_idx]  # Training labels for the current fold
            X_cross_val = X_train[val_idx]  # Validation data for the current fold
            y_cross_val = y_train[val_idx]  # Validation labels for the current fold
            
            # Initialize weights based on the specified method
            if self.theta_init == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.theta_init == 'xavier':
                m = X_train.shape[0]
                lower, upper = -(1.0 / math.sqrt(m)), (1.0 / math.sqrt(m))
                numbers = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + numbers * (upper - lower)

            # Training loop
            for epoch in range(self.num_epochs):
                perm = np.random.permutation(X_cross_train.shape[0])  # Shuffle data
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]
                
                # Perform optimization based on the specified method
                if self.method == 'sto':
                    for batch_idx in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[batch_idx].reshape(1, -1) 
                        y_method_train = y_cross_train[batch_idx] 
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                        X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                        train_loss = self._train(X_method_train, y_method_train)
                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                # Compute validation loss
                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)
                
                # Early stopping if validation loss stops improving
                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new
        
            self.kfold_scores.append(val_loss_new)  # Store validation score for the fold
            print(f"Fold {fold}: {val_loss_new}")
            
                    
    def _train(self, X, y):
        """
        Performs a single training step (weight update) using gradient descent.

        Args:
            X (np.array): Input features.
            y (np.array): Target values.

        Returns:
            float: Training loss (MSE).
        """
        yhat = self.predict(X)
        m = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)  # Compute gradient
        prev_step = 0
        
        # Apply momentum if specified
        if self.momentum == "without":
            step = self.lr * grad
        else:
            step = self.lr * grad + self.momentum * prev_step

        self.theta -= step  # Update weights
        prev_step = step
        return self.mse(y, yhat)
    
    def predict(self, X):
        """
        Predicts target values using the trained model.

        Args:
            X (np.array): Input features.

        Returns:
            np.array: Predicted target values.
        """
        return X @ self.theta  
    
    def _coef(self):
        """
        Returns the model coefficients (excluding the bias term).

        Returns:
            np.array: Model coefficients.
        """
        return self.theta[1:]  
                         
    def _bias(self):
        """
        Returns the bias term of the model.

        Returns:
            float: Bias term.
        """
        return self.theta[0]
    
    def feature_importance(self):
        """
        Plots the feature importance based on the absolute values of the model coefficients.
        """
        feature_names = ["name", "engine", "mileage"]
        importance_values = [abs(self._coef()[0]), abs(self._coef()[1]), abs(self._coef()[2])]

        plt.figure(figsize=(8, 6))
        plt.barh(feature_names, importance_values, color='blue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Graph')
        plt.xlim([0, max(self._coef()) * 1.3])  
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.show()

# Regularization Classes
class NormalPenalty:
    """
    No regularization (normal linear regression).
    """
    
    def __init__(self, l):
        self.l = l 
        
    def __call__(self, theta): 
        return 0
        
    def derivation(self, theta):
        return 0

class LassoPenalty:
    """
    L1 regularization (Lasso).
    """
    
    def __init__(self, l):
        self.l = l 
        
    def __call__(self, theta): 
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    """
    L2 regularization (Ridge).
    """
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): 
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    """
    ElasticNet regularization (combination of L1 and L2).
    """
    
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
# Subclasses for specific regularization methods
class Normal(LinearRegression):
    """
    Linear Regression with no regularization.
    """
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = NormalPenalty(l)
        super().__init__(self.regularization, lr, method, theta_init, momentum)

class Lasso(LinearRegression):
    """
    Linear Regression with L1 regularization (Lasso).
    """
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, theta_init, momentum)
        
class Ridge(LinearRegression):
    """
    Linear Regression with L2 regularization (Ridge).
    """
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, theta_init, momentum)
        
class ElasticNet(LinearRegression):
    """
    Linear Regression with ElasticNet regularization.
    """
    
    def __init__(self, method, lr, theta_init, momentum, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, theta_init, momentum)
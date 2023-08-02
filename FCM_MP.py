from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class FCM_MP(BaseEstimator, ClassifierMixin):

    def __init__(self, W1: np.ndarray = None, T: int = 10,
                 phi: float = 0.8, slope: float = 1.0, offset: float = 0.0):
        """
        Hybrid Fuzzy Cognitive Model (FCM_MP) for decision-making.
        
        Arguments:
        - W1: Weights defining the interaction between input variables (default: None).
        - T: Number of iterations to be performed during FCM reasoning (default: 10).
        - phi: Nonlinearity coefficient in the reasoning process (default: 0.8).
        - slope: Initial slope of the sigmoid function (default: 1.0).
        - offset: Initial offset of the sigmoid function (default: 0.0).
        """
        self.T = T
        self.phi = phi
        self.slope = slope
        self.offset = offset

        self.W1 = W1
        self.W2 = None
        self.P = None
        self.E = None

    def sigmoid(self, X: np.ndarray, slope: float = 1.0,
                offset: float = 0.0, l: float = 0.0, u: float = 1.0) -> np.ndarray:
        """
        Sigmoid function used for clipping concepts' activation values.
        
        Arguments:
        - X: Input to the sigmoid function.
        - slope: Initial slope of the sigmoid function (default: 1.0).
        - offset: Initial offset of the sigmoid function (default: 0.0).
        - l: Lower limit of the sigmoid function (default: 0.0).
        - u: Upper limit of the sigmoid function (default: 1.0).
        
        Returns:
        - Sigmoid activation values.
        """
        return l + (u - l) / (1.0 + np.exp(-slope * (X - offset)))

    def inverse(self, Y: np.ndarray, slope: float = 1.0,
                offset: float = 0.0, l: float = 0.0, u: float = 1.0) -> np.ndarray:
        """
        Inverse of the sigmoid function.
        
        Arguments:
        - Y: Output of the sigmoid function.
        - slope: Initial slope of the sigmoid function (default: 1.0).
        - offset: Initial offset of the sigmoid function (default: 0.0).
        - l: Lower limit of the sigmoid function (default: 0.0).
        - u: Upper limit of the sigmoid function (default: 1.0).
        
        Returns:
        - Inverse sigmoid values.
        """
        return offset + (1.0 / slope) * np.log((Y - l) / (u - Y))

    def reasoning(self, X: np.ndarray) -> np.ndarray:
        """
        Perform the reasoning process using the FCM model.
        
        Arguments:
        - X: Data to be used as initial activation vectors.
        
        Returns:
        - X: Output after the reasoning process.
        """
        X0 = X
        for t in range(1, self.T + 1):
            X = self.phi * self.sigmoid(
                np.dot(X, self.W1), self.slope, self.offset) + (1 - self.phi) * X0
        return X

    def coefficients(self, X_train: np.ndarray) -> np.ndarray:
        """
        Compute the coefficients of multiple regression models.
        
        Arguments:
        - X_train: Training data.
        
        Returns:
        - C: Matrix of coefficients.
        """
        n, m = X_train.shape
        T1 = np.sum(X_train, axis=0)
        T3 = np.sum(X_train ** 2, axis=0)

        C = np.random.random((m, m))

        for i in range(0, m):
            for j in range(0, m):
                d = n * T3[i] - T1[i] ** 2
                if d != 0:
                    C[i, j] = (n * np.sum(X_train[:, i] * X_train[:, j]) - T1[i] * T1[j]) / d

        np.fill_diagonal(C, 0)
        return C / np.max(np.abs(C))

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Fit the FCM model to the training data.
        
        Arguments:
        - X_train: Training data.
        - Y_train: Target values.
        
        Performs the following steps:
        1. Computes the coefficients matrix W1 if not provided.
        2. Performs the reasoning process to obtain the H matrix.
        3. Computes the weight matrix W using the MP inverse.
        4. Computes the expected values for calibration purposes.
        5. Performs normalization of learned output weights.
        
        Updates the following attributes:
        - W2: Weight matrix connected to the last layer.
        - E: Expected values for calibration purposes.
        """

        if self.W1 is None:
            self.W1 = self.coefficients(X_train)

        H = self.reasoning(X_train)
        self.W2 = np.dot(np.linalg.pinv(H), self.inverse(Y_train))
        self.P = [(self.slope, self.offset)] * Y_train.shape[1]

        if self.phi == 1.0:
            self.W2[self.W2 < -100] = -100
            self.W2[self.W2 > 100] = 100

        self.E = np.mean(X_train, axis=0)
        mx = np.max(np.abs(self.W2))

        if mx > 1:
            self.W2 /= mx
            self.P = [(self.slope * mx, self.offset / mx)] * Y_train.shape[1]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted FCM model.
        
        Arguments:
        - X_test: Test data.
        
        Returns:
        - output: Predicted values.
        """
        H = self.reasoning(X_test)
        sigmoid_inputs = np.dot(H, self.W2)

        output = np.empty_like(sigmoid_inputs)

        for i, params in enumerate(self.P):
            output[:, i] = self.sigmoid(sigmoid_inputs[:, i], *params)

        return output

    def weight_probabilities(self) -> np.ndarray:
        """
        Compute the probability matrix of weights in the last layer.
        
        Returns:
        - Probabilities of output weights.
        """
        return 1 - np.multiply(np.abs(self.W2), self.E.reshape(-1, 1))

    def remove_and_calibrate(self, i: int, j: int) -> list:
        """
        Remove the wij weight and perform calibration.
        
        Arguments:
        - i: Index of the first concept.
        - j: Index of the second concept.
        
        Returns:
        - parameters: Updated sigmoid parameters.
        """
        parameters = list(self.P[j])
        parameters[0] -= self.W2[i, j] * self.E[i]
        self.P[j] = tuple(parameters)

        self.W2[i, j] = 0
        return parameters

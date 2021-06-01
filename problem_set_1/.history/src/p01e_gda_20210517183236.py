import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = 1/m * np.sum(y)
        mu0 = np.dot(x.T, np.sum(1-y)) / np.sum(1-y)
        mu1 = np.dot(x.T, np.sum(y)) / np.sum(y)
        y_shaped = np.reshape(y, (-1, 1))#(m, 1)
        mux = y_shaped * mu1 + (1-y_shaped) * mu0
        x_centered = x - mux
        sigmia = 1 / m *np.sum(np.dot(x_centered.T, x_centered))
        sigmia_inv = np.linalg.inv(sigmia)
        theta = sigmia_inv * (mu1 - mu0)
        theta0 = mu0.T @ sigmia_inv @ mu0 - mu1.T @ sigmia_inv @ mu1 - np.log((1-phi) / phi)
        self.theta = np.insert(theta, 0, theta0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return util.add_intercept(x) @ self.theta >= 0
        # *** END CODE HERE

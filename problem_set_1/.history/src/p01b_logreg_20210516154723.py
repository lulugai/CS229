import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        old_theta = self.theta#(n,)
        forward = 1 / (1 + np.exp(-np.dot(x, old_theta)))#(m,)
        gradient = -1 / m * np.dot(x.T, (y - forward))#(n,)
        hessian = 1 / m * np.dot(x.T, (forward*(1-forward)*x))#(n,n)
        new_theta = old_theta - np.dot(np.linalg.inv(hessian), gradient)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:#L1范数
            old_theta = new_theta
            forward = 1 / (1 + np.exp(-np.dot(x, old_theta)))#(m,)
            gradient = -1 / m * np.dot(x.T, (y - forward))#(n,)
            hessian = 1 / m * np.dot(x.T, (forward*(1-forward)*x))#(n,n)
            new_theta = old_theta - np.dot(np.linalg.inv(hessian), gradient)
        self.theta = new_theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
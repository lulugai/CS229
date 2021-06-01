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
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    # util.plot(x_train, y_train, theta=logreg.theta)
    print("Theta is: ", logreg.theta)
    print("The accuracy on training set is: ", np.mean(logreg.predict(x_train) == y_train))
    util.plot(x_val, y_val, theta=logreg.theta, save_path=pred_path)
    print("The accuracy on valid set is: ", np.mean(logreg.predict(x_val) == y_val))
    print(logreg.predict(x_val))
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
        forward = np.reshape(forward, (-1, 1))#(m, 1)
        hessian = 1 / m * np.dot(x.T, (forward*(1-forward)*x))#(n,n)
        new_theta = old_theta - np.dot(np.linalg.inv(hessian), gradient)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:#L1范数
            old_theta = new_theta
            forward = 1 / (1 + np.exp(-np.dot(x, old_theta)))#(m,)
            gradient = -1 / m * np.dot(x.T, (y - forward))#(n,)
            forward = np.reshape(forward, (-1, 1))#(m, 1)
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
        return x @ self.theta >= 0
        # *** END CODE HERE ***

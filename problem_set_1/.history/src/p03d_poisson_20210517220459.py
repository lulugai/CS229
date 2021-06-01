import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    logreg = PoissonRegression()
    logreg.fit(x_train, y_train)
    print("Theta is: ", logreg.theta)
    print("The accuracy on training set is: ", np.mean(logreg.predict(x_train) == y_train))
    print("The accuracy on valid set is: ", np.mean(logreg.predict(x_val) == y_val))
    np.savetxt(pred_path, logreg.predict(x_val))
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def h(self, theta, x):
        return np.exp(x @ theta)

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        theta = self.theta
        
        next_theta = theta + self.step_size / m * x.T @ (y - self.h(theta, x))
        while np.linalg.norm(self.step_size / m * x.T @ (y - self.h(theta, x)), 1) >= self.eps:
            theta = next_theta
            next_theta = theta + self.step_size / m * x.T @ (y - self.h(theta, x))
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(self.theta, x)
        # *** END CODE HERE ***

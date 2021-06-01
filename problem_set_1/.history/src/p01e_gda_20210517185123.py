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
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    logreg = GDA()
    logreg.fit(x_train, y_train)
    # util.plot(x_train, y_train, theta=logreg.theta)
    print("Theta is: ", logreg.theta)
    print("The accuracy on training set is: ", np.mean(logreg.predict(x_train) == y_train))
    # util.plot(x_val, y_val, theta=logreg.theta, save_path=pred_path)
    print("The accuracy on valid set is: ", np.mean(logreg.predict(x_val) == y_val))
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
        mu0 = np.dot(x.T, 1-y) / np.sum(1-y)#(n, )
        mu1 = np.dot(x.T, y) / np.sum(y)
        y_shaped = np.reshape(y, (-1, 1))#(m, 1)
        mux = y_shaped * mu1 + (1-y_shaped) * mu0
        x_centered = x - mux
        sigmia = 1 / m * np.dot(x_centered.T, x_centered)
        sigmia_inv = np.linalg.inv(sigmia)
        theta = sigmia_inv @ (mu1 - mu0)
        theta0 = mu0.T @ sigmia_inv @ mu0 / 2 - mu1.T @ sigmia_inv @ mu1 / 2 - np.log((1-phi) / phi)
        self.theta = np.insert(theta, 0, theta0)
        print(mu0.shape, y_shaped.shape, sigmia.shape)#(2,) (800, 1) (2, 2)
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

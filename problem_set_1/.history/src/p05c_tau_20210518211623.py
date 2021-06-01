from operator import add
import matplotlib.pyplot as plt
import numpy as np
import util
import math

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lowest_mse = math.inf
    best_tau = None

    def plot(x, y_label, y_pred, title):
        plt.figure()
        plt.plot(x[:,-1], y_label, 'bx', label='label')
        plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')
        plt.show()

    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_valid_pred = clf.predict(x_valid)

        mse = np.mean((y_valid_pred - y_valid)**2)
        if mse < lowest_mse:
            lowest_mse = mse
            best_tau = tau

        plot(x_valid, y_valid, y_valid_pred, f'Validation Set (Tau = {tau}, MSE = {mse})')

    print(f'Tau = {best_tau} achieves the lowest MSE on the validation set.')
    y_test_pred = clf.predict(x_test)
    plot(x_test, y_test, y_test_pred, f'Test Set (MSE = {np.mean((y_test_pred - y_test)**2)})')
    # *** END CODE HERE ***

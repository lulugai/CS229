import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_train , y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_val, y_val = util.load_dataset(test_path, label_col='y', add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t')
    _, t_val = util.load_dataset(test_path, label_col='t')
    logreg = LogisticRegression()
    logreg.fit(x_train, t_train)
    print("Theta is: ", logreg.theta)
    print("The accuracy on training set is: ", np.mean(logreg.predict(x_train) == t_train))
    print("The accuracy on valid set(t) is: ", np.mean(logreg.predict(x_val) == t_val))
    np.savetxt(pred_path_c, logreg.predict(x_val))
    logreg.fit(x_train, y_train)
    print("Theta is: ", logreg.theta)
    print("The accuracy on training set is: ", np.mean(logreg.predict(x_train) == y_train))
    print("The accuracy on valid set(y) is: ", np.mean(logreg.predict(x_val) == y_val))
    np.savetxt(pred_path_d, logreg.predict(x_val))

    def sigmod(theta, x):
        return 1 / (1 + np.exp(-np.dot(x, theta)))
    def predict(x):
        return sigmod(logreg.theta, x) / alphi >= 0.5

    v_plus = x_val[y_val==1]
    alphi = sigmod(logreg.theta, v_plus).mean()
    print("The accuracy on valid set(modify) is: ", np.mean(predict(x_val) == y_val))
    np.savetxt(pred_path_e, predict(x_val))


    # *** END CODER HERE

import os
import sys
sys.path.append(os.path.join('..'))
import matplotlib.pyplot as plt
import numpy as np
import problem_set_2.src.util as util

def plot(x, y, title):
    plt.figure()
    plt.plot(x[y==1, 1], x[y==1, 2], 'bx', linewidth=2)

def main():
    x_train_a, y_train_a = util.load_csv('data/ds1_a.csv', add_intercept=True)
    x_train_b, y_train_b = util.load_csv('data/ds1_b.csv', add_intercept=True)
    print(x_train_a.shape, x_train_a)


if __name__ == '__main__':
    main()
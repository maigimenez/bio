# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
#from sympy import *


def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-tr", "--train", type=argparse.FileType('r'),
                        help="Scores for train data", metavar="TR",
                        dest="train_file")
    parser.add_argument("-te", "--test", type=argparse.FileType('r'),
                        help="Scores for test data", metavar="TE",
                        dest="test_file")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="Make plot",
                        dest="plot")

    try:
        args = parser.parse_args()
        if args.train_file is None or args.test_file is None:
            load_default()
        else:
            # Load train and test scores
            train  = np.loadtxt(args.train_file)
            test = np.loadtxt(args.test_file)

            # Normalization
            for score in xrange(train.shape[1] - 1):
                train[:, score] = np.divide(
                    train[:, score] - np.mean(train[:, score]),
                    np.std(train[:, score]))
                test[:, score] = np.divide(
                    test[:, score] - np.mean(test[:, score]),
                    np.std(test[:, score]))

            # Split clients & impostors
            c_train = train[train[:, 2] == 1]
            i_train = train[train[:, 2] == 0]
            c_test = test[train[:, 2] == 1]
            i_test = test[train[:, 2] == 0]

            return (c_train, i_train), (c_test, i_test), args.plot

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def aprox_AUR(w, clients, impostors):
    """ Get the aproximated area under the roc """

    # Delete clients/impostors tag
    c_scores = clients[:, :-1]
    i_scores = impostors[:, :-1]
    num_scores = c_scores.shape[1]

    # Step function
    heaviside = lambda x: 0.5 if x == 0 else 0 if x < 0 else 1
    sum_scores = 0.0
    for c in c_scores:
        for i in i_scores:
            for score in xrange(num_scores):
                subs_scores = w[score] * c[score] - i[score]
                sum_scores += heaviside(subs_scores)
    aprox_aur = sum_scores / float(c_scores.shape[0] * i_scores.shape[0])
    return aprox_aur


def aprox_w(clients, impostors):
    # Normalizar
    clients = clients - np.mean(clients) / np.std(clients)
    impostors = impostors - np.mean(impostors) / np.std(impostors)
    aurW = {}
    wp = itertools.product(np.arange(0, 1.1, 0.1), repeat=2)
    weights = [w for w in wp if sum(w) == 1.0]
    for w in weights:
        aur = aprox_AUR(w, clients, impostors)
        if aur in aurW.keys():
            aurW[aur].append(w)
        else:
            aurW[aur] = [w]

    maxAUR = max(aurW.keys())
    print ("El valor máximo del área bajo la curva ROC= "
           "%4f \nCon los pesos:" % maxAUR)
    for v in aurW[maxAUR]:
        print("  [ %.2f, %.2f ]" % (v[0], v[1]))


if __name__ == "__main__":
    (c_train, i_train), (c_test, i_test), p = get_data()
    aprox_w(c_train, i_train)
    if p:
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        ax1.set_ylabel("Score 1")
        ax2.set_ylabel("Score 1")
        ax2.set_xlabel("Score 2")
        ax1.plot(c_train[:, 0], c_train[:, 1], 'o', color='green')
        ax1.plot(i_train[:, 0], i_train[:, 1], 'o', color='red')
        ax1.set_title('Train Scores')
        ax2.plot(c_test[:, 0], c_test[:, 1], 'o', color='green')
        ax2.plot(i_test[:, 0], i_test[:, 1], 'o', color='red')
        ax2.set_title('Test Scores')
        plt.show()

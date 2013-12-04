import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-c", "--clients", type=argparse.FileType('r'),
                        help="Clients filename", metavar="C",
                        dest="clients_file")
    parser.add_argument("-i", "--impersonators", type=argparse.FileType('r'),
                        help="Impersonators filename", metavar="I",
                        dest="impersonators_file")
    try:
        args = parser.parse_args()
        if args.impersonators_file is None or args.clients_file is None:
            load_default()
        else:
            c_id, c_score = np.loadtxt(args.clients_file, unpack=True,
                                       dtype=float)
            i_id, i_score = np.loadtxt(args.impersonators_file, unpack=True,
                                       dtype=float)
            return c_score, i_score

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def solve_roc():
    c_score, i_score = get_data()

    # Get all possible threasholds and inserts a 0.
    thr = np.insert(np.unique(np.concatenate([c_score, i_score])), 0, 0)

    # Get false negative ratio
    fnr = np.divide(map(lambda x: np.sum(c_score <= x), thr),
                    float(len(c_score)))
    # Get true positive ratio
    tpr = 1.0 - fnr
    # Get false positive ratio
    fpr = np.divide(map(lambda x: np.sum(i_score > x), thr),
                    float(len(i_score)))
    # Get true negative ratio
    tnr = 1.0 - fpr
    print fpr
    print tpr

    plt.fill(fpr, tpr, 'r')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    solve_roc()

import argparse
import numpy


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
            c_id, c_score = numpy.loadtxt(args.clients_file, unpack=True)
            i_id, i_score = numpy.loadtxt(args.impersonators_file, unpack=True)
            return c_score, i_score

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def solve_roc():
    c_score, i_score = get_data()
    print c_score


if __name__ == "__main__":
    solve_roc()

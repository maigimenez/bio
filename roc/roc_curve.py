import argparse
import numpy as np
import ConfigParser

from roc_data import RocData


def load_default():
    # Read configuration file
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    # Get path where default recognition scores are
    clients_file = config.get('Default BioSystem', 'Clients')
    impersonators_file = config.get('Default BioSystem', 'Impersonators')
    c_id, c_score = np.loadtxt(clients_file, unpack=True,
                               dtype=float)
    i_id, i_score = np.loadtxt(impersonators_file, unpack=True,
                               dtype=float)
    data = RocData("",c_score,i_score)

    return data


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
    parser.add_argument("-fp", type=float,
                        help="False positive", metavar="FP",
                        dest="fp")
    parser.add_argument("-fn", type=float,
                        help="False negative", metavar="FN",
                        dest="fn")
    parser.add_argument("-p","--plot", action='store_true',
                        help="Make plot",
                        dest="plot")
    parser.add_argument("-a","--aur", action='store_true',
                        help="Get area under the ROC curve",
                        dest="aur")
    parser.add_argument("-d","--dprime", action='store_true',
                        help="Get dprime",
                        dest="dprime")
    try:
        args = parser.parse_args()
        if args.impersonators_file is None or args.clients_file is None:
            data = load_default()
        else:
            c_id, c_score = np.loadtxt(args.clients_file, unpack=True,
                                       dtype=float)
            i_id, i_score = np.loadtxt(args.impersonators_file, unpack=True,
                                       dtype=float)
            data = RocData("",c_score,i_score)
        return data, args.fp, args.fn, args.plot, args.aur, args.dprime

    except SystemExit:
        data = load_default()
        return data, False, False, False, False, False

def find_nearest_pos(scores, value):
    return (np.abs(scores-value)).argmin()


def get_fn(data,fp):
    pos = find_nearest_pos(data.fpr, fp)
    print("Dado el valor de fp: {0}, el valor de fnr es: {1} y el umbral: {2} "
          .format(fp,data.fnr[pos],data.thrs[pos]))


def get_fp(data,fn):
    pos = find_nearest_pos(data.fnr, fn)
    print("Dado el valor de fn: {0}, el valor de fp es: {1} y el umbral: {2} "
          .format(fn,data.fpr[pos],data.thrs[pos]))


if __name__ == "__main__":
    data, fp, fn, plot, aur, dprime = get_data() 
    if data: data.solve_ratios()
    if fp:
        get_fn(data,fp)
    if fn:
        get_fp(data,fn)
    if aur:
        data.aur(plot)
    if dprime:
        data.dprime(plot)
    elif not aur and not dprime and plot:
        data.plot()

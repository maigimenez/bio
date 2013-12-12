import argparse
import numpy as np
import ConfigParser
from scipy import interpolate
from roc_data import RocData


def load_default():
    """ Load a default set of data """
    # Read configuration file
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    # Get path where default recognition scores are
    clients_file = config.get('Default BioSystem', 'Clients')
    impersonators_file = config.get('Default BioSystem', 'Impersonators')
    # Load default data
    c_id, c_score = np.loadtxt(clients_file, unpack=True,
                               dtype=float)
    i_id, i_score = np.loadtxt(impersonators_file, unpack=True,
                               dtype=float)
    # Init RocData
    data = RocData("",c_score,i_score)
    return data


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    # Add arguments 
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-c", "--clients", type=argparse.FileType('r'),
                        help="Clients filename", metavar="C",
                        dest="clients_file")
    parser.add_argument("-i", "--impersonators", type=argparse.FileType('r'),
                        help="Impostors filename", metavar="I",
                        dest="impostors_file")
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
                        help="Get area under the ROC curve using trapezoidal rule ",
                        dest="aur")
    parser.add_argument("-aA","--aurAprox", action='store_true',
                        help="Get area under the ROC curve using an aproximation",
                        dest="aur_aprox")
    parser.add_argument("-d","--dprime", action='store_true',
                        help="Get dprime",
                        dest="dprime")
    try:
        # Parse arguments
        args = parser.parse_args()
        # If some filename is missing -> load default arguments
        if args.impostors_file is None or args.clients_file is None:
            data = load_default()
        # Load user arguments
        else:
            c_id, c_score = np.loadtxt(args.clients_file, unpack=True,
                                       dtype=float)
            i_id, i_score = np.loadtxt(args.impostors_file, unpack=True,
                                       dtype=float)
            data = RocData("",c_score,i_score)
        return (data, args.fp, args.fn, args.plot, args.aur, args.aur_aprox, 
                args.dprime)

    except SystemExit:
        data = load_default()
        return data, False, False, False, False, False, False



def get_fn(data,fp):
    if fp in data.fpr:
        pos =  np.where(data.fpr==fp)
        fnr, thr =  np.mean(data.fnr[pos]), np.mean(data.thrs[pos])
    else:
        # Set data for interpolation
        x = np.sort(data.fpr)
        # Set new arange whichs includes the wanted value
        xnew = np.arange(fp, x[-1])
        # Interpolate the FN
        y = np.sort(data.tpr)
        f = interpolate.interp1d(x, y)
        tpr = f(xnew)[0]
        fnr = 1 - tpr
        # Interpolate the threashold
        y = np.sort(data.thrs)
        f = interpolate.interp1d(x, y)
        thr = f(xnew)[0]
    print("Dado el valor de fp: {0}, el valor de fnr es: {1} y el umbral: {2} "
          .format(fp,fnr,thr))


def get_fp(data,fn):
    if fn in data.fnr:
        pos =  np.where(data.fnr==fn)
        fpr, thr =  np.mean(data.fpr[pos]), np.mean(data.thrs[pos])
    else:
        # Set data for interpolation
        x = np.sort(data.tpr)
        # Set new arange whichs includes the wanted value
        xnew = np.arange(fn, x[-1])
        # Interpolate the FN
        y = np.sort(data.fpr)
        f = interpolate.interp1d(x, y)
        fpr = f(xnew)[0]
        # Interpolate the threashold
        y = np.sort(data.thrs)
        f = interpolate.interp1d(x, y)
        thr = f(xnew)[0]
    print("Dado el valor de fn: {0}, el valor de fpr es: {1} y el umbral: {2} "
          .format(fn,fpr,thr))



if __name__ == "__main__":
    data, fp, fn, plot, aur,aur_aprox, dprime = get_data() 
    data.solve_ratios()
    if fp >= 0.0:
        get_fn(data,fp)
    if fn >= 0.0:
        get_fp(data,fn)
    if aur:
        data.aur(plot)
    if aur_aprox:
        data.aur_aprox(plot)
    if dprime:
        data.dprime(plot)
    elif not aur and not aur_aprox  and not dprime and plot:
        data.plot()

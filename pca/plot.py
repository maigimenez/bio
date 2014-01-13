# -*- coding: utf-8 -*-
import argparse
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    """ Get data to plot.
    """
    # Add arguments 
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-p", "--path",
                        help="Path with PCA prediction ouputs", metavar="PCA",
                        dest="predictions")
    parser.add_argument("-s", "--save",
                        help="Path where save PCA predictions plot", metavar="FILENAME",
                        dest="save")
    try:
        args = parser.parse_args()
        predictions_files = [ join(args.predictions,f) for f in listdir(args.predictions) if f.endswith(".txt")]
        return predictions_files, args.save
    
    except SystemExit:
        sys.exit(-1)

if __name__ == "__main__":
    predictions_files, save_path = get_data() 
    x = []
    y = []
    for p_file in predictions_files:
        tag, results = np.loadtxt(p_file, usecols = (0,3), dtype='str', unpack=True)
        x.append(tag[0])
        y.append(float(len([res for res in results if res=='False' ]))/len(results))

    plt.plot(x, y, 'r')
    plt.grid(True)
    plt.title('Curva de error PCA')
    plt.ylabel('Tasa de error')
    plt.xlabel('Dimensiones')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
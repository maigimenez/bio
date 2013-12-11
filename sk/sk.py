import argparse
import numpy as np


def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-f", "--faces", type=argparse.FileType('r'),
                        help="Filename with faces data", metavar="F",
                        dest="faces_file")
    parser.add_argument("-n", "--not_faces", type=argparse.FileType('r'),
                        help="Filename with not faces data", metavar="NF",
                        dest="notfaces_file")

    try:
        args = parser.parse_args()
        if args.notfaces_file is None or args.faces_file is None:
            load_default()
        else:
            print np.loadtxt(args.faces_file)
            return None, None

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


if __name__ == "__main__":
    faces = get_data() 

#f = np.loadtxt('data/caras/dfFaces_21x21_norm')
#>>> img.show()
#>>> import matplotlib.pyplot as plt
#>>> plt.imshow((f[0].reshape((21,21))))
#<matplotlib.image.AxesImage object at 0x104dd7a90>
#>>> plt.gray()
#>>> plt.show()
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil,sqrt
from numpy import linalg as la

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
            faces_21x21 = np.loadtxt(args.faces_file)
            not_faces_21x21 = np.loadtxt(args.notfaces_file)

            faces = []
            for face in faces_21x21 :
                faces.append(face.reshape((21,21))[:-1,:-1])

            not_faces = []
            for nface in not_faces_21x21:
                not_faces.append(nface.reshape((21,21))[:-1,:-1])

            #plt.imshow((faces[0].reshape((20,20)))) 
            #plt.gray()
            #plt.show()
            #print type(faces)
            return np.array(faces), np.array(not_faces)

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()

def train(data):
    #n, d = data.shape
    for image in data:
        print image.shape


if __name__ == "__main__":
    per_train = 0.8
    per_test = 1 - per_train
    faces, not_faces = get_data() 

    # Shuffles data, and get train and test sets for faces. 
    np.random.shuffle(faces)
    examples, d1, d2 = faces.shape
    sep = ceil(per_train*examples)
    faces_train = faces[0:sep]
    faces_test = faces[sep:] 
    #print faces.shape, faces_test.shape, faces_train.shape

    # Shuffles data, and get train and test sets for not faces. 
    np.random.shuffle(not_faces)
    examples, d1, d2 = not_faces.shape
    sep = ceil(per_train*examples)
    not_faces_train = not_faces[0:sep]
    not_faces_test = not_faces[sep:]
    #print examples, not_faces_train.shape,  not_faces_test.shape

    #Train
    train(faces_train)


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
                faces.append(face.reshape((21,21))[:-1,:-1].flatten())

            not_faces = []
            for nface in not_faces_21x21:
                not_faces.append(nface.reshape((21,21))[:-1,:-1].flatten())

            #plt.imshow((faces[0].reshape((20,20)))) 
            #plt.gray()
            #plt.show()
            #print type(faces)
            return np.array(faces), np.array(not_faces)

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()

def PCA(data):
    pass

if __name__ == "__main__":
    per_train = 0.8
    per_test = 1 - per_train
    faces, not_faces = get_data() 

    # Shuffles data, and get train and test sets for faces. 
    np.random.shuffle(faces)
    examples, dimension = faces.shape
    sep = ceil(per_train*examples)
    faces_train = faces[0:sep]
    faces_test = faces[sep:] 

    # PCA    
    face = faces_train[0].reshape(20,20)
    mu = np.mean(face)
    A = face-mu
    C = np.dot(A,A.T)
    #Delta=eigenvalues B=eigenvectors
    D,B = la.eigh(C) 
    #print A.shape
    #print D.shape, B.shape
    ##print examples, faces_train.shape,  faces_test.shape
    #Y = np.dot(B.T,A)
    ##print Y.shape

    #tmp = np.dot(A.T,B).T #this is the compact trick
    #V = tmp[::-1] #reverse since last eigenvectors are the ones we want
    #S = sqrt(D)[::-1] #reverse since eigenvalues are in increasing order
    #print S

    # Shuffles data, and get train and test sets for faces. 
    np.random.shuffle(not_faces)
    examples, dimension = not_faces.shape
    sep = ceil(per_train*examples)
    not_faces_train = not_faces[0:sep]
    not_faces_test = not_faces[sep:]
    #print examples, not_faces_train.shape,  not_faces_test.shape



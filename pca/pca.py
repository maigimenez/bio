import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil,sqrt
from numpy import linalg as la
#from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import cv2

def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-p", "--path", type=string,
                        help="Path with faces data", metavar="F",
                        dest="faces_path")
    try:
        args = parser.parse_args()
        if args.faces_path is None:
            load_default()
        else:
            print faces
            #faces = np.loadtxt(args.faces_file)
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
    #np.random.shuffle(faces)
    #examples, dimension = faces.shape
    #sep = ceil(per_train*examples)
    #faces_train = faces[0:sep]
    #faces_test = faces[sep:] 

    # PCA
    #face = faces[0]
    #print face.shape
    #face = faces_train[0].reshape(21,21)
    #n,d = faces.shape
    #mu = faces.mean(axis=0)
    #A = faces-mu

    # Covariance matrix
    #C =  np.dot(A,A.T)
    #C = np.dot(A,A.T)

    #Delta=eigenvalues B=eigenvectors
    #D,B = la.eigh(C) 

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



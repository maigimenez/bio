import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil,sqrt
from numpy import linalg as la
#from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import cv
import pylab
from os import walk
from os.path import join

def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-p", "--path",
                        help="Path with faces data", metavar="F",
                        dest="faces_path")
    try:
        args = parser.parse_args()
        if args.faces_path is None:
            load_default()
        else:
            image_files = []
            for root, dirs, files in walk(args.faces_path):
                image_files.extend([ f for f in files if f.endswith(".pgm")])

            # TODO: check if there aren't images, otherwise it will fail. 
            # Get image dimensions
            image = cv.LoadImage(join(args.faces_path,image_files[0]),cv.CV_LOAD_IMAGE_GRAYSCALE)
            dim = np.asarray(cv.GetMat(image)).flatten().shape[0]
            # Init numpy array 
            images = np.empty([len(image_files), dim])
            # Fill it with images
            img_no = 0
            for image_file in image_files:
                image = cv.LoadImage(join(args.faces_path,image_file),cv.CV_LOAD_IMAGE_GRAYSCALE)
                images[img_no] = np.asarray(cv.GetMat(image)).flatten()
                img_no += 1
                #print img.shape
                #plt.imshow(img) 
                #plt.gray()
                #plt.show()
            return images

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()

def PCA(data):
    pass

if __name__ == "__main__":
    faces = get_data() 

    # PCA
    #num_data,dim = faces[0].shape
    #A = []
    #for face in faces:
    #    A.append(face.flatten())
    #print faces.flatten()
    #mu = A.mean(axis=0)
    #A = A-mu
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




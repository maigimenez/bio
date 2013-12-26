# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil,sqrt
from numpy import linalg as la
#from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import cv
import cv2
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
                # create the window
                #cv.NamedWindow('Face', cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage('Face', image) # show the image
                #cv.WaitKey() # the window will be closed with a (any)key press
            return images

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()

def PCA(X, d_prime):
    print X.shape
    n,d = X.shape
    # mu: vector promedio
    mu = X.mean(axis=0)
    # Restamos la media 
    for i in range(n):
        X[i] -= mu 
    A = X.T

    if d>100:
        # C: Matriz de covarianzas
        C_prime = 1.0/d * np.dot(A.T,A)
        #Delta=eigenvalues B=eigenvectors
        D_prime,B_prime = la.eigh(C_prime) 
        B = np.dot(A, B_prime)
        D = d/n * D_prime
        print B.shape, D.shape
        # Ordenamos los vectores propios, primero los que más varianza recogen 
        order = np.argsort(D)[::-1] # sorting the eigenvalues
        # Ordenamos los vectores propios & los valores propios
        B = B[:,order]
        D = D[order]
        #V = tmp[::-1]
        #print V.shape
    else:
        #C = 
        pass

    print "B: ", B.shape
    print "X: ",X.shape
    Y = np.empty([n, d_prime])
    for i in range(n):
        aux = B.T[i]*X[i]
        Y[i] = aux[range(d_prime)] 

    print "Face:", X[0].reshape(112,92)
    #plt.imshow(X[0].reshape((92,112))) 
    ##plt.gray()
    #plt.show()
    cv2.imwrite("demo.pgm",X[0].reshape(112,92))
    cv.NamedWindow('Face', cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage('Face', X[0].reshape(112,92)) # show the image
    cv.WaitKey() # the window will be closed with a (any)key press
    # Proyección con todas las componentes
    #tmp = np.dot(B.T,A)
    #print "tmp:", tmp.shape
        
    # Cogemos únicamente los vectores propios de la proyección con d_prime componentes
    #if d_prime<d:
    #    tmp = tmp[range(d_prime),:]
    #print tmp.shape
    #print B.T.shape
    #print X.shape
    #print A.shape
    #Y = np.dot(B.T,A)
    #print Y.shape

    #print B.T.shape, X.shape
    #print X.T.shape, B.shape
    #print B.shape, X.T.shape
    
    #Y = np.dot(X.T,B)
    #print Y.shape

if __name__ == "__main__":
    # X: vectores de entrenamiento
    X = get_data() 

    # PCA
    PCA(X,1000)
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
#from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import cv
import cv2
import pylab
from os import walk
from os.path import join,basename
import Image
from random import shuffle


def load_default():
    print "TODO: load default scores"
    return None, None


def get_data(per_train, per_test):
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="PCA Algorithm")
    parser.add_argument("-p", "--path",
                        help="Path with faces data", metavar="F",
                        dest="faces_path")
    parser.add_argument("-d", "--dprime",
                        type=float,
                        help="Number of dimensions to project", metavar="DIM",
                        dest="d_prime")
    try:
        args = parser.parse_args()
        if args.faces_path is None:
            load_default()
        else:
            train_image_files = {}
            test_image_files = {}
            for root, dirs, files in walk(args.faces_path):
                images = [ f for f in files if f.endswith(".pgm")]
                shuffle(images)
                tag = basename(root)
                if len(images)>0:
                    train_image_files[tag] = images[:int(len(images)*per_train)]
                    test_image_files[tag] = images[-int(len(images)*per_test):]

            # TODO: check if there aren't images, otherwise it will fail. 
            # Get image dimensions
            image = cv.LoadImage(join(args.faces_path,tag,train_image_files[tag][0]),
                                 cv.CV_LOAD_IMAGE_GRAYSCALE)
            dim = np.asarray(cv.GetMat(image)).flatten().shape[0]

            # Init numpy array 
            train_images = {}
            test_images = {}
            #images = np.empty([len(image_files), 400])
            # Fill it with images
            for tag, image_files in train_image_files.iteritems():
                img_no = 0
                train_images[tag] =  np.empty([len(image_files), dim])
                for image_file in image_files:
                    image = cv.LoadImage(join(args.faces_path,tag,image_file),
                                         cv.CV_LOAD_IMAGE_GRAYSCALE)
                    train_images[tag][img_no] = np.asarray(cv.GetMat(image)).flatten()
                    img_no += 1

            for tag, image_files in test_image_files.iteritems():
                img_no = 0
                test_images[tag] = np.empty([len(image_files), dim])
                for image_file in image_files:
                    image = cv.LoadImage(join(args.faces_path,tag,image_file),
                                         cv.CV_LOAD_IMAGE_GRAYSCALE)
                    test_images[tag][img_no] = np.asarray(cv.GetMat(image)).flatten()
                    img_no += 1

                #print 'Size:',image.width,image.height
                # create the window
                #cv.NamedWindow('Face', cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage('Face', image) # show the image
                #cv.WaitKey() # the window will be closed with a (any)key press
            return train_images, test_images, args.d_prime

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def pca(X, d_prime):
    d,n = X.shape
    # mu: vector promedio
    mu = X.mean(axis=0)
    # Restamos la media 
    for i in range(n):
        X[i] -= mu 
    A = X.copy()

    if d>200 and n<3*d:
        if d_prime > n:
            d_prime = n
        # C: Matriz de covarianzas
        C_prime = 1.0/d * np.dot(A.T,A)
        #Delta=eigenvalues B=eigenvectors
        D_prime,B_prime = la.eigh(C_prime)
        #print "B prime: ", B_prime.shape, "- delta: ",  D_prime.shape

        for i in xrange(n):
            B_prime[:,i] = B_prime[:,i]/np.linalg.norm(B_prime[:,i])

        B = np.dot(A, B_prime)
        D = d/n * D_prime
        #print "B complete: ", B.shape, "- delta: ",  D.shape
        # Ordenamos los vectores propios, primero los que más varianza recogen 
        order = np.argsort(D, axis=0)[::-1] 
        # Ordenamos los vectores propios & los valores propios
        B = B[:,order]
        D = D[order]

    else:
        C = 1.0/n * np.dot(A,A.T)
        D,B = la.eigh(C) 
        # Ordenamos los vectores propios, primero los que más varianza recogen 
        order = np.argsort(D)[::-1] # sorting the eigenvalues
        # Ordenamos los vectores propios & los valores propios
        B = B[:,order]
        D = D[order]

    # B_dprime (d'xn)
    #print "B: ", B.shape, " - ", B[:,:d_prime].shape
    #print "D: ", D.shape
    #print "X: ", X.shape
    #print "d': ",d_prime
    #print "mu: ", mu.shape
    #Proyectamos los datos en d'
    B_dprime = B[:,:d_prime]
    y = np.dot(B_dprime.T,X)
    #print y[0]
    #print 
    #print
    #return ['B_dprime':B_dprime,D,B,mu,X]
    return {'B':B, 'B_dprime':B_dprime,'mu':mu,'y':y}, d_prime


def predict(pca_train, test):
    _, n_images = test.shape
    distances={}
    for tag,pca in pca_train.iteritems():
        #Normalizar la imagen de test
        #print test.T[0]
        #print pca['mu'][0]
        test_norm = test - pca['mu']
        #print test_norm[0]
        # Proyectar imagen de test
        #print test_norm.shape, pca['B'].shape
        y_test = np.dot(pca['B'].T,test_norm)
        #print y_test[0]
        #print
        #print y_test.shape
        minium = float('inf')
        for i in range(n_images):
            dif = (pca['y']-y_test[:,i])**2
            euclidean = np.sqrt(dif.sum(axis=0))

        #print dif[0]
        #euclidean = np.sqrt(np.sum(dif**2,  axis=1)) 
        distances[tag]= min(euclidean)

    #print distances
    predictions = []
    for i in range(n_images):
        minium = float('inf')
        prediction = ''
        #print
        for tag, distance in distances.iteritems():
            if distance < minium:
                prediction = tag
                minium = distance
                #print tag
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    # X: vectores de entrenamiento
    per_train = 0.5
    per_test = 1 - per_train
    train, test, d_prime = get_data(per_train,per_test)
    #print train['s1']

    # PCA
    pca_train = {}
    for k,v in train.iteritems():
       pca_train[k], d_prime = pca(train[k].T, d_prime)

    for tag, images in test.iteritems():
        predictions = predict(pca_train, test[tag].T)
        for image in range(len(images)):
            print d_prime, tag, predictions[image], tag==predictions[image]


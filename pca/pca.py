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
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-p", "--path",
                        help="Path with faces data", metavar="F",
                        dest="faces_path")
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
            return train_images, test_images

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def pca(X, d_prime):
    n,d = X.shape
    # mu: vector promedio
    mu = X.mean(axis=0)
    # Restamos la media 
    for i in range(n):
        X[i] -= mu 
    A = X.T

    if d>200:
        # C: Matriz de covarianzas
        C_prime = 1.0/d * np.dot(A.T,A)
        #Delta=eigenvalues B=eigenvectors
        D_prime,B_prime = la.eigh(C_prime)
        
        for i in xrange(n):
            B_prime[:,i] = B_prime[:,i]/np.linalg.norm(B_prime[:,i])

        # Ordenamos los vectores propios, primero los que más varianza recogen 
        order = np.argsort(D_prime)[::-1] # sorting the eigenvalues
        # Ordenamos los vectores propios & los valores propios
        B_prime = B_prime[:,order]
        D_prime = D_prime[order]
        #print B_prime.shape, D_prime.shape
        B = np.dot(A, B_prime)
        D = d/n * D_prime
        #print "B complete: ", B.shape, "- delta: ",  D.shape

    else:
        C = 1.0/n * np.dot(A,A.T)
        D,B = la.eigh(C) 
        # Ordenamos los vectores propios, primero los que más varianza recogen 
        order = np.argsort(D)[::-1] # sorting the eigenvalues
        # Ordenamos los vectores propios & los valores propios
        B = B[:,order]
        D = D[order]


    #print "B: ", B.shape, " - ", B[:d_prime].shape
    #print "D: ", D.shape
    #print "X: ",X.shape
    #print "d': ",d_prime
    
    #Proyectamos los datos en d'
    B_dprime = np.dot(B[:d_prime],X)

    return [B_dprime,D,B,mu,X]


def predict(pca_train, test):
    distances={}
    #Finding weights
    for k,v in pca_train.iteritems():
        #Normalizar la imagen de test
        shifted_in = test - v[3]
        w = np.dot(v[0],v[4].T) 
        w_in = np.dot(v[0],shifted_in)

        # Distancia euclídea
        df = np.asarray(w.T - w_in)    
        dst = np.sqrt(np.sum(df**2,  axis=0))     
        distances[k]= dst

    minimun = 1000000000000
    tag = ""
    for k in distances.keys():
        k_min = distances[k].min()
        if k_min < minimun :
            minimun = distances[k].min()
            tag=k
    return tag

if __name__ == "__main__":
    # X: vectores de entrenamiento
    per_train = 0.6
    per_test = 1 - per_train
    train, test = get_data(per_train,per_test)

    # PCA
    pca_train = {}
    for k,v in train.iteritems():
       pca_train[k] = pca(train[k], 1000)

    #[evalues, evectors, mean_image, shifted_images]=pca(train['s1'],300)
    for tag, images in test.iteritems():
        for image in images:
            predict_tag = predict(pca_train, image)
            print tag == predict_tag
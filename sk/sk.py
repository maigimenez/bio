# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
#import matplotlib.pyplot as plt
from math import ceil
from scipy.cluster.vq import kmeans, vq, whiten, kmeans2
from numpy import vstack
import sys
import cv

def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="SK Algorithm")
    parser.add_argument("-f", "--faces", type=argparse.FileType('r'),
                        help="Filename with faces data", metavar="F",
                        dest="faces_file")
    parser.add_argument("-n", "--not_faces", type=argparse.FileType('r'),
                        help="Filename with not faces data", metavar="NF",
                        dest="notfaces_file")
    parser.add_argument("-ft", "--faces_test", type=argparse.FileType('r'),
                        help="Filename with faces data", metavar="FT",
                        dest="faces_test_file")
    parser.add_argument("-nt", "--not_faces_test", type=argparse.FileType('r'),
                        help="Filename with not faces data", metavar="NFT",
                        dest="notfaces_test_file")
    parser.add_argument("-l", "--lambda", type=float,
                        help="Lambda value", metavar="L",
                        dest="lambda_value")
    parser.add_argument("-d","--dev", action='store_true',
                        help="Tune lambda value",
                        dest="dev")
    parser.add_argument("-test","--test", action='store_true',
                        help="Test SK Algorithm",
                        dest="test")
    parser.add_argument("-i", "--image",
                        help="Image where identify faces", metavar="I",
                        dest="test_image_path")
    try:
        args = parser.parse_args()
        if args.notfaces_file is None or args.faces_file is None:
            load_default()
        else:
            faces_21x21 = np.loadtxt(args.faces_file)
            not_faces_21x21 = np.loadtxt(args.notfaces_file)

            # Eliminar un píxel para hacer la imagen cuadrada 20x20
            faces = []
            for face in faces_21x21:
                faces.append(face.reshape((21, 21))[:-1, :-1])

            not_faces = []
            for nface in not_faces_21x21:
                not_faces.append(nface.reshape((21, 21))[:-1, :-1])

            faces_test_21x21 = np.loadtxt(args.faces_test_file)
            not_faces_test_21x21 = np.loadtxt(args.notfaces_test_file)

            # Eliminar un píxel para hacer la imagen cuadrada 20x20
            faces_test = []
            for face in faces_test_21x21:
                faces_test.append(face.reshape((21, 21))[:-1, :-1])

            not_faces_test = []
            for nface in not_faces_21x21:
                not_faces_test.append(nface.reshape((21, 21))[:-1, :-1])

            image_test = None
            if args.test_image_path:
                image = cv.LoadImage(args.test_image_path,
                                    cv.CV_LOAD_IMAGE_GRAYSCALE)
                image_test = np.asarray(cv.GetMat(image))

                #cv.NamedWindow('Face', cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage('Face', image) # show the image
                #cv.WaitKey() # the window will be closed with a (any)key press

            #plt.imshow((faces[0].reshape((20,20))))
            #plt.gray()
            #plt.show()
            #print type(faces)
            return (np.array(faces), np.array(not_faces), args.lambda_value, 
                    args.test, image_test, args.dev, faces_test, not_faces_test)

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def get_regions(width, height, region_dim):
    #Get the regions
    regions = []
    for i in range(0, height, region_dim):
        for j in range(0, width, region_dim):
            regions.append(((i, i + region_dim), (j, j + region_dim)))

    return regions


def split_image(image,regions, image_regions, flatten):
    for region in regions:
        image_region = np.array(image[region[0][0]:region[0][1],
                            region[1][0]:region[1][1]])
        if flatten:
            image_region.flatten()

        image_regions.append(image_region)


def split_images(images,regions, image_regions, flatten):
    for image in images:
        split_image(image,regions, image_regions, flatten)

def quantification(image_regions, q_levels):
    # Quantification
    #print 
    #print image_regions
    data = vstack(image_regions)
    #print data, q_levels
    whitened = whiten(data)
    centroids, _ = kmeans(whitened, q_levels, iter=1)
    idx, _ = vq(data, centroids)
    #print "@", idx, idx.shape
    return idx


def train(faces, not_faces, num_regions, q_levels):

    num_faces = len(faces)
    #print "Faces:", len(faces)
    #print "Not faces:", len(not_faces)

    # Split into regions
    # TODO: Check if there are no faces
    width, height = faces[0].shape
    region_dim = int(np.sqrt((width * height) / num_regions))
    regions = get_regions(width, height, region_dim)

    # Get regions from train images
    image_regions = []
    # Split faces in regions
    split_images(faces,regions,image_regions, True)
    # Split not faces in regions
    split_images(not_faces,regions,image_regions, True)

    # Quantification
    idx = quantification(image_regions, q_levels)
    faces_q = idx[:num_faces * num_regions]
    notFaces_q = idx[num_faces * num_regions + 1:]

    # Estimating v_faces
    # Try to use  a dictionary insted of an array
    # because if there are many 0s a lot space unused (spare matrix)
    v_faces = np.ones(q_levels)
    #v_faces = dict.fromkeys(set(faces_q), 0)
    for q in faces_q:
        v_faces[q] += 1
    # Normalized
    p_q_faces = v_faces / num_faces

    v_notFaces = np.ones(q_levels)
    #v_notFaces = dict.fromkeys(set(notFaces_q), 0)
    for q in notFaces_q:
        v_notFaces[q] += 1
    p_q_notFaces = v_notFaces / num_faces

    # For every face create a dictionary where keys are the regions 
    # and values are the quantification tag for that image and that position.
    tagged_faces = []
    for i in xrange(0, len(faces_q), num_regions):
        tagged_face = dict.fromkeys(xrange(num_regions))
        for j in xrange(0, num_regions):
            tagged_face[j] = faces_q[i + j]
        tagged_faces.append(tagged_face)

    p_pos_q_notFaces = 1.0 / num_regions

    m_faces = np.ones((num_regions, q_levels))
    for face in tagged_faces:
        for region, q in face.iteritems():
            m_faces[region][q] += 1

    sum_column = np.sum(m_faces, axis=0)

    #print m_faces[m_faces!=1.0]
    p_pos_q_faces = m_faces.copy()
    rows, columns = m_faces.shape
    for col in range(columns):
        p_pos_q_faces[:,col] = m_faces[:,col]/sum_column[col]
    #print p_q_faces
    #print
    #print p_q_notFaces, 
    #print 
    #print p_pos_q_faces, p_pos_q_faces.shape
    return p_q_faces, p_q_notFaces, p_pos_q_faces, p_pos_q_notFaces, width


def dev(image, p_q_faces, p_q_notFaces, p_pos_q_faces,
        p_pos_q_notFaces, num_regions):
    width, height =image.shape
    region_dim = int(np.sqrt((width * height) / num_regions))
    regions = get_regions(width, height, region_dim)
    image_regions = []
    split_image(image, regions, image_regions, True)
    # Quantification
    q = quantification(image_regions, q_levels)
    if not (q[q<0].shape[0]!=0 or  q[q>256].shape[0]!=0):
        prob = 1.0
        for i in range(num_regions):
            num = p_pos_q_faces[i][q[i]] * p_q_faces[q[i]]
            den = p_q_notFaces[q[i]] * p_pos_q_notFaces 
            prob *= num / den
    else:
        prob = 0.0
        #print p_pos_q_faces[i][q[i]], "*", p_q_faces[q[i]], "/", p_q_notFaces[q[i]], "*", p_pos_q_notFaces
        #, "*", p_pos_q_notFaces[i]    print
   # print "*", len(regions)
    #print "!", p_pos_q_faces.shape
    return prob

if __name__ == "__main__":
    num_regions = 16
    q_levels = 256
    per_train = 0.8
    per_test = 1 - per_train
    (faces, not_faces, lambdav, test_mode, image_test, dev_mode,
     faces_test, not_faces_test ) = get_data()

    # Shuffles data, and get train and test sets for faces.
    np.random.shuffle(faces)
    examples, d1, d2 = faces.shape
    sep = ceil(per_train * examples)
    faces_train = faces[0:sep]
    faces_dev = faces[sep:]

    #print faces.shape, faces_test.shape, faces_train.shape

    # Shuffles data, and get train and test sets for not faces.
    np.random.shuffle(not_faces)
    examples, d1, d2 = not_faces.shape
    sep = ceil(per_train * examples)
    not_faces_train = not_faces[0:sep]
    not_faces_dev = not_faces[sep:]
    #print examples, not_faces_train.shape,  not_faces_test.shape

    #Train
    (p_q_faces, p_q_notFaces, 
     p_pos_q_faces, p_pos_q_notFaces, 
     window) = train(faces_train, not_faces_train, num_regions, q_levels)

    sys.stdout.write('\a')
    sys.stdout.flush()

    # Development. Used to tune lambda value properly
    if (dev_mode):
        num_faces = len(faces_dev)
        prob_faces = np.zeros(num_faces)
        for i in xrange(num_faces):
            prob_faces[i]=dev(faces_dev[i], p_q_faces, p_q_notFaces, p_pos_q_faces,
                              p_pos_q_notFaces, num_regions)
            #print prob_faces[i], lambdav, prob_faces[i]-lambdav
        
        #print
        true_negatives = 0
        false_negatives = 0
        num_notFaces = len(not_faces_dev)
        prob_notfaces = np.zeros(num_notFaces)
        for i in xrange(num_notFaces):
            prob_notfaces[i] = dev(not_faces_dev[i], p_q_faces, p_q_notFaces, p_pos_q_faces,
                                   p_pos_q_notFaces, num_regions)
            #print prob_notfaces[i], lambdav, prob_notfaces[i]-lambdav
        print np.mean(prob_faces), np.mean(prob_notfaces)

    # Development. Used to tune lambda value properly
    if (test_mode):
        true_positives = 0
        false_positives = 0
        num_faces = len(faces_test)
        prob_faces = np.zeros(num_faces)
        for i in xrange(num_faces):
            prob_faces[i]=dev(faces_test[i], p_q_faces, p_q_notFaces, p_pos_q_faces,
                              p_pos_q_notFaces, num_regions)
            if prob_faces[i]<lambdav:
                true_positives += 1
            else:
                false_positives +=1

        true_negatives = 0
        false_negatives = 0
        num_notFaces = len(not_faces_test)
        prob_notfaces = np.zeros(num_notFaces)
        for i in xrange(num_notFaces):
            prob_notfaces[i] = dev(not_faces_test[i], p_q_faces, p_q_notFaces, p_pos_q_faces,
                                   p_pos_q_notFaces, num_regions)
            if prob_notfaces[i]>lambdav:
                true_negatives += 1
            else:
                false_negatives +=1
        
        print "+", true_positives, false_positives, true_positives/num_faces, false_positives/num_faces
        print "-", true_negatives, false_negatives, true_negatives/num_faces, false_negatives/num_faces
        print true_positives/num_faces, false_negatives/num_faces
        #print np.mean(prob_faces), np.mean(prob_notfaces)


    if image_test is not None:
        image_normalized = image_test - np.mean(image_test) / np.std(image_test)
        width, height = image_normalized.shape
        print image_normalized

        regions_image = get_regions(width, height, window)
        image_windows = []
        split_image(image_normalized, regions_image, image_windows, False)
        #print
        #print
        for i in xrange(len(image_windows)):
            w, h = image_windows[i].shape
            if  w == window and h==window:
                print dev(image_windows[i], p_q_faces, p_q_notFaces, p_pos_q_faces,
                          p_pos_q_notFaces, num_regions)

                #print dev(subimage, p_q_faces, p_q_notFaces, p_pos_q_faces,
                #p_pos_q_notFaces, num_regions)
        #    print dev(window, p_q_faces, p_q_notFaces, p_pos_q_faces,
        #    p_pos_q_notFaces, num_regions)
       

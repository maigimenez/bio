# -*- coding: utf-8 -*-
import argparse
import numpy as np
#import matplotlib.pyplot as plt
from math import ceil
from scipy.cluster.vq import kmeans, vq, whiten
from numpy import vstack


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

            # Eliminar un píxel para hacer la imagen cuadrada 20x20
            faces = []
            for face in faces_21x21:
                faces.append(face.reshape((21, 21))[:-1, :-1])

            not_faces = []
            for nface in not_faces_21x21:
                not_faces.append(nface.reshape((21, 21))[:-1, :-1])

            #plt.imshow((faces[0].reshape((20,20))))
            #plt.gray()
            #plt.show()
            #print type(faces)
            return np.array(faces), np.array(not_faces)

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()


def train(faces, not_faces):

    num_faces = len(faces)
    #print "Faces:", len(faces)
    #print "Not faces:", len(not_faces)

    # Split into regions
    # TODO: Check if there are no faces
    len_w, len_h = faces[0].shape
    num_regions = 16
    q_levels = 256
    region_dim = int(np.sqrt((len_w * len_h) / num_regions))

    #Get the regions
    regions = []
    for i in range(0, len_h, region_dim):
        for j in range(0, len_w, region_dim):
            regions.append(((i, i + region_dim), (j, j + region_dim)))

    # Split faces in regions
    image_regions = []
    for face in faces:
        for region in regions:
            image_regions.append(
                np.array(face[region[0][0]:region[0][1],
                              region[1][0]:region[1][1]]) .flatten())

    # Split not faces in regions
    for image in not_faces:
        for region in regions:
            image_regions.append(
                np.array(image[region[0][0]:region[0][1],
                               region[1][0]:region[1][1]]).flatten())

    # Quantification
    data = vstack(image_regions)
    whitened = whiten(data)
    centroids, _ = kmeans(whitened, q_levels, thresh=1e-02)
    idx, _ = vq(data, centroids)
    print "id:", idx.shape
    faces_q = idx[:num_faces * num_regions]
    notFaces_q = idx[num_faces * num_regions + 1:]

    # Estimating v_faces
    # Try to use  a dictionary insted of an array
    # because if there are many 0s a lot space unused (spare matrix)
    print faces_q.shape
    v_faces = np.zeros(q_levels)
    #v_faces = dict.fromkeys(set(faces_q), 0)
    for q in faces_q:
        v_faces[q] += 1
    # Normalized
    p_q_faces = v_faces / num_faces

    v_notFaces = np.zeros(q_levels)
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

    m_faces = np.zeros((num_regions, q_levels))
    for face in tagged_faces:
        for region, q in face.iteritems():
            m_faces[region][q] += 1

    print "****", m_faces
    #m_faces[m_faces==0]=1.0
    #print m_faces
    sum_column = np.sum(m_faces, axis=0)
    print sum_column, np.sum(sum_column)
    m_faces[:,sum_column==0]=1
    sum_column = np.sum(m_faces, axis=0)

    print sum_column, np.sum(sum_column)

    #m_faces = np.divide(m_faces[],np.sum(m_faces, axis=0))
    #print m_faces
    #i = 0
    #for column in m_faces.T:
    #    if sum_column[i] != 0 :
    #        print column / sum_column[i]
    #        print
    #    i+=1


if __name__ == "__main__":
    per_train = 0.8
    per_test = 1 - per_train
    faces, not_faces = get_data()

    # Shuffles data, and get train and test sets for faces.
    np.random.shuffle(faces)
    examples, d1, d2 = faces.shape
    sep = ceil(per_train * examples)
    faces_train = faces[0:sep]
    faces_test = faces[sep:]
    #print faces.shape, faces_test.shape, faces_train.shape

    # Shuffles data, and get train and test sets for not faces.
    np.random.shuffle(not_faces)
    examples, d1, d2 = not_faces.shape
    sep = ceil(per_train * examples)
    not_faces_train = not_faces[0:sep]
    not_faces_test = not_faces[sep:]
    #print examples, not_faces_train.shape,  not_faces_test.shape

    #Train
    train(faces_train, not_faces_train)

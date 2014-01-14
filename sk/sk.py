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


def get_regions(width, height, num_regions):
    region_dim = int(np.sqrt((width * height) / num_regions))

    #Get the regions
    regions = []
    for i in range(0, height, region_dim):
        for j in range(0, width, region_dim):
            regions.append(((i, i + region_dim), (j, j + region_dim)))

    return regions


def split_image(image,regions, image_regions):
    for region in regions:
        image_regions.append(
            np.array(image[region[0][0]:region[0][1],
                            region[1][0]:region[1][1]]) .flatten())


def split_images(images,regions, image_regions):
    for image in images:
        split_image(image,regions, image_regions)


def train(faces, not_faces, num_regions, q_levels):

    num_faces = len(faces)
    #print "Faces:", len(faces)
    #print "Not faces:", len(not_faces)

    # Split into regions
    # TODO: Check if there are no faces
    len_w, len_h = faces[0].shape
    regions = get_regions(len_w, len_h, num_regions)

    # Get regions from train images
    image_regions = []
    # Split faces in regions
    split_images(faces,regions,image_regions)
    # Split not faces in regions
    split_images(not_faces,regions,image_regions)

    # Quantification
    data = vstack(image_regions)
    whitened = whiten(data)
    centroids, _ = kmeans(whitened, q_levels, thresh=1e-02)
    idx, _ = vq(data, centroids)
    faces_q = idx[:num_faces * num_regions]
    notFaces_q = idx[num_faces * num_regions + 1:]

    # Estimating v_faces
    # Try to use  a dictionary insted of an array
    # because if there are many 0s a lot space unused (spare matrix)
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

    m_faces = np.ones((num_regions, q_levels))
    for face in tagged_faces:
        for region, q in face.iteritems():
            m_faces[region][q] += 1

    sum_column = np.sum(m_faces, axis=0)

    p_pos_q_faces = m_faces.copy()
    rows, columns = m_faces.shape
    for col in range(columns):
        p_pos_q_faces = m_faces[:,col]/sum_column[col]

    return p_q_faces, p_q_notFaces, p_pos_q_faces, p_pos_q_notFaces


def test(image, p_q_faces, p_q_notFaces, p_pos_q_faces,
        p_pos_q_notFaces, num_regions):
    width, height =image.shape
    regions = get_regions(width, height, num_regions)
    image_regions = []
    split_image(image, regions, image_regions)


if __name__ == "__main__":
    num_regions = 16
    q_levels = 256
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
    (p_q_faces, p_q_notFaces, p_pos_q_faces, p_pos_q_notFaces) = train(faces_train, not_faces_train, 
                                                                       num_regions, q_levels)

    #Test
    for face in faces_test:
        test(face, p_q_faces, p_q_notFaces, p_pos_q_faces,
            p_pos_q_notFaces, num_regions)


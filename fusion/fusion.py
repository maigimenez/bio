import argparse
import numpy as np
from sympy.functions.special.delta_functions import Heaviside
import matplotlib.pyplot as plt


def load_default():
    print "TODO: load default scores"
    return None, None


def get_data():
    """ Get scores data.
    If there are no arguments in command line load default

    """
    parser = argparse.ArgumentParser(description="Solve the ROC curve")
    parser.add_argument("-tr", "--train", type=argparse.FileType('r'),
                        help="Scores for train data", metavar="TR",
                        dest="train_file")
    parser.add_argument("-te", "--test", type=argparse.FileType('r'),
                        help="Scores for test data", metavar="TE",
                        dest="test_file")
    parser.add_argument("-p","--plot", action='store_true',
                        help="Make plot",
                        dest="plot")

    try:
        args = parser.parse_args()
        if args.train_file is None or args.test_file is None:
            load_default()
        else:
            train  = np.loadtxt(args.train_file)
            test = np.loadtxt(args.test_file)
            #return train, test, args.plot
            c_train = train[train[:,2]==1]
            i_train = train[train[:,2]==0]
            c_test = test[train[:,2]==1]
            i_test = test[train[:,2]==0]
            return (c_train,i_train), (c_test,i_test),args.plot

    except SystemExit:
        #TODO: load default scores filenames
        print "Default"
        load_default()

def score_norm():
    # u is the slope and v is the displacement of the sigmoid. z is the score
    sigmoid = lambda u,v,z: 1 / (1 + np.exp(u*(v-z)))
    # Fused scores restictions:
    # sum(w) = 1 for all w
    # w >= 0 for all w
    
    # f(u,v,w)(z) = wT * sigmoid(u,v)(z) in [0,1]  

def aprox_AUR(clients, impostors):
    # Delete clients/impostors tag
    c_scores = clients[:,:-1]
    i_scores = impostors[:,:-1]
    num_scores = c_scores.shape[1]
    #print c_scores.shape,i_scores.shape, num_scores
    res = np.array([c-i for c in c_scores for i in i_scores])


if __name__ == "__main__":
    (c_train,i_train),(c_test,i_test), p= get_data() 
    aprox_AUR(c_train,i_train)
    if p:
        f, (ax1, ax2) = plt.subplots(2,sharex=True, sharey=True)
        #c_train = train[train[:,2]==1]
        #i_train = train[train[:,2]==0]
        ax1.set_ylabel("Score 1")
        ax2.set_ylabel("Score 1")
        ax2.set_xlabel("Score 2")
        ax1.plot(c_train[:,0],c_train[:,1],'o', color='green')
        ax1.plot(i_train[:,0],i_train[:,1],'o', color='red')
        ax1.set_title('Train Scores')
        #c_test = test[test[:,2]==1]
        #i_test = test[test[:,2]==0]
        ax2.plot(c_test[:,0],c_test[:,1],'o', color='green')
        ax2.plot(i_test[:,0],i_test[:,1],'o', color='red')
        ax2.set_title('Test Scores')
        plt.show()

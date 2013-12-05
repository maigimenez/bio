# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import matplotlib.mlab as mlab

class RocData(object):
    def __init__(self, name="", c_score=None,i_score=None):
        self.name = name 
        self.c_score = c_score
        self.i_score = i_score
        # Get all possible threasholds and inserts a 0.
        self.thrs = np.insert(np.unique(
                np.concatenate([self.c_score, self.i_score])),0, 0)
        self.fpr = None
        self.fnr = None
        self.tpr = None
        self.tnr = None
        
    def solve_ratios(self):
        # Get false negative ratio
        self.fnr = np.divide(map(lambda x: np.sum(self.c_score <= x), self.thrs),
                             float(len(self.c_score)))
        # Get true positive ratio
        self.tpr = 1.0 - self.fnr
        # Get false positive ratio
        self.fpr = np.divide(map(lambda x: np.sum(self.i_score > x), self.thrs),
                             float(len(self.i_score)))
        # Get true negative ratio
        self.tnr = 1.0 - self.fpr

    def plot(self):
        plt.plot(self.fpr,self.tpr, linewidth=2.0)
        plt.xlabel("FP")
        plt.ylabel("1-FN")
        plt.show()

    def func(self,x):
        return (x - 3) * (x - 5) * (x - 7) + 85


    def aur(self,plot):
        aur = trapz(self.tpr, np.sort(self.fpr))
        if plot:
            a,b = self.fpr[-1], self.fpr[0]
            fig, ax = plt.subplots()
            plt.plot(self.fpr, self.tpr, 'r',linewidth=2)
            plt.ylim(ymin=self.tpr.min()-0.03, ymax=self.tpr.max()*1.05)
            plt.xlim(xmin=self.fpr.min()-0.03)
            plt.fill_between(self.fpr, self.tpr, facecolor='gray', alpha='0.5')
            aur_string = "aur= "+str(aur)
            plt.text(a+b/2, a+b/2, aur_string,
                     horizontalalignment='center', fontsize=14)
            plt.xlabel("FP")
            plt.ylabel("1-FN")
            plt.show()
        else:
            print(u"El área bajo la curva roc es igual a {0}".format(aur))

    def dprime(self, plot):
        mu_pos = np.mean(self.c_score)
        mu_neg = np.mean(self.i_score)
        var_pos = np.var(self.c_score)
        var_neg = np.var(self.i_score)
        print self.c_score, mu_pos, var_pos
        print self.i_score, mu_neg, var_neg
        if plot:
            x_pos = np.linspace(self.c_score.min()-10, self.c_score.max(), 100)
            x_neg = np.linspace(self.i_score.min(), self.i_score.max(), 100)
            std_pos = np.std(self.c_score)
            std_neg = np.std(self.i_score)
            plt.plot(x_pos,mlab.normpdf(x_pos,mu_pos,std_pos))
            plt.plot(x_neg,mlab.normpdf(x_neg,mu_neg,std_neg))
            plt.show()
        

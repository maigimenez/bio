# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import matplotlib.mlab as mlab
from math import sqrt
import scipy.stats as stats
import matplotlib.patches as patches
import matplotlib.path as path
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

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
        std_pos = np.sqrt(var_pos)
        std_neg = np.sqrt(var_neg)
        dprime = np.divide((mu_pos-mu_neg),(np.sqrt(var_pos+var_neg)))
        print(u"El factor dprime es {0}".format(dprime))
        if plot:
            bins = len(self.c_score)/50
            text_pos = '$Clientes$\n$\mu=%.2f$\n$\sigma=%.2f$'%(mu_pos, std_pos)
            text_neg = '$Impostores$\n$\mu=%.2f$\n$\sigma=%.2f$'%(mu_neg, std_neg)
            plt.hist(self.c_score, bins= bins, histtype='stepfilled', 
                normed=True, color='b', label=text_pos)
            plt.hist(self.i_score, bins=bins, histtype='stepfilled', 
                normed=True, color='r', alpha=0.5, label=text_neg)
            plt.xlabel("Scores")
            plt.ylabel("Frecuencias")
            plt.title("Scores")

            # Gaussian distribution
            #textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(mu_pos, std_pos)
            #x_pos = np.linspace(mu_pos-std_pos,mu_pos+std_pos)
            #x_neg = np.linspace(mu_neg-std_neg,mu_neg+std_neg)
            #plt.plot(x_pos,mlab.normpdf(x_pos,mu_pos,std_pos))
            #plt.plot(x_neg,mlab.normpdf(x_neg,mu_neg,std_neg))
            #plt.xlim(xmin=-left_margin*1.2, xmax=right_margin*1.2)
            plt.legend()
            plt.show()
        

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import timeit


class RocData(object):
    def __init__(self, name="", c_score=None, i_score=None):
        self.name = name
        self.c_score = c_score
        self.i_score = i_score
        # Get all possible threasholds and inserts a 0.
        self.thrs = np.unique(
            np.insert(np.concatenate([self.c_score, self.i_score]), 0, 0))
        self.fpr = None
        self.fnr = None
        self.tpr = None
        self.tnr = None

    def solve_ratios(self):
        """ Given a set of scores from a biometric system solve the ratios """
        # Get false negative ratio
        self.fnr = np.divide(map(lambda x: np.sum(self.c_score <= x),
                                 self.thrs), float(len(self.c_score)))
        # Get true positive ratio
        self.tpr = 1.0 - self.fnr
        # Get false positive ratio
        self.fpr = np.divide(map(lambda x: np.sum(self.i_score > x),
                                 self.thrs), float(len(self.c_score)))
        # Get true negative ratio
        self.tnr = 1.0 - self.fpr

    def plot(self, save_path):
        """ Plot ROC curve """
        plt.plot(self.fpr, self.tpr, '--ro', linewidth=0.75)
        plt.xlabel("FP")
        plt.ylabel("1-FN")
        if (save_path):
            plt.savefig(save_path)
        else:
            plt.show()

    def aur(self, plot, save_path):
        """ Calculate AUR curve using Trapezoidal method """
        start_time = timeit.default_timer()
        aur = np.abs(np.trapz(self.tpr, x=self.fpr))
        #simps(self.tpr, x=self.fpr)
        if plot:
            self.plot_aur(aur, save_path)
        else:
            elapsed = timeit.default_timer() - start_time
            print(u"El área bajo la curva roc es igual a"
                  " {0} (Coste: {1})".format(aur, elapsed))

    def aur_aprox(self, plot, save_path):
        """ Calculate AUR curve using aproximation """
        start_time = timeit.default_timer()
        heaviside = lambda x: 0.5 if x == 0 else 0 if x < 0 else 1
        sum_scores = 0.0
        for c in self.c_score:
            for i in self.i_score:
                subs_scores = c - i
                sum_scores += heaviside(subs_scores)

        aprox_aur = (sum_scores /
                     float(self.c_score.shape[0] * self.i_score.shape[0]))
        if plot:
            self.plot_aur(aprox_aur, save_path)
        else:
            elapsed = timeit.default_timer() - start_time
            print(u"El área bajo la curva roc es igual a"
                  " {0} (Coste: {1})".format(aprox_aur, elapsed))

    def plot_aur(self, aur, save_path):
        """ Plot AUR curve """
        a, b = self.fpr[-1], self.fpr[0]
        fig, ax = plt.subplots()
        plt.plot(self.fpr, self.tpr, 'r', linewidth=2)
        plt.ylim(ymin=self.tpr.min() - 0.03, ymax=self.tpr.max() * 1.05)
        plt.xlim(xmin=self.fpr.min() - 0.03)
        plt.fill_between(self.fpr, self.tpr, facecolor='gray', alpha='0.5')
        aur_string = "aur= " + str(aur)
        plt.text(a + b / 2, a + b / 2, aur_string,
                 horizontalalignment='center', fontsize=14)
        plt.xlabel("FP")
        plt.ylabel("1-FN")
        if (save_path):
            plt.savefig(save_path)
        else:
            plt.show()

    def dprime(self, plot):
        """ Get d' value"""
        mu_pos = np.mean(self.c_score)
        mu_neg = np.mean(self.i_score)
        var_pos = np.var(self.c_score)
        var_neg = np.var(self.i_score)
        std_pos = np.sqrt(var_pos)
        std_neg = np.sqrt(var_neg)
        dprime = np.divide((mu_pos - mu_neg), (np.sqrt(var_pos + var_neg)))
        print(u"El factor dprime es {0}".format(dprime))
        if plot:
            # Get the bins used to split the histogram
            len_c_score = len(self.c_score)
            len_i_score = len(self.i_score)
            if len_c_score > 10:
                bins_pos = len_c_score * 0.02
            else:
                bins_pos = len_c_score / 2
            if len_i_score > 10:
                bins_neg = len_i_score * 0.02
            else:
                bins_neg = len_i_score / 2
            # Set the legend
            text_pos = ('$Clientes$\n$\mu=%.2f$\n$\sigma=%.2f$'
                        % (mu_pos, std_pos))
            text_neg = ('$Impostores$\n$\mu=%.2f$\n$\sigma=%.2f$'
                        % (mu_neg, std_neg))
            # Plot
            plt.hist(self.c_score, bins=bins_pos, histtype='stepfilled',
                     normed=True, color='b', label=text_pos)
            plt.hist(self.i_score, bins=bins_neg, histtype='stepfilled',
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

import numpy as np
from math import floor


i,pos,neg=np.loadtxt("test.txt", unpack=True)
mu_pos = np.mean(pos)
mu_neg = np.mean(neg)
lambda_value = floor(mu_pos+mu_neg/2.0)
print lambda_value

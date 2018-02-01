"""
To create chart for number of candidates

Xia Cui
02/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def collector(domain):
    input_file = open("../work/%s/cpwords_ppmi_overlap.dat-expansion.csv")
    # skip the first two lines
    next(input_file)
    next(input_file)
    y= []
    for line in input_file:
        y.append(p[4])
    return y


def drawer(domains):
    for domain in domains:
        plt.plot(collector(domain))
    plt.show()
    pass

domains = ['TR','CR','SUBJ','MR']
drawer(domains)
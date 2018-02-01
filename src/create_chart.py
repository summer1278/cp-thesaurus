"""
To create chart for number of candidates

Xia Cui
02/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 16}
matplotlib.rc('font', **font)

def collector(domain):
    input_file = open("../work/%s/cpwords_ppmi_overlap.dat-expansion.csv"%domain,"r")
    # skip the first two lines
    next(input_file)
    next(input_file)
    y= []
    for line in input_file:
        p = line.strip().split(',')
        # print p
        if len(p) < 4:
            continue
        y.append(float(p[4]))
    return y


def drawer(domains,ks,title):
    fig, ax = plt.subplots(figsize=(8,6))
    index = np.arange(len(ks))
    markers = ['o','s']*(len(domains)/2)
    # linestyles= ['--',':']*(len(domains)/2)
    m_i = 0
    for domain in domains:
        y = collector(domain)
        p = plt.plot(index,y,linestyle='--',label = domain,marker = markers[m_i],markersize=10,fillstyle='none')
        color = p[0].get_color()
        for i in range(0,len(index)):
            if y[i] == max(y):
                plt.plot(index[i],y[i],marker = markers[m_i],markersize=10,color=color)
            else:
                plt.plot(index[i],y[i],marker = markers[m_i],markersize=10,color=color,fillstyle='none')
        m_i +=1
    plt.xticks(index,ks)
    plt.title(title)
    plt.xlabel("$k$ (#candidates)")
    plt.ylabel("Accuracy")
    #right box
    # box = ax.get_position()
    # ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])
    # ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
    #           fancybox=True, shadow=True, ncol=1)
    plt.legend(fontsize=14)
    plt.grid()
    if title == 'DA':
        plt.ylim([0.65,0.85])
    plt.savefig('../%s.png'%title)
    plt.show()
    pass

ks = [10,100,500,1000]
title = 'Non-DA'
domains = ['TR','CR','SUBJ','MR']
# title = 'DA'
# domains = ['B-D','B-E','B-K','D-B','D-E','D-K','E-B','E-D','E-K','K-B','K-D','K-E']
drawer(domains,ks,title)
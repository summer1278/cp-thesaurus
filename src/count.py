"""
Count CP

Xia Cui
02/2018
"""
import numpy,math

def count_CP(CP_fname):
    all_peris = []
    with open(CP_fname) as CP_file:
        core = 0
        for line in CP_file:
            p = line.strip().split()
            core +=1
            peris = sum([1 for x in p[2:]])
            print "current = ",core,peris
            all_peris.append(peris)
    print CP_fname
    print "all = ", core, all_peris
    average = float(sum(all_peris))/float(core)
    print "average = ", average
    pass

if __name__ == '__main__':
    # dataset = "TR"
    # CP_fname = "../data/%s/cpwords_ppmi_overlap1.dat" %dataset
    # CP_fname = "../data/cp-overlap.ppmi"
    CP_fname = "../data/cp-nonoverlap.ppmi"
    count_CP(CP_fname)
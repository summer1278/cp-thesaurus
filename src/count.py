"""
Count CP

Xia Cui
02/2018
"""
import numpy,math,sys

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
    # print "all = ", core, all_peris
    average = float(sum(all_peris))/float(core)
    print "average = ", average
    pass

def count_by_option(option,dataset="TR"):
    # 
    CP_fnames = {1:"../data/%s/cpwords_ppmi_overlap1.dat" %dataset,
        2:"../data/cp-overlap.ppmi",
        3:"../data/cp-nonoverlap.ppmi"}
    
    count_CP(CP_fnames[option])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        option = int(sys.argv[1])
        count_by_option(option)
    elif len(sys.argv) > 2:
        option = int(sys.argv[1])
        dataset = sys.argv[2]
        count_by_option(option,dataset)
    else:
        print "usage: <option: 1:overlap+coress, 2:overlap, 3:nonoverlap> <dataset: for 1 only>"
    pass
    
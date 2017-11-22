"""
Script for running km and cp in together

Xia Cui
Novermber 2017
"""

import preprocess
import expand
import os.path
from subprocess import call
import sys

def compute_ppmi_coreness(domain):
    preprocess.compute_ppmi_coreness(domain,1000)
    pass

def runner_nonoverlap(domain):
    # run km (nonoverlap)
    call('time ../../kmcpp/./km ../data/ppmi_links.dat '+\
        '../data/%s/ppmi_coreness.dat ../data/%s/result_ppmi_nonoverlap.dat 100 1 10'%(domain,domain))
    preprocess.convert_cp_nonoverlap(domain,'ppmi')
    expand.main(domain)
    pass

def runner_overlap(domain):
    # run km_overlap
    call('time ../../kmcpp/./km_overlap ../data/ppmi_links.dat '+\
        '../data/%s/ppmi_coreness.dat ../data/%s/result_ppmi_overlap.dat 10'%(domain,domain))
    preprocess.convert_cp_overlap(domain,'ppmi')
    expand.main(domain)
    pass

def check_file_exists(fname):
    return os.path.isfile(fname)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        option = sys.argv[1]
        domain = sys.argv[2]
        # if ppmi corenesss is not generated, create it before we run 
        if not check_file_exists("../data/%s/ppmi_coreness.dat"%domain):
            # compute_ppmi_coreness(domain)
            print "ppmi computed"
        if option == 'nonoverlap':
            runner_nonoverlap(domain)
        elif option == "overlap":
            runner_overlap(domain)
        else:
            print "usage:<option:overlap or nonoverlap> <dataset or domain>"
    else:
        print "usage:<option:overlap or nonoverlap> <dataset or domain>"
    pass

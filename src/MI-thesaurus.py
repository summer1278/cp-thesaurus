"""
Create a thesaurus using PPMI values.

Danushka Bollegala
18 May 2017
"""

from expand import CP_EXPANDER
import numpy as np


def comp_func(x, y):
    """
    (peri, rank, score)
    """
    if x[1] == y[1]:
        return int(x[2] - y[2])
    else:
        return x[1] - y[1]


def inverted_thesaurus_generator(cp_fname, inv_fname):
    """
    Read the CP thesaurus and create an inverted index from peri to core.
    For each peri rank the cores in the descending order of their association scores.
    This rank will be the score indicating how close a core is to a peri.
    """
    CP = CP_EXPANDER()
    CP.load_CP_Dictionary(cp_fname, 10000)
    R = {}
    id2word = {}
    for word in CP.wid:
        id2word[CP.wid[word]] = word

    for core in CP.D:
        for (peri, score) in CP.D[core]["peris"]:
            R.setdefault(peri, []).append((core, score))

    h = {}
    for peri in R:
        R[peri].sort(lambda x, y: -1 if x[1] > y[1] else 1)
        for i in range(len(R[peri])):
            (core, score) = R[peri][i]
            h.setdefault(core, []).append((peri, i, score))

    with open(inv_fname, 'w') as inv_file:
        for core in h:
            h[core].sort(comp_func)
            core_word = id2word[core]
            L = []
            for (peri, rank, score) in h[core]:
                L.append("%s,%f" % (id2word[peri], score))
            inv_file.write("%s %f %s\n" % (core_word, CP.D[core]["coreness"], " ".join(L)))
    pass


def PMI_thesaurus_generator(bigram_fname, thesaurus_fname):
    h = {}
    count = 0
    with open(bigram_fname) as F:
        for line in F:
            p = line.strip().split()
            count += 1
            if count % 1e+6 == 0:
                print count
            first = p[0]
            second = p[1]
            val = float(p[2])
            h.setdefault(first, []).append((second, val))
            h.setdefault(second, []).append((first, val))
    print "Loaded bigrams successfully..."        

    with open(thesaurus_fname, 'w') as G:
        for word in h:
            h[word].sort(lambda x,y: -1 if x[1] > y[1] else 1)
            L = []
            for (second, val) in h[word][:10000]:
                L.append("%s,%f" % (second, val))
            G.write("%s 0 %s\n" % (word, " ".join(L)))
    pass


def PMI_computation():
    bi_total = 1405946955.73
    uni_total = 0
    h = {}
    with open("../data/unigrams.sorted") as F:
        for line in F:
            p = line.strip().split('\t')
            if len(p) != 2:
                continue
            h[p[0]] = float(p[1])
            uni_total += float(p[1])

    G = open("../data/bigrams.sorted")
    P = open("../data/ppmi.values", 'w')
    for line in G:
        p = line.strip().split()
        first = p[0]
        second = p[1]
        val = float(p[2])
        ppmi = np.log((val / bi_total) / ((h[first] / uni_total) * (h[second] / uni_total)))
        if ppmi > 0:
            P.write("%s %s %f\n" % (first, second, ppmi))
    P.close()
    G.close()
    pass


def assign_ppmi_scores_to_CP(cp_raw_fname, ppmi_fname, cp_thesaurus_fname):
    """
    Assign ppmi scores to the CP thesaurus and save the result.
    """
    h = {}
    with open(ppmi_fname) as ppmi_file:
        for line in ppmi_file:
            p = line.strip().split()
            h[(p[0], p[1])] = float(p[2])

    with open(cp_raw_fname) as F:
        with open(cp_thesaurus_fname, 'w') as G:
            for line in F:
                p = line.strip().split(',')
                try:
                    source = p[0]
                    coreness = float(p[1])
                    peris = p[2:]
                    G.write("%s %f " % (source, coreness))
                    for peri in peris:
                        if (source, peri) in h:
                            val = h[(source, peri)]
                        else:
                            val = h[(peri, source)]
                        G.write("%s,%f " % (peri, val))
                    G.write("\n")
                except ValueError as e:
                    print e
                    print p
    pass


if __name__ == '__main__':
    #PMI_computation()
    # PMI_thesaurus_generator("../data/ppmi.values", "../data/PMI-thesaurus")
    #inverted_thesaurus_generator("../data/cp-overalp.ppmi", "../data/inv-cp-overlap.ppmi")
    #assign_ppmi_scores_to_CP("../data/cpwords/cpwords_overlap.dat", "../data/ppmi.values", "../data/cp-overlap.ppmi")

    dataset = 'TR'
    assign_ppmi_scores_to_CP("../data/%s/cpwords_overlap.dat", "../data/ppmi.values", "../data/%s/cp-overlap.ppmi"%dataset)
    # assign_ppmi_scores_to_CP("../data/%s/cpwords_nonoverlap.dat", "../data/ppmi.values", "../data/%s/cp-nonoverlap.ppmi"%dataset)




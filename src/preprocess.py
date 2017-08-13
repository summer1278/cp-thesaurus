"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""
def compute_links():
    F = open(".../work/bigram.sorted","r")
    output = open(".../work/bigram_links.dat","w")
    output.write("i \t j \t wij\n")
    for line in F:
        p = line.strip().split(' ')
        i = get_id(p[0])
        j = get_id(p[1])
        w = float(p[2])
        output.write("%s \t %s \t %f\n"%(i, j, w))
    pass

# get node id for a word in D
# generate from unigrams or bigrams?
def get_id(word):
    pass
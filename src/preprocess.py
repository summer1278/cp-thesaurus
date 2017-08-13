"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""
def compute_links():
    F = open("../work/bigram.sorted","r")
    output = open("../work/bigram_links.dat","w")
    output.write("i \t j \t wij\n")
    for line in F:
        p = line.strip().split(' ')
        i = get_id(p[0])
        j = get_id(p[1])
        w = float(p[2])
        output.write("%s \t %s \t %f\n"%(i, j, w))
    pass

# get node id for a word in D
def get_id(word):
    pass

# generate from unigrams or bigrams?
# firstly, try using bigrams
def word_ids_generator():
    words = []
    F = open("../work/bigram.sorted","r")
    for line in F:
        p = line.strip().split(' ')
        word_i = p[0]
        word_j = p[1]
        words.append(word_i)
        words.append(word_j)
    print len(set(words)),len(words)
    pass

if __name__ == '__main__':
    word_ids_generator()
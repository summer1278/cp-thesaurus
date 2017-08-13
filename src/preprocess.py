"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""
def compute_links():
    F = open("../work/bigram.sorted","r")
    G = open("../work/bigram_links.dat","w")
    G.write("i \t j \t wij\n")
    for line in F:
        p = line.strip().split(' ')
        i = get_id(p[0])
        j = get_id(p[1])
        w = float(p[2])
        G.write("%s \t %s \t %f\n"%(i, j, w))
    F.close()
    G.close()
    pass

# get node id for a word in D
# from word_ids pre-generated
def get_id(word):
    F=open("../work/word_ids","r")
    words = [line[:-1] for line in F]
    word_id = words.index(word)
    return word_id

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
    F.close()
    print "words loaded = ",len(set(words))
    G = open("../work/word_ids","w")
    # i = 0
    for word in set(words):
        # G.write('%d %s\n'%(i,word))
        G.write('%s\n'%(word))
        # i+=1
    G.close()
    pass

def compute_coreness():
    
    pass

if __name__ == '__main__':
    # word_ids_generator()
    compute_links()
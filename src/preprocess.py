"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""
def compute_links():
    print "generate links..."
    F = open("../data/bigrams.sorted","r")
    G = open("../data/bigram_links.dat","w")
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
    F=open("../data/word_ids","r")
    words = [line[:-1] for line in F]
    F.close()
    word_id = words.index(word) if word in words else -1
    return word_id

# generate from unigrams or bigrams?
# firstly, try using bigrams
def word_ids_generator():
    words = []
    F = open("../data/bigrams.sorted","r")
    for line in F:
        p = line.strip().split(' ')
        word_i = p[0]
        word_j = p[1]
        words.append(word_i)
        words.append(word_j)
    F.close()
    print "words loaded = ",len(set(words))
    G = open("../data/word_ids","w")
    # i = 0
    for word in set(words):
        # G.write('%d %s\n'%(i,word))
        G.write('%s\n'%(word))
        # i+=1
    G.close()
    pass

# coreness = pivothood = freq in domains = min(h(w,S), h(w,T))
# if single domain?
def compute_coreness(domain):
    fname = "../data/%s/train"%domain
    write_original_sentences(fname)
    freq_dict = {}
    count_freq(fname,freq_dict)
    F=open("../data/word_ids","r")
    words = [line[:-1] for line in F]
    F.close()
    features = set(freq_dict.keys())|set(words)
    G = open("../data/%s/freq_coreness.dat"%domain,"w")
    G.write("id \t coreness\n")
    for feat in features:
        if get_id(feat) != -1:
            feat_id = get_id(feat)
            G.write("%d \t %d\n"%(feat_id,freq_dict.get(feat,0)))
    G.close()
    pass

# if domain adaptation?
def compute_coreness_DA(source,target):
    source_fname = "../data/%s/train"%source
    target_fname = "../data/%s/train"%target
    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    src_freq = {}
    tgt_freq = {}
    count_freq(source_fname,src_freq)
    count_freq(target_fname,tgt_freq)
    # s = {}
    F=open("../data/word_ids","r")
    words = [line[:-1] for line in F]
    F.close()
    features = (set(src_freq.keys()) | set(tgt_freq.keys()))| set(words)
    G = open("../data/%s-%s/freq_coreness.dat"%(source,target),"w")
    G.write("id \t coreness\n")
    for feat in features:
        # s[feat] = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
        if get_id(feat) != -1:
            feat_id = get_id(feat)
            G.write("%d \t %d\n"%(feat_id,min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))))
    G.close()
    pass

# count frequency and return a dict h
def count_freq(fname, h):
    for line in open("%s-sentences" % (fname)):
        for feat in line.strip().split(','):
            h[feat] = h.get(feat, 0) + 1
    pass

def write_original_sentences(fname):
    lines = [line for line in open(fname)] 
    res_file = open("%s-sentences" % (fname), 'w')
    for line in open(fname):
        res_file.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    res_file.close()  
    print "original sentences in %s have been written to the disk"%fname 
    pass

if __name__ == '__main__':
    # word_ids_generator()
    compute_links()
    # domian = "TR"
    # compute_coreness(domain)
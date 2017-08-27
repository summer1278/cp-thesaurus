"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""
def compute_links():
    wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            wids[line.strip()] = wid_count
            wid_count += 1
            
    print "generate links..."
    F = open("../data/bigrams.sorted","r")
    G = open("../data/bigram_links.dat","w")
    G.write("i \t j \t wij\n")
    for line in F:
        p = line.strip().split(' ')
        i, j, w = wids[p[0]], wids[p[1]], float(p[2])
        G.write("%s \t %s \t %f\n" % (i, j, w))
    F.close()
    G.close()
    pass

# get node id for a word in D
# from word_ids pre-generated
# def get_id(word):
#     F=open("../data/word_ids","r")
#     words = [line[:-1] for line in F]
#     F.close()
#     word_id = words.index(word) if word in words else -1
#     return word_id

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
# if single domain? train = source, test = target
def compute_coreness(domain):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    src_freq = {}
    tgt_freq = {}
    count_freq(source_fname,src_freq)
    count_freq(target_fname,tgt_freq)

    # read word ids and process
    G = open("../data/%s/freq_coreness.dat"%domain,"w")
    G.write("id \t coreness\n")
    # wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            feat = line.strip()
            # wids[feat] = wid_count        
            coreness = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
            G.write("%d \t %d\n"%(wid_count,coreness))
            wid_count += 1
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
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            feat = line.strip()
            # wids[feat] = wid_count        
            coreness = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
            G.write("%d \t %d\n"%(wid_count,coreness))
            wid_count += 1
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

# convert cp-nonoverlap results from kmcpp to
# core,coreness,peri1,peri2... 
# replace word_ids with words
def convert_cp_nonoverlap(domain):
    wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            wids[line.strip()] = wid_count
            wid_count += 1
    
    cores = {}
    # F = open("../data/%s/result_nonoverlap.dat"%domain,"r")
    # next(F) # skip the first line of the read file
    # for line in F:
    #     p = line.strip().split()
    #     if int(p[3])==1:
    #         cores[int(p[0])]={"coreness":float(p[2]),"peris":[]}
    # F.close()
    # # add peris
    # F = open("../data/%s/result_nonoverlap.dat"%domain,"r")
    # next(F) # skip the first line of the read file
    # for line in F:
    #     p = line.strip().split()
    #     print p
    #     if int(p[3])==0:
    #         cores[int(p[1])]["peris"].append(int(p[0]))

    # print cores

    cores = {}
    cp_pairs = set()
    F = open("../data/%s/result_nonoverlap.dat"%domain,"r")
    next(F) # skip the first line of the read file
    for line in F:
        p = line.strip().split()
        cp_pairs.append(int(p[1]))
    print cp_pairs,len(cp_pairs)

    # write each core with coreness and its peris as a line
    # G = open("../data/%s/result_cp_nonoverlap.dat"%domain ,"r")
    # for core in cores.keys():
    #     print ("%s,%f,"%(wids.keys()[wids.values().index(core)],cores.get(core,0)))
    #     for line in F:
    #         p = line.strip().split()

    pass

# convert cp-overlap results from kmcpp to
# core,coreness,peri1,peri2...
# replace word_ids with words


if __name__ == '__main__':
    # word_ids_generator()
    # compute_links()
    domain = "TR"
    # compute_coreness(domain)
    convert_cp_nonoverlap(domain)

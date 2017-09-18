"""
Preprocess files for CP thesaurus

Xia Cui
August 2017
"""

import numpy,math

def compute_links():
    wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            wids[line.strip()] = wid_count
            wid_count += 1
            
    print "generate links..."
    F = open("../data/ppmi.values","r")
    G = open("../data/ppmi_links.dat","w")
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
    F = open("../data/ppmi.values","r")
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
def compute_freq_coreness(domain):
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
            G.write("%d \t %f\n"%(wid_count,coreness))
            wid_count += 1
    G.close()
    pass

def compute_ppmi_coreness(domain,k):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    # source labelled
    # fname = source_fname
    # F=open(fname+'-pos','w')
    # G=open(fname+'-neg','w')
    # for line in open(fname):
    #     if line.strip().split(' ')[0]=='+1':
    #         F.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    #     elif line.strip().split(' ')[0]=='-1':
    #         G.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    # F.close()
    # G.close()
    # count_reviews(source_fname,'pos')
    # count_reviews(source_fname,'neg')
    src_reviews = float(count_reviews(source_fname,'all'))
    tgt_reviews = float(count_reviews(target_fname,'all'))
    total_reviews = float(src_reviews+tgt_reviews)
    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    features = set(features_list(source_fname+'-sentences')).union(set(features_list(target_fname+'-sentences')))
    # print features
    x_src = reviews_contain_x(features_list(source_fname+'-sentences'),source_fname+'-sentences')
    x_tgt = reviews_contain_x(features_list(target_fname+'-sentences'),target_fname+'-sentences')
    x_total = combine_dicts(x_src,x_tgt)
    

    ppmi_dict={}
    for x in features:
        if x_total.get(x,0) > 0 and x_src.get(x,0) > 0 and x_tgt.get(x,0) > 0:
            # print x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews
            src_ppmi = ppmi(x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews) 
            tgt_ppmi = ppmi(x_total.get(x,0), x_tgt.get(x,0), tgt_reviews, total_reviews)
            ppmi_dict[x] = (src_ppmi-tgt_ppmi)**2
    L = ppmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    top_feats = [x for (x,val) in L[:k]]
    print top_feats, len(top_feats)

    # read word ids and process
    G = open("../data/%s/ppmi_coreness.dat"%domain,"w")
    G.write("id \t coreness\n")
    # wids = {}
    wid_count = 0
    nonzeros = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            feat = line.strip()
            # wids[feat] = wid_count        
            coreness = ppmi_dict.get(feat,0) if feat in top_feats else 0
            if coreness >0:
                print wid_count,coreness
                nonzeros += 1
            G.write("%d \t %f\n"%(wid_count,coreness))
            wid_count += 1
    print nonzeros
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

# write separately original sentences to positive and negative
def count_reviews(fname,opt):
    count = 0
    if opt == "pos":
        count = len([line for line in open(fname) if line.strip().split(' ')[0]=='+1'])

    elif opt == "neg":
        count = len([line for line in open(fname) if line.strip().split(' ')[0]=='-1'])
    else:
        count = len([line for line in open(fname)])
    return count
    pass

def features_list(fname):
    return list(set([word for line in open(fname) for word in line.strip().split(',')]))

def reviews_contain_x(features, fname):
    # for x in features:
    #     for line in open(fname):
    #         if x in line.strip().split():
    #             h[x] = h.get(x, 0) + 1
    features = list(features)
    feautres_vector = numpy.zeros(len(features), dtype=float)
    for line in open(fname):
        # print line
        for x in set(line.strip().split(',')):
            i = features.index(x)
            feautres_vector[i] += 1
    return dict(zip(features,feautres_vector))

def ppmi(joint_x, x_scale, y, N):
    prob_y = float(y / N)
    prob_x = float(joint_x / N)
    prob_x_scale = float(x_scale / N)
    val = float(prob_x_scale / (prob_x * prob_y))
    return math.log(val) if math.log(val) > 0 else 0


# method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])

# convert cp-nonoverlap results from kmcpp to
# core,coreness,peri1,peri2... 
# replace word_ids with words
def convert_cp_nonoverlap(domain,method):
    wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            wids[line.strip()] = wid_count
            wid_count += 1

    cores = {}
    cp_pairs = []
    F = open("../data/%s/result_%s_nonoverlap.dat"%(domain,method),"r")
    # F = open("../../kmcpp/result.dat","r")
    next(F) # skip the first line of the read file
    for line in F:
        p = line.strip().split()
        cp_pairs.append(int(p[1]))
        if int(p[3])==1:
            cores[int(p[0])]={"coreness":float(p[2]),"cp_pair":int(p[1]),"peris":[]}
    F.close()

    core_keys=[]
    for cp_pair in set(cp_pairs):
        h = [value['coreness'] for value in cores.values() if value['cp_pair']==cp_pair]
        if h:
            core_key = cores.keys()[[idx for idx,value in enumerate(cores.values()) if value['cp_pair']==cp_pair and value['coreness']==max(h)][0]]
            core_keys.append(core_key)

    print len(core_keys),len(set(cp_pairs))

    # only use max(coreness) as core
    new_cores = {k: cores[k] for k in core_keys}
    # print new_cores

    F = open("../data/%s/result_%s_nonoverlap.dat"%(domain,method),"r")
    # F = open("../../kmcpp/result.dat","r")
    next(F)
    for line in F:
        p =line.strip().split()
        if int(p[0]) not in core_keys:
            h = [idx for idx,value in enumerate(new_cores.values()) if value['cp_pair']==int(p[1])]
            # print h
            if h:
                temp_key = new_cores.keys()[h[0]]
                new_cores[temp_key]['peris'].append(int(p[0]))
    F.close()

    coreness_list = get_coreness_list(domain,wids)

    # write each core with coreness and its peris as a line
    G = open("../data/%s/cpwords_%s_nonoverlap.dat"%(domain,method) ,"w")
    for core in new_cores:
        G.write("%s,%f,"%(wids.keys()[wids.values().index(core)],new_cores[core]['coreness']))
        # print ("%s,%f,"%(wids.keys()[wids.values().index(core)],new_cores[core]['coreness']))
        temp_peris = [wids.keys()[wids.values().index(peri)] for peri in new_cores[core]['peris']]
        peris = sort_peris(temp_peris,coreness_list)
        G.write('%s\n'%','.join(peris))
        # print ('%s\n'%','.join(peris))

    G.close()
    pass

# convert cp-overlap results from kmcpp to
# core,coreness,peri1,peri2...
# replace word_ids with words
def convert_cp_overlap(domain,method):
    wids = {}
    wid_count = 0
    with open("../data/word_ids") as wid_file:
        for line in wid_file:
            wids[line.strip()] = wid_count
            wid_count += 1

    new_cores = {}
    # cp_pairs = [] # in case there are multiple cores in the cp_pair
    F = open("../data/%s/result_%s_overlap.dat"%(domain,method),"r")
    # F = open("../../kmcpp/result_overlap.dat","r")
    next(F)
    for line in F:
        p = line.strip().split()
        new_cores[int(p[0])]={"coreness":float(p[2]),"cp_pair":int(p[1]),"peris":map(int,p[3:])}
    F.close()
    # print new_cores   

    coreness_list = get_coreness_list(domain,wids)
    # print coreness_list

    G = open("../data/%s/cpwords_%s_overlap.dat"%(domain,method) ,"w")
    for core in new_cores:
        G.write("%s,%f,"%(wids.keys()[wids.values().index(core)],new_cores[core]['coreness']))
        # print ("%s,%f,"%(wids.keys()[wids.values().index(core)],new_cores[core]['coreness']))
        temp_peris = [wids.keys()[wids.values().index(peri)] for peri in new_cores[core]['peris']]
        peris = sort_peris(temp_peris,coreness_list)
        G.write('%s\n'%','.join(peris))
        # print ('%s\n'%','.join(peris))
    G.close() 
    pass

# sort peris by coreness in decsending order
# also assign the coreness at the same time
def sort_peris(peris_list,coreness_list):
    new_peris = []
    for (word,coreness) in coreness_list:
        if word in peris_list:
            new_peris.append(word+','+str(coreness))
    
    return new_peris

def get_coreness_list(domain,wids):
    coreness_list = []
    F = open("../data/%s/ppmi_coreness.dat"%domain,"r")
    next(F)
    for line in F:
        p = line.strip().split()
        coreness_list.append((wids.keys()[wids.values().index(int(p[0]))],float(p[1])))
    F.close()

    coreness_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    return coreness_list


if __name__ == '__main__':
    # word_ids_generator()
    # compute_links()
    domain = "TR"
    method = "ppmi"
    # compute_ppmi_coreness(domain,1000)
    # print get_coreness_list(domain)[:10]
    # compute_freq_coreness(domain)
    # convert_cp_nonoverlap(domain)
    convert_cp_overlap(domain,method)


"""
Test Methods for CP thesaurus
modified from expand.py written by Danushka Bollegala

Xia Cui
July 2017
"""
import sys
from sklearn import linear_model
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import expand 


def get_indices(predict, target, value):
    return find_indices(np.sign(np.multiply(predict,target)),value)

def find_indices(my_list, value):
    indices = [i for i, x in enumerate(my_list) if x == value]
    return indices

# print term instances
def append_peri_value(CP_file,sentences,output_indices,k,res_file):
    feats = []
    for idx,line in enumerate(sentences):
        if idx in output_indices:
            feats+=[word.replace(':1','') for word in line.strip().split(' ')[1:]]
    feats=set(feats)

    for feat in feats:
        core_id = CP_file.wid[feat]
        if core_id in CP_file.D:
            res_file.write('%s %f '%(feat,CP_file.D[core_id]["coreness"]))
            for (peri_id, peri_val) in CP_file.D[core_id]["peris"][:k]:
                res_file.write("%s,%f "%(CP_file.wid.keys()[CP_file.wid.values().index(peri_id)],peri_val))
            res_file.write("\n")
    res_file.close()
    pass

def train_with_CV(X_train, y_train, X_test, y_test,value):
    # find the best classifier
    print "cross validation.."
    theta_vals = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]
    cv_res = []
    for theta  in theta_vals:
        clf = linear_model.LogisticRegression(penalty='l2', C=theta, solver='sag')
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        #print scores
        cv_res.append(np.mean(scores))

    # print cv_res
    best_theta = theta_vals[np.argmax(cv_res)]
    print best_theta
    # best_theta = 1.0

    clf = linear_model.LogisticRegression(penalty='l2', C=best_theta)
    clf.fit(X_train, y_train)
    print "classifier learnt."
    # print np.sign(np.multiply(clf.predict(X_test),y_test))
    indices = get_indices(clf.predict(X_test),y_test,value)
    test_acc = 0.5 + 0.5 * np.mean(np.sign(clf.predict(X_test) * y_test))
    return indices,test_acc

def train_without_CV(X_train, y_train, X_test, y_test,value,best_theta):
    # find the best classifier
    # print "cross validation.."
    # theta_vals = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]
    # cv_res = []
    # for theta  in theta_vals:
    #     clf = linear_model.LogisticRegression(penalty='l2', C=theta, solver='sag')
    #     scores = cross_val_score(clf, X_train, y_train, cv=5)
    #     #print scores
    #     cv_res.append(np.mean(scores))

    # # print cv_res
    # best_theta = theta_vals[np.argmax(cv_res)]
    print best_theta
    # best_theta = 1.0

    clf = linear_model.LogisticRegression(penalty='l2', C=best_theta)
    clf.fit(X_train, y_train)
    print "classifier learnt."
    # print np.sign(np.multiply(clf.predict(X_test),y_test))
    indices = get_indices(clf.predict(X_test),y_test,value)
    test_acc = 0.5 + 0.5 * np.mean(np.sign(clf.predict(X_test) * y_test))
    return indices,test_acc


# short text classification
def evaluate(CP,bench,k,res_file,fname):
    train_fname = "%s/train" % bench
    test_fname = "%s/test" % bench
    train_feats = CP.get_feature_set(train_fname)
    print "k = %d" % k
    print "Total no. of train features =", len(train_feats)
    test_feats = CP.get_feature_set(test_fname)
    print "Total no. of test features =", len(test_feats)
    for feat in train_feats:
        CP.wid.setdefault(feat, len(CP.wid))
    for feat in test_feats:
        CP.wid.setdefault(feat, len(CP.wid))
    print "Total no. of all features =", len(CP.wid)


    # no expansion
    train_data = np.array([CP.get_feat_vect(line) for line in open(train_fname)])     
    test_data = np.array([CP.get_feat_vect(line) for line in open(test_fname)])     

    X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)
   
    # best_theta, train_acc, test_acc = CP.train_with_CV(X_train, y_train, X_test, y_test)
    correct_indices,test_acc = train_with_CV(X_train, y_train, X_test, y_test,1)
    # correct_indices,test_acc = train_without_CV(X_train, y_train, X_test, y_test,1,1.0)
    print "\n ---- NO Expansion ----"
    print "Test corrects =", len(correct_indices)

    # expansion for proposed methods
    train_data = np.array([CP.expand_weighted(line, k) for line in open(train_fname)])     
    test_data = np.array([CP.expand_weighted(line, k) for line in open(test_fname)])    

    X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)

    wrong_indices,test_acc = train_with_CV(X_train, y_train, X_test, y_test,-1)
    # wrong_indices,test_acc = train_without_CV(X_train, y_train, X_test, y_test,-1,1.0)
    print "\n ---- With Expansion ----"
    print "Test incorrects =", len(wrong_indices)

    output_indices=set(correct_indices)&set(wrong_indices)
    print "\nintersection of both = ", len(output_indices)
    test_data = [line for line in open(test_fname)] 
    # test_data = np.array([CP.expand_weighted(line, k) for line in open(test_fname)])

    append_peri_value(CP,test_data,output_indices,k,res_file)
    write_original_sentences(test_data,output_indices,fname)
    # for idx,line in enumerate(test_data):
    #     if idx in output_indices:
            # res_file.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
            # res_file.write("%s %s\n" % (line[1:], line[0]))
    return test_acc



def evaluate_projection(CP,bench,k,res_file,fname):
    train_fname = "%s/train" % bench
    test_fname = "%s/test" % bench
    train_feats = CP.get_feature_set(train_fname)
    print "k = %d" % k
    print "Total no. of train features =", len(train_feats)
    test_feats = CP.get_feature_set(test_fname)
    print "Total no. of test features =", len(test_feats)
    for feat in train_feats:
        CP.wid.setdefault(feat, len(CP.wid))
    for feat in test_feats:
        CP.wid.setdefault(feat, len(CP.wid))
    print "Total no. of all features =", len(CP.wid)


    # no expansion
    train_data = np.array([CP.get_feat_vect(line) for line in open(train_fname)])     
    test_data = np.array([CP.get_feat_vect(line) for line in open(test_fname)])     

    X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)
   
    # best_theta, train_acc, test_acc = CP.train_with_CV(X_train, y_train, X_test, y_test)
    correct_indices,test_acc = train_with_CV(X_train, y_train, X_test, y_test,1)

    print "\n ---- NO Expansion ----"
    print "Test corrects =", len(correct_indices)
    # expansion for projected method
    # expanded by cores
    train_data = np.array([CP.get_feat_vect(line) for line in open(train_fname)])     
    test_data = np.array([CP.get_feat_vect(line) for line in open(test_fname)])     

    X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)

    # Build a matrix between cores and peris.
    D = len(CP.D)
    CP_mat = np.zeros((D, len(CP.wid)), dtype=np.float)
    core_list = list(CP.D.keys())
    core_list.sort()

    print "Building CP_matrix...",
    for (i, core_id) in enumerate(core_list):
        peris = CP.D[core_id]["peris"]
        coreness = CP.D[core_id]["coreness"]
        peri_val_total = sum([x[1] for x in peris])
        for peri in peris:
            CP_mat[i,peri[0]] = peri[1] / peri_val_total
    print "Done."

    # Expand by projecting onto cores.        
    print "Expanding train data...",
    csr_CP_mat = csr_matrix(CP_mat.T)
    train_proj = csr_matrix(X_train).dot(csr_CP_mat).todense()
    SP = StandardScaler()
    train_proj = SP.fit_transform(train_proj)
    X_train = np.concatenate((X_train, train_proj), axis=1)
    #X_train = np.concatenate((X_train, X_train.dot(CP_mat.T)), axis=1)
    print "Done."

    print "Expanding test data...",
    test_proj = csr_matrix(X_test).dot(csr_CP_mat).todense()
    test_proj = SP.transform(test_proj)
    X_test = np.concatenate((X_test, test_proj), axis=1)
    #X_test = np.concatenate((X_test, X_test.dot(CP_mat.T)), axis=1)
    print "Done."
    
    wrong_indices,test_acc = train_with_CV(X_train, y_train, X_test, y_test,-1)

    print "\n ---- With Expansion ----"
    print "Test incorrects =", len(wrong_indices)

    output_indices=set(correct_indices)&set(wrong_indices)
    print "\nintersection of both = ", len(output_indices)
    test_data = [line for line in open(test_fname)] 

    append_peri_value(CP,test_data,output_indices,k,res_file)
    write_original_sentences(test_data,output_indices,fname)
    # for idx,line in enumerate(test_data):
    #     if idx in output_indices:
    #         res_file.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    return test_acc

def write_original_sentences(test_data,output_indices,fname):
    res_file = open("%s-sentences" % (fname), 'w')
    for idx,line in enumerate(test_data):
        if idx in output_indices:
            res_file.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    res_file.close()            
    pass

def batch_expansion(CP, res_file, dataset,k,fname):
    print dataset
    
    test_acc = evaluate(CP,"../data/%s" % dataset, k, res_file,fname)
    # res_file.write("%f, %f, %f\n" % (l2, train_acc, test_acc))
    return test_acc

def batch_projection(CP, res_file, dataset,k,fname):
    print dataset
    # res_file.write("%s, " % dataset)
    test_acc = evaluate_projection(CP,"../data/%s" % dataset, k, res_file,fname)
    # res_file.write("%f, %f, %f\n" % (l2, train_acc, test_acc))
    return test_acc

# from output original sentences and term features (words), 
# connect into the format as sentences with words for observation
def link_sentences_with_words(fname):
    combined_file = open('%s-combined'%fname,'w')
    # in sentFile, each line is a sentence, each word is split by ','
    sentFile = open('%s-sentences'%fname, 'r')
    sents = []
    for line in sentFile:
        sent = line.strip().split(',')
        sents.append(sent)
    # print sents
    # in wordFile, each line is a word and its peris, 
    # format: word coreness peri1,ppmi1 peri2,ppmi2 ...
    wordFile = open('%s-words'%fname, 'r')
    # get a list of all cores have been found in these instances
    core_words = [line.strip().split()[0] for line in wordFile]
    core_peris = [line.strip().split()[2:] for line in wordFile]
    print core_words
    for sent in sents:
        combined_file.write('%s\n'%' '.join([word for word in sent]))
        for word in sent:
            if word in core_words:
                combined_file.write('%s'%' '.join(core_peris[core_words.index(word)]))
        combined_file.write('\n')
    combined_file.close()
    pass

def main():
    # dict_name = "cp-overalp.ppmi"
    #dict_name = "PMI-thesaurus"
    dict_name = sys.argv[1]
    # res_file.write("dataset, k, l2, true_instances, false_instances\n")
    # k=1000
    # kvals = [10,100,500,1000]
    kvals = [20]
    # datasets = ["TR"]
    dataset = 'TR'

    # datasets = ["TR", "CR", "SUBJ","MR", "B-D", "B-E", "B-K", "D-B", "D-E", "D-K", "E-B", "E-D", "E-K", "K-B", "K-D", "K-E"]
    
    acc_file = open('../work/%s-%s-projection-acc'%(dict_name,dataset),'w')
    # acc_file = open('../work/%s-%s-proposed-acc'%(dict_name,dataset),'w')
    # for dataset in datasets:
    for k in kvals:
        CP = expand.CP_EXPANDER()
        CP.load_CP_Dictionary("../data/%s" % dict_name, k)
        # fname = "../work/%s-%s-proposed-%d" % (dataset,dict_name,k)
        # res_file = open("%s-words"%fname, 'w')
        # test_acc = batch_expansion(CP, res_file, dataset, k, fname)
        # res_file.close()
        
        fname = "../work/%s-%s-projection-%d" % (dataset,dict_name,k)
        res_file = open("%s-words"%fname, 'w')
        test_acc=batch_projection(CP, res_file, dataset, k, fname)
        res_file.close()
        print test_acc
        acc_file.write('%f,%f\n'%(k, test_acc))
    acc_file.close()

def test():
    dict_name = sys.argv[1]
    k = 20
    dataset = 'TR'
    fname = "../work/%s-%s-proposed-%d" % (dataset,dict_name,k)
    link_sentences_with_words(fname)
    pass

if __name__ == '__main__':
    # main()
    test()
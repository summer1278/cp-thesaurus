
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


def get_indices(predict,target,value):
    return find_indices(np.sign(np.multiply(predict,target)),value)

def find_indices(my_list,value):
    indices = [i for i, x in enumerate(my_list) if x == value]
    return indices

def train_with_CV(X_train, y_train, X_test, y_test,value):
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
    # print best_theta
    best_theta = 1.0

    clf = linear_model.LogisticRegression(penalty='l2', C=best_theta)
    clf.fit(X_train, y_train)
    print "classifier learnt."
    # print np.sign(np.multiply(clf.predict(X_test),y_test))
    indices = get_indices(clf.predict(X_test),y_test,value)

    return indices

# short text classification
def evaluate(CP,bench,k,res_file):
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
   
    # best_theta, train_acc, test_acc = self.train_with_CV(X_train, y_train, X_test, y_test)
    correct_indices = train_with_CV(X_train, y_train, X_test, y_test,1)

    print "\n ---- NO Expansion ----"
    print "Test corrects =", len(correct_indices)

    # expansion for proposed methods
    train_data = np.array([CP.expand_weighted(line, k) for line in open(train_fname)])     
    test_data = np.array([CP.expand_weighted(line, k) for line in open(test_fname)])    

    X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)

    wrong_indices = train_with_CV(X_train, y_train, X_test, y_test,-1)
    print "\n ---- With Expansion ----"
    print "Test incorrects =", len(wrong_indices)

    output_indices=set(correct_indices)&set(wrong_indices)
    print "\nintersection of both = ", len(output_indices)
    test_data = [line for line in open(test_fname)] 

    for idx,line in enumerate(test_data):
        if idx in output_indices:
            res_file.write("%s\n" % ','.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    pass

def batch_expansion(CP, res_file, dataset):
    print dataset
    # res_file.write("%s, " % dataset)
    evaluate(CP,"../data/%s" % dataset, 100, res_file)
    # res_file.write("%f, %f, %f\n" % (l2, train_acc, test_acc))
    pass


def main():
    # dict_name = "cp-overalp.ppmi"
    #dict_name = "PMI-thesaurus"
    dict_name = sys.argv[1]
    res_file = open("../work/%s-test" % dict_name, 'w')
    # res_file.write("dataset, k, l2, true_instances, false_instances\n")

    datasets = ["TR"]
    # datasets = ["TR", "CR", "SUBJ","MR", "B-D", "B-E", "B-K", "D-B", "D-E", "D-K", "E-B", "E-D", "E-K", "K-B", "K-D", "K-E"]
    for dataset in datasets:
        CP = expand.CP_EXPANDER()
        CP.load_CP_Dictionary("../data/%s" % dict_name, 100)
        # batch_process(CP, res_file, dataset)
        batch_expansion(CP, res_file, dataset)
    res_file.close()

if __name__ == '__main__':
    main()
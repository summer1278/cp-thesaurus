
"""
Test Methods for CP thesaurus

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


def get_true_instances():

    pass

def get_false_instances():
    pass

def find_indices(my_list,value):
    indices = [i for i, x in enumerate(my_list) if x == value]
    return indices

def train_with_CV(X_train, y_train, X_test, y_test):
    # find the best classifier
    print "cross validation.."
    theta_vals = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]
    cv_res = []
    for theta  in theta_vals:
        clf = linear_model.LogisticRegression(penalty='l2', C=theta, solver='sag')
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        #print scores
        cv_res.append(np.mean(scores))

    #print cv_res
    best_theta = theta_vals[np.argmax(cv_res)]
    print best_theta

    clf = linear_model.LogisticRegression(penalty='l2', C=best_theta)
    clf.fit(X_train, y_train)
    print "classifier learnt."
    print np.sign(np.multiply(clf.predict(X_train),y_test))
    print find_indices(np.sign(np.multiply(clf.predict(X_train),y_test)),1)
    pass

def non_expansion():
    pass


def main():
    dict_name = "cp-overalp.ppmi"
    #dict_name = "PMI-thesaurus"
    # dict_name = sys.argv[1]
    res_file = open("../work/%s-batchres.csv" % dict_name, 'w')
    res_file.write("dataset, k, l2, true_instances, false_instances\n")

    datasets = ["TR"]
    # datasets = ["TR", "CR", "SUBJ","MR", "B-D", "B-E", "B-K", "D-B", "D-E", "D-K", "E-B", "E-D", "E-K", "K-B", "K-D", "K-E"]
    for dataset in datasets:
        CP = expand.CP_EXPANDER()
        CP.load_CP_Dictionary("../data/%s" % dict_name, 100)
        # batch_process(CP, res_file, dataset)
        #non_expansion(CP, res_file, dataset)
    res_file.close()

if __name__ == '__main__':
    main()
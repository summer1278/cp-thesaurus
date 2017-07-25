
"""
Load the CP thesaurus and performs various types of feature expansions.

Danushka Bollegala
May 2017
"""

import sys
from sklearn import linear_model
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class CP_EXPANDER():

    def __init__(self):        
        pass

    def append_PPMI_values(self, CP_fname, PPMI_fname, out_fname):
        """
        Append PPMI values for the (core, peri) pairs in each peri list.
        """
        sys.stderr.write("Loading PPMI values from %s ..." % PPMI_fname)
        sys.stderr.flush()
        PPMI = {}
        n = 0
        with open(PPMI_fname) as PPMI_file:
            for line in PPMI_file:
                n += 1
                if n % 1000000 == 0:
                    print n
                p = line.strip().split()
                #print p
                (first, second, value) = (p[0], p[1], float(p[2]))
                if first > second:
                    (first, second) = (second, first)
                PPMI[(first, second)] = value
        sys.stderr.write("Done\n")
        sys.stderr.flush()
        F = open(CP_fname)
        G = open(out_fname, 'w')
        for line in F:
            p = line.strip().split(',')
            core = p[0]
            coreness = float(p[1])
            peris = p[2:]
            G.write("%s %f " % (core, coreness))
            for peri in peris:
                if core < peri:
                    value = PPMI.get((core,peri), None)
                else:
                    value = PPMI.get((peri, core), None)
                if value is not None:
                    G.write("%s,%f " % (peri, value))
            G.write("\n")
        F.close()
        G.close()
        pass

    def load_CP_Dictionary(self, CP_fname, cutoff):
        """
        Load the CP dictioanry from the specified file. 
        Do not load more than cutoff no. of peris for each core.
        D["core"] = {"coreness": <how good is this core word> (float), "peris" : [(peri, score)]}
        wid["word"] = id (int)
        """
        self.D = {}
        self.wid = {}
        with open(CP_fname) as CP_file:
            for line in CP_file:
                p = line.strip().split()
                core = p[0]
                self.wid.setdefault(core, len(self.wid))
                self.D[self.wid[core]] = {"coreness":float(p[1]), "peris":[]}
                for x in p[2:]:
                    t = x.split(',')
                    (peri, val) = (t[0], float(t[1]))
                    self.wid.setdefault(peri, len(self.wid))
                    if len(self.D[self.wid[core]]["peris"]) > cutoff:
                        break
                    self.D[self.wid[core]]["peris"].append((self.wid[peri],val))
        print "Total core words =", len(self.D)
        print "Vocabulary size  =", len(self.wid)
        print self.wid
        pass

    def get_feature_set(self, fname):
        """
        Return the set of features in the instance file fname.
        """
        feats = set()
        with open(fname) as F:
            for line in F:
                p = line.strip().split()
                feats = feats.union(set([x.split(':')[0] for x in p[1:]]))
        return feats

    def get_feat_vect(self, feat_str):
        """
        Return the expanded feature vector.
        """
        x = np.zeros(len(self.wid) + 1, dtype=float)
        p = feat_str.strip().split()
        x[0] = p[0]
        for feat in p[1:]:
            fstr = feat.split(':')[0]
            x[self.wid[fstr]] = 1
        return x

    def expand_unweighted(self, feat_str, k):
        """
        Expand the feature vector given by feat_str.
        """
        x = np.zeros(len(self.wid) + 1, dtype=float)
        p = feat_str.strip().split()
        x[0] = p[0]
        for feat in p[1:]:
            fstr = feat.split(':')[0]
            core_id = self.wid[fstr]
            x[core_id] = 1
            if core_id in self.D:
                for (peri_id, peri_val) in self.D[core_id]["peris"][:k]:
                    x[peri_id] = 1
        return x


    def expand_weighted(self, feat_str, k):
        """
        Expand the feature vector given by feat_str. Use the normalised PPMI
        over each peripheral set as the value of expanded features.
        """
        x = np.zeros(len(self.wid) + 1, dtype=float)
        p = feat_str.strip().split()
        x[0] = p[0]
        z = np.zeros(len(self.wid) + 1, dtype=float)
        for feat in p[1:]:
            fstr = feat.split(':')[0]
            core_id = self.wid[fstr]
            x[core_id] = 1
            if core_id in self.D:
                for (peri_id, peri_val) in self.D[core_id]["peris"][:k]:
                    z[peri_id] = peri_val
        norm_z = np.linalg.norm(z)
        if norm_z > 0:
            x += (z / norm_z)
        return x


    def train_with_CV(self, X_train, y_train, X_test, y_test):
        """
        Perform CV on l2 parameters and return the best results.
        """
        theta_vals = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]
        cv_res = []
        for theta  in theta_vals:
            clf = linear_model.LogisticRegression(penalty='l2', C=theta, solver='sag')
            scores = cross_val_score(clf, X_train, y_train, cv=5)
            #print scores
            cv_res.append(np.mean(scores))

        #print cv_res
        best_theta = theta_vals[np.argmax(cv_res)]

        clf = linear_model.LogisticRegression(penalty='l2', C=best_theta)
        clf.fit(X_train, y_train)
        train_acc = 0.5 + 0.5 * np.mean(np.sign(clf.predict(X_train) * y_train))
        test_acc = 0.5 + 0.5 * np.mean(np.sign(clf.predict(X_test) * y_test))
        return (best_theta, train_acc, test_acc)

    def evaluate(self, bench, k):
        """
        Perform short text classification using expanded features.
        """
        train_fname = "%s/train" % bench
        test_fname = "%s/test" % bench
        train_feats = self.get_feature_set(train_fname)
        print "k = %d" % k
        print "Total no. of train features =", len(train_feats)
        test_feats = self.get_feature_set(test_fname)
        print "Total no. of test features =", len(test_feats)
        for feat in train_feats:
            self.wid.setdefault(feat, len(self.wid))
        for feat in test_feats:
            self.wid.setdefault(feat, len(self.wid))
        print "Total no. of all features =", len(self.wid)

        if 0:
            # no expansion
            train_data = np.array([self.get_feat_vect(line) for line in open(train_fname)])     
            test_data = np.array([self.get_feat_vect(line) for line in open(test_fname)])     

            X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
            X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)
           
            best_theta, train_acc, test_acc = self.train_with_CV(X_train, y_train, X_test, y_test)

            print "\n ---- NO Expansion ----"
            print "Train accuracy =", train_acc
            print "Test accuracy =", test_acc
            print "Best theta =", best_theta

        if 1:
            # Expansion per features. (Proposed method)
            train_data = np.array([self.expand_weighted(line, k) for line in open(train_fname)])     
            test_data = np.array([self.expand_weighted(line, k) for line in open(test_fname)])    

            X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
            X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)
            
            best_theta, train_acc, test_acc = self.train_with_CV(X_train, y_train, X_test, y_test)

            print "\n ---- With Expansion ----"
            print "Train accuracy =", train_acc
            print "Test accuracy =", test_acc
            print "Best theta =", best_theta

        if 0:
            # Expansion by cores (Projection method)
            train_data = np.array([self.get_feat_vect(line) for line in open(train_fname)])     
            test_data = np.array([self.get_feat_vect(line) for line in open(test_fname)])     

            X_train, y_train = train_data[:,1:], train_data[:,0].astype(int)
            X_test, y_test = test_data[:,1:], test_data[:,0].astype(int)

            # Build a matrix between cores and peris.
            D = len(self.D)
            CP_mat = np.zeros((D, len(self.wid)), dtype=np.float)
            core_list = list(self.D.keys())
            core_list.sort()

            print "Building CP_matrix...",
            for (i, core_id) in enumerate(core_list):
                peris = self.D[core_id]["peris"]
                coreness = self.D[core_id]["coreness"]
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
            
            print "Cross-validation..."
            best_theta, train_acc, test_acc = self.train_with_CV(X_train, y_train, X_test, y_test)

            print "\n ---- With Expansion ----"
            print "Train accuracy =", train_acc
            print "Test accuracy =", test_acc
            print "Best theta =", best_theta            

        return (best_theta, train_acc, test_acc)   

    def evaluate_text_classification(self, bench):
        """
        Perform short text classification using expanded features.
        """
        train_fname = "%s/train" % bench
        test_fname = "%s/test" % bench
        train_feats = self.get_feature_set(train_fname)
        print "Total no. of train features =", len(train_feats)
        test_feats = self.get_feature_set(test_fname)
        print "Total no. of test features =", len(test_feats)
        for feat in train_feats:
            self.wid.setdefault(feat, len(self.wid))
        for feat in test_feats:
            self.wid.setdefault(feat, len(self.wid))
        print "Total no. of all features =", len(self.wid)

        train_data = np.array([self.get_feat_vect(line) for line in open(train_fname)])     
        test_data = np.array([self.get_feat_vect(line) for line in open(test_fname)])     

        X_train, y_train = train_data[:,1:], train_data[:,0]
        X_test, y_test = test_data[:,1:], test_data[:,0]
        predictor = linear_model.LogisticRegression()
        predictor.fit(X_train, y_train)

        print "\n ---- NO Expansion ----"
        print "Train accuracy =", 0.5 + 0.5 * np.mean(np.sign(predictor.predict(X_train) * y_train))
        print "Test accuracy =", 0.5 + 0.5 * np.mean(np.sign(predictor.predict(X_test) * y_test))

        # Expansion using all peri features ignoring weights
        train_data = np.array([self.expand(line) for line in open(train_fname)])     
        test_data = np.array([self.expand(line) for line in open(test_fname)])    

        X_train, y_train = train_data[:,1:], train_data[:,0]
        X_test, y_test = test_data[:,1:], test_data[:,0]
        predictor = linear_model.LogisticRegression()
        predictor.fit(X_train, y_train)

        print "\n ---- With Expansion ----"
        print "Train accuracy =", 0.5 + 0.5 * np.mean(np.sign(predictor.predict(X_train) * y_train))
        print "Test accuracy =", 0.5 + 0.5 * np.mean(np.sign(predictor.predict(X_test) * y_test))
        pass


def batch_process(CP, res_file, dataset): 
    print dataset 
    res_file.write("%s\n" % dataset)
    kvals = [100]
    #kvals =  [10, 100, 500, 1000]
    for k in kvals:
        l2, train_acc, test_acc = CP.evaluate("../data/%s" % dataset, k)
        res_file.write(", %d, %f, %f, %f\n" % (k, l2, train_acc, test_acc))
        res_file.flush()
    res_file.write("\n")
    pass


def non_expansion(CP, res_file, dataset):
    print dataset
    res_file.write("%s, " % dataset)
    l2, train_acc, test_acc = CP.evaluate("../data/%s" % dataset, 0)
    res_file.write("%f, %f, %f\n" % (l2, train_acc, test_acc))
    pass


def main():
    #dict_name = "cp-overalp.ppmi"
    #dict_name = "PMI-thesaurus"
    dict_name = sys.argv[1]
    res_file = open("../work/%s-batchres.csv" % dict_name, 'w')
    res_file.write("dataset, k, l2, train_acc, test_acc\n")

    datasets = ["TR"]
    # datasets = ["TR", "CR", "SUBJ","MR", "B-D", "B-E", "B-K", "D-B", "D-E", "D-K", "E-B", "E-D", "E-K", "K-B", "K-D", "K-E"]
    for dataset in datasets:
        CP = CP_EXPANDER()
        CP.load_CP_Dictionary("../data/%s" % dict_name, 100)
        batch_process(CP, res_file, dataset)
        #non_expansion(CP, res_file, dataset)
    res_file.close()


if __name__ == '__main__':
    main()
    









import expand.CP_EXPANDER as CP

def get_true_instances():
    pass

def get_false_instances():
    pass

def train_with_CV(X_train, y_train, X_test, y_test):
    # find the best classifier
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
    
    pass
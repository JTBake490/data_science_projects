import numpy as np 
import pandas as pd 
from warnings import filterwarnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss

def make_pretty_metrics(model, X_test, y_test, model_name=None):
    pred = model.predict(X_test)
    print(f'{model_name} Accuracy: {np.round(model.score(X_test, y_test), 5)}')
    print(f'{model_name} Precision: {np.round(precision_score(y_test, pred), 5)}')
    print(f'{model_name} Recall: {np.round(recall_score(y_test, pred), 5)}')
    print(f'{model_name} F1: {np.round(f1_score(y_test, pred), 5)}\n')

def get_f1_thresholds(model, thresholds, X_test, y_test, model_name=None):
    model_test_probs = model.predict_proba(X_test)[:, 1]

    f1_scores = np.array([])
    for p in thresh_ps:
        model_test_labels = model_test_probs > p
        f1_scores = np.hstack((f1_scores, f1_score(y_test, model_test_labels)))

        best_f1_score = np.max(f1_scores)
        best_thresh_p = thresholds[np.argmax(f1_scores)]
        
    if model_name != None:    
        print(f'{model_name}')
    print(f'Best F1 Score: {best_f1_score}')
    print(f'Threshold For F1 score of {best_f1_score}: {best_thresh_p}')
    
    return f1_scores, best_f1_score, best_thresh_p


if __name__ == '__main__':
    filterwarnings('ignore')
    cc = pd.read_csv('creditcard.csv/creditcard.csv')

    sns.countplot(cc['Class'])
    plt.show()

    print(cc.Class.value_counts())

    X = cc.drop(columns='Class')
    y = cc['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    cc_dum = DummyClassifier(random_state=10).fit(X_train, y_train)
    cc_log = LogisticRegression(random_state=10).fit(X_train, y_train)

    print(f'Dummy Accuracy Score: {np.round(cc_dum.score(X_test, y_test), 5)}')
    print(f'Unbalanced Logistic Regression Accuracy Score: {np.round(cc_log.score(X_test, y_test),5)}\n')

    make_pretty_metrics(cc_dum, X_test, y_test, 'Dummy')
    make_pretty_metrics(cc_nb, X_test, y_test, 'Gaussian Naive Bayes')

    sns.heatmap(confusion_matrix(y_test, cc_dum.predict(X_test)),cmap=plt.cm.Reds, annot=True, square=True, fmt='d')
    plt.show()

    sns.heatmap(confusion_matrix(y_test, cc_nb.predict(X_test)),cmap=plt.cm.Reds, annot=True, square=True, fmt='d')
    plt.show()

    smote = SMOTE(random_state=10)
    adasyn = ADASYN(random_state=10)
    nm1 = NearMiss(random_state=10)
    nm2 = NearMiss(random_state=10, version=2)

    X_tr_smote, y_tr_smote = smote.fit_sample(X_train, y_train) # SMOTE oversampling
    X_tr_ada, y_tr_ada = adasyn.fit_sample(X_train, y_train)    # ADASYN oversampling
    X_tr_nm1, y_tr_nm1 = nm1.fit_sample(X_train, y_train)       # NearMiss Version 1 undersampling
    X_tr_nm2, y_tr_nm2 = nm2.fit_sample(X_train, y_train)       # NearMiss Version 2 undersampling

    # Displaying the size of each new training set.
    print(f'Imbal_shape: {X_train.shape}')
    print(f'SMOTE_shape: {X_tr_smote.shape}')
    print(f'ADASYN_shape: {X_tr_ada.shape}')
    print(f'NearMiss1_shape: {X_tr_nm1.shape}')
    print(f'NearMiss2_shape: {X_tr_nm2.shape}')
    
    # Logistic Regression with Cross-Validation for each of the resampling techniques.
    log_orig = LogisticRegressionCV(cv= 5, random_state=10).fit(X_train, y_train)
    log_smote = LogisticRegressionCV(cv=5, random_state=10, class_weight='balanced').fit(X_tr_smote, y_tr_smote)
    log_ada = LogisticRegressionCV(cv=5, random_state=10, class_weight='balanced').fit(X_tr_ada, y_tr_ada)
    log_nm1 = LogisticRegressionCV(cv=5, random_state=10, class_weight='balanced').fit(X_tr_nm1, y_tr_nm1)
    log_nm2 = LogisticRegressionCV(cv=5, random_state=10, class_weight='balanced').fit(X_tr_nm2, y_tr_nm2)

    # Gaussian Naive Bayes for each of the resampling techniques.
    gnb_orig = GaussianNB().fit(X_train, y_train)
    gnb_smote = GaussianNB().fit(X_tr_smote, y_tr_smote)
    gnb_ada = GaussianNB().fit(X_tr_ada, y_tr_ada)
    gnb_nm1 = GaussianNB().fit(X_tr_nm1, y_tr_nm1)
    gnb_nm2 = GaussianNB().fit(X_tr_nm2, y_tr_nm2)

    # Random Forest Classifier for each of the resampling techniques
    rfc_orig = RandomForestClassifier(random_state=10).fit(X_train, y_train)
    rfc_smote = RandomForestClassifier(random_state=10, class_weight='balanced_subsample').fit(X_tr_smote, y_tr_smote)
    rfc_ada = RandomForestClassifier(random_state=10, class_weight='balanced_subsample').fit(X_tr_ada, y_tr_ada)
    rfc_nm1 = RandomForestClassifier(random_state=10, class_weight='balanced_subsample').fit(X_tr_nm1, y_tr_nm1)
    rfc_nm2 = RandomForestClassifier(random_state=10, class_weight='balanced_subsample').fit(X_tr_nm2, y_tr_nm2)

    # Gradient Boosting Classifier for each the resampling techniques
    gbc_orig = GradientBoostingClassifier(random_state=10, max_features='sqrt').fit(X_train, y_train)
    gbc_smote = GradientBoostingClassifier(random_state=10, max_features='sqrt').fit(X_tr_smote, y_tr_smote)
    gbc_ada = GradientBoostingClassifier(random_state=10, max_features='sqrt').fit(X_tr_ada, y_tr_ada)
    gbc_nm1 = GradientBoostingClassifier(random_state=10, max_features='sqrt').fit(X_tr_nm1, y_tr_nm1)
    gbc_nm2 = GradientBoostingClassifier(random_state=10, max_features='sqrt').fit(X_tr_nm2, y_tr_nm2)

    log_models = np.array([log_orig, log_smote, log_ada, log_nm1, log_nm2])
    gnb_models = np.array([gnb_orig, gnb_smote, gnb_ada, gnb_nm1, gnb_nm2])
    rfc_models = [rfc_orig, rfc_smote, rfc_ada, rfc_nm1, rfc_nm2]           # important not to use a numpy array.
    gbc_models = [gbc_orig, gbc_smote, gbc_ada, gbc_nm1, gbc_nm2]           # important not to use a numpy array.

    make_pretty_metrics(log_orig, X_test, y_test, 'Log_Imbal')
    make_pretty_metrics(log_smote, X_test, y_test, 'Log_w_SMOTE')
    make_pretty_metrics(log_ada, X_test, y_test, 'Log_w_ADASYN')
    make_pretty_metrics(log_nm1, X_test, y_test, 'Log_w_NM1')
    make_pretty_metrics(log_nm2, X_test, y_test, 'Log_w_NM2')

    make_pretty_metrics(gnb_orig, X_test, y_test, 'Gauss_Naive_Bayes_Imbal')
    make_pretty_metrics(gnb_smote, X_test, y_test, 'Gauss_Naive_Bayes_SMOTE')
    make_pretty_metrics(gnb_ada, X_test, y_test, 'Gauss_Naive_Bayes_ADASYN')
    make_pretty_metrics(gnb_nm1, X_test, y_test, 'Gauss_Naive_Bayes_NM1')
    make_pretty_metrics(gnb_nm2, X_test, y_test, 'Gauss_Naive_Bayes_NM2')

    make_pretty_metrics(rfc_orig, X_test, y_test, 'Rand_Forest_Imbal')
    make_pretty_metrics(rfc_smote, X_test, y_test, 'Rand_Forest_SMOTE')
    make_pretty_metrics(rfc_ada, X_test, y_test, 'Rand_Forest_ADASYN')
    make_pretty_metrics(rfc_nm1, X_test, y_test, 'Rand_Forest_NM1')
    make_pretty_metrics(rfc_nm2, X_test, y_test, 'Rand_Forest_NM2')

    make_pretty_metrics(gbc_orig, X_test, y_test, 'Grad_Boost_Imbal')
    make_pretty_metrics(gbc_smote, X_test, y_test, 'Grad_Boost_SMOTE')
    make_pretty_metrics(gbc_ada, X_test, y_test, 'Grad_Boost_ADASYN')
    make_pretty_metrics(gbc_nm1, X_test, y_test, 'Grad_Boost_NM1')
    make_pretty_metrics(gbc_nm2, X_test, y_test, 'Grad_Boost_NM2')

    thresh_ps = np.linspace(0.1, 0.9, 5000) # This range will be the thresholds to be test for the random forest classifier

    rand_smote_f1s, rand_smote_best_f1, rand_smote_best_thresh = get_f1_thresholds(rfc_smote, thresh_ps, X_test, y_test, 'Rand_For_SMOTE')
    print('\n')
    rand_ada_f1s, rand_ada_best_f1, rand_ada_best_thresh = get_f1_thresholds(rfc_ada, thresh_ps, X_test, y_test, 'Rand_For_Adasyn')

    print(precision_score(y_test, rfc_smote.predict_proba(X_test)[:,1]>0.6001))
    print(recall_score(y_test, rfc_smote.predict_proba(X_test)[:,1]>0.6001))
    print(f1_score(y_test, rfc_smote.predict_proba(X_test)[:,1]>0.6001))

    print(precision_score(y_test, rfc_ada.predict_proba(X_test)[:,1]>0.6001))
    print(recall_score(y_test, rfc_ada.predict_proba(X_test)[:,1]>0.6001))
    print(f1_score(y_test, rfc_ada.predict_proba(X_test)[:,1]>0.6001))

    # Plotting all the threshold values against the f1 scores. (Random Forest Classifier with SMOTE oversampling)
    plt.plot(thresh_ps, rand_smote_f1s)
    plt.title('Fraud Probability Threshold vs F1 Scores')
    plt.xlabel('Probability Thresholds', labelpad=10)
    plt.ylabel('F1 \nScores', rotation=0, labelpad=30)
    plt.show()

    plt.plot(thresh_ps, rand_ada_f1s)
    plt.title('Fraud Probability Threshold vs F1 Scores')
    plt.xlabel('Probability Thresholds', labelpad=10)
    plt.ylabel('F1 \nScores', rotation=0, labelpad=30)

    feature_importances = pd.DataFrame(rfc_smote.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',  ascending=False)
    print(feature_importances)

    model = RandomForestClassifier(random_state=10, class_weight='balanced_subsample').fit(X, y)
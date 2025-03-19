import pickle
import time
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score

IMAGE_SIZE = 64

clfs = {
        'RF': RandomForestClassifier(n_estimators=200, random_state=1),
    }

param_grid = {
    'RF' : {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy']
    },
}

ListAlgo = {
    'RF': RandomForestClassifier(random_state=1),
}

def open_pickle():
    with open('/artifacts/best_model.pkl', 'rb') as f:
        return pickle.load(f)

def check_for_new_pickle():
    df1 = pd.read_csv('/data/prod_data.csv', sep=';')
    if len(df1) > 3:
        MergeData()
        doTraining()
        open_pickle()

def SaveFeedBackData(image_vector,classe):
    with open('/data/prod_data.csv', 'a') as f:
        f.write(image_vector+";"+classe+"\n")

def MergeData():
    df1 = pd.read_csv('/data/prod_data.csv', sep=';')
    df2 = pd.read_csv('/data/ref_data.csv', sep=';')
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv('/data/ref_data.csv', sep=';', index=False)
    open('/data/prod_data.csv', 'w').close()

def Apprentissage(df) :
    bg = BaggingClassifier(n_jobs=-1)
    np_array = df.to_numpy()
    variables = np_array[:, :-1]
    status = np_array[:, -1]

    creation_pipeline(variables, status, bg, "NORM", pow(IMAGE_SIZE, 2) * 3)

def doTraining():
    df = pd.read_csv('/data/ref_data.csv', sep=';')
    Apprentissage(df)

def run_classifier(clfs,X,Y):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    combined_scorer = make_scorer(custom_scorer,greater_is_better=True)
    max = 0
    max_name = "NBS"
    for i in clfs:
        clf = clfs[i]
        time_start = time.time()
        score = cross_val_score(clf, X, Y, cv=kf, scoring = combined_scorer)
        duration = time.time() - time_start
        print("Résultats pour {0} :".format(i))  
        print("  - score : {1:.3f} ± {2:.3f}".format(i, np.mean(score), np.std(score)))           
        print("  - Temps moyen par fold : {0:.3f} sec".format(duration / 10))  
        
        if (np.mean(score) > max) :
            max = np.mean(score)
            max_name = i
    return max_name,max


def custom_scorer(y_true, y_pred):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return balanced_acc, acc



def creation_pipeline(X, y, model, strategy, nb_selected_features):
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)

    if strategy == 'BASE':
        P = Pipeline([
            ('fs', SelectFromModel(clf, max_features=nb_selected_features)),
            ('classifier', model)
        ])
    
    elif strategy == 'NORM':
        P = Pipeline([
            ('ss', StandardScaler()),
            ('fs', SelectFromModel(clf, max_features=nb_selected_features)),
            ('classifier', model)
        ])
    
    else:  # Normalisation + PCA
        P = Pipeline([
            ('ss', StandardScaler()),
            ('fu', FeatureUnion([
                ('ss', StandardScaler()),
                ('pca', PCA(n_components=3))
            ])),
            ('fs', SelectFromModel(clf, max_features=nb_selected_features)),
            ('classifier', model)
        ])

    P.fit(X, y)

    with open('/artifacts/best_model.pkl', 'wb') as f:
        pickle.dump(P, f)

    print("Pipeline sauvegardé sous 'best_model.pkl'")
    return P

def bestVariable(name,X,Y) : 
    algo = clfs[name]
    clf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
    clf.fit(X, Y)
    importances=clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    scores= 0
    bestFeatures = []
    indice = 0
    for f in np.arange(0, X_train.shape[1]+1):
        X1_f = X_train[:,sorted_idx[:f+1]]
        X2_f = X_test [:,sorted_idx[:f+1]]
        algo.fit(X1_f,Y_train)
        yAlgo=algo.predict(X2_f)
        if (scores < np.round(custom_scorer(Y_test,yAlgo),3)) :
            scores =  np.round(custom_scorer(Y_test,yAlgo),3)  
            bestFeatures = X[:,sorted_idx[:f+1]]
            indice = f
            
    return bestFeatures,indice


def classifier_choose(X,Y):
    return run_classifier(clfs,X,Y)




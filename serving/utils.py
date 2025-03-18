import pickle
import time
import numpy as np
import pandas as pd

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

clfs = {
        'NBS':GaussianNB(),
        'CART': tree.DecisionTreeClassifier(criterion="gini",random_state=1),
        'IDE3': tree.DecisionTreeClassifier(criterion="entropy",random_state=1),
        'DS': tree.DecisionTreeClassifier(max_depth=1,random_state=1),
        'MLP': MLPClassifier(hidden_layer_sizes=(20,10),random_state=1),
        'KNN': KNeighborsClassifier(n_neighbors=5,n_jobs=-1),
        'BAGGING': BaggingClassifier(n_estimators=200,random_state=1),
        'ADA_BOOST': AdaBoostClassifier(n_estimators=200,random_state=1),
        'RF': RandomForestClassifier(n_estimators=200, random_state=1),
        'XGBOOST': XGBClassifier(n_estimators=200,random_state=1)
    }

param_grid = {
    'NBS' : {
        'var_smoosing' : [1e-9]
    },
    'ADA_BOOST' : {
        'n_estimators': [100, 200, 500]
    },
    'RF' : {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy']
    },
    'CART': {
        'max_depth': [1, 2, 3, 4, 5, None], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'ID3': {
        'max_depth': [1, 2, 3, 4, 5, None], 
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4]
    },
    'KNN' : {
        'n_neighbors': [1, 3, 5, 10]
    },
    'MLP' : {
        'hidden_layer_sizes': [(100,), (200,), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic']
    },
    'BAGGING' : {
        'n_estimators': [100, 200, 500]
    },
    'XGBOOST': {
        'n_estimators': [100, 200, 500],  
        'max_depth': [3, 5, 7, 10],  
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  
        'subsample': [0.6, 0.8, 1.0],  
        'colsample_bytree': [0.6, 0.8, 1.0],  
        'gamma': [0, 0.1, 0.2, 0.5],  
        'lambda': [0, 1, 10],  
        'alpha': [0, 1, 10]  
    },
}

ListAlgo = {
    'NBS':GaussianNB(),
    'CART': tree.DecisionTreeClassifier(criterion="gini",random_state=1),
    'IDE3': tree.DecisionTreeClassifier(criterion="entropy",random_state=1),
    'DS': tree.DecisionTreeClassifier(random_state=1),
    'MLP': MLPClassifier(random_state=1),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'BAGGING': BaggingClassifier(random_state=1),
    'ADA_BOOST': AdaBoostClassifier(random_state=1),
    'RF': RandomForestClassifier(random_state=1),
    'XGBOOST': XGBClassifier(random_state=1)
}

def open_pickle():
    with open('../artifacts/best_model.pkl', 'rb') as f:
        return pickle.load(f)

def SaveFeedBackData(image_vector,classe):
    with open('../data/prod_data.csv', 'a') as f:
        f.write(image_vector+";"+classe+"\n")

def MergeData():
    df1 = pd.read_csv('../data/prod_data.csv', sep=';')
    df2 = pd.read_csv('../data/ref_data.csv', sep=';')
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv('../data/ref_data.csv', sep=';', index=False)
    open('../data/prod_data.csv', 'w').close()

def Apprentissage(X,y) :
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    features = []

    print("--------------------------------------------------------------")
    print("On récupere le meilleur classifieur pour chaque type de donnée")
    max_base_data = classifier_choose(X,y)
    print("Pour les données non traitées :", max_base_data[0])
    max_norm = classifier_choose(X_norm,y)
    print("Pour les données normalisées :", max_norm[0])
    max_pca = classifier_choose(X_pca,y)
    print("Pour les données PCA :", max_pca[0])
    
    print("--------------------------------------------------------------")
    print("On choisi le meilleur classifieur")
    bestAlgo = ""
    X_update = []
    strategy = ""
    if (max_base_data[1] > max_norm[1] and max_base_data[1] > max_pca[1]) :
        bestAlgo = max_base_data[0]
        X_update = X   
        strategy = "BASE"
    if (max_pca[1] > max_norm[1] and max_pca[1] > max_base_data[1]) :
        bestAlgo = max_pca[0]
        X_update = X_pca
        strategy = "PCA"
    if(max_norm[1] > max_base_data[1] and max_norm[1] > max_pca[1]) :
        bestAlgo = max_norm[0]
        X_update = X_norm 
        strategy = "NORM"
    print("Le meilleur est :", bestAlgo)
    
    print("--------------------------------------------------------------")
    features,indice = bestVariable(bestAlgo, X_update, y)
    print("On a choisi", indice, "variable(s)")
    print("--------------------------------------------------------------")
    print("Strategie utilisé : ", strategy)
    algo =  ListAlgo[bestAlgo]
    combined_scorer = make_scorer(custom_scorer ,greater_is_better=True)
    grid_search = GridSearchCV(estimator=algo,param_grid=param_grid[bestAlgo],scoring=combined_scorer, cv=5) 
    creation_pipeline(X,y,grid_search,strategy,indice)


def doTraining():
    df = pd.read_csv('ref_data.csv', sep=';')
    X = df.drop(columns=['Classe']).values 
    y = df['Classe'].values
    Apprentissage(X,y)

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

    with open('../artifacts/best_model.pkl', 'wb') as f:
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




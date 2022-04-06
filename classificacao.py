from email.mime import base
from os import access
from random import shuffle
from tabnanny import verbose
import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid', {"axes.grid": False})
sns.set_context('notebook')
np.random.seed(42)




def nayvebayes():
    from sklearn.naive_bayes import GausianNB, MultinomialNb, BernoulyNB
    baseDados =pd.read_csv('data/base_NaiveBayes.csv')

    X = baseDados[['X1','X2']]
    Y = baseDados["Y"]

    model = GausianNB()#vale um teste com o L2 tambem Ridge e Lasso
    model.fit(X,Y)

    h= .005
    
    pred = model.predict(X)
    print(pred)
    print(model.predict_proba(X))

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import RadiusNeighborsClassifier

    baseDados =pd.read_csv('data/base_knn.csv')

    h = .01

    print('knn')

    X = baseDados[['X1','X2']]
    Y = baseDados["Y"]

    KNN = KNeighborsClassifier(n_neighbors = 1)#isso aqui muda bastante viu
    KNN.fit(X,Y)

    KNN.predict(X)

    plt.scatter([2.5],[2.5], s = 100, c='darkgreen' if KNN.predict([[2.5,2.5]]) == 0 else 'black')
    plt.scatter([1.5],[1.0], s = 100, c='darkgreen' if KNN.predict([[1.5,1.0]]) == 0 else 'black')
    plt.scatter([3.5],[3.0], s = 100, c='darkgreen' if KNN.predict([[3.5,3.0]]) == 0 else 'black');


def regLog():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolinomialFeatures
    baseDados =pd.read_csv('data/base_regLog.csv')


    X = baseDados[['X1','X2']]
    grau = 10
    #X = PolinomialFeatures(degree = grau).fit_transform(baseDados[baseDados.columns[:-1]])
    Y = baseDados["Y"]

    model = LogisticRegression(solver='liblinear', penalty = 'l1', C=10000 )#vale um teste com o L2 tambem Ridge e Lasso
    model.fit(X,Y)

    h= .005
    #modo polinomial
    #z = model.predict(PolinomialFeatures(degree = grau).fit_transform(np.c_[xx.ravel(),yy.ravel()]))





def svm():
    from sklearn.svm import SVC
    baseDados =pd.read_csv('data/base_svm.csv')

    SVM = SVC(kernel = 'linear', C=1)
    #polinomial
    #SVM = SVC(kernel = 'poly', C=1)   
    #sigmoidal
    #SVM = SVC(kernel = 'sigmoid', C=1)
    #gaussiano
    #SVM = SVC(kernel = 'rbf', C=1)

    X = baseDados[:, baseDados.columns != 'Y']
    Y = baseDados.Y

    SVM.fit(X,Y)


def redeNeural():

    from sklearn.neural_network import MLPClassifier
    baseDados =pd.read_csv('data/base_redeNeural.csv')

    MLP = MLPClassifier(activation = 'relu',
                        hidden_layer_sizes = (40,20,10),
                        learning_rate = 'constant',
                        learning_rate_init = 0.001,
                        max_iter = 5000)

    X = baseDados[:, baseDados.columns != 'Y']
    Y = baseDados.Y

    MLP.fit(X,Y)

    h= .02
    


def decision_tree():

    from sklearn.tree import DecisionTreeClassifier
    baseDados =pd.read_csv('data/base_tree.csv')

    DT = DecisionTreeClassifier(criterion = 'entropy', max_depth=2)#maxDepth muda as coisas

    X = baseDados[:, baseDados.columns != 'Y']
    Y = baseDados.Y

    DT.fit(X,Y)

    ##same aos outros

    from sklearn import tree

    plt.figure(figsize=(12,5))
    tree.plot_tree(DT,
                    feature_names = X.columns,
                    filled = True,
                    rounded = True,
                    class_names = True);


def RandomF():
    from sklearn.ensemble import RandomForestClassifier
    baseDados =pd.read_csv('data/base_tree.csv')

    RF = RandomForestClassifier(criterion = 'entropy', max_depth=2, n_estimators = 10)#maxDepth muda as coisas

    X = baseDados[:, baseDados.columns != 'Y']
    Y = baseDados.Y

    RF.fit(X,Y)



def GradientBoosting():
    from sklearn.ensemble import GradientBoostingClassifier
    baseDados =pd.read_csv('data/base_tree.csv')
    GB = GradientBoostingClassifier(learning_rate = 0.5, max_depth=2, n_estimators = 10)#maxDepth muda as coisas
    X = baseDados[:, baseDados.columns != 'Y']
    Y = baseDados.Y
    GB.fit(X,Y)



#plot visual de generalização
def plotVisual(baseDados,model,X,Y, h):
    x_min, x_max = baseDados.X1.min() - 1, baseDados.X1.max() + 1
    y_min, y_max = baseDados.X2.min() - 1, baseDados.X2.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))


    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    #modo polinomial
    #z = model.predict(PolinomialFeatures(degree = grau).fit_transform(np.c_[xx.ravel(),yy.ravel()]))
    z = Z.reshape(xx.shape)

    plt.figure(figsize=(12,8))
    plt.pcolormesh(xx, yy, Z, cmap= plt.cm.Accent)

    pred = model.predict(X)
    #nada é feit com o pred, é necessario avaiar os resultados

    plt.scatter(baseDados.X1[Y == 0], baseDados.x2[Y == 0], c = 'darkgreen', marker="_")
    plt.scatter(baseDados.X1[Y == 1], baseDados.x2[Y == 1], c = 'black', marker="+");

def plotSeco(baseDados):
    
    plt.figure(figsize=(12,8))
    plt.scatter(baseDados.X1[baseDados.Y == 0],baseDados.X2[baseDados.Y == 0], c = 'darkgreen',marker = '_')
    plt.scatter(baseDados.X1[baseDados.Y == 1],baseDados.X2[baseDados.Y == 1], c = 'darkgreen',marker = '+');

def TreinoTeste_holdout():
    from sklearn.model_selection import train_teste_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import GridSearchCV



    data_imdb = pd.read_csv("data/imdb.csv", encoding="latin1")
    X = data_imdb.text
    Y = data_imdb.sentiment

    #holdout

    x_train, x_test, y_train, y_test = train_teste_split(X, Y, test_size = 0.2, random_state = 42,shuffle = True)

    transformer = CountVectorizer(decode_error="ignore",binary=True)
    x_train = transformer.fit_transform(x_train).toarray()
    y_test = transformer.transform(x_test).toarray()


    #exemplo overfitted
    model = LogisticRegression(solver='liblinear',penalty='l1', c= 1e10)
    model.fit(x_train,y_train)

    print('Acuracia do treino: %.2f' % model.score(x_train,y_train))
    print('Acuracia do teste: %.2f' % model.score(x_test,y_test))

    #exemplo bom
    model = LogisticRegression(penalty='l2', c= 0.1)
    model.fit(x_train,y_train)
    print('Acuracia do treino: %.2f' % model.score(x_train,y_train))
    print('Acuracia do teste: %.2f' % model.score(x_test,y_test))

    #exemplo underfitted
    model = LogisticRegression(penalty='l2', c= 0.0001)
    model.fit(x_train,y_train)

    print('Acuracia do treino: %.2f' % model.score(x_train,y_train))
    print('Acuracia do teste: %.2f' % model.score(x_test,y_test))

    #kfold

    model = LogisticRegression(penalty='l2', C=0.1)
    score = cross_val_score(model,x_train,y_train, cv= KFold(n_splits = 10,shuffle=True),scoring='accuracy')
    print('Acurácia do teste: %.2f +/- %.2f' % (scores.mean(),2* scores.std()))

    #kfold

    model = LogisticRegression(penalty='l2', C=0.1)
    score = cross_val_score(model,x_train,y_train, cv= LeaveOneOut(),scoring='accuracy')
    print('Acurácia do teste: %.2f +/- %.2f' % (scores.mean(),2* scores.std()))


    #gridsearch

    gridS = GridSearchCV(estimator = LogisticRegression(solver='liblinear'),
                         param_grid={'C':[0.01,0.1,1,10,100], 'penalty' : ['l1','l2']},
                         cv= 3,
                         n_jobs = -1,
                         verbose =1,
                         return_train_score = True)

    gridS.fit(x_train,y_train)
    results = pd.DataFrame(gridS.cv_results_)
    results.sort_values(by='mean_test_score', ascending=False).round(3)[['mean_test_score','mean_train_score','param_C','param_penalty']]
    print(gridS.best_params_)


    ################################ ###########################################
    ##Metricas de classificação#####
    ################################ #####3

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

    gini_score = lambda y_true,y_score: 2*roc_auc_score(y_true,y_score)-1


    model = LogisticRegression(C=0.1,penalty='l2')
    model.fit(x_train,y_train)

    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:,1]

    print(confusion_matrix(y_test,pred))

    df_cm = pd.DataFrame(confusion_matrix(y_test,pred),
                         index = [i for i in ['0','1']],
                         columns = [i for i in ['0','1']])
    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm,annot=True,annot_kws={'size':12},fmt='d', cmap=plt.cm.Blues)
    plt.xlabel('REAl')
    plt.ylabel('PREDITO');

    print('Acurácia: %.2f' % accuracy_score(y_test,pred))
    print('Precision: %.2f' % precision_score(y_test,pred))
    print('Recall: %.2f' % recall_score(y_test,pred))
    print('F1: %.2f' % f1_score(y_test,pred))
    print('AUC: %.2f' % roc_auc_score(y_test,score))
    print('GINI: %.2f' % gini_score(y_test,score))

    x,y,_ = roc_curve(y_test,score)
    plt.figure(figsize=(12,8))
    plt.plot([0]+x.tolist()+[1],[0]+y.tolist()+[1])
    plt.plot([0,1],[0,1]);













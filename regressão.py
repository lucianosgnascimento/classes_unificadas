#from lib2to3.pytree import Results
from re import X
from tkinter.font import names
import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

sns.set_style('whitegrid', {"axes.grid": False})
sns.set_context('notebook')
np.random.seed(42)

X = np.array([5,6,4,8,8,3,2,3,6,9,12])
Y = np.array([25,26,24,28,28,23,22,23,26,29,42])

df = pd.DataFrame()
df['x'] = X
df['y'] = Y

def normalize(x):
   num = x - np.min(x)
   denom = np.max(x) - np.min(x)
   return (num/denom)

def regressao_linear():
   from sklearn.linear_model import LinearRegression
   

   sns.regplot(x='x', y='y', data=df)
   plt.show()


def regressao_linear_multipla():
   from sklearn.linear_model import LinearRegression
   
   y = [1,2,3,4,3,4,5,3,5,5,4,5,4,5,4,5,6,0,6,3,1,3,1] 
   x = [[0,2,4,1,5,4,5,9,9,9,3,7,8,8,6,6,5,5,5,6,6,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,6,8,9,2,1,5,6],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,4,3,1,2,3,4,1,3,9,7]]

   X_b = np.ones(len(x[0]))

   X_sm = sm.add_constant(np.column_stack((x[0], X_b)))

   for i in x[1:]:
       X_sm = sm.add_constant(np.column_stack((i, X_sm)))

   results = sm.OLS(y, X_sm).fit()
   print(results.summary)

def regressao_linear_multipla_2():
   from sklearn.linear_model import LinearRegression
   
   y = [1,2,3,4,3,4,5,3,5,5,4,5,4,5,4,5,6,0,6,3,1,3,1]
   x = [[0,2,4,1,5,4,5,9,9,9,3,7,8,8,6,6,5,5,5,6,6,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,6,8,9,2,1,5,6],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,4,3,1,2,3,4,1,3,9,7]]

   #esse modelo funciona com dataframes apenas 

   reg = LinearRegression()
   reg = reg.fit(x, y)
   print(reg.predict())
   print(reg.score())
   print(reg.intercept_)
   print(reg.coef_)


def regularizacaoRidge():
   from sklearn.linear_model import Ridge
   from sklearn.model_selection import cross_val_score
   from sklearn.model_selection import GridSearchCV

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']
   ridge_reg = Ridge(alpha=1)
   score_ridge = cross_val_score(ridge_reg,x,y,cv=10,scoring='neg_mean_squared_error')
   print(score_ridge.mean())



def ridge_MSE():
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import Ridge

   #modelo basico

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']

   reg = Ridge()
   x_train, y_train,x_test, y_test = train_test_split(x,y,random_state=13)
   reg.fit(x_train,y_train)

   pred = reg.predict(x_test)

   mse = np.mean((pred - y_test)**2)
   print(mse)
   print(reg.score(x_train,y_train))

def Ridge_Real_Life():
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import Ridge
   from sklearn.model_selection import GridSearchCV
   #from sklearn.model_selection import cross_val_score
   from sklearn import metrics

   #modelo basico

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']
   x_train, y_train,x_test, y_test = train_test_split(x,y,random_state=13)

   reg = Ridge()
   param_grid = {'alpha':[0.1,1,10]}

   score = scoring=['neg_mean_squared_error','r2']

   grid_cv_ridge = GridSearchCV(reg,param_grid,scoring=score,cv=10,verbose=3,refit='r2')

   grid_cv_ridge.fit(x_train,y_train)
   print("R2::{}".format(grid_cv_ridge.best_score_))
   print("hiperparametro melhor::{}".format(grid_cv_ridge.best_params_))

   pd.DataFrame(data = grid_cv_ridge.cv_results_).head(3)

   best_model = grid_cv_ridge.best_estimator_

   y_pred = best_model.predict(x_test)

   r2_score = best_model.score(x_test,y_test)

   print("R2:{:.3f}".format(r2_score))
   print("MSE:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))
   print("MAE:{:.3f}".format(np.sqrt(metrics.mean_absolute_error(y_test,y_pred))))




def Lasso_Real_Life():
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import Lasso
   from sklearn.model_selection import GridSearchCV
   #from sklearn.model_selection import cross_val_score
   from sklearn import metrics

   #modelo basico

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']
   x_train, y_train,x_test, y_test = train_test_split(x,y,random_state=13)

   reg = Lasso()
   param_grid = {'alpha':[0.1,0.0001,0.000002]}

   score = scoring=['neg_mean_squared_error','r2']

   #validar isso aqui
   grid_cv_lasso = GridSearchCV(reg,param_grid,scoring=score,cv=10,verbose=3,refit='r2')

   grid_cv_lasso.fit(x_train,y_train)
   print("R2::{}".format(grid_cv_lasso.best_score_))
   print("hiperparametro melhor::{}".format(grid_cv_lasso.best_params_))

   pd.DataFrame(data = grid_cv_lasso.cv_results_).head(3)

   best_model = grid_cv_lasso.best_estimator_

   y_pred = best_model.predict(x_test)

   r2_score = best_model.score(x_test,y_test)

   print("R2:{:.3f}".format(r2_score))
   print("MSE:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))

def ElasticNet_Real_Life():
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import ElasticNet
   from sklearn.model_selection import GridSearchCV
   #from sklearn.model_selection import cross_val_score
   from sklearn import metrics

   #modelo basico

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']
   x_train, y_train,x_test, y_test = train_test_split(x,y,random_state=13)

   reg = ElasticNet()
   param_grid = {'alpha':[1,0.0001,0.000002],
                  'l1_ratio' : [0.5, 0.8, 1]}

   score = scoring=['neg_mean_squared_error','r2']

   #validar isso aqui
   grid_cv_elastic = GridSearchCV(reg,param_grid,scoring=score,cv=10,verbose=3,refit='r2')

   grid_cv_elastic.fit(x_train,y_train)
   print("R2::{}".format(grid_cv_elastic.best_score_))
   print("hiperparametro melhor::{}".format(grid_cv_elastic.best_params_))

   pd.DataFrame(data = grid_cv_elastic.cv_results_).head(3)

   best_model = grid_cv_elastic.best_estimator_

   y_pred = best_model.predict(x_test)

   r2_score = best_model.score(x_test,y_test)

   print("R2:{:.3f}".format(r2_score))
   print("MSE:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))




def reg_GradDesc():
   n=0.0001
   iteractions = 10
   m=10

   X_b  = np.c_[np.ones((11,1)),X]

   beta = (9,1) #np.random.randn(2,1) #inicializaçao aleatorio 

   for i in range(iteractions):
      gradients = 2/m * X_b.T.dot(X_b.dot(beta)-Y)
      beta = beta - n * gradients

   a=beta[1]
   b=beta[0]


   predict = (a*X)+ b
   print(predict)


   ######### avaliação desse cara com estatisticas do stats model
   #adicionando constante x a matriz
   X_sm = sm.add_constant(X)
   #ols ordinary least squares, o fit treina
   results = sm.OLS(Y,X_sm).fit()

   print(results.summary())
   print(results.predict(X_sm))
   #foco (rquare, fstatistic, prob do fstatistic,coluna coef dos coeficientes e coluna P>[t] que mostra os pvalues para cada coeficiente)



def reg_tree():
   import numpy as np
   import pandas as pd
   from matplotlib import pyplot as plt
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.tree import export_graphviz
   from matplotlib.colors import ListedColormap


   np.random.seed(42)
   df= pd.DataFrame({'feature': np.sort(np.random.uniform(-10,10,200)),'var_aux':np.random.normal(0,70,200)})
   df['target'] = df['feature']**3 + df['var_aux']
   #df['feature_2'] = np.random.lognormal(0,1,200)
   df = df.drop(columns=['var_aux']).round(2)
   x = df.drop(columns=['target']).values
   y = df['target'].values


   scatter_plot = df.plot.scatter(x='feature',y='target')
   scatter_plot.plot()
   plt.show()


   reg_1 = DecisionTreeRegressor(max_depth=1)#min_samples_leaf = 20 - minimo de observações em uma folha
   reg_2 = DecisionTreeRegressor(max_depth=2)#min_samples_split = 40 - minimo de observações para que seja feito oprocesso de divisão
   reg_3 = DecisionTreeRegressor(max_depth=5)#max_features = 2 - numero maximo de variaveis a serem consideradas no processo de divisão

   reg_1.fit(x,y)
   reg_2.fit(x,y)
   reg_3.fit(x,y)

   y_pred1 = reg_1.predict(x)
   y_pred2 = reg_2.predict(x)
   y_pred3 = reg_3.predict(x)


   plt.figure(figsize=(20,8))

   plt.subplot(1,3,1)
   plt.scatter(x,y,s=20,edgecolors='black',c='darkorange')
   plt.plot(x,y_pred1,color='cornflowerblue',linewidth=2)
   plt.title('max_depth = 1')

   plt.subplot(1,3,2)
   plt.scatter(x,y,s=20,edgecolors='black',c='darkorange')
   plt.plot(x,y_pred2,color='cornflowerblue',linewidth=2)
   plt.title('max_depth = 2')

   plt.subplot(1,3,3)
   plt.scatter(x,y,s=20,edgecolors='black',c='darkorange')
   plt.plot(x,y_pred3,color='cornflowerblue',linewidth=2)
   plt.title('max_depth = 5')

   plt.subtitle('decision tree regression')
   plt.show()


   export_graphviz(reg_1,out_file='tree.dot',feature_names=['x'])
   #www.webgraphviz.com


   est_folha_media = df.query('feature < -6.345').mean().round(2)[1]
   est_folha_mediana = df.query('feature < -6.345').median()[1]
   print('a média e mediana dos resultados com x observações é')
   print(est_folha_media)
   print(est_folha_mediana)


def reg_tree_realLife():
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.model_selection import GridSearchCV
   #from sklearn.model_selection import cross_val_score
   from sklearn import metrics
   from sklearn.metrics import make_scorer

   #modelo basico

   names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
   df = pd.read_csv(r'.data\housing.data',delim_whitespace=True,names=names)
   x = df.drop(['MEDV'],axis=1)
   y = df['MEDV']
   x_train, y_train,x_test, y_test = train_test_split(x,y,random_state=13)

   reg = DecisionTreeRegressor()
   param_grid = {'criterion':['mse','mae'],
                  'min_samples_split' : [5,10,20],
                  'max_depth' : [2,4,6,8],
                  'min_samples_leaf' : [8,12,15]}

   score = scoring=['neg_mean_squared_error','r2']

   #validar isso aqui
   grid_cv_RegTree = GridSearchCV(reg,param_grid,scoring=score,cv=8,verbose=2,refit='r2')

   grid_cv_RegTree.fit(x_train,y_train)
   print("R2::{}".format(grid_cv_RegTree.best_score_))
   print("hiperparametro melhor::{}".format(grid_cv_RegTree.best_params_))

   pd.DataFrame(data = grid_cv_RegTree.cv_results_).head(3)

   best_model = grid_cv_RegTree.best_estimator_

   y_pred = best_model.predict(x_test)

   r2_score = best_model.score(x_test,y_test)

   print("R2:{:.3f}".format(r2_score))
   print("MSE:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))


def custo_complex_prune():

   import numpy as np
   import pandas as pd
   from matplotlib import pyplot as plt
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.tree import export_graphviz
   from matplotlib.colors import ListedColormap

   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_moons

   #tambem é possivel avaliar com mse mae e half position deviance

   x,y = make_moons(n_samples=10000,noise=0.4,random_state=42)

   x_train, y_train,x_test, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

   reg_ccp = DecisionTreeRegressor()
   reg_ccp_fit = reg_ccp.fit(x_train,y_train)
   path = reg_ccp_fit.cost_complexity_pruning_path(x_train,y_train)

   ccp_alphas = path.ccp_alphas
   ccp_alphas = ccp_alphas[100:130]

   clf_dts = []

   for ccp_alpha in ccp_alphas:
      clf_dt = DecisionTreeRegressor(random_state=0,ccp_alpha=ccp_alpha)
      clf_dt.fit(x_train,y_train)
      clf_dts.append(clf_dt)

   score = [clf_dt.score(x_train,y_train) for clf_dt in clf_dts]
   score_test = [clf_dt.score(x_test,y_test) for clf_dt in clf_dts]

   fig,ax = plt.subplots()
   ax.set_xlabel('ccp_alphs')
   ax.set_ylabel('score')
   ax.set_title('Escolha do ccp_alpha')
   ax.plot(ccp_alphas,score,marker='o',label='train',drawstyle='step-post')
   ax.plot(ccp_alphas,score_test,marker='o',label='train',drawstyle='step-post')
   ax.legend()
   plt.show()



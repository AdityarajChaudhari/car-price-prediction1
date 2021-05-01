import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'D:\Datasets\car data.csv')

#dropping catergorical feature
data.drop('Car_Name',axis=1,inplace=True)

#dropping CNG
data.drop(data.index[[18,35]],inplace=True)

#Dropping third hand vehicle
data.drop(85,inplace=True,axis=0)

#FEATURE ENGINEERING

data['Fuel_Type'] = np.where(data['Fuel_Type'] == 'Petrol',1,0)
#seller
data['Seller_Type'] = np.where(data['Seller_Type'] == 'Individual',0,1)

#Transmission

data['Transmission'] = np.where(data['Transmission'] == 'Manual',0,1)

#Checking Multicollinearity and deleting the feature with high vif score



#dropping Year as it has multicollinearity
indep_features = data.drop('Selling_Price',axis=1)
#sbn.heatmap(indep_features.corr(),cmap='rocket',annot=True,linewidths=1,linecolor='black')
#plt.show()
indep_features.drop('Year',axis=1,inplace=True)

#Performing preprocessing


x = indep_features
y = data['Selling_Price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=75)



from lazypredict.Supervised import LazyRegressor
lazy_reg = LazyRegressor()
models,scores = lazy_reg.fit(x_train,x_test,y_train,y_test)
print(models.head(6))
print()

#Gradient Boosting
print()
print("-----Gradient Boosting Regressor-----")
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train,y_train)

print("Training Score :- ",gbr.score(x_train,y_train))
print("Testing Score :- ",gbr.score(x_test,y_test))
print()
print("Model is overfitting so let's do Hyperparameter Tuning")
print()
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'learning_rate' : [float(x) for x in np.linspace(0.1,1,10)],
    'max_depth' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'max_features' : ['auto','sqrt'],
    'min_samples_leaf' : [i for i in range(1,50,1)],
    'min_samples_split' : [i for i in range(2,50,1)],
    'n_estimators' : [int(i) for i in np.linspace(100,3000,30)],
    'alpha' : [a for a in np.linspace(0.1,0.99,20)],
    'loss' : ['ls','lad', 'huber', 'quantile']
}
rscv_gbr = RandomizedSearchCV(estimator=gbr,param_distributions=param_grid,n_iter=100,cv=5,n_jobs=-1,verbose=True)
rscv_gbr.fit(x_train,y_train)

gbr_best=rscv_gbr.best_estimator_

y_pred = gbr_best.predict(x_test)
from sklearn import metrics
print('MAE:', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('MSE:', round(metrics.mean_squared_error(y_test, y_pred),4))

print()
print("After Hyperparameter Tuning :- ")
print("Training Score :- ",gbr_best.score(x_train,y_train))
print("Testing Score :- ",gbr_best.score(x_test,y_test))
print("Now it looks Generalized model !!!")

print()
print()

print("-----Decision Tree Regressor-----")
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print("Training Score :- ",dtr.score(x_test,y_test))
print("Testing Score :- ",dtr.score(x_train,y_train))
print()
print("Model is overfitting so let's do Hyperparameter Tuning")
print()
grid_params = {
    'criterion' : ['mse','friedman_mse'],
    'max_depth' : [i for i in range(1,50,1)],
    'min_samples_leaf' : [i for i in range(1,50,1)],
    'min_samples_split' : [i for i in range(2,50,1)],
    'max_features' : ['auto','sqrt','log2',None],
    'ccp_alpha' : [float(i) for i in np.linspace(0,1,100)]
}
randomcv_dt = RandomizedSearchCV(dtr,param_distributions=grid_params,cv=5,n_iter=100,verbose=True,n_jobs=-1)
randomcv_dt.fit(x_train,y_train)
randomcv_dt_best = randomcv_dt.best_estimator_
y_pred = randomcv_dt_best.predict(x_test)

print('MAE:', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('MSE:', round(metrics.mean_squared_error(y_test, y_pred),4))
print()
print("After HyperparameterTuning :- ")
print("Training Score :- ",round(randomcv_dt_best.score(x_train,y_train),4))
print("Testing Score :- ",round(randomcv_dt_best.score(x_test,y_test),4))
print("Now it looks generalized model !!!")

print()
print()

print("-----Random Forest Regressor-----")
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
print("Training Score :- ",rfr.score(x_train,y_train))
print("Testing Score :- ",rfr.score(x_test,y_test))
print()
print("Model is overfitting so let's do Hyperparameter Tuning")
print()
params = {
    'criterion':['mse','friedman_mse'],
    'max_features' : ['sqrt','auto','log2',None],
    'max_depth' : [i for i in range(1,50,1)],
    'n_estimators' : [i for i in range(100,3000,30)],
    'min_samples_split' : [i for i in range(2,50,1)],
    'min_samples_leaf' : [i for i in range(1,50,1)]
}
randomcv_rfr = RandomizedSearchCV(rfr,param_distributions=params,n_jobs=-1,cv=5,n_iter=100,verbose=True)
randomcv_rfr.fit(x_train,y_train)
randomcv_rfr_best = randomcv_rfr.best_estimator_
print()
y_pred = randomcv_rfr_best.predict(x_test)

print('MAE:', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('MSE:', round(metrics.mean_squared_error(y_test, y_pred),4))
print()
print("After Hyperparameter Tuning :- ")
print("Training Score :- ",round(randomcv_rfr_best.score(x_train,y_train),4))
print("Testing Score :- ",round(randomcv_rfr_best.score(x_test,y_test),4))
print()
print("Now the model looks more Generalized !!!")

print()
print()

print("----- Bagging Regressor -----")
from sklearn.ensemble import BaggingRegressor
bgr = BaggingRegressor()
bgr.fit(x_train,y_train)
print("Training Score :- ",bgr.score(x_train,y_train))
print("Testing Score :- ",bgr.score(x_test,y_test))
print()
print("Model is overfitting so let's do Hyperparameter Tuning")
print()

print("Using various models in Bagging")
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.svm import SVR
svr = SVR()

params = {
    'base_estimator' : [svr,lr,knr],
    'n_estimators' : [i for i in range(10,300,10)]
}

randomcv_bgr = RandomizedSearchCV(bgr,param_distributions=params,n_jobs=-1,verbose=True,cv=5,n_iter=100)
randomcv_bgr.fit(x_train,y_train)
print()
randomcv_bgr_best = randomcv_bgr.best_estimator_
print(randomcv_bgr_best)
print()
print("So Knn Regressor used in bagging")
print()
y_pred = randomcv_bgr_best.predict(x_test)
print('MAE:', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('MSE:', round(metrics.mean_squared_error(y_test, y_pred),4))
print()
print("After Hyperparameter Tuning :- ")
print("Training Score :- ",round(randomcv_bgr_best.score(x_train,y_train),4))
print("Testing Score :- ",round(randomcv_bgr_best.score(x_test,y_test),4))
print()
print("Now the model looks more Generalized !!!")

print()
print()

print("-----XGBoost Regressor-----")
from xgboost import XGBRegressor
xgbr = XGBRegressor()
xgbr.fit(x_train,y_train)
print("Training Score :- ",xgbr.score(x_train,y_train))
print("Testing Score :- ",xgbr.score(x_test,y_test))
print()
print("Model is overfitting so let's do Hyperparameter Tuning")
print()
param = {

    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7, 8, 9],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 1000],
    "subsample": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

}
randomcv_xgbr = RandomizedSearchCV(xgbr,param_distributions=param,cv=5,n_iter=100,verbose=True,n_jobs=-1)
randomcv_xgbr.fit(x_train,y_train)
randomcv_xgbr_best = randomcv_xgbr.best_estimator_
print()
y_pred = randomcv_xgbr_best.predict(x_test)

print('MAE:', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('MSE:', round(metrics.mean_squared_error(y_test, y_pred),4))
print()

print("After Hyperparameter Tuning :- ")
print("Training Score :- ",round(randomcv_xgbr_best.score(x_train,y_train),4))
print("Testing Score :- ",round(randomcv_xgbr_best.score(x_test,y_test),4))
print()
print("Now the model looks more Generalized !!!")

print()
print()
print("So we have trained five different models lets compare them!!")

models = {
    'Algorithm' : ['GradientBoostingRegressor','DecisionTreeRegressor','RandomForestRegressor','BaggingRegressor','XGBoostRegressor'],
    'Train_Acc' : [round(gbr_best.score(x_train,y_train),4),round(randomcv_dt_best.score(x_train,y_train),4),round(randomcv_rfr_best.score(x_train,y_train),4),round(randomcv_bgr_best.score(x_train,y_train),4),round(randomcv_xgbr_best.score(x_train,y_train),4)],
    'Test_Acc'  : [round(gbr_best.score(x_test,y_test),4),round(randomcv_dt_best.score(x_test,y_test),4),round(randomcv_rfr_best.score(x_test,y_test),4),round(randomcv_bgr_best.score(x_test,y_test),4),round(randomcv_xgbr_best.score(x_test,y_test),4)]
}
print(pd.DataFrame(models))
print()

best_model_test = pd.DataFrame(models).sort_values(by='Test_Acc',ascending=False)
best_model_train = pd.DataFrame(models).sort_values(by='Train_Acc',ascending=False)

print()
print("Model with best testing accuracy :- ")
print(best_model_test)

print()
print("Model with best training accuracy :- ")
print(best_model_train)

plt.figure(figsize=(10,7))
plt.subplot(2,2,1)
sbn.barplot(x='Algorithm',y='Test_Acc',data=best_model_test,palette='Set1',edgecolor='black')
plt.xlabel('Algorithm', fontsize=15)
plt.ylabel('Test_Acc', fontsize=15)
plt.xticks(rotation=90)
plt.subplot(2,2,2)
sbn.barplot(x='Algorithm',y='Train_Acc',data=best_model_train,palette='Set2',edgecolor='black')
plt.xlabel('Algorithm', fontsize=15)
plt.ylabel('Train_Acc', fontsize=15)
plt.xticks(rotation=90)
plt.suptitle('ACCURACY OF MODELS',fontsize=25)
plt.show()

import pickle

pickle.dump(randomcv_xgbr_best,open('model.pkl','wb'),protocol=2)
model = pickle.load(open('model.pkl','rb'))
print(model.get_booster().feature_names)




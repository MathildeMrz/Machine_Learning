import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


plt.style.use('bmh')
df = pd.read_csv('insurance.csv')

#Check empty values
print(df.isnull().sum())
df.head()

#Analyse : On voit qu'aucune donnée n'est manquante dans le jeu de données.
#          Il nous faudra voir si toutes les données sont pertinentes, mais pour le moment, nous n'avons pas à en supprimer.

#Box plots
plt.figure("Smoker vs charges")
sns.boxplot(x='smoker', y='charges', data=df)
plt.savefig('box_plot_smoker_charges.png')
plt.figure("Region vs charges")
sns.boxplot(x='region', y='charges', data=df)
plt.savefig('box_plot_region_charges.png')
plt.figure("Sex vs charges")
sns.boxplot(x='sex', y='charges', data=df)
plt.savefig('box_plot_sex_charges.png')

#Analyse : On peut remarquer directement grâce au boxPlot fumeurs/charges que les fumeurs ont une médiane et des valeurs min/max beaucoup plus élevées que les non-fumeurs

#Afin de pouvoir analyser des données qui sont pour le moment textuelles, nous allons devoir les transformer en int.
#Il s'agit de "sex" et smoker => 0/1 et region => 0/1/2...?????

#Parse string to int
df = pd.get_dummies(df, columns=['sex', 'smoker'], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=['region'], dtype=int)

#Histograms
df.hist(bins=40, figsize=(15, 10))

#Scatter Matrix plot
sns.pairplot(df)

#Scatter plots of charges vs all (prettier to see than the matrix)
for i in range(0, len(df.columns), 5):
    sns.pairplot(data=df, x_vars=df.columns[i:i + 5], y_vars=['charges'])

#Check correlations => create a map to see if variables are very correlated
dfcorr = df.corr()['charges'][:-1]
golden_features_list = dfcorr.sort_values(ascending=False)
print(
    "\nThere are {} lowly correlated values and 1 highly correlated value with Charges:\n{}"
    .format(len(golden_features_list) - 2, golden_features_list))

corr = df.drop('charges',
               axis=1).corr()  # We already examined Charges correlations
plt.figure(figsize=(12, 10))
plt.savefig('correlation_with_charges.png')

sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)],
            cmap='viridis',
            vmax=1.0,
            vmin=-1.0,
            linewidths=0.1,
            annot=True,
            annot_kws={"size": 8},
            square=True)

###################################Explore the dataset###################################
#regression lin peut être très sensible aux points extremes : si un point est très loin, cela fera bouger la droite
#MKernel : bcp moins d'influence des variables extremes

#Train Test split

X = df.drop(['charges'], axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

#SVM Model x Confusion Matrix

#X_train = X_train.astype(int)
#y_train = y_train.astype(int)
#X_test = X_test.astype(int)
#y_test = y_test.astype(int)
#clf = svm.SVC().fit(X_train, y_train)
#acc_train = clf.score(X_train, y_train)
#y_pred = clf.predict(X_test)
#acc_test = accuracy_score(y_pred, y_test)
#print('accuracy_train: ', round(acc_train * 100, 2), '%')
#print('accuracy_test: ', round(acc_test * 100, 2), '%')
#print('precision_score: ', round(precision_score(y_test, y_pred), 2))
#print('recall_score: ', round(recall_score(y_test, y_pred), 2))
#print('f1_score: ', round(f1_score(y_test, y_pred), 2))
#cm = confusion_matrix(y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot()

##################################################Linear Regression Model##################################################"
#Model building

print("\n\nDébut entrainement régression linéaire")
model = LinearRegression().fit(X_train, y_train)
r_sq = model.score(X_train, y_train)
print(f"coefficient of determination : {r_sq}")
for idx, col_name in enumerate(X_train.columns):
    print("Linear regression model coefficient for {} is {}".format(
        col_name, model.coef_[idx]))

intercept = model.intercept_
print("The intercept for linear regression model is {}".format(intercept))

# sklearn regression module
y_pred_sk_linearRegression = model.predict(X_test)

#Model validation
# Check for Linearity
f = plt.figure(figsize=(14, 5))
ax = f.add_subplot(121)
sns.scatterplot(x=y_test, y=y_pred_sk_linearRegression, ax=ax, color='r')
ax.set_title(
    'Check for Linearity:\n Actual (x : charges observées) Vs Predicted value (y : charges prédites)'
)

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.histplot((y_test - y_pred_sk_linearRegression), ax=ax, color='b')
ax.axvline((y_test - y_pred_sk_linearRegression).mean(),
           color='k',
           linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual error')

# Check for Multivariate Normality
# Quantile-Quantile plot
f, ax = plt.subplots(1, 2, figsize=(14, 6))
import scipy as sp

_, (_, _, r) = sp.stats.probplot((y_test - y_pred_sk_linearRegression),
                                 fit=True,
                                 plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')
plt.savefig('linear_regression_model_multivariate_normality.png')

#Check for Homoscedasticity
sns.scatterplot(y=(y_test - y_pred_sk_linearRegression),
                x=y_pred_sk_linearRegression,
                ax=ax[1],
                color='r')
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted')
plt.savefig('linear_regression_modelhomoscedasticity.png')

##################################################Ridge Regression Model##################################################"
print("\n\nDébut entrainement Ridge régression")

ridgeReg = Ridge(alpha=10).fit(X_train, y_train)

r_sq = ridgeReg.score(X_train, y_train)
print(f"coefficient of determination for Ridge regression model: {r_sq}")
for idx, col_name in enumerate(X_train.columns):
    print("Ridge model coefficients for {} is {}".format(
        col_name, ridgeReg.coef_[idx]))

intercept = ridgeReg.intercept_
print("The intercept for Ridge model is {}".format(intercept))

y_pred_sk_rigdeReg = ridgeReg.predict(X_test)

#Model validation
# Check for Linearity
f = plt.figure(figsize=(14, 5))
ax = f.add_subplot(121)
sns.scatterplot(x=y_test, y=y_pred_sk_rigdeReg, ax=ax, color='r')
ax.set_title(
    'Ridge Regression : Check for Linearity:\n Actual (x : charges observées) Vs Predicted value (y : charges prédites)'
)
plt.savefig('ridge_regression_model_linearity_observed_predicted.png')


# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.histplot((y_test - y_pred_sk_rigdeReg), ax=ax, color='b')
ax.axvline((y_test - y_pred_sk_rigdeReg).mean(), color='k', linestyle='--')
ax.set_title(
    'Ridge Regression : Check for Residual normality & mean: \n Residual error'
)

# Check for Multivariate Normality
# Quantile-Quantile plot
f, ax = plt.subplots(1, 2, figsize=(14, 6))
import scipy as sp

_, (_, _, r) = sp.stats.probplot((y_test - y_pred_sk_rigdeReg),
                                 fit=True,
                                 plot=ax[0])
ax[0].set_title(
    'Ridge Regression : Check for Multivariate Normality: \nQ-Q Plot')
plt.savefig('ridge_regression_model_multivariate_normality.png')

#Check for Homoscedasticity
sns.scatterplot(y=(y_test - y_pred_sk_rigdeReg),
                x=y_pred_sk_rigdeReg,
                ax=ax[1],
                color='r')
ax[1].set_title(
    'Ridge Regression : Check for Homoscedasticity: \nResidual Vs Predicted')

plt.savefig('ridge_regression_modelhomoscedasticity.png')


##################################################Kernel Ridge Regression Model##################################################

'''print("\n\nDébut entrainement Kernel Ridge Régression")

#Test meilleurs paramètres
k = 5
kr_param_grid = {
    "alpha": [100, 50, 25, 10, 5, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "kernel": ['linear', 'rbf', 'poly'],
    "gamma": [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
}

kr = GridSearchCV(KernelRidge(), cv=k, param_grid=kr_param_grid)
kr.fit(X_train, np.asarray(y_train).ravel())
train_rmse = sqrt(mean_squared_error(y_train, kr.predict(X_train)))
test_rmse = sqrt(mean_squared_error(y_test, kr.predict(X_test)))
print(train_rmse)
print(test_rmse)
print(kr.best_params_)

kf = KFold(n_splits=k, shuffle=False, random_state=42)
reg = KernelRidge(alpha=kr.best_params_['alpha'],
                  kernel=kr.best_params_['kernel'],
                  gamma=kr.best_params_['gamma'])
i = 1
validation = pd.DataFrame()
results = pd.DataFrame()
resid = pd.DataFrame()

cols = df.columns
train_y = pd.DataFrame(data=df['Charges'], columns=['Charges'])
train_x = pd.DataFrame(data=df[cols], columns=cols)

for train_index, test_index in kf.split(train_x):

    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = np.asarray(train_x)[train_index], np.asarray(
        train_x)[test_index]
    y_train, y_test = np.asarray(train_y)[train_index], np.asarray(
        train_y)[test_index]

    reg.fit(X_train, np.asarray(y_train).ravel())
    #     print(reg.predict(test_x))
    temp_val = pd.DataFrame(data=[[
        sqrt(mean_squared_error(y_train, reg.predict(X_train))),
        sqrt(mean_squared_error(y_test, reg.predict(X_test)))
    ]],
                            columns=['train', 'test'])
    validation = pd.concat([validation, temp_val], axis=0)

    temp_resid = pd.DataFrame(data=reg.predict(train_x),
                              columns=['res{0}'.format(i)])
    resid = pd.concat([resid, temp_resid], axis=1)

    temp_submission = pd.DataFrame(data=reg.predict(test_x),
                                   columns=['res{0}'.format(i)])
    results = pd.concat([results, temp_submission], axis=1)
    i += 1

validation.index = range(1, k + 1)
plt.plot(validation['train'])
plt.plot(validation['test'])
plt.title('RMSE')

print(validation['train'].mean())
print(validation['test'].mean())'''

#######################################Report generation#######################################

#à laisser tout en bas
#plt.show()
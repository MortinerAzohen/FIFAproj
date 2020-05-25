# Data manipulation
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import collections
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import scatter_matrix

# Machine Learning Algorithms
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVR

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def parseValue(x):
    x = str(x).replace('€', '')
    if('M' in str(x)):
        x = str(x).replace('M', '')
        x = float(x) * 1000000
    elif('K' in str(x)):
        x = str(x).replace('K', '')
        x = float(x) * 1000
    return float(x)

def parsePosition(x,uniq):
    x = str(x).replace(x,str(uniq.index(x)))
    x = int(x)
    return x

#funkcja do podmiany pozycji na cyfry
fifa_raw_dataset = pd.read_csv('input/FutBinCards19.csv')
seen = set()
uniq = [x for x in fifa_raw_dataset['Position'] if x not in seen and not seen.add(x)]
fifa_raw_dataset['Position'] = fifa_raw_dataset['Position'].apply(parsePosition,args=[uniq])
print(fifa_raw_dataset['Position'])
#wybrane atrybuty to przewidywania wartości pilakrza
features = ['Price','WeakFoot','SkillsMoves','Pace','Shooting','Passing','Dribbling','Defending','Phyiscality','Position']
fifa_dataset = fifa_raw_dataset[[*features]]
#podmianka wartosci pilkarza z literkami na cyfry
fifa_dataset['Price'] = fifa_dataset['Price'].apply(parseValue)

plt.figure(figsize=(10, 6))
sb.countplot(fifa_dataset["Price"], palette="muted")
plt.title('Rozklad wartosci pilkarzy')
plt.show()
print(fifa_dataset["Price"].value_counts())


plt.figure( figsize=(12, 6))
plt.title('Korelacja cech')
sb.heatmap(fifa_dataset.corr(), annot=True)
plt.show()

fifa_dataset.hist(bins=50, figsize=(20,15))
plt.title('Histogram cech')
plt.show()

X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.isnull().any())
print(np.isnan(X_train).any())

# regresja liniowa
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
lin_reg = LinearRegression(normalize=True, n_jobs=-1)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print('Coefficients: \n', lin_reg.coef_)
print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#drzwko decyzyjne
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#random forest

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)

y_pred = forest_reg.predict(X_test)
print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#extra trees reg

extra_reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
extra_reg.fit(X_train, y_train)
y_pred = extra_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

#gradient boost regressor

grad_reg = GradientBoostingRegressor(random_state=0)
grad_reg.fit(X_train,y_train)
y_pred = grad_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

#cross-validation
scores = cross_val_score(tree_reg, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

scores = cross_val_score(lin_reg, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

scores = cross_val_score(forest_reg, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
scores = cross_val_score(grad_reg, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
scores = cross_val_score(extra_reg, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
#grid_search - forest_reg
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, final_predictions)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, final_predictions))
#grid search extratrees regressor
grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, final_predictions)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, final_predictions))

#grid search gradient regressor
param_grid = {
    'loss': ['ls','lad'],
    'learning_rate': [0.1,0.5,0.9],
    'criterion': ['mse'],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [2, 4,6],
    'n_estimators': [100, 200]
}
grid_search = GridSearchCV(grad_reg, param_grid, cv = 3, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, final_predictions)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, final_predictions))

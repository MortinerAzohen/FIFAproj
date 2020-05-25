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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# For Missing Values
from sklearn.impute import SimpleImputer
# funkcja do podmiany liter przy wartościach na cyfry
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


#wykres zliczajacy  ile pilkarzy jest dla danej wartosci
plt.figure(1, figsize=(18, 7))
sb.set(style="whitegrid")
sb.countplot( x= 'Price', data=fifa_dataset)
plt.title('Value distribution of all players')
plt.show()

#histogram wybranych atrybutów
fifa_dataset.hist(bins=50, figsize=(20,15))
plt.show()

#szukanie korelacji cech
plt.title('Korelacja cech')
corr_matrix = fifa_dataset.corr()
corr_matrix["Price"].sort_values(ascending=False)

#pokazanie zaleznosci ceny od danej cechy
attributes = ['Price','WeakFoot','SkillsMoves','Pace','Shooting','Passing','Dribbling','Defending','Phyiscality','Position']
scatter_matrix(fifa_dataset[attributes], figsize=(12, 8))
plt.show()

#identyczny wynik z kazdego odpalenia programu
np.random.seed(42)

#Podzial na trenignowy i testowy zestaw
train_set, test_set = train_test_split(fifa_dataset, test_size=0.2, random_state=42)

print('Train',' ','Test')
print(len(train_set),'+',len(test_set),'=',len(train_set)+len(test_set))

# Machine learning algorythm

l = list(train_set['Price'] == 0)
print('Zeros in output label: ',len([v for v in l if v==True] ))
print('\nNaN values in following features:')
print(train_set.isnull().any())

train_set = train_set.replace(0, pd.np.nan)
#zastapienie NaN mediana
imputer = SimpleImputer(strategy="median")

imputer.fit(train_set)

tf = imputer.transform(train_set)

fifa_dataset_tf = pd.DataFrame(tf, columns=fifa_dataset.columns)

#test czy sa jeszcze puste wartosci
print(fifa_dataset_tf.isnull().any())

fifa_dataset_features = fifa_dataset_tf.drop("Price", axis=1) # drop labels for training set
fifa_dataset_labels = fifa_dataset_tf["Price"].copy()

#regresja liniowa
lin_reg = LinearRegression()
lin_reg.fit(fifa_dataset_features, fifa_dataset_labels)

fifa_dataset_predictions = lin_reg.predict(fifa_dataset_features)
lin_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)
lin_rmse = np.sqrt(lin_mse)
score = r2_score(fifa_dataset_labels, fifa_dataset_predictions)
print('Accuracy linear regretion:',format(score*100,'.2f'),'%')

#drzewko decyzyjne
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(fifa_dataset_features, fifa_dataset_labels)

fifa_dataset_predictions = tree_reg.predict(fifa_dataset_features)
tree_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)
tree_rmse = np.sqrt(tree_mse)

score = r2_score(fifa_dataset_labels, fifa_dataset_predictions)
print('Accuracy tree_reg:',format(score*100,'.2f'),'%')

#random forest
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(fifa_dataset_features, fifa_dataset_labels)

fifa_dataset_predictions = forest_reg.predict(fifa_dataset_features)
forest_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)
forest_rmse = np.sqrt(forest_mse)

score = r2_score(fifa_dataset_labels, fifa_dataset_predictions)
print('Accuracy random forest:',format(score*100,'.2f'),'%')

#Evaluation using Cross-Validation
scores = cross_val_score(tree_reg, fifa_dataset_features, fifa_dataset_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, fifa_dataset_features, fifa_dataset_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

forest_scores = cross_val_score(forest_reg, fifa_dataset_features, fifa_dataset_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#fine-tune model grid search
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(fifa_dataset_features, fifa_dataset_labels)
print(grid_search.best_params_)
print("--------------")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

test_set = test_set.replace(0, np.nan)
tf = imputer.transform(test_set)
fifa_dataset_tf = pd.DataFrame(tf, columns=fifa_dataset.columns)

fifa_dataset_features = fifa_dataset_tf.drop("Price", axis=1)
fifa_dataset_labels = fifa_dataset_tf["Price"].copy()
print(grid_search.best_estimator_)
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(fifa_dataset_features)

final_mse = mean_squared_error(fifa_dataset_labels, final_predictions)
final_rmse = np.sqrt(final_mse)

final_model_score = r2_score(fifa_dataset_labels, final_predictions)
print('Accuracy final:',format(final_model_score*100,'.2f'),'%')

# Machine Learning Algorithms
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from Data_operations.Prepare_data import DataOperations
# Model Selection and Evaluation
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sb
# Performance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


fifa_data = DataOperations()
fifa_dataset_all = fifa_data.import_data()
fifa_dataset = fifa_dataset_all[1]

X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
#grid_search - forest_reg
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5,
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
grid_search = GridSearchCV(ExtraTreesRegressor(), param_grid, cv=5,
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
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv = 3,scoring='neg_mean_squared_error',
                           )

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, final_predictions)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, final_predictions))

# Machine Learning Algorithms
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from Data_operations.Prepare_data import DataOperations
from sklearn.model_selection import  KFold, cross_validate


def display_scores(scores):
    rmse_scores = np.sqrt(-scores["test_neg_mean_squared_error"])
    print('R2_score : %.2f' %scores["test_r2"].mean())
    print('Mean Squared Error: %.2f' %rmse_scores.mean())
    print('Standard deviation: %.2f' %rmse_scores.std())


fifa_data = DataOperations()
fifa_dataset = fifa_data.import_data()


X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']

lin_reg = LinearRegression(normalize=True, n_jobs=-1)

tree_reg = DecisionTreeRegressor()

forest_reg = RandomForestRegressor(n_estimators=100 )

extra_reg = ExtraTreesRegressor(n_estimators=100)

grad_reg = GradientBoostingRegressor()

#cross-validation
print("\nDecision Tree Regresor scores:")
scores = cross_validate(tree_reg, X, y,scoring=('r2','neg_mean_squared_error'), cv=KFold(n_splits = 10, shuffle = True, random_state = 42))
display_scores(scores)
print("\nLinear Regresion scores:")
scores = cross_validate(lin_reg, X, y,scoring=('r2','neg_mean_squared_error'), cv=KFold(n_splits = 10, shuffle = True, random_state = 42))
display_scores(scores)
print("\nRandom Forest Regressor")
scores = cross_validate(forest_reg, X, y,scoring=('r2','neg_mean_squared_error'), cv=KFold(n_splits = 10, shuffle = True, random_state = 42))
display_scores(scores)
print("\nGradient Boosting Regressor")
scores = cross_validate(grad_reg, X, y,scoring=('r2','neg_mean_squared_error'), cv=KFold(n_splits = 10, shuffle = True, random_state = 42))
display_scores(scores)
print("\nExtra Trees Regressor")
scores = cross_validate(extra_reg, X, y,scoring=('r2','neg_mean_squared_error'), cv=KFold(n_splits = 10, shuffle = True, random_state = 42))
display_scores(scores)
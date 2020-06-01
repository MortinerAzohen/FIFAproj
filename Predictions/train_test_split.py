
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from Data_operations.Prepare_data import DataOperations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


fifa_data = DataOperations()
fifa_dataset = fifa_data.import_data()

X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# regresja liniowa
print("\n regresja liniowa test/train set")
lin_reg = LinearRegression(normalize=True, n_jobs=-1)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#drzwko decyzyjne
print("\n drzewko decyzyjne test/train set")
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#random forest
print("\n random forest test/train set")
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)

y_pred = forest_reg.predict(X_test)
print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
#extra trees reg
print("\n extra trees reg test/train set")
extra_reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
extra_reg.fit(X_train, y_train)
y_pred = extra_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

#gradient boost regressor
print("\n gradient boost regresor test/train set")
grad_reg = GradientBoostingRegressor(random_state=0)
grad_reg.fit(X_train,y_train)
y_pred = grad_reg.predict(X_test)

print('Mean squared error: %.2f'
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
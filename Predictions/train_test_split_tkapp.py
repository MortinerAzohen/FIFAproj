import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from Data_operations.Prepare_data import DataOperations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class TrainTestTkkApp:
    def __init__(self,fifa_data = DataOperations()):
        self.fifa_dataset = fifa_data.import_data()
        self.X = self.fifa_dataset.drop('Price', axis=1)
        self.y = self.fifa_dataset['Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def linearReg(self):
        array = []
        lin_reg = LinearRegression(normalize=True, n_jobs=-1)
        lin_reg.fit(self.X_train, self.y_train)
        y_pred = lin_reg.predict(self.X_test)
        array.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        array.append(r2_score(self.y_test, y_pred))
        return array


    def decisionTree(self):
        array = []
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.X_train, self.y_train)
        y_pred = tree_reg.predict(self.X_test)
        array.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        array.append(r2_score(self.y_test, y_pred))
        return array


    def randomForest(self):
        array = []
        forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        forest_reg.fit(self.X_train, self.y_train)
        y_pred = forest_reg.predict(self.X_test)
        array.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        array.append(r2_score(self.y_test, y_pred))
        return array


    def extraTree(self):
        array = []
        extra_reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
        extra_reg.fit(self.X_train, self.y_train)
        y_pred = extra_reg.predict(self.X_test)
        array.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        array.append(r2_score(self.y_test, y_pred))
        return array


    def gradientBoost(self):
        array=[]
        grad_reg = GradientBoostingRegressor(random_state=0)
        grad_reg.fit(self.X_train, self.y_train)
        y_pred = grad_reg.predict(self.X_test)
        array.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        array.append(r2_score(self.y_test, y_pred))
        return array


    def  checkNewFifaPlayer(self,x_user):
        New_X = pd.DataFrame(x_user,columns = ['WeakFoot', 'SkillsMoves', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending',
                    'Phyiscality', 'Position','Country','Club','WorkRate'])
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.X_train, self.y_train)
        y_pred = tree_reg.predict(New_X)
        print(y_pred)





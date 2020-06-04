
# Fifa game cards price predicting. 

## 1. IDE used for creating program - PyCharm IDE.
Program was written in python language. 
Libraries used in project: 
numpy
sklearn
seaborn
matplotlib
tkinter
## 2. Directory input contains data used for machine learning algorythms.
## 3. Data_operations directory contains scrips for reading data from csv file and analysing data.
## 4. Predictions directory contains machine learning scrypts. 
- train_test_split.py use simple train_test_split for data. 20% data is for test data and 80% is for train algorythms. Result of script is 
comparison of five regression algorythms. The result is given as r2 score and mean squared error. 
- cross_validation_results.py use cross_validate function for spliting data by KFold function. Result is given as mean of r2 score and mean squared error results.
- grid_search_results.py finds the best parameters for 3 diffrent machine learnign methods. This script returns scores of those methods using the best founded estimator.
- train_test_split_tkapp.py contains functions used in GUI app. 
## 5. In program directory you can find GUI app. 
Top 5 buttons in that program show scores of ML algorythms. This app allows you to create your own fifa card and it shows you potential price of that card on the game market.






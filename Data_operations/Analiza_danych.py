from sklearn import preprocessing
from Data_operations.Prepare_data import DataOperations
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from pandas.plotting import scatter_matrix

fifa_data = DataOperations()
fifa_dataset = fifa_data.import_data()

plt.figure(figsize=(10, 6))
sb.countplot(fifa_dataset["Price"], palette="muted")
plt.title('Rozklad wartosci pilkarzy')
plt.show()

plt.figure( figsize=(12, 6))
plt.title('Korelacja cech')
sb.heatmap(fifa_dataset.corr(), annot=True)
plt.show()

fifa_dataset.hist(bins=50, figsize=(20,15))
plt.title('Histogram cech')
plt.show()

attributes = ['WeakFoot','SkillsMoves','Pace','Shooting','Passing','Dribbling','Defending','Phyiscality','Position','Country','Club','WorkRate']

fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize = (18,11), sharey=True)
fig.subplots_adjust(wspace=.2, hspace=.35)

sb.scatterplot(x=fifa_dataset.WeakFoot, y=fifa_dataset.Price, ax=ax[0, 0], color='#69547C')
sb.scatterplot(x=fifa_dataset.SkillsMoves, y=fifa_dataset.Price, ax=ax[0, 1], color='#69547C')
sb.scatterplot(x=fifa_dataset.Pace, y=fifa_dataset.Price, ax=ax[0, 2], color='#69547C')
sb.scatterplot(x=fifa_dataset.Shooting, y=fifa_dataset.Price, ax=ax[1, 0], color='#69547C')
sb.scatterplot(x=fifa_dataset.Passing, y=fifa_dataset.Price, ax=ax[1, 1], color='#69547C')
sb.scatterplot(x=fifa_dataset.Dribbling, y=fifa_dataset.Price, ax=ax[1, 2], color='#69547C')
sb.scatterplot(x=fifa_dataset.Defending, y=fifa_dataset.Price, ax=ax[2, 0], color='#69547C')
sb.scatterplot(x=fifa_dataset.Phyiscality, y=fifa_dataset.Price, ax=ax[2, 1], color='#69547C')
sb.scatterplot(x=fifa_dataset.Position, y=fifa_dataset.Price, ax=ax[2, 2], color='#69547C')
sb.scatterplot(x=fifa_dataset.Club, y=fifa_dataset.Price, ax=ax[3, 0], color='#69547C')
sb.scatterplot(x=fifa_dataset.Country, y=fifa_dataset.Price, ax=ax[3, 1], color='#69547C')
sb.scatterplot(x=fifa_dataset.WorkRate, y=fifa_dataset.Price, ax=ax[3, 2], color='#69547C')
plt.show()

X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)

model = ExtraTreesClassifier()
model.fit(X,training_scores_encoded)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=[ 'WeakFoot', 'SkillsMoves', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending',
                    'Phyiscality', 'Position','Club','Country','WorkRate'])
feat_importances.nlargest(12).plot(kind='barh')
plt.title('Ważność cech za pomocą Extra Trees Classifier')
plt.show()
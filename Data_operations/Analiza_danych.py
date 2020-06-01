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



attributes = ['WeakFoot','SkillsMoves','Pace','Shooting','Passing','Dribbling','Defending','Phyiscality','Position']
for x in attributes:
    att = ['Price', x]
    scatter_matrix(fifa_dataset[att])
    plt.show()


X = fifa_dataset.drop('Price', axis = 1)
y = fifa_dataset['Price']
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)

model = ExtraTreesClassifier()
model.fit(X,training_scores_encoded)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=[ 'WeakFoot', 'SkillsMoves', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending',
                    'Phyiscality', 'Position'])
feat_importances.nlargest(9).plot(kind='barh')
plt.title('Ważność cech za pomocą Extra Trees Classifier')
plt.show()
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('PICO.csv')
X = data.drop('ID', axis=1)
y = data['ID']

clf = DecisionTreeClassifier(max_depth=None, random_state=42)
clf.fit(X, y)

importances = clf.feature_importances_

features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

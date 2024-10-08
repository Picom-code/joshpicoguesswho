import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('PICO2.csv')
X = data.drop('ID', axis=1)  
y = data['ID']               

clf = DecisionTreeClassifier(max_depth=None, random_state=22)
clf.fit(X, y)

plt.figure(figsize=(40,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=sorted(y.unique()), proportion=False, precision=2)
plt.show()

def count_leaf_nodes(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    leaf_count = sum(children_left[i] == -1 and children_right[i] == -1 for i in range(n_nodes))
    return leaf_count

num_leaves = count_leaf_nodes(clf)
print(f"Number of leaf nodes: {num_leaves}")

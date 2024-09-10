import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


data = pd.read_csv('PICO.csv')
X = data.drop('ID', axis=1)  
y = data['ID']               
clf = DecisionTreeClassifier(max_depth=None, random_state=42, ccp_alpha=0.01)
clf.fit(X, y)

plt.figure(figsize=(30,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=sorted(y.unique()), proportion=False, precision=2)
plt.show()


def get_leaf_node_names(tree, class_labels):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    leaf_names = []
    
    for i in range(n_nodes):
        if children_left[i] == -1 and children_right[i] == -1: 
            class_index = tree.tree_.value[i, 0].argmax()  
            leaf_names.append(class_labels[class_index])  
    
    return leaf_names

leaf_node_names = get_leaf_node_names(clf, clf.classes_)
print("Names in leaf nodes:", leaf_node_names)

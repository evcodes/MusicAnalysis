import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


kNN_params = []
tree_params = []

def clean_df(dataframe):
    #Clean out all ID fields, and non attribute fields
    #return the dataframe as a numerical matrix
    return dataframe.values

def test_model(model, X, Y):
    return 0

URL = "OUR CSV URL"
cols = [] # Our attribute list

df = pd.read_csv(URL, names=cols, sep=',')


feats  = clean_df(df)
target = feats[:, 'gender'] #-1 until we know what column gender will be in
SA = feats[:, 'SA tuple'] #same as above, this is for the sentiment analysis tuple
feats = np.delete(feats, 'gender', 1)

SAX_train, SAX_test, SAY_train, SAY_test = train_test_split(SA, target)
allX_train, allX_test, allY_train, allY_test = train_test_split(feats, target)

kNN = KNeighborsClassifier(kNN_params)
all_Tree = tree.DecisionTreeClassifier(tree_params)
SA_tree = tree.DecisionTreeClassifier(tree_params)

all_Tree.fit(allX_train, allY_train)
kNN.fit(SAX_train, SAY_train)
SA_tree.fit(SAX_train, SAY_train)

models = [all_Tree, SA_tree, kNN]

for i, mod in enumerate(models):
    if i == 0:
        test_model(mod, allX_test, allY_test)
    else:
        test_model(mod, SAX_test, SAY_test)

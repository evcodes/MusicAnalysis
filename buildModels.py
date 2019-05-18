#TODO: Comment all the files with function comments (can be sparse), header
# comments, and clarification on confusing lines, buildSA can be left for Dylan,
# unless someone else wants to figure out what its doing and comment it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


kNN_params = []
tree_params = []

def clean_df(dataframe):
    #TODO: Write the code to strip all non numerical attributes from the dataframe
    #Clean out all ID fields, and non attribute fields
    #return the dataframe as a numerical matrix
    return dataframe.values

# TODO: Implement testing code, doesn't have to be fancy, basically any of the
#  stuff we've done for homeworks, with finding accuracy, recall, precision,
#  f1 and making graphs of all that shit.
def test_model(model, X, Y):
    return 0

URL = "https://docs.google.com/spreadsheets/d/197gXGzGUupdhpf71XfsWZtpHlmBRrpvg69taNCDwKVg/export?format=csv&gid=1500097129"
cols = [] # Our attribute list

df = pd.read_csv(URL, names=cols, sep=',')


feats  = clean_df(df)
target = feats[:, 'gender'] #-1 until we know what column gender will be in
SA = feats[:, 'sentiment'] #same as above, this is for the sentiment analysis tuple
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

#TODO: Make pretty visuals of all of our results, testing accuracy, what
# happens with unusual data (old songs, foreign songs etc. Get general statistics
# on the data as well, whats the ratio of male:female artists in our data set,
# genre stuff maybe?
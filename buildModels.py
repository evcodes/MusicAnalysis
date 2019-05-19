#TODO: Comment all the files with function comments (can be sparse), header
# comments, and clarification on confusing lines, buildSA can be left for Dylan,
# unless someone else wants to figure out what its doing and comment it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC

from pprint import pprint as pp
# import warnings
# warnings.filterwarnings('always')


def clean_df(dataframe):
    #TODO: Write the code to strip all non numerical attributes from the dataframe
    #Clean out all ID fields, and non attribute fields
    #return the dataframe as a numerical matrix

    # Replacing all 'female' tags with 1, all 'male' tags with 0  
    dataframe['gender'].replace(['female', 'male'], [1, 0], inplace=True)
    
    # Get rid of non-numeric columns
    dataframe = dataframe.drop(['title', 'artist_name'], axis=1)

    # Get rid of rows with NaN values
    dataframe = dataframe.dropna()

    # Convert all stringified numeric data to numeric values
    dataframe['danceability'] = pd.to_numeric(dataframe['danceability'],errors='coerce')
    dataframe['duration'] = pd.to_numeric(dataframe['duration'],errors='coerce')
    dataframe['end_of_fade_in'] = pd.to_numeric(dataframe['end_of_fade_in'],errors='coerce')
    dataframe['energy'] = pd.to_numeric(dataframe['end_of_fade_in'],errors='coerce')
    dataframe['key'] = pd.to_numeric(dataframe['key'],errors='coerce')
    dataframe['key_confidence'] = pd.to_numeric(dataframe['key_confidence'],errors='coerce')
    dataframe['loudness'] = pd.to_numeric(dataframe['loudness'],errors='coerce')
    dataframe['mode'] = pd.to_numeric(dataframe['mode'],errors='coerce')
    dataframe['mode_confidence'] = pd.to_numeric(dataframe['mode_confidence'],errors='coerce')
    dataframe['song_hotttnesss'] = pd.to_numeric(dataframe['song_hotttnesss'],errors='coerce')
    dataframe['start_of_fade_out'] = pd.to_numeric(dataframe['start_of_fade_out'],errors='coerce')
    dataframe['tempo'] = pd.to_numeric(dataframe['tempo'],errors='coerce')
    dataframe['time_signature'] = pd.to_numeric(dataframe['time_signature'],errors='coerce')
    dataframe['time_signature_confidence'] = pd.to_numeric(dataframe['time_signature_confidence'],errors='coerce')
    dataframe['year'] = pd.to_numeric(dataframe['year'],errors='coerce')
    dataframe['neutral'] = pd.to_numeric(dataframe['neutral'],errors='coerce')
    dataframe['happy'] = pd.to_numeric(dataframe['happy'],errors='coerce')
    dataframe['sad'] = pd.to_numeric(dataframe['sad'],errors='coerce')
    dataframe['hate'] = pd.to_numeric(dataframe['hate'],errors='coerce')
    dataframe['anger'] = pd.to_numeric(dataframe['anger'],errors='coerce')

    return dataframe.values


# TODO: Implement testing code, doesn't have to be fancy, basically any of the
#  stuff we've done for homeworks, with finding accuracy, recall, precision,
#  f1 and making graphs of all that shit.
def test_model(model_tuple, X, Y):

    model = model_tuple[0]
    model_name = model_tuple[1]
    print("\n",model_name)

    predictions = model.predict(X)
    print("Predictions: ", predictions)

    confusion_matrix = metrics.confusion_matrix(Y, predictions)
    print("Confusion Matrix: ", confusion_matrix)

    print("Precision: ", metrics.precision_score(Y, predictions))
    print("Recall: ", metrics.recall_score(Y, predictions))
    print("Accuracy: ", metrics.accuracy_score(Y, predictions))

    fpr, tpr, _ = roc_curve(Y, predictions)
    auc = metrics.roc_auc_score(Y, predictions)
    plt.clf()
    plt.plot(fpr, tpr, label = "Classifier, AUC = " + str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name + ' ROC Curve')
    plt.legend(loc = "lower right")
    plt.show()

    return 0


URL = "https://docs.google.com/spreadsheets/d/197gXGzGUupdhpf71XfsWZtpHlmBRrpvg69taNCDwKVg/export?format=csv&gid=1500097129"
cols = ['title',
        'artist_name',
        'gender',
        'danceability',
        'duration',
        'end_of_fade_in',
        'energy',
        'key',
        'key_confidence',
        'loudness',
        'mode',
        'mode_confidence',
        'song_hotttnesss',
        'start_of_fade_out',
        'tempo',
        'time_signature',
        'time_signature_confidence',
        'year',
        "neutral",
        "happy",
        "sad",
        "hate",
        "anger"] # Our attribute list

feats_list = ['danceability',
        'duration',
        'end_of_fade_in',
        'energy',
        'key',
        'key_confidence',
        'loudness',
        'mode',
        'mode_confidence',
        'song_hotttnesss',
        'start_of_fade_out',
        'tempo',
        'time_signature',
        'time_signature_confidence',
        'year',
        "neutral",
        "happy",
        "sad",
        "hate",
        "anger"]

emotions = ["neutral", "happy", "sad", "hate", "anger"]


df = pd.read_csv(URL, names=cols, sep=',', header=0)

pp(df)
print(df.shape)

feats  = clean_df(df)

# print("Feats:", feats)
print(feats.shape)
target = feats[:, 0] # gender column
feats = np.delete(feats, 0, 1) #Remove target col
print("target", end=" ")
pp(target)
SA = feats[:, -5:] # last 5 columns for SA
# SA = np.delete(SA, 0, 0)
# print("\n\nSA:", SA)
# feats = np.delete(feats, 0, 0) # delete row containing column names
print("updated feats", end=" ")
pp(feats)
# print("\nFeats Shape:", feats.shape)
# print("\nSA Shape:", SA.shape)
# print("\nTarget Shape:", target.shape)

#TODO: (Maybe) use cross_val_scoring instead for a better training set
SAX_train, SAX_test, SAY_train, SAY_test = train_test_split(SA, target)
allX_train, allX_test, allY_train, allY_test = train_test_split(feats, target)

# print("\n\nAllX train:", allX_train.shape)
# print("\n\nAllX test:", allX_train.shape)
# print("\n\nAllY train:", allY_train.shape)
# print("\n\nAllY test:", allY_train.shape)
# print(type(allY_train[0]))
# print("\n\nSAX train:", SAX_train)
# print("\n\nSAX test:", SAX_train)
# print("\n\nSAY train:", SAY_train)
# print("\n\nSAY test:", SAY_train)

allY_test = allY_test.astype(float)
allY_train = allY_train.astype(float)
SAY_test = SAY_test.astype(float)
SAY_train = SAY_train.astype(float)

all_kNN = KNeighborsClassifier()
SA_kNN = KNeighborsClassifier()
all_Tree = tree.DecisionTreeClassifier()
SA_Tree = tree.DecisionTreeClassifier()
all_RandFor = RFC(n_estimators=25)
SA_RandFor = RFC(n_estimators=25)

all_kNN.fit(allX_train, allY_train)
SA_kNN.fit(SAX_train, SAY_train)
all_Tree.fit(allX_train, allY_train)
SA_Tree.fit(SAX_train, SAY_train)
all_RandFor.fit(allX_train, allY_train)
SA_RandFor.fit(SAX_train, SAY_train)

model_tuples = [(all_Tree, 'All Tree Classifier'),
                (SA_Tree, 'SA Tree Classifier'),
                (all_kNN, "All kNN Classfier"),
                (SA_kNN, 'SA kNN Classifier'),
                (all_RandFor, "All Random Forest Classifier"),
                (SA_RandFor, 'SA Random Forest Classifier')]


for mod in model_tuples:
    if mod[1].startswith("All"):
        test_model(mod, allX_test, allY_test)
    else:
        test_model(mod, SAX_test, SAY_test)




# TODO: Make pretty visuals of all of our results, testing accuracy, what
# happens with unusual data (old songs, foreign songs etc. 



# TODO: Get general statistics on the data as well, whats the ratio of 
# male:female artists in our data set, genre stuff maybe?
# General Stats
print("\nGeneral Data Statistics:\n", df.describe(include='all'))



# Ratio of male:female Artists
gender_counts = df['gender'].value_counts() 
print("\n\nRatio of male:female Artists:\n", gender_counts[0]/gender_counts[1])



# Most Common Year of Composition for each Gender
gender_year_df = df[['year', 'gender']]

male_year_df = gender_year_df[gender_year_df['gender']==0]
print("\n\nMost Common Year of Composition for Male Artists:", male_year_df.mode()['year'])

female_year_df = gender_year_df[gender_year_df['gender']==1]
print("Most Common Year of Composition for Female Artists:", female_year_df.mode()['year'])
# print(gender_year_df)



# Most Common Energy Score for each Gender
gender_energy_df = df[['energy', 'gender']]

male_energy_df = gender_energy_df[gender_energy_df['gender']==0]
print("\n\nMost Common Energy Score for Male Artists:", male_energy_df.mode()['energy'])

female_energy_df = gender_energy_df[gender_energy_df['gender']==1]
print("Most Common Energy Score for Female Artists:", female_energy_df.mode()['energy'])
# print(gender_energy_df)



# Average Song Tempo for each Gender
gender_tempo_df = df[['tempo', 'gender']]

male_tempo_df = gender_tempo_df[gender_tempo_df['gender']==0]
print("\n\nAverage Song Score for Male Artists:", male_tempo_df['tempo'].astype(float).mean())

female_tempo_df = gender_tempo_df[gender_tempo_df['gender']==1]
print("Average Song Tempo for Females:", female_tempo_df['tempo'].astype(float).mean())
# print(gender_tempo_df)



# Average Song Loudness for each Gender
gender_loudness_df = df[['loudness', 'gender']]

male_loudness_df = gender_loudness_df[gender_loudness_df['gender']==0]
print("\n\nAverage Loudness Score for Male Artists:", male_loudness_df['loudness'].astype(float).mean())

female_loudness_df = gender_loudness_df[gender_loudness_df['gender']==1]
print("Average Song Loudness for Females:", female_loudness_df['loudness'].astype(float).mean())
# print(gender_loudness_df)

print("\nAll Tree feature Importance:")
for i, feat_imp in enumerate(all_Tree.feature_importances_):
    print(feats_list[i], ":",  feat_imp)

print("\nSA Tree feature importance:")
for i, feat_imp in enumerate(SA_Tree.feature_importances_):
    print(emotions[i], ":",  feat_imp)

print("\nAll Random Forest feature Importance:")
for i, feat_imp in enumerate(all_RandFor.feature_importances_):
    print(feats_list[i], ":",  feat_imp)

print("\nSA Random Forest feature importance:")
for i, feat_imp in enumerate(SA_RandFor.feature_importances_):
    print(emotions[i], ":",  feat_imp)


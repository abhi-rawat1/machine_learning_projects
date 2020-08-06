'''
    ***** RECOMMENDATION SYSTEM USING SURPRISE LIBRARY for User/Movie Rating *****

A. This notebook has implementation of recommendation system for Movie Rating database using SURPRISE library with following 
    algorithms:
    a. SVD
    b. SVD with GridSearchCV
    c. KNNBaseline default
    d. KNNBaseline item-item similarity
    e. KNNBaseline user-user similarity
    
B. This notebook has different implementation of loading/referring data as supported by Surprise Library such as:
    a. Dataset.load_builtin
    b. Dataset.load_from_df
    
C. Also, this notebook has different implementation of splitting data into training and testing dataset such as:
    a. model_selection.train_test_split
    b. build_full_trainset
    c. build_anti_testset (from build_full_trainset)
    
'''


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import SVDpp
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import pandas as pd
import pickle
from surprise import KNNBaseline
from surprise.model_selection import LeaveOneOut
from collections import defaultdict
import itertools 





# Part-1 - Load the 'ml-100k' data and split into trainign and testing data

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25)



# Part-2 - Train the model using SVD algo with use of 'GridSearchCV' configuration and evaluate model accuracy 

param_grid = {'n_factors': [110, 120, 140, 160], 'n_epochs': [90, 100, 110], 'lr_all': [0.001, 0.003, 0.005, 0.008],
              'reg_all': [0.08, 0.1, 0.15]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



'''
Part - 3 - Train the model using SVD algorithm and test the model in different combination like 
a. test with the build model
b. test with built model pickle file
c. test set of records via dataframe with model

'''

algo = SVD(n_factors=160, n_epochs=2, lr_all=0.005, reg_all=0.1)
algo.fit(trainset)

with open('movie_recomm_svd_pkl.pkl', 'wb') as fid:
    pickle.dump(algo, fid)

test_pred = algo.test(testset)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)

print("***** SVD Model Prediction Result *****")
accuracy.rmse(test_pred, verbose=True)
accuracy.mae(test_pred, verbose=True)
print(df.head())


with open('movie_recomm_svd_pkl.pkl', 'rb') as fid:
    sv = pickle.load(fid)

test_pred = sv.test(testset)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)

print("\n***** SVD Model Prediction Result via model file*****")
accuracy.rmse(test_pred, verbose=True)
accuracy.mae(test_pred, verbose=True)
print(df.head())


with open('movie_recomm_svd_pkl.pkl', 'rb') as fid:
    sv = pickle.load(fid)


ratings_dict = {'movieId': [735, 642],
                'userID': [916, 848],
                'rating': [4, 5]}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.95)

pred = sv.test(testset)
odf = pd.DataFrame(pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
odf['err'] = abs(odf.est - odf.rui)

print("\n***** SVD Model Prediction Result via model file for Two record*****")
accuracy.rmse(pred, verbose=True)
accuracy.mae(test_pred, verbose=True)
print(odf.head())



# Part-4 - Train the model using KNNBaseline item-item similarity

sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(trainset)

test_pred = simsAlgo.test(testset)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)
print("****************KNNBaseline item-item similarity: Accuracy Score *****************")
accuracy.rmse(test_pred, verbose=True)
accuracy.mae(test_pred, verbose=True)
print(df.head())


# Part-5 - Train the model using KNNBaseline User-User similarity and get the Top-10 movies predictions for each user 

sim_options = {'name': 'pearson_baseline', 'user_based': True}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(trainset)

test_pred = simsAlgo.test(testset)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)

print("******* KNNBaseline User-User similarity: Accuracy Score*****************")
accuracy.rmse(test_pred, verbose=True)
accuracy.mae(test_pred, verbose=True)
print(df.head().to_string())


def GetTopN(predictions, n=10, minimumRating=4.0):
    topN = defaultdict(list)
    for uid, iid, rui, est, _ in predictions:
        if (est >= minimumRating):
            topN[int(uid)].append((int(iid), est))

    for uid, ratings in topN.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        topN[int(uid)] = ratings[:n]

    return topN

topN = GetTopN(test_pred, n=10)
print('****** Top 10 Predictions for first 10 users ****')
dict(itertools.islice(topN.items(), 10))



'''
Part- 6 - Train the model using KNNBaseline item-item similarity. 

Trainign and testing Data is generated using  'build_full_trainset' and 'build_anti_testset' api respectively.

'''


fullTrainSet = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

test_pred = simsAlgo.test(testset)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)

print("**************** KNNBaseline item-item similarity: Accuracy Score with Test Set *****************")
accuracy.rmse(test_pred, verbose=True)
print(df.head().to_string())


bigTestSet = fullTrainSet.build_anti_testset()
test_pred = simsAlgo.test(bigTestSet)
df = pd.DataFrame(test_pred, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['err'] = abs(df.est - df.rui)
print("\n**************** KNNBaseline item-item similarity: Accuracy Score with Anti-Testset *****************")
accuracy.rmse(test_pred, verbose=True)
df.head()




'''
    ***** RECOMMENDATION SYSTEM USING DEEP MATRIX FACTORIZATION for Movie Rating *****

A. This notebook has implementation of recommendation system for Movie Rating database using Deep Matrix Factorization technique
    
Note:  This has been referenced from opensource project.

'''



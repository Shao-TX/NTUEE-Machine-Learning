#%%
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#%%
TRAIN_PATH = 'ml2021spring-hw1\covid.train.csv'

original_data = pd.read_csv(TRAIN_PATH)
data = original_data.iloc[:, 1:94]
target = original_data.iloc[:,-1]

data = np.array(data)
target = np.array(target)

#%%
New_Data = SelectKBest(score_func = f_regression, k = 75)
fit = New_Data.fit(data, target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(75,'Score'))  #print 75 best features
#%%
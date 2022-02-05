#%%
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

#%%
TRAIN_PATH = 'ml2021spring-hw1\covid.train.csv'

original_data = pd.read_csv(TRAIN_PATH)
data = original_data.iloc[:, 1:94]
target = original_data.iloc[:,-1]

data = np.array(data)
target = np.array(target)

#%%

fn = input("SelectKBest(1) or SelectPercentile(2) : ")

if(fn == "1"):
    k = int(input("Number of feature : "))
    selector = SelectKBest(score_func = f_regression, k = k)                     # 篩選特定數量
elif(fn == "2"):
    percent = int(input("Percentage of feature : "))
    selector = SelectPercentile(score_func = f_regression, percentile = percent) # 篩選特定比例

selector.fit(data, target) # 擬和

Scores = selector.scores_
Pvalues = selector.pvalues_

Feature_Position = selector.get_support(True) # 篩選過的特徵位置
Feature_Position = Feature_Position.tolist()  # numpy to list

#%%
print(Feature_Position)

# %%

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
# selector = SelectKBest(score_func = f_regression, k = 75)             # 篩選特定數量
selector = SelectPercentile(score_func = f_regression, percentile = 80) # 篩選特定比例
selector.fit(data, target)

Scores = selector.scores_
Pvalues = selector.pvalues_

Feature_Position = selector.get_support(True) # 篩選過的特徵位置
Feature_Position = Feature_Position.tolist()  # numpy to list

#%%
print(Feature_Position)

# %%

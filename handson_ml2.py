#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
assert sys.version_info >= (3, 5)


# In[5]:


import sklearn
assert sklearn.__version__ >= "0.20"


# In[6]:


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# In[8]:


import os
datapath = os.path.join("datasets", "lifesat", "")


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[11]:


import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# 데이터 적재
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# 데이터 준비
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# 데이터 시각화
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# 선형 모델 선택
model = sklearn.linear_model.LinearRegression()

# 모델 훈련
model.fit(X, y)

# 키프로스에 대한 예측
X_new = [[22587]]  # 키프로스 1인당 GDP
print(model.predict(X_new)) # 출력 [[ 5.96242338]]


# In[27]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()


# In[28]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[31]:


housing = load_housing_data()
housing.head()


# In[33]:


housing.info()


# In[35]:


housing['ocean_proximity'].value_counts()


# In[37]:


housing.describe()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize = (20, 15))
plt.show()


# In[50]:


import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indices[:test_set_size]
    train_indicies = shuffled_indices[test_set_size]
    return data.iloc[train_indicies], data.iloc[test_indicies]
train_set, test_set = split_train_test(housing, 0.2)

len(train_set)


# In[51]:


len(test_set)


# In[53]:


from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 **32


# In[55]:


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[59]:


import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# In[61]:


housing_with_id = housing.reset_index()   # `index` 열이 추가된 데이터프레임을 반환합니다
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[63]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[66]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


# In[68]:


housing['income_cat'] = pd.cut(housing['median_income'],
                              bins = [0, 1.5, 3.0, 4.5, 6., np.inf],
                              labels = [1, 2, 3, 4, 5])


# In[70]:


housing['income_cat'].hist()


# In[73]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    


# In[76]:


strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[79]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace = True)


# In[81]:


housing = strat_train_set.copy()


# In[83]:


housing.plot(kind = "scatter", x='longitude', y='latitude')


# In[85]:


housing.plot(kind = "scatter", x='longitude', y='latitude', alpha = 0.1)


# In[89]:


housing.plot(kind = "scatter", x="longitude", y="latitude", alpha = 0.4, s=housing['population']/100, 
             label="population", figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()


# In[93]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=True)


# In[100]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[102]:


housing.plot(kind = "scatter", x = 'median_income', y = 'median_house_value', alpha =  0.1)


# In[104]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[107]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# In[109]:


housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[111]:


median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace = True)


# In[113]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')


# In[115]:


housing_num = housing.drop('ocean_proximity', axis = 1)


# In[117]:


imputer.fit(housing_num)


# In[120]:


imputer.statistics_


# In[122]:


housing_num.median().values


# In[124]:


X = imputer.transform(housing_num)


# In[126]:


housing_tr = pd.DataFrame(X, columns = housing_num.columns,
                         index = housing_num.index)
housing_tr


### RecSys Sales Script

"""
Title: Sales Recommender System

Description:
This script processes raw sales data to create a recommender system for products using collaborative filtering.
The SVD algorithm from the Surprise library is employed for recommendations.
The script also calculates evaluation metrics and generates top 10 recommendations for each user.

Imports:
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from surprise import accuracy

Steps:

1. Data Loading and Preprocessing:
   - Load raw sales data into a DataFrame.
   - Preprocess the data using LabelEncoder, MinMaxScaler, and StandardScaler.
   - Aggregate the data to create a new DataFrame with relevant features for building a recommender system.
   - Save the aggregated data to a CSV file.

2. Data Preparation for Machine Learning:
   - Normalize the count_rate values.
   - Drop any duplicates.
   - Create separate DataFrames for products and user ratings.
   - Save these DataFrames to CSV files for further use.

3. Recommender System Training and Evaluation:
   - Use the SVD algorithm from the Surprise library to create a recommender system.
   - Calculate evaluation metrics like RMSE, MAE, MAPE, and R2 score for the model.
   - Generate top 10 recommendations for each user in the training set.
   - Sort the recommendations by predicted rating.
   - Export the recommendations to a CSV file.
"""

import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from surprise import accuracy


# SFR data manipulation
path = r'C:\Users\retzer.matan\Desktop\RecSys_Sales\ZIM'  # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))

dfs = {}
for f in all_files:
    df = pd.read_csv(f)
    file_name = os.path.basename(f)
    key = os.path.splitext(file_name)[0]
    dfs[key] = df

sfr = dfs['SFR_RECSYS_2018_2023']
customers = dfs['ZIM_Customers']
commodity = dfs['ZIM_Commodity']

sfr = pd.merge(sfr, customers, left_on='CUSTOMER', right_on='customer', how='left')
sfr = pd.merge(sfr, commodity, on='ZGC_COMOD', how='left')
df = sfr.dropna(axis=1, thresh=int((1 - 0.91) * len(sfr)))

# Create commodity column
df['2_ZGC_COMOD']=df['ZGC_COMOD'].str[:2].fillna('').astype(int, errors='ignore')

# Product column
df['product'] = df['ZEC_PACPT'] + "_" + df['ZEC_PLOAD'] + "_" + df['ZEC_PDISC'] + "_" + df['ZEC_PDEST'] + "_" + df['2_ZGC_COMOD']

# Filter the data
counts = df['product'].value_counts()
df_filtered = df[df['product'].isin(counts[counts > 1].index)]
counts = df_filtered['ZCR_COCUS'].value_counts()
df_filtered = df_filtered[df_filtered['ZCR_COCUS'].isin(counts[counts > 2].index)]
df_filtered = df_filtered[df_filtered['CALMONTH'].astype(str).str[:4].astype(int) > 2018]

# Create the RecSys dataset
df_recsys = pd.DataFrame(df_filtered.groupby(['product', 'CUST_DESC', 'CALMONTH'])['product'].count().reset_index(name='count'))
df_recsys['total_count'] = df_recsys.groupby(['product','CUST_DESC'])['count'].transform('sum')

# Add rating columns
scaler = MinMaxScaler(feature_range=(1, 5))
df_recsys['count_rate'] = scaler.fit_transform(df_recsys[['count']]).round(1)
df_recsys['total_count_rate'] = scaler.fit_transform(df_recsys[['total_count']]).round(1)

scaler = StandardScaler()
df_recsys['count_r_rate'] = scaler.fit_transform(df_recsys[['count']])
df_recsys['count_r_rate'] = ((df_recsys['count_r_rate'] - df_recsys['count_r_rate'].min()) / (df_recsys['count_r_rate'].max() - df_recsys['count_r_rate'].min()) * (5 - 1) + 1).round().astype(int)
df_recsys['total_r_count_rate'] = scaler.fit_transform(df_recsys[['total_count']])
df_recsys['total_r_count_rate'] = ((df_recsys['total_r_count_rate'] - df_recsys['total_r_count_rate'].min()) / (df_recsys['total_r_count_rate'].max() - df_recsys['total_r_count_rate'].min()) * (5 - 1) + 1).round().astype(int)

df_recsys = df_recsys.rename(columns={'CUST_DESC': 'c_customer'})

# Prepare for ML training
df_recsys['timestamp'] = pd.to_datetime(df_recsys['CALMONTH'], format='%Y%m')
data = df_recsys[['c_customer', 'product', 'count_rate', 'timestamp']]
data.loc[:, 'trade'] = data['product'].str[6:17]
data['normalized_rate'] = data.groupby('trade')['count_rate'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

data = data.dropna().drop_duplicates()

# Data for general recsys
products = data[['product', 'trade']].drop_duplicates()
products['productId'] = products.reset_index().index
products.to_csv("zim_products_recsys.csv", index=False)

zim_ratings = data[['c_customer', 'product', 'count_rate', 'timestamp']]
zim_ratings['userId'] = zim_ratings.groupby('c_customer').ngroup()
zim_ratings = zim_ratings.merge(products, on='product')
ratings = zim_ratings[['userId', 'productId', 'count_rate', 'timestamp']]
ratings.to_csv("zim_ratings_recsys.csv", index=False)
zim_ratings[['c_customer', 'userId']].drop_duplicates().to_csv("zim_customer_userId.csv", index=False)

# RecSys Training
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['c_customer', 'product', 'count_rate']], reader)

param_grid = {'n_factors': [50, 100, 150], 'reg_all': [0.02, 0.05, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(dataset)

print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']
trainset = dataset.build_full_trainset()
algo.fit(trainset)

trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
algo = SVD(n_factors=50, reg_all=0.05)
algo.fit(trainset)
predictions = algo.test(testset)

threshold = 2
n_hits = sum([1 for p in predictions if p.r_ui >= threshold and p.est >= threshold])
hit_rate = n_hits / len(predictions)
print(f'Hit rate: {hit_rate:.2f}')

trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Use SVD for recommendations
algo = SVD(n_factors=50, reg_all=0.02)
algo.fit(trainset)
predictions = algo.test(testset)

y_true = [p.r_ui for p in predictions]
y_pred = [p.est for p in predictions]
rmse = accuracy.rmse(predictions)
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'Mape: {mape:.2f}')
print(f'R2 score: {r2:.2f}')

# Recommendations
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['c_customer', 'product', 'count_rate']], reader)
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

train_user_indices = trainset.all_users()
test_user_indices = set([uid for (uid, _, _) in testset])

print(f'Number of users in the training set: {len(train_user_indices)}')
print(f'Number of users in the test set: {len(test_user_indices)}')

algo = SVD(n_factors=50, reg_all=0.02)
algo.fit(trainset)

all_items = trainset.all_items()
items = list(set(data['product']) & set(trainset.to_raw_iid(i) for i in all_items))

user_loc = 10
user_id = trainset.to_raw_uid(user_loc)

print(f'User: {user_id}')
print('Top rated items:')

item_ratings = []
for item in items:
    item_idx = trainset.to_inner_iid(item)
    rating = algo.predict(trainset.to_raw_uid(user_loc), item_idx).est
    item_ratings.append((item, rating))

item_ratings.sort(key=lambda x: x[1], reverse=True)

print(f'\nTop 10 recommended items for user at location {user_loc} in the training set:')
for i in range(10):
    print(f'{i+1}. {item_ratings[i][0]} ({item_ratings[i][1]:.2f})')

# Export recommendations
recommendations = pd.DataFrame(columns=['user_id', 'product', 'rating'])

for user_loc in range(trainset.n_users):
    user_id = trainset.to_raw_uid(user_loc)
    item_ratings = []
    for item in items:
        item_idx = trainset.to_inner_iid(item)
        rating = algo.predict(trainset.to_raw_uid(user_loc), item_idx).est
        item_ratings.append((user_id, item, rating))

    item_ratings.sort(key=lambda x: x[2], reverse=True)
    recommendations = recommendations.append(pd.DataFrame(item_ratings[:10], columns=['user_id', 'product', 'rating']), ignore_index=True)

recommendations.to_csv('sales_recsys_top_10_recommendations_com.csv', index=False)

### End of script
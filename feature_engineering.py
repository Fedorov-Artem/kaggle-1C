import pandas as pd
import features_time_series
import numpy as np
import time
import gc

start = time.time()

# About this file in general:
# Here I perform the main data-aggregation
# And produce all the features that will be used to build the models

# load the data (sales and test are from kaggle, items_and_vectors.csv is an output from features_from_text.py)
path_sales = 'Input/sales_train_v2.csv'
path_test = 'Input/test.csv'
path_items = 'Input/items_and_vectors.csv'

df_sales = pd.read_csv(path_sales)
df_test = pd.read_csv(path_test)
df_items = pd.read_csv(path_items)

# Here are some manual data fixes after data exploration
# fixing shop_id's
df_sales['shop_id'] = [57 if x == 0 else x for x in df_sales['shop_id']]
df_sales['shop_id'] = [58 if x == 1 else x for x in df_sales['shop_id']]
df_sales['shop_id'] = [10 if x == 11 else x for x in df_sales['shop_id']]

# removing shops that all the time start and stop working, as I do not make predictions for them
df_sales = df_sales.loc[df_sales['shop_id'] != 9]
df_sales = df_sales.loc[df_sales['shop_id'] != 20]

# shop_id = 27 did not work at 32-nd month - no sales and only one return. I exclude this data from model.
df_sales = df_sales.loc[(df_sales['shop_id'] != 27) | (df_sales['date_block_num'] != 32)]

# here I aggregate the data and add rows for zero sales
df = df_sales.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price': 'median'})
df = df.rename(columns={'item_cnt_day': 'item_cnt_month'}).reset_index()
kaggle_1C = features_time_series.FeaturesTimeSeries('shop_id', 'date_block_num', 'item_cnt_month')
df = kaggle_1C.add_zero_sales_rows(df, 'item_id')
df['item_cnt_month'] = df['item_cnt_month'].fillna(0)
df['item_price'] = df.groupby(['item_id', 'shop_id'])['item_price'].ffill()
df['item_price'] = df['item_price'].fillna(0)

# for this task target is clipped to (0,20), but I only clip after preparing all the features
# here I decrease target values only for the few biggest outliers and do not touch values less than 400
df['item_cnt_month'] = [400 + ((x - 400) * 0.25) if x > 400 else x for x in df['item_cnt_month']]

# add month and number of working days per shop
df_sales['date'] = pd.to_datetime(df_sales['date'], format='%d.%m.%Y')
df_sales['month'] = df_sales['date'].dt.month
df = kaggle_1C.add_aggregate(df_sales, 'date_block_num', 'month', 'mean', 'month', df)
df = kaggle_1C.add_aggregate(df_sales, ['date_block_num', 'shop_id'], 'date', 'nunique', 'days', df)

# preparing test data and combining it with main dataframe
df['ID'] = -1
df_test['month'] = 11
df_test['date_block_num'] = 34
df_test['item_cnt_month'] = -1
df_test['days'] = 30
df_test['item_price'] = 0
df = pd.concat([df, df_test], sort=True)

# merging main df's with item_data
df_items = kaggle_1C.reduce_memory(df_items)
df = pd.merge(df, df_items, on='item_id', how='left')

# now delete the ticket categories as I do not need to predict ticket sales
df = df.loc[df['item_category_id'] != 8]
df = df.loc[df['item_category_id'] != 80]

# on_sale feature
df['on_sale'] = 1
df.loc[(df['shop_id'] != 12) & (df['item_category_id'] == 9), 'on_sale'] = 0
df.loc[(df['shop_id'] != 55) & (df['is_digital'] == 1), 'on_sale'] = 0
df.loc[(df['shop_id'] == 55) & (df['is_digital'] == 0), 'on_sale'] = 0
del df['is_digital']

# 'technical' features for previous months sales and days, shop was working
df = kaggle_1C.add_lag_feature(df, range(1, 7), 'item_id', 'item_cnt_month')
df = kaggle_1C.add_lag_feature(df, range(1, 7), 'item_id', 'days')

# feature for month after first sale of this item_id in any shop (0 for new items)
df['month_after_start'] = df['date_block_num'] - df.groupby('item_id')['date_block_num'].transform('min')

# feature for a day when item was first time sold in a shop
# is used to improve predicting for second month sales, so it is set to 0 for all months but second
df_2 = df_sales.groupby(['item_id', 'shop_id']).agg({'date_block_num': 'min', 'date': 'min'}).reset_index()
df_2['first_month'] = df_2.groupby('item_id')['date_block_num'].transform('min')
df_2['first_day'] = df_2['date'].dt.day
df_2.loc[df_2['first_month'] < df_2['date_block_num'], 'first_day'] = 35
df_2 = df_2[['shop_id', 'item_id', 'first_day']]
df = pd.merge(df, df_2, on=['shop_id', 'item_id'], how='left')
df['first_day'] = df['first_day'].fillna(35)
df.loc[df['month_after_start'] != 1, 'first_day'] = 0

# Features generated from item-description sometimes should be used only for the first month item is sold
# Prediction for later months can be better predicted with data from previous periods
df.loc[df['month_after_start'] > 0, ['month_after_same_text', 'same_text_this_month']] = 0
df.loc[df['on_sale'] == 0, 'lang'] = -1

# lag_0.5 feature (sales for the last 14 days of month)
df_sales['day'] = df_sales['date'].dt.day
df = kaggle_1C.last_n_days_of_previous_month_sales(df_sales, 14, df)

# I will only use aggregated data after this point. Will now release some memory and collect garbage.
del df_2
del df_sales
del df_items
gc.collect()
df = kaggle_1C.reduce_memory(df)

# calculate 3 and 6 month shop trends - average sales per day, per shop and per category (or per type)
df = kaggle_1C.add_n_month_trends(df, 'item_category_id', 3)
df = kaggle_1C.add_n_month_trends(df, 'item_category_id', 6)

# adding new lag features, all calculated per day the shop worked that month
df['lag_1_feature'] = 0
df.loc[df['days_lag_1'] > 0, 'lag_1_feature'] = df['item_cnt_month_lag_1']/df['days_lag_1']
df = kaggle_1C.add_n_month_average(df, 3)
df = kaggle_1C.add_n_month_average(df, 6)

# item_sales_stability - share of items sold last months to total sales last three months (all the sales are per day)
df['3_month_stability_feature'] = 0
df.loc[(df['3_month_average_feature'] > 0) & (df['month_after_start'] > 2), '3_month_stability_feature'] = \
    df['lag_1_feature']/df['3_month_average_feature']

# another second month feature - all shops average sales per day after after first sale
df = kaggle_1C.add_aggregate(df, 'date_block_num', 'days_lag_1', 'max', 'max_days')
df['per_day_after_start'] = 0
df.loc[(df['item_cnt_month_lag_1'] > 0) & (df['first_day'] > 0), 'per_day_after_start'] = df['item_cnt_month_lag_1'] /\
                                                                                (df['max_days'] - df['first_day'] + 1)
df = kaggle_1C.add_aggregate(df.loc[(df['item_cnt_month_lag_1'] > 0) & (df['first_day'] > 0)],
                             ['date_block_num', 'item_id'], 'per_day_after_start', 'mean', 'average_per_day_after_start', df)
df['average_per_day_after_start'] = df['average_per_day_after_start'].fillna(0)
df.drop(columns=['per_day_after_start', 'max_days'], inplace=True)

# feature for average item's sales last month
df = kaggle_1C.add_aggregate(df, ['item_id', 'date_block_num'], 'lag_1_feature', 'mean', 'item_popularity_lag_1')

# price features
df = kaggle_1C.add_lag_feature(df, [1], 'item_id', 'item_price')
df['max_price_per_shop'] = df.groupby(['item_id', 'shop_id'])['item_price_lag_1'].transform(pd.Series.cummax)
df = kaggle_1C.add_aggregate(df, ['item_id', 'date_block_num'], 'max_price_per_shop', 'max', 'max_price')
df.drop(columns=['item_price', 'item_price_lag_1', 'max_price_per_shop'], inplace=True)

df = kaggle_1C.add_aggregate(df, ['item_category_id', 'date_block_num'], 'max_price', 'median', 'median_max_price')
df = kaggle_1C.add_cumsum_average(df, ['item_category_id'], 'median_max_price_agg', agg_value='median_max_price')
df.loc[df['median_max_price'] == 0, 'relative_price_median'] = 1
df.loc[df['median_max_price_agg'] > 0, 'relative_price_median'] = df['max_price'] / df['median_max_price_agg']
df['relative_price_median'] = df['relative_price_median'].fillna(0)
del df['median_max_price']
del df['median_max_price_agg']

# average per category and per type encodings
df = kaggle_1C.add_cumsum_average(df.loc[df['on_sale'] == 1], ['type_code'], 'type_average', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[df['on_sale'] == 1], ['item_category_id'], 'category_average', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[df['on_sale'] == 1], ['type_code', 'shop_id'], 'type_average_per_shop', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[df['on_sale'] == 1], ['item_category_id', 'shop_id'],
                                  'category_average_per_shop', df_to=df)
df.loc[df['on_sale'] == 0, ['type_average', 'category_average', 'type_average_per_shop', 'category_average_per_shop']] = -1

# average per category and per type encodings for new (first month) items
df = kaggle_1C.add_cumsum_average(df.loc[(df['date_block_num'] > 1) & (df['month_after_start'] == 0) & (df['on_sale'] == 1)],
                                           ['type_code'], 'type_average_1m', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[(df['date_block_num'] > 1) & (df['month_after_start'] == 0) & (df['on_sale'] == 1)],
                                           ['item_category_id'], 'category_average_1m', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[(df['date_block_num'] > 1) & (df['month_after_start'] == 0) & (df['on_sale'] == 1)],
                                           ['type_code', 'shop_id'], 'type_average_per_shop_1m', df_to=df)
df = kaggle_1C.add_cumsum_average(df.loc[(df['date_block_num'] > 1) & (df['month_after_start'] == 0) & (df['on_sale'] == 1)],
                                           ['item_category_id', 'shop_id'], 'category_average_per_shop_1m', df_to=df)
df.loc[df['on_sale'] == 0, ['type_average_1m', 'category_average_1m', 'type_average_per_shop_1m',
                            'category_average_per_shop_1m']] = -1

df.loc[df['month_after_start'] > 0, ['type_average_1m', 'category_average_1m', 'type_average_per_shop_1m']] = 0
df.loc[df['month_after_start'] > 0, 'category_average_per_shop_1m'] = 0

# delete 'technical' features
features_to_delete = [col for col in df.columns if ('item_cnt_month_lag_' in col) | ('days_lag_' in col)]
df.drop(columns=features_to_delete, inplace=True)
del df['days']
del df['on_sale']

# now I remove data for the first 6 months, it will not be used for training models
df = df.loc[df['date_block_num'] > 5]

# check before saving data
columns_with_nan = df.isna().any()
print(columns_with_nan)

# save data into file
export_file_final_features = 'Input/' + 'final_features.csv'
df.to_csv(export_file_final_features, encoding='utf-8', index=False)

end = time.time()
print(end - start)

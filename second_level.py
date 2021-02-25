import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
#from sklearn.ensemble import RandomForestRegressor


# About this file in general:
# here I join together results of several models and run the linear regression on them to get the final result
# I divide data from date_block_num=33 to cross-validate the linear regression to get an estimation
# And then use models on the test data to get the final results


def assemble_the_df(end_part, all_models, ids_used=False, mid_part=None, date_blocks=None):
    for lvl1_model in all_models:
        if date_blocks:
            for block in date_blocks:
                file_path = 'Output/' + lvl1_model + mid_part + str(block) + end_part
                df_i = pd.read_csv(file_path)
                df_i['date_block_num'] = block
                if block == date_blocks[0]:
                    df_m = df_i
                else:
                    df_m = pd.concat([df_m, df_i], sort=True)
            ['item_id', 'shop_id'].append('date_block_num')
        else:
            file_path = 'Output/' + lvl1_model + end_part
            df_m = pd.read_csv(file_path)
        if ids_used:
            merge_on = ['ID']
            prediction_field_name_model = 'predicted_' + lvl1_model
            df_m = df_m.rename(columns={'item_cnt_month': prediction_field_name_model})
        else:
            merge_on = ['item_id', 'shop_id']
        if lvl1_model == all_models[0]:
            df = df_m
        else:
            df = pd.merge(df, df_m, on=merge_on, how='inner')
    return df


# load the Y data for date_block_num=33
path_y_33 = 'Input/y_33.csv'
df_y_33 = pd.read_csv(path_y_33)
df_y_33['item_cnt_day'] = df_y_33['item_cnt_day'].clip(0, 20)

# combine the data from multiple csv files into a few dataframes
blocks = [30, 31, 32]
#models = ['cat_100', 'cat_10', 'cat_25', 'rf_25', 'cat_03']
models = ['cat_50', 'cat_10', 'cat_25', 'rf_25']
df_cv = assemble_the_df('_cv_33_short.csv', models, date_blocks=blocks, mid_part='_iter_')
df_cv = pd.merge(df_cv, df_y_33, on=['item_id', 'shop_id'], how='inner')
df_test_fin = assemble_the_df('_result_test.csv', models, ids_used=True)
df_33_test_fin = assemble_the_df('_iter_33_for_test.csv', models)
df_33_test_fin = pd.merge(df_33_test_fin, df_y_33, on=['item_id', 'shop_id'], how='inner')

# definition of a model
reg = LinearRegression()
#rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=200, n_jobs=2, max_depth=3, verbose=2)
model = reg

# specification of X and Y
x_cols = [col for col in df_cv.columns if 'predicted_' in col]
kf = KFold(n_splits=6, random_state=44, shuffle=True)
X = df_cv[x_cols]
Y = df_cv['item_cnt_day']
X_test_fin = df_test_fin[x_cols]

# fitting the model and saving the results
scores = []
scores_cv_33 = []
n = 0
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    column_name = 'result_' + str(n)
    df_test_fin[column_name] = model.predict(df_test_fin[x_cols])
    df_33_test_fin[column_name] = model.predict(df_33_test_fin[x_cols])
    n = n + 1

# visualisation of results
print(scores)
print(np.mean(scores))
to_sum_up = [col for col in df_test_fin.columns if 'result_' in col]
df_33_test_fin['predicted_average'] = df_33_test_fin[to_sum_up].mean(axis=1).clip(0, 20)
cv_error_clip = sqrt(mean_squared_error(df_33_test_fin['item_cnt_day'], df_33_test_fin['predicted_average']))
print('CV_error: ' + str(cv_error_clip))

# final export
df_test_fin['item_cnt_month'] = df_test_fin[to_sum_up].mean(axis=1).clip(0, 20)
df_result = df_test_fin[['ID', 'item_cnt_month']].copy()
export_file_results = 'Output/result_new.csv'
df_result.to_csv(export_file_results, encoding='utf-8', index=False)

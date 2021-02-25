import run_time_series
import pandas as pd

# Here is the definition of the 1st level models
# It is possible to run only one model "cat_25" and get good results
# I use several models and then run linear regression on top of them to get to the top in leaderboard

cols_to_del = ['on_sale']
categorical_features = ['month', 'shop_id', 'type_code', 'item_category_id', 'lang']
excl_fields = ['item_id', 'ID']

cat25 = run_time_series.RunTimeSeriesCatboost('cat_25', 25, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_cat25 = {'iterations': 100, 'cat_features': categorical_features, 'max_depth': 9, 'learning_rate': 0.25,
                    'max_ctr_complexity': 2}
cat25.init_model(parameters_cat25)

cat50 = run_time_series.RunTimeSeriesCatboost('cat_50', 50, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_cat50 = {'iterations': 100, 'cat_features': categorical_features, 'max_depth': 9, 'learning_rate': 0.20,
                     'max_ctr_complexity': 2}
cat50.init_model(parameters_cat50)

cat100 = run_time_series.RunTimeSeriesCatboost('cat_100', 100, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_cat100 = {'iterations': 100, 'cat_features': categorical_features, 'max_depth': 9, 'learning_rate': 0.15,
                     'max_ctr_complexity': 2}
cat100.init_model(parameters_cat100)

cat10 = run_time_series.RunTimeSeriesCatboost('cat_10', 10, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_cat10 = {'iterations': 100, 'cat_features': categorical_features, 'max_depth': 9, 'learning_rate': 0.30,
                    'max_ctr_complexity': 2}
cat10.init_model(parameters_cat10)

cat03 = run_time_series.RunTimeSeriesCatboost('cat_03', 3, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_cat03 = {'iterations': 100, 'cat_features': categorical_features, 'max_depth': 9, 'learning_rate': 0.35,
                    'max_ctr_complexity': 2}
cat03.init_model(parameters_cat03)


def encode_1hot(df):
    df = pd.get_dummies(data=df, columns=['lang', 'month'])
    return df


rf25 = run_time_series.RunTimeSeriesRandomForest('rf_25', 25, del_columns=cols_to_del, fields_to_exclude=excl_fields)
parameters_rf25 = {'n_estimators': 200, 'min_samples_leaf': 150, 'n_jobs': 2, 'max_depth': 10, 'max_features': 'sqrt',
              'verbose': 2}
rf25.init_model(parameters_rf25)

# This cycle runs the calculation

#for model in [cat50, cat25, cat100, cat10, rf25]:
for model in [cat25, cat50, cat10, rf25]:
    model.import_prepare_data('Input/final_features.csv', 'item_cnt_month', 'date_block_num', ids='ID')
    #for i in (30, 31, 32, 33):
    for i in (32, 33):
        model.prepare_model(i)
        model.save_importances(i)
        if i < 33:
            model.export_predictions(i, '.csv')
            model.export_predictions(i, '_cv_33_short.csv', 33)
        else:
            model.export_predictions(i, '_for_test.csv', 33)
            model.export_predictions_with_ids('_result_test.csv')


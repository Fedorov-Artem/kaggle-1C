import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle


class RunTimeSeries():
    '''
        This class will nor be used itself, only its child classes will be used.
    '''

    def __init__(self, model_name, clip, del_columns=None, load_model=False, use_second_level=True,
                 fields_to_exclude=None, prepare_func=None):
        self.model_name = model_name
        self.load_model = load_model
        self.clip = clip
        self.del_columns = del_columns
        self.use_second_level = use_second_level
        self.fields_to_exclude = fields_to_exclude
        self.prepare_func = prepare_func

        self.all_features_df = None
        self.target_field = None
        self.time_field = None
        self.ids = None
        self.test_period = None
        self.cv2_period = None
        self.model = None
        self.total_fields_to_exclude = None

    @staticmethod
    def reduce_memory(df_g):
        ints = df_g.select_dtypes('int64').columns
        df_g[ints] = df_g[ints].astype(np.int32)
        floats = df_g.select_dtypes('float64').columns
        df_g[floats] = df_g[floats].astype(np.float32)
        return df_g

    def import_prepare_data(self, path_to_features, target_field, time_field, ids=None):
        self.target_field = target_field
        self.time_field = time_field
        if self.prepare_func is None:
            self.all_features_df = pd.read_csv(path_to_features)
        else:
            self.all_features_df = self.prepare_func(pd.read_csv(path_to_features))
        self.all_features_df = self.reduce_memory(self.all_features_df)
        self.all_features_df = self.all_features_df[self.all_features_df.columns.difference(self.del_columns)]
        self.all_features_df[self.target_field] = self.all_features_df[self.target_field].clip(0, self.clip)
        if self.fields_to_exclude is None:
            self.total_fields_to_exclude = [self.target_field] + [self.time_field]
        else:
            self.total_fields_to_exclude = [self.target_field] + [self.time_field] + self.fields_to_exclude
        self.test_period = self.all_features_df[self.time_field].max()
        if self.use_second_level:
            self.cv2_period = self.test_period - 1
        if ids:
            self.ids = ids

    def calculate_scores(self, X, Y, X_cv, Y_cv, period):
        Y_training_predicted = self.model.predict(X)
        Y_cv_predicted = self.model.predict(X_cv)
        training_error = sqrt(mean_squared_error(Y, Y_training_predicted))
        test_error = sqrt(mean_squared_error(Y_cv, Y_cv_predicted))
        Y = Y.clip(0, 20)
        Y_training_predicted = Y_training_predicted.clip(0, 20)
        training_error_clip = sqrt(mean_squared_error(Y, Y_training_predicted))
        Y_cv = Y_cv.clip(0, 20)
        Y_cv_predicted = Y_cv_predicted.clip(0, 20)
        test_error_clip = sqrt(mean_squared_error(Y_cv, Y_cv_predicted))
        print(self.model_name + '_period_' + str(period) + 'training error ' + str(training_error))
        print(self.model_name + '_period_' + str(period) + 'training error clip ' + str(training_error_clip))
        print(self.model_name + '_period_' + str(period) + 'test error ' + str(test_error))
        print(self.model_name + '_period_' + str(period) + 'test error clip ' + str(test_error_clip))

    def extract_period(self, period, all_periods_before=False):
        if all_periods_before:
            df_cv = self.all_features_df.loc[self.all_features_df[self.time_field] < period].copy()
        else:
            df_cv = self.all_features_df.loc[self.all_features_df[self.time_field] == period].copy()
        X = df_cv[df_cv.columns.difference(self.total_fields_to_exclude)]
        Y = df_cv[self.target_field]
        return X, Y

    def export_predictions(self, period, suffix, export_period=None):
        if export_period is None:
            export_period = period
        prediction_field_name = 'predicted_' + self.model_name
        X_period, Y_period = self.extract_period(export_period)
        df_export = self.all_features_df.loc[self.all_features_df[self.time_field] == export_period].copy()
        df_export[prediction_field_name] = self.model.predict(X_period).clip(0, 20)
        export_file_path = 'Output/' + self.model_name + '_iter_' + str(period) + suffix
        df_export[['item_id', 'shop_id', prediction_field_name]].to_csv(export_file_path, encoding='utf-8', index=False)

# I use this function only to export data for test period
    def export_predictions_with_ids(self, suffix, period=None):
        if period is None:
            export_file_path = 'Output/' + self.model_name + suffix
            prediction_field_name = self.target_field
        else:
            export_file_path = 'Output/' + self.model_name + '_iter_' + str(period) + suffix
            prediction_field_name = 'predicted_' + self.model_name
        X_period, Y_period = self.extract_period(self.test_period)
        df_export = self.all_features_df.loc[self.all_features_df[self.time_field] == self.test_period].copy()
        df_export[prediction_field_name] = self.model.predict(X_period).clip(0, 20)
        df_export = df_export[[self.ids, self.target_field]].copy()
        df_export[self.target_field] = df_export[self.target_field].clip(0, 20)
        df_export.to_csv(export_file_path, encoding='utf-8', index=False)

    def prepare_model(self, period, model_path=None):
        if self.load_model:
            if model_path is None:
                model_path = self.model_name + '_model_iter_' + str(period) + '.bin'
            self.load_model(model_path)
        else:
            X, Y = self.extract_period(period, True)
            X_cv, Y_cv = self.extract_period(period)
            self.fit_model(X, Y, X_cv, Y_cv)
            self.calculate_scores(X, Y, X_cv, Y_cv, period)
            self.save_model(period)

    def save_importances(self, period):
        importances = self.get_importances()
        cols = self.all_features_df.columns.difference(self.total_fields_to_exclude)
        df_imp = pd.DataFrame(cols, columns=['column_name'])
        df_imp['importance'] = importances
        export_importance = 'Output/' + self.model_name + '_iter_' + str(period) + 'importance.csv'
        df_imp.to_csv(export_importance, encoding='utf-8', index=False)
        del df_imp


class RunTimeSeriesCatboost(RunTimeSeries):
    '''
        Child class for catboost
    '''

    def init_model(self, params):
        self.model = CatBoostRegressor(**params)

    def fit_model(self, X, Y, X_cv, Y_cv):
        self.model.fit(X, Y, eval_set=(X_cv, Y_cv))

    def save_model(self, period):
        model_file_name = self.model_name + '_model_iter_' + str(period) + '.bin'
        self.model.save_model(model_file_name)

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def get_importances(self):
        importances = self.model.get_feature_importance()
        return importances

class RunTimeSeriesRandomForest(RunTimeSeries):
    '''
        Child class for sklearn random forest
    '''

    def init_model(self, params):
        self.model = RandomForestRegressor(**params)

    def fit_model(self, X, Y, X_cv, Y_cv):
        self.model.fit(X, Y)

    def save_model(self, period):
        model_file_name = self.model_name + '_model_iter_' + str(period) + '.sav'
        pickle.dump(self.model, open(model_file_name, 'wb'))

    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def get_importances(self):
        importances = self.model.feature_importances_
        return importances

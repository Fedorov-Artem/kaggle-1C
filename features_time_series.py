import pandas as pd
import numpy as np


class FeaturesTimeSeries():
    '''
        This class implements useful functions for feature extraction for time series data
    '''

    def __init__(self, POS, time, quantity):
        self.POS = POS
        self.time = time
        self.quantity = quantity

    # method to add lags (corresponding value for the previous period)
    def add_lag_feature(self, df_g, lags, col_to_groupby, col):
        df_short = df_g[[self.time, self.POS, col_to_groupby, col]]
        for j in lags:
            df_tmp = df_short.copy()
            col_new = col + '_lag_' + str(j)
            df_tmp = df_tmp.rename(columns={col: col_new})
            df_tmp[self.time] = df_tmp[self.time] + j
            df_g = pd.merge(df_g, df_tmp, on=[self.time, self.POS, col_to_groupby], how='left')
            df_g[col_new] = df_g[col_new].fillna(0)
        return df_g

    # simple static method to add aggregated values
    @staticmethod
    def add_aggregate(df_from, col_groupby, col_agg, agg_type, new_col_name, df_to=None):
        if df_to is None:
            df_to = df_from
        df_g2 = df_from.groupby(col_groupby).agg({col_agg: agg_type})
        df_g2 = df_g2.rename(columns={col_agg: new_col_name}).reset_index()
        df_g = pd.merge(df_to, df_g2, on=col_groupby, how='left')
        return df_g

    # simple static method to reduce memory usage
    @staticmethod
    def reduce_memory(df_g):
        ints = df_g.select_dtypes('int64').columns
        df_g[ints] = df_g[ints].astype(np.int32)
        floats = df_g.select_dtypes('float64').columns
        df_g[floats] = df_g[floats].astype(np.float32)
        return df_g

    # simple static method to calculate sum of several lags, created by add_lag_feature method
    @staticmethod
    def sum_lags(df_g, column, lags, sum_column):
        lags_to_sum = []
        for n in lags:
            lag_n = column + '_lag_' + str(n)
            lags_to_sum.append(lag_n)
        df_g[sum_column] = df_g[lags_to_sum].sum(axis=1)
        return df_g

    # method to calculate per day average sales for a number of periods
    def add_n_month_trends(self, df_g, column, n_months):
        self.sum_lags(df_g, 'days', range(1, n_months + 1), 'total_days')
        df_g_2 = df_g.groupby([column, self.POS, self.time]).agg({self.quantity: 'sum', 'total_days': 'mean'})
        df_g_2 = df_g_2.reset_index()
        feature_name = column + '_trend_' + str(n_months) + '_months'
        lags = range(1, n_months + 1)
        df_g_2 = self.add_lag_feature(df_g_2, lags, column, self.quantity)
        lag_feature_names = self.quantity + '_lag_'
        features_to_sum = [col for col in df_g_2.columns if lag_feature_names in col]
        df_g_2['total_sold'] = df_g_2[features_to_sum].sum(axis=1)
        df_g_2[feature_name] = 0
        df_g_2.loc[df_g_2['total_days'] > 0, feature_name] = df_g_2['total_sold'] / df_g_2['total_days']
        df_g_2 = df_g_2[[column, self.POS, self.time, feature_name]]
        df_g = pd.merge(df_g, df_g_2, on=[column, self.POS, self.time], how='left')
        del df_g['total_days']
        return df_g

    # method to calculate per day average sales for a number of periods
    def add_zero_sales_rows(self, df_g, column):
        for i in range(df_g[self.time].min(), (df_g[self.time].max() + 1)):
            df_g_i = df_g[df_g[self.time] == i]
            df_g_i = df_g_i.set_index([self.time, self.POS, column])
            new_index = pd.MultiIndex.from_product(df_g_i.index.levels)
            df_g_i = df_g_i.reindex(new_index)
            df_g_i = df_g_i.reset_index()
            if i == df_g[self.time].min():
                df = df_g_i
            else:
                df = pd.concat([df, df_g_i])
        df = df.rename(index=str, columns={"level_0": self.time, "level_1": self.POS, "level_2": column})
        return df

    def last_n_days_of_previous_month_sales(self, df_g_sales, n, df_g):
        column_name = 'lag_last_' + str(n) + '_days'
        df_g_sales = self.add_aggregate(df_g_sales, self.time, 'day', 'max', 'total_days')
        df_g_sales['end_of_month'] = df_g_sales['total_days'] - n
        df_g_2 = df_g_sales.loc[df_g_sales['day'] > df_g_sales['end_of_month']].groupby(
            [self.time, self.POS, 'item_id']).agg({'item_cnt_day': 'sum'})
        df_g_2 = df_g_2.rename(columns={'item_cnt_day': column_name}).reset_index()
        df_g_2[self.time] = df_g_2[self.time] + 1
        df_g = pd.merge(df_g, df_g_2, on=[self.time, 'item_id', self.POS], how='left')
        df_g[column_name] = df_g[column_name].fillna(0)
        return df_g

    # method to calculate cumulative averages for all past periods
    def add_cumsum_average(self, df_from, columns_to_av, feature_name, df_to=None, aggregate='mean', agg_value=None):
        if df_to is None:
            df_to = df_from
        if agg_value is None:
            agg_value = self.quantity
        groupby_columns = columns_to_av + [self.time]
        if aggregate == 'mean':
            df_g = df_from.sort_values(self.time).groupby(groupby_columns).agg({agg_value: 'mean'})
        if aggregate == 'median':
            df_g = df_from.sort_values(self.time).groupby(groupby_columns).agg({agg_value: 'median'})
        df_g = df_g.rename(columns={agg_value: feature_name}).reset_index()
        df_g['counter'] = df_g.groupby(columns_to_av)[feature_name].cumcount() + 1
        df_g[feature_name] = df_g.groupby(columns_to_av)[feature_name].transform(pd.Series.cumsum)
        df_g[feature_name] = df_g[feature_name] / df_g['counter']
        del df_g['counter']
        df_g[self.time] = df_g[self.time] + 1
        df_to = pd.merge(df_to, df_g, on=groupby_columns, how='left')
        df_to[feature_name] = df_to.groupby(columns_to_av)[feature_name].ffill()
        df_to[feature_name] = df_to[feature_name].fillna(0)
        return df_to

    # method to calculate cumulative averages for all past periods
    def add_cumsum_average_per_day(self, df_from, columns_to_av, feature_name, df_to=None, aggregate='mean'):
        if df_to is None:
            df_to = df_from
        groupby_columns = columns_to_av + [self.time]
        if aggregate == 'mean':
            df_g = df_from.sort_values(self.time).groupby(groupby_columns).agg({'per_day_after_start': 'mean'})
        if aggregate == 'median':
            df_g = df_from.sort_values(self.time).groupby(groupby_columns).agg({'per_day_after_start': 'median'})
        df_g = df_g.rename(columns={'per_day_after_start': feature_name}).reset_index()
        df_g = df_g.sort_values(by=self.time)
        df_g['counter'] = df_g.groupby(columns_to_av)[feature_name].cumcount() + 1
        df_g[feature_name] = df_g.groupby(columns_to_av)[feature_name].transform(pd.Series.cumsum)
        df_g[feature_name] = df_g[feature_name] / df_g['counter']
        del df_g['counter']
        df_g[self.time] = df_g[self.time] + 1
        df_to = pd.merge(df_to, df_g, on=groupby_columns, how='left')
        df_to[feature_name] = df_to.groupby(columns_to_av)[feature_name].ffill()
        df_to[feature_name] = df_to[feature_name].fillna(0)
        return df_to

    # method to calculate per day averages for n past periods
    def add_n_month_average(self, df_g, n_months):
        feature_name = str(n_months) + '_month_average_feature'
        df_g[feature_name] = 0
        for n in range(1, n_months + 1):
            self.sum_lags(df_g, 'days', range(1, n + 1), 'total_days')
            self.sum_lags(df_g, 'item_cnt_month', range(1, n + 1), 'total_sales')
            if n == n_months:
                df_g.loc[(df_g['total_days'] > 0) & (df_g['month_after_start'] > n - 1), feature_name] = \
                    df_g['total_sales'] / df_g['total_days']
            else:
                df_g.loc[(df_g['total_days'] > 0) & (df_g['month_after_start'] == n), feature_name] = \
                    df_g['total_sales'] / df_g['total_days']
        del df_g['total_sales']
        del df_g['total_days']
        return df_g



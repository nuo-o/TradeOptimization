from utils.import_packages import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calender


class LagDataGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, lags):
        # if np.min(lags) < 96:
        #     raise ValueError('Invalid feature! No true data is available within the same day.')
        self.lags = lags

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        shifted = [x.shift(lag) for lag in self.lags]
        return pd.concat(shifted, axis=1)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        return x.loc[:, self.columns]


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        assert(isinstance(columns, list))
        self.columns = columns

    def fit(self, x, y= None):
        return self

    def transform(self, x):
        return x.drop(self.columns, axis = 1)


class DropNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.dropna()


def make_lag_feat(lag_target_name, lags, df):
    raw = ColumnSelector(lag_target_name).fit_transform(df)
    lagged = LagDataGenerator(lags).fit_transform(raw)

    rename_columns = []
    for l in lags:
        rename_columns.append(lag_target_name + '_' + str(l))

    lagged.columns = rename_columns

    return lagged


def single_ax_plot(df, x, y, figsize = (10,10), **kwargs):
    # f, a = plt.subplots(figsize = figsize)
    f,a = plt.subplots()
    df.plot(x=x, y=y, ax=a, **kwargs)
    return f


def plot_feature_importance(rf, cols):
    importances = pd.DataFrame()
    importances.loc[:, 'importances'] = rf.feature_importances_
    importances.loc[:, 'features'] = cols
    importances.sort_values('importances', inplace = True)
    # return single_ax_plot(importances, 'features', 'importances', \
    #                       kind = 'barh', color = 'b')
    return importances


def sin_cos_time(time_data, time_cycle):
    sin_ = np.sin(2 * np.pi * time_data / time_cycle)
    cos_ = np.cos(2 * np.pi * time_data / time_cycle)

    return sin_, cos_


def make_date_feature(dateCol, df):
    date = df[dateCol]
    holidays = calender().holidays(start=date.values[0], end = date.values[len(date)-1])
    weekday, month, day,year=date.dt.weekday, date.dt.month, date.dt.day, date.dt.year
    df['holiday'] = [1 if d in holidays else 0 for d in date]
    df['weekday'] = weekday
    df['day'] = day
    df['month'] = month
    df['year'] = year
    # df['sin_weekday'], df['cos_weekday'] = sin_cos_time(weekday, 7)
    # df['sin_month'], df['cos_month'] = sin_cos_time(month, 12)
    # df['sin_day'], df['cos_day'] = sin_cos_time(day, 31)
    # df['sin_PTE'], df['cos_PTE'] = sin_cos_time(df['PTE'], 96)
    return df


def dummy_feat(df):
    df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='weekday')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['PTE'], prefix='PTE')], axis=1)
    df.drop(['weekday'], axis=1)
    return df


def standardize_feat(df, on_cols):
    # standard_col = list(set(df.columns) - set(not_standar_cols))
    # df = DropNanRows().fit_transform(df)
    df[on_cols] = StandardScaler().fit_transform(df[on_cols])
    return df

def train_test_split(df, dateCol, split_date = None, splitBySize = False, train_size=0.8):
    if splitBySize:
        used = set()
        unique_days = [day for day in df[dateCol] if day not in used and (used.add(day) or True)]
        split_date = unique_days[math.floor(len(unique_days) * train_size)]

    split_index = df.index[df[dateCol] == split_date].tolist()[0]
    # train_df = df.iloc[:split_index]
    # test_df = df.iloc[split_index:]

    # return train_df, test_df
    return split_index






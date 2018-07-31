from utils.import_packages import *
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance,train_test_split


def make_feat_pipeline(target, valCol, dateCol, lag_dict, df, standardize = True, dropDate=True):
    # extract weekday, month, dayofyear, year, isholiday
    df = make_date_feature(dateCol, df)
    # df = df.drop([dateCol], axis =1)

    # hot-encoding categorical features
    dummyCol = []
    for col in df.columns:
        if ('missing' in col):
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy], axis = 1)
            dummyCol.append(col)

    if len(dummyCol)>0:
        df = df.drop(dummyCol, axis = 1)

    # generate lag features
    # lag_col_name = []
    for lag_column, lag_value in lag_dict.items():
        lag_feat, lag_name = make_lag_feat(lag_column, lag_value, df)

        # fill missing values
        temp_df = pd.concat([df[dateCol],lag_feat], axis = 1)
        temp_val = ','.join(lag_name)
        ts = TimeSeriesData(temp_df, dateCol, temp_val)
        ts.extract_PTE()
        # ts.fill_nan_by_avg(lag_name)

        df = pd.concat([df, ts.file[lag_name]], axis = 1)
        # if (0 in lag_value) | (lag_column == target):
        #     df = pd.concat([df, ts.file[lag_name]],axis=1)
        # else:
        #     df = pd.concat([df.drop([lag_column], axis = 1), ts.file[temp_val]],axis=1)
        # lag_col_name.extend(lag_name)

    # fill missing values for lag data
    # ts = TimeSeriesData(df, dateCol, ','.join(lag_col_name))
    # ts.extract_PTE()
    # ts.fill_nan_by_avg(imputingCols = lag_col_name)
    # df.loc[:lag_col_name] = ts.file[lag_col_name]

    # # standardize
    if standardize:
        num_col = list(set(valCol) - set(dummyCol)).append([lag_feat.columns])
        df = standardize_feat(df, num_col)

    # return feats and targets
    if target!=None:
        y = df[target]
        if dropDate:
            X = df.drop([target, dateCol],axis=1)
        else:
            X = df.drop([target], axis = 1)
    else:
        y = None
        if dropDate:
            X = df.drop([target, dateCol], axis= 1)
        else:
            X = df.drop([target], axis = 1)
    return X, y

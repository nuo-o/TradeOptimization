from utils.import_packages import *
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance,train_test_split


def make_feat_pipeline(target, valCol, dateCol, lag_dict, df, targetCol, standardize = True):
    # extract weekday, month, dayofyear, year, isholiday
    df = make_date_feature(dateCol, df)
    df = df.drop([dateCol], axis =1)

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
    for lag_column, lag_value in lag_dict.items():
        lag_feat = make_lag_feat(lag_column, lag_value, df)

        if (0 in lag_value) | (lag_column == target):
            df = pd.concat([df, lag_feat],axis=1)
        else:
            df = pd.concat([df.drop([lag_column])],axis=1)

    # # standardize
    if standardize:
        num_col = list(set(valCol) - set(dummyCol)).append(lag_feat.columns)
        df = standardize_feat(df, num_col)

    # return feats and targets
    y = df[targetCol]
    X = df.drop([targetCol],axis=1)

    return X, y

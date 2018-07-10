from utils.import_packages import *
import utils.hardcode_parameters as param
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance,train_test_split
from models.CrossValidate import cross_validation, Evaluator, save_result_to_file


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

    # standardize
    if standardize:
        num_col = list(set(valCol) - set(dummyCol)).append(lag_feat.columns)
        df = standardize_feat(df, num_col)

    # drop rows that missing baseline forecast data
    y = df[targetCol]
    X = df.drop([targetCol],axis=1)

    return X, y


if __name__ == '__main__':
    df, df_config = Configuration().readFile('final-diff')
    dateCol,valCol = df_config.date_col, df_config.forecast_v
    target = 'Diff'
    df = TimeSeriesData(df, df_config.date_col, df_config.forecast_v).file
    df = df[df['missing_First_Forecast_Volume'] == 0]

    split_index= train_test_split(df, dateCol,splitBySize=False,split_date=param.hold_out_date_begin)
    train_df, test_df = df[:split_index], df[split_index:]
    print('train size = {}'.format(round(len(train_df)/len(df),2)))

    lag_dict = {target: [96, 96 * 2, 96 * 3]}
    # out, feat_cols = cross_validation(valCol,dateCol,target, \
    #                                   train_df, XGBRegressor, make_feat_pipeline, lag_dict, \
    #                                   classification=False,stand=False)
    #
    # cv_mae = sum([float(mae) for mae in out['MAE']])/len(out['MAE'])
    # cv_mpe = sum([float(mpe) for mpe in out['MPE']])/len(out['MPE'])
    # print('cross validation:\nauc = {}\nacc = {}\n'.format(cv_mae, cv_mpe))

    X, y = make_feat_pipeline(target, valCol, dateCol, lag_dict, df, target, standardize=False)
    train_x, test_x = X.iloc[train_index], X.iloc[test_index]
    train_y, test_y = y[train_index], y[test_index]
    feat_cols = train_x.columns

    model = XGBRegressor
    cv_model = model()
    cv_model.fit(train_x, train_y)

    best_model = cv_model

    feat_imp = plot_feature_importance(best_model, feat_cols)[-10:]
    print(feat_imp)

    test_x, test_y = make_feat_pipeline(target, train_config.forecast_v, train_config.date_col, lag_dict, test_df, target, standardize=False)
    # test_pred = best_model2.predict_proba(test_x)[:, 1]
    test_pred = best_model.predict(test_x)

    print('hold-out:')
    metrics = Evaluator(test_pred, test_y).classification_metrics()

    # saving results
    save_result_to_file(param.data_folder_path + '/results/DA_TAKE_auc/', metrics['Accuracy'], lag_dict)
    print(metrics)
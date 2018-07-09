from utils.import_packages import *
from xgboost import XGBClassifier, XGBRegressor
import utils.hardcode_parameters as param
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance
from models.evaluation import cross_validation, Evaluator
from sklearn.preprocessing import StandardScaler
from features.train_test_split import train_test_split
from data_gathering.Configure import Configuration



def make_feat_pipeline(valCol, dateCol, lagdict, df, targetCol, standardize = True):
    # extract weekday, month, dayofyear, year, isholiday
    df = make_date_feature(dateCol, df)
    df = df.drop([dateCol], axis =1)

    # hot-encoding categorical features
    for col in df.columns:
        if ('missing' in col):
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop([col], axis = 1), dummy], axis = 1)

    # generate lag features
    for lag_column, lag_value in lag_dict.items():
        lag_feat = make_lag_feat(lag_column, lag_value, df)
        df = pd.concat([df, lag_feat], axis=1)

    # # standardize
    if standardize:
        df = standardize_feat(df, lag_feat.columns.append(valCol))

    # drop rows that missing baseline forecast data
    y = df[targetCol]
    X = df.drop([targetCol],axis=1)

    return X, y


if __name__ == '__main__':
    configuration = Configuration()
    df, df_config = configuration.readFile('final-diff')
    selected_cols = [df_config.val_col, ]
    dateCol = df_config.date_col
    valCol = df_config.forecast_v.split(',')
    save_result_df = pd.DataFrame()

    df = df[df['missing_First_Forecast_Volume'] == 0]
    df = df[]

    # train_df,test_df = train_test_split(df, dateCol,splitBySize = True, train_size=0.8)

    lag_values = {'yesterday': [96], \
                  'two_days': np.arange(96, 96 * 3), \
                  'same_time_before': [96, 96 * 2, 96 * 3], \
                  'around_time_before': [96, 97, 95 * 2, 96 * 2, 97 * 2]}
    lag_dict = {'Diff': lag_values['same_time_before']}

    X,y = make_feat_pipeline(valCol, dateCol, lag_dict, df, 'Diff', standardize=False)
    split = train_test_split(df, dateCol,splitBySize = True, train_size=0.8)
    train_x,test_x = X.iloc[:split], X.iloc[split:]
    train_y,test_y = y[:split], y[split:]
    model = XGBRegressor().fit(train_x,train_y)
    test_pred = model.predict(test_x)
    print('test:')
    metrics = Evaluator(test_pred, test_y).regression_metrics()
    print(metrics)

    # save prediction file
    save_result_df['DeliveryDate'] = df.iloc[split:][df_config.date_col]
    save_result_df['true_diff'] = test_y
    save_result_df['predict'] = test_pred
    pathName = param.data_folder_path + '/results/diff_MAPE/'+ str(int(metrics['MAPE']))+ '.xlsx'
    print('feature used:')
    print(train_x.columns)
    print('save result to:{}'.format(pathName))
    save_result_df.to_excel(pathName, index = False)

    # out, feat_cols = cross_validation(valCol, dateCol, 'Diff', train_df, XGBRegressor, make_feat_pipeline, lag_dict, stand=False ,n_folds = 2, classification=False)
    #
    # model_performance = out['test_WMAPE']
    # best_model2 = out['models'][model_performance.index(max(model_performance))]
    #
    # feat_imp = plot_feature_importance(best_model2, feat_cols)[-10:]
    # print(feat_imp)

    # test_x, test_y = make_feat_pipeline(valCol, dateCol, lag_dict, test_df, 'Diff',standardize=False)
    # test_proba = best_model2.predict_proba(test_x)[:, 1]

    # test_pred = best_model2.predict(test_x)

    # print('test:')
    # metrics = Evaluator(test_pred, test_y).classification_metrics()

    # saving results
    # save_result_to_file(param.data_folder_path + '/results/classification_acc/', metrics['Accuracy'], lag_dict)
    # print(metrics)
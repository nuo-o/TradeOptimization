from utils.import_packages import *
import utils.hardcode_parameters as param
from models.CrossValidate import cross_validation, Evaluator
from features.make_feature import make_feat_pipeline
from features.features import plot_feature_importance


if __name__ == '__main__':
    model = XGBRegressor()
    # model = LogisticRegression()
    model_name = 'XGB'
    df, df_config = Configuration().readFile('diff-MAPE') #add plant availability feature
    df = TimeSeriesData(df, df_config.date_col, df_config.forecast_v).file
    df = df.drop(['ActualVolumes','Diff'], axis = 1)
    target = 'MPE'

    # train-hold split
    hold_split_index = train_test_split(df, df_config.date_col, splitBySize=False, split_date=param.hold_out_date_begin)
    train_df, hold_df = df[:hold_split_index], df[hold_split_index:]
    # lag_dict = {target: [96]}
    lag_dict = {}
    X, Y = make_feat_pipeline(target, df_config.forecast_v, df_config.date_col, lag_dict, df, standardize=False)

    # cross-validate
    if model_name == 'LR':
        Y = [int(y) for y in Y]
        train_x, hold_x = X[:hold_split_index], X[hold_split_index:]
        train_y, hold_y = Y[:hold_split_index], Y[hold_split_index:]
    else:
        train_x, hold_x = X[:hold_split_index], X[hold_split_index:]
        train_y, hold_y = Y[:hold_split_index], Y[hold_split_index:]

    out, feat_cols = cross_validation(train_x,train_y, model, classification=False)

    # feature importance
    if model_name == 'XGB':
        feat_imp = plot_feature_importance(model, feat_cols)[-10:]
        print(feat_imp)

    cv_max_score = max(out['test_WMAPE(%)'])
    model = out['models'][out['test_WMAPE(%)'].index(cv_max_score)]
    cv_avg_score = np.mean(np.array(out['test_WMAPE(%)']))
    print('cross validation:\nWMAPE(%)={}'.format(round(cv_avg_score,4)))

    # hold-out test
    print('hold-out:')
    X,y = make_feat_pipeline(target, df_config.forecast_v, df_config.date_col, lag_dict, hold_df, standardize=False)
    prediction = model.predict(X)
    hold_metrics = Evaluator(prediction, y).regression_metrics()
    print(hold_metrics)

    # save prediction file
    hold_prediction = pd.DataFrame()
    hold_prediction['DeliveryDate'] = df.iloc[hold_split_index:][df_config.date_col]
    hold_prediction['true_diff'] = y
    hold_prediction['predict_diff'] = prediction
    prediction_path = param.data_folder_path + '/results/hold-out-prediction/' + target + '_WMAPE_' + str(round(hold_metrics['WMAPE(%)'],4)) + '.xlsx'
    hold_prediction.to_excel(prediction_path, index=False)
    print('save hold-out prediction to {}'.format(prediction_path))

    # save result to file
    save_result_path = param.data_folder_path + '/results/train_results/'+target+'_WMAPE_'+str(round(hold_metrics['WMAPE(%)'],4)) + '.txt'
    with open(save_result_path, 'w') as f:
        f.write('train_size : test_size = {}:{}\n'.format(round(len(train_df) / len(df), 2), round(len(hold_df) / len(df), 2)))

        f.write('\nCross Validation Score:\n')
        f.write('WMAPE = {}\n'.format(cv_max_score))

        f.write('\nTest Score:\n')
        f.write('WMAPE = {}\n'.format(hold_metrics['WMAPE(%)']))

        f.write('\nmodel:\n')
        f.write(str(model))

        f.write('\nfeature used:\n')
        for feat in X.columns:
            f.write('{}\n'.format(feat))

        if model_name == 'XGB':
            f.write('\ntop 10 important features:\n')
            for imp, feat in zip(feat_imp['importances'], feat_imp['features']):
                f.write('feat_{}:\t{}\n'.format(feat, imp))


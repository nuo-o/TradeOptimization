from utils.import_packages import *
import utils.hardcode_parameters as param
from features.features import plot_feature_importance
from models.CrossValidate import cross_validation, Evaluator
from features.make_feature import make_feat_pipeline
import pickle

if __name__ == '__main__':
    model = XGBRegressor()
    model_name = 'XGB'

    df, df_config = Configuration().readFile('OTC-DA')
    df = TimeSeriesData(df, df_config.date_col, df_config.forecast_v, pteCol=df_config.pte_col).file
    target = 'DA'

    train_date_start = datetime(2015,1,3)
    hold_out_date_start = datetime(2017,8,22)
    hold_out_date_end = datetime(2018,7, 23)

    df = df[(df[df_config.date_col] >= train_date_start) & (df[df_config.date_col]<=hold_out_date_end)]
    df = df.dropna(subset=['VWAP','DA'])

    hold_split_index = train_test_split(df, df_config.date_col, splitBySize=False, split_date=hold_out_date_start)
    train_df, hold_df = df[:hold_split_index], df[hold_split_index:]
    print('train:test = {}:{}'.format(round( len(train_df)/len(df), 2),round( len(hold_df)/len(df), 4)))

    # train-evaluate split
    lag_dict = {target:[96],
               'VWAP': [96,96*2]}
    X, y = make_feat_pipeline(target, df_config.forecast_v, df_config.date_col, lag_dict, df, standardize=False)

    train_x, hold_x = X[:hold_split_index], X[hold_split_index:]
    train_y, hold_y = y[:hold_split_index], y[hold_split_index:]

    # cross validation
    out, feat_cols = cross_validation(train_x, train_y, model, classification=False,n_folds=10)

    # get the best model
    cv_max_score = min(out['test_MAPE(%)'])
    model = out['models'][out['test_MAPE(%)'].index(cv_max_score)]
    pickle.dump(model, open("DA_OTC_XGB.pickle.dat", "wb"))

    # cv_avg_score = np.mean(np.array(out['test_MAPE(%)']))
    print('cross validation:\nMAPE(%)={}'.format(round(cv_max_score,6)))

    # feature importance
    if model_name == 'XGB':
        feat_imp = plot_feature_importance(model, feat_cols)[-10:]
        print(feat_imp)

    # hold-out test
    print('hold-out:')
    model = pickle.load(open("DA_OTC_XGB.pickle.dat","rb"))
    X,y = make_feat_pipeline(target, df_config.forecast_v, df_config.date_col, lag_dict, hold_df, standardize=False)
    prediction = model.predict(X)
    hold_metrics = Evaluator(prediction, y).regression_metrics()
    print(hold_metrics)

    # save prediction file
    hold_prediction = pd.DataFrame()
    hold_prediction['DeliveryDate'] = df.iloc[hold_split_index:][df_config.date_col]
    hold_prediction['true_DA'] = y
    hold_prediction['predict_DA'] = prediction
    prediction_path = param.data_folder_path + '/results/hold-out-prediction/DA_MAPE_' + str(round(hold_metrics['MAPE(%)'],5)) + '.xlsx'
    hold_prediction.to_excel(prediction_path, index=False)
    print('save hold-out prediction to {}'.format(prediction_path))

    # save result to file
    save_result_path = param.data_folder_path + '/results/train_results/DA_MAPE_'+str(round(hold_metrics['MAPE(%)'],5)) + '.txt'
    with open(save_result_path, 'w') as f:
        f.write('train_size : test_size = {}:{}\n'.format(round(len(train_df) / len(df), 2), round(len(hold_df) / len(df), 5)))

        f.write('\nCross Validation Score:\n')
        f.write('MAPE(%) = {}\n'.format(cv_max_score))

        f.write('\nTest Score:\n')
        f.write('MAPE(%) = {}\n'.format(hold_metrics['MAPE(%)']))

        f.write('\nmodel:\n')
        f.write(str(model))

        f.write('\nfeature used:\n')
        for feat in X.columns:
            f.write('{}\n'.format(feat))

        if model_name == 'XGB':
            f.write('\ntop 10 important features:\n')
            for imp, feat in zip(feat_imp['importances'], feat_imp['features']):
                f.write('feat_{}:\t{}\n'.format(feat, imp))


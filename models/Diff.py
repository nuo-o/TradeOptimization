from utils.import_packages import *
import utils.hardcode_parameters as param
from models.CrossValidate import cross_validation, Evaluator
from data_gathering.Configure import Configuration
from data_gathering.CleanData import TimeSeriesData
from features.make_feature import make_feat_pipeline
from features.features import plot_feature_importance
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit



if __name__ == '__main__':
    model = XGBRegressor(learning_rate=0.005)
    df, df_config = Configuration().readFile('final-diff')
    date_col,val_col = df_config.date_col, df_config.forecast_v
    target = 'Diff'

    df = df[df['missing_First_Forecast_Volume'] == 0]
    ts = TimeSeriesData(df, date_col, val_col)
    df = ts.file

    # train-hold split
    hold_split_index = train_test_split(df, date_col, splitBySize=False, split_date=param.hold_out_date_begin)
    train_df, hold_df = df[:hold_split_index], df[hold_split_index:]
    lag_dict = {target: [96, 96*2, 96*3]}
    X, Y = make_feat_pipeline(target, val_col, date_col, lag_dict, df, target, standardize=False)
    train_x, hold_x = X[:hold_split_index], X[hold_split_index:]
    train_y, hold_y = Y[:hold_split_index], Y[hold_split_index:]

    # cross-validate
    out, feat_cols = cross_validation(train_x,train_y, model, classification=False)

    cv_score = max(out['test_WMAPE(%)'])
    model = out['models'][out['test_WMAPE(%)'].index(cv_score)]

    # feature importance
    feat_imp = plot_feature_importance(model, feat_cols)[-10:]
    print(feat_imp)

    # hold-out test
    print('hold-out:')
    X,y = make_feat_pipeline(target, val_col, date_col, lag_dict, hold_df, target, standardize=False)
    prediction = model.predict(X)
    hold_metrics = Evaluator(prediction, y).regression_metrics()
    print(hold_metrics)

    # save prediction file
    hold_prediction = pd.DataFrame()
    hold_prediction['DeliveryDate'] = df.iloc[hold_split_index:][df_config.date_col]
    hold_prediction['true_diff'] = y
    hold_prediction['predict_diff'] = prediction
    prediction_path = param.data_folder_path + '/results/hold-out-prediction/diff_MAE_' + str(int(hold_metrics['MAE'])) + '.xlsx'
    hold_prediction.to_excel(prediction_path, index=False)
    print('save hold-out prediction to {}'.format(prediction_path))

    # save result to file
    save_result_path = param.data_folder_path + '/results/train_results/diff_MAE_'+str(int(hold_metrics['MAE'])) + '.txt'
    with open(save_result_path, 'w') as f:
        f.write('train_size : test_size = {}:{}\n'.format(round(len(train_df) / len(df), 2), round(len(hold_df) / len(df), 2)))

        f.write('\nCross Validation Score:\n')
        f.write('WMAPE = {}\n'.format(cv_score))

        f.write('\nTest Score:\n')
        f.write('WMAPE = {}\n'.format(hold_metrics['WMAPE(%)']))

        f.write('\nmodel:\n')
        f.write(str(model))

        f.write('\nfeature used:\n')
        for feat in X.columns:
            f.write('{}\n'.format(feat))

        f.write('\ntop 10 important features:\n')
        for imp, feat in zip(feat_imp['importances'], feat_imp['features']):
            f.write('feat_{}:\t{}\n'.format(feat, imp))

    # lag_dict = {target: [96, 96*2, 96*3]}
    # X,y = make_feat_pipeline(target, valCol.split(','), dateCol, lag_dict, train_df, target, standardize=False)
    # validate_split_index = train_test_split(train_df, dateCol,splitBySize = True, train_size=0.8)
    # train_x,valid_x,train_y,valid_y = X.iloc[:validate_split_index], X.iloc[validate_split_index:], y[:validate_split_index], y[validate_split_index:]

    # print('train:valid:test = {}:{}:{}'.format(\
    #     round( len(train_x)/len(df), 2),round( len(valid_x)/len(df), 2),round( len(hold_df)/len(df), 2)))
    #
    # model = model.fit(train_x,train_y)
    # train_pred = model.predict(train_x)
    # valid_pred = model.predict(valid_x)
    # train_metrics = Evaluator(train_pred, train_y).regression_metrics()
    # valid_metrics = Evaluator(valid_pred, valid_y).regression_metrics()
    # print('train:')
    # print(train_metrics)
    # print('validation:')
    # print(valid_metrics)

    # # hold-out test
    # print('hold-out:')
    # X,y = make_feat_pipeline(target, valCol.split(','), dateCol, lag_dict, hold_df, target, standardize=False)
    # prediction = model.predict(X)
    # hold_metrics = Evaluator(prediction, y).regression_metrics()
    # print(hold_metrics)
    #
    # # save prediction file
    # hold_prediction = pd.DataFrame()
    # hold_prediction['DeliveryDate'] = df.iloc[hold_split_index:][df_config.date_col]
    # hold_prediction['true_diff'] = y
    # hold_prediction['predict_diff'] = prediction
    # prediction_path = param.data_folder_path + '/results/hold-out-prediction/diff_MAE_' + str(int(hold_metrics['MAE']))+ '.xlsx'
    # hold_prediction.to_excel(prediction_path, index=False)
    # print('save hold-out prediction to {}'.format(prediction_path))
    #
    # # save train result to txt file
    # save_result_path = param.data_folder_path + '/results/train_results/diff_MAE_'+str(int(hold_metrics['MAE'])) + '.txt'
    # with open(save_result_path, 'w') as f:
    #     f.write('train:valid:test = {}:{}:{}\n'.format( \
    #         round(len(train_x) / len(df), 2), round(len(valid_x) / len(df), 2), round(len(hold_df) / len(df), 2)))
    #     f.write('\nfeature used:\n')
    #     for feat in X.columns:
    #         f.write('{}\n'.format(feat))
    #     f.write('\nTrain Metrics:\n')
    #     f.write(str(train_metrics))
    #     f.write('\nValid Metrics\n')
    #     f.write(str(valid_metrics))
    #     f.write('\nTest Metrics\n')
    #     f.write(str(hold_metrics))
    #     f.write('\nModel:\n')
    #     f.write(str(model))
    #
    # print('\nfeature used:')
    # print(X.columns)
    # print('save train figure to:{}'.format(save_result_path))

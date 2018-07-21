from utils.import_packages import *
import utils.hardcode_parameters as param
from features.features import plot_feature_importance
from models.CrossValidate import cross_validation, Evaluator
from features.make_feature import make_feat_pipeline

if __name__ == '__main__':
    model = XGBClassifier(learning_rate=0.001)
    # df, df_config = Configuration().readFile('final-take')
    df, df_config = Configuration().readFile('final-take2')
    date_col, val_col, target = df_config.date_col, df_config.forecast_v, 'DA>TAKE'
    ts = TimeSeriesData(df, date_col, val_col).file

    # checked, no da price is missing
    df = df[df['missing_take_from_system_kWhPTE'] == 0]
    ts = TimeSeriesData(df, df_config.date_col, df_config.forecast_v)
    df = ts.file

    # train-test split (as cross-validation, no evaluate set is split)
    hold_split_index = train_test_split(df, date_col, splitBySize=False, split_date=param.hold_out_date_begin)
    train_df, hold_df = df[:hold_split_index], df[hold_split_index:]
    print('train:test = {}:{}'.format(round( len(train_df)/len(df), 2),round( len(hold_df)/len(df), 2)))

    # train-evaluate split
    lag_dict = {target: [96, 96 * 2, 96 * 3],
                'DA': [96],
                'take_from_system_kWhPTE': [96]}
    X, y = make_feat_pipeline(target, val_col, date_col, lag_dict, df, target, standardize=False)

    train_x, hold_x = X[:hold_split_index], X[hold_split_index:]
    train_y, hold_y = y[:hold_split_index], y[hold_split_index:]

    train_data_to_dataRobot = train_x
    train_data_to_dataRobot[target] = train_y
    test_data_to_dataRobot = hold_x
    test_data_to_dataRobot[target] = hold_y
    train_data_to_dataRobot.to_excel(param.data_folder_path + '/data_robot/DA_TAKE_train_feat.xlsx', index = False)
    test_data_to_dataRobot.to_excel(param.data_folder_path + '/data_robot/DA_TAKE_test_feat.xlsx', index = False)

    # cross validation
    out, feat_cols = cross_validation(train_x, train_y, model, classification=True)

    cv_auc = max(out['auc'])
    best_model = out['models'][out['auc'].index(cv_auc)]
    cv_acc = out['acc'][out['auc'].index(cv_auc)]
    model = best_model

    # feature importance
    feat_imp = plot_feature_importance(model, feat_cols)[-10:]
    print(feat_imp)

    # hold-out test
    print('hold-out:')
    X,y = make_feat_pipeline(target, val_col, date_col, lag_dict, hold_df, target, standardize=False)
    prediction = model.predict(X)
    prediction_proba = model.predict_proba(X)[:,1]
    hold_metrics = Evaluator(prediction, y).classification_metrics()
    print(hold_metrics)

    # save prediction file
    hold_prediction = pd.DataFrame()
    hold_prediction['DeliveryDate'] = df.iloc[hold_split_index:][df_config.date_col]
    hold_prediction['true_DA>TAKE'] = y
    hold_prediction['predict_DA>TAKE_proba'] = prediction_proba
    hold_prediction['predict_DA>TAKE'] = prediction
    prediction_path = param.data_folder_path + '/results/hold-out-prediction/TAKE_AUC_' + str((hold_metrics['AUC'])) + '.xlsx'
    hold_prediction.to_excel(prediction_path, index=False)
    print('save hold-out prediction to {}'.format(prediction_path))

    # save result to file
    save_result_path = param.data_folder_path + '/results/train_results/TAKE_AUC_'+str((hold_metrics['AUC'])) + '.txt'
    with open(save_result_path, 'w') as f:
        f.write('train_size : test_size = {}:{}\n'.format(round(len(train_df) / len(df), 2), round(len(hold_df) / len(df), 2)))

        f.write('pos class ratio:\n')
        train_df_pos_ratio = round(get_pos_class_ratio(train_df, 1, target), 2)
        test_df_pos_ratio = round(get_pos_class_ratio(hold_df, 1, target), 2)
        f.write('train = {}, test = {}\n'.format(train_df_pos_ratio, test_df_pos_ratio))

        f.write('\nCross Validation Score:\n')
        f.write('auc = {}\n'.format(cv_auc))
        f.write('acc = {}\n'.format(cv_acc))

        f.write('\nTest Score:\n')
        f.write('auc = {}\n'.format(hold_metrics['AUC']))
        f.write('acc = {}\n'.format(hold_metrics['Accuracy']))

        f.write('\nmodel:\n')
        f.write(str(model))

        f.write('\nfeature used:\n')
        for feat in X.columns:
            f.write('{}\n'.format(feat))

        f.write('\ntop 10 important features:\n')
        for imp, feat in zip(feat_imp['importances'], feat_imp['features']):
            f.write('feat_{}:\t{}\n'.format(feat, imp))

from utils.import_packages import *
import utils.hardcode_parameters as param
from features.features import plot_feature_importance
from models.CrossValidate import cross_validation, Evaluator, save_result_to_file
from features.make_feature import make_feat_pipeline

if __name__ == '__main__':
    model = XGBClassifier(learning_rate=0.001)
    df, df_config = Configuration().readFile('final-take')
    date_col = df_config.date_col
    val_col = df_config.forecast_v
    ts = TimeSeriesData(df, date_col, val_col).file
    target = 'DA>TAKE'

    # checked, no da price is missing
    df = df[df['missing_take_from_system_kWhPTE'] == 0]
    ts = TimeSeriesData(df, df_config.date_col, df_config.forecast_v)
    df = ts.file

    pos_train = df[df[target] == 1]
    neg_train = df[df[target] == 0]
    a = len(pos_train) / len(df) * 100
    print('pos class proportion:{}'.format(round(a,2)))

    # train-holdout split
    hold_split_index = train_test_split(df, date_col, splitBySize=False, split_date=param.hold_out_date_begin)
    train_df, hold_df = df[:hold_split_index], df[hold_split_index:]

    # train-evaluate split
    lag_dict = {target: [96, 96 * 2, 96 * 3],
                'DA': [96],
                'take_from_system_kWhPTE': [96]}
    X,y = make_feat_pipeline(target, val_col.split(','), date_col, lag_dict, train_df, target, standardize=False)
    validate_split_index = train_test_split(train_df, date_col,splitBySize = True, train_size=0.8)
    train_x,valid_x,train_y,valid_y = X.iloc[:validate_split_index], X.iloc[validate_split_index:], y[:validate_split_index], y[validate_split_index:]
    print('train:valid:test = {}:{}:{}'.format(\
        round( len(train_x)/len(df), 2),round( len(valid_x)/len(df), 2),round( len(hold_df)/len(df), 2)))

    # cross validation
    out, feat_cols = cross_validation(val_col, date_col,target, \
                                      train_df, model, make_feat_pipeline, lag_dict, stand=False)

    cv_auc = sum([float(auc) for auc in out['auc']])/len(out['auc'])
    cv_acc = sum([float(acc) for acc in out['acc']])/len(out['acc'])
    print('cross validation:\nauc = {}\nacc = {}\n'.format(cv_auc, cv_acc))

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
    hold_prediction['predict_DA>TAKE'] = prediction_proba
    prediction_path = param.data_folder_path + '/results/hold-out-prediction/TAKE_AUC_' + str((hold_metrics['AUC'])) + '.xlsx'
    hold_prediction.to_excel(prediction_path, index=False)
    print('save hold-out prediction to {}'.format(prediction_path))
    # test_pred = best_model2.predict_proba(test_x)[:, 1]

    # save train result to txt file
    save_result_path = param.data_folder_path + '/results/train_results/TAKE_AUC_'+str((hold_metrics['AUC'])) + '.txt'
    with open(save_result_path, 'w') as f:
        f.write('train:valid:test = {}:{}:{}\n'.format( \
            round(len(train_x) / len(df), 2), round(len(valid_x) / len(df), 2), round(len(hold_df) / len(df), 2)))
        f.write('\nfeature used:\n')
        for feat in X.columns:
            f.write('{}\n'.format(feat))
        f.write('\nCross Validation Score:\n')
        f.write('auc = {}\n'.format(cv_auc))
        f.write('acc = {}\n'.format(cv_acc))
        f.write('\nTest Score:\n')
        f.write('auc = {}\n'.format(hold_metrics['AUC']))
        f.write('acc = {}\n'.format(hold_metrics['Accuracy']))
        f.write('\nmodel:\n')
        f.write(str(model))

    print('all done')
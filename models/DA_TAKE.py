from utils.import_packages import *
from xgboost import XGBClassifier, XGBRegressor
import utils.hardcode_parameters as param
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance, FileSaver
from models.evaluation import cross_validation, Evaluator, save_result_to_file
from sklearn.preprocessing import StandardScaler


def make_feat_pipeline(lagdict, df, targetCol):
    # df = lag_dict['df']
    df = make_date_feature(df)
    # hot-encoding weekday and pte
    dummy_date = pd.get_dummies(df['weekday'], prefix='weekday')
    dummy_PTE = pd.get_dummies(df['PTE'], prefix='PTE')
    df = pd.concat([df, dummy_date, dummy_PTE], axis=1)
    df.drop(['weekday'], axis=1)

    # lag feat
    for lag_column, lag_value in lag_dict.items():
        lag_feat = make_lag_feat(lag_column, lag_value, df)
        df = pd.concat([df, lag_feat], axis=1)

    # standardize
    clasy_cols = ['DA>FEED', 'DA>TAKE', 'Date', 'DateTime', 'day', 'month', 'PTE', 'weekday',
                  'week', 'sin_weekday', 'sin_month', 'sin_day', 'sin_PTE',
                  'take_from_system_price', 'feed_into_system_price', 'UTC']
    clasy_cols.extend(dummy_date.columns)
    clasy_cols.extend(dummy_PTE.columns)

    df = standardize_feat(df, clasy_cols)

    # select x,y
    # targetCol = lag_dict[target]
    y = df[targetCol]
    X = df.drop(['DA>FEED', 'DA>TAKE', 'take_from_system_price', 'feed_into_system_price', \
                 'DA-price', 'system_purchase_vol', 'system_sell_vol', 'system_absolute_vol', \
                 'DateTime', 'week', 'Date', 'PTE', 'weekday', 'UTC'], \
                axis=1)

    return X, y


if __name__ == '__main__':
    # train_df = pd.read_excel(param.data_folder_path + '/train_imb_no_demd.xlsx')
    # test_df = pd.read_excel(param.data_folder_path + '/test_imb_no_demd.xlsx')
    train_df = pd.read_excel(param.data_folder_path + '/train_imb_demd.xlsx')
    test_df = pd.read_excel(param.data_folder_path + '/test_imb_demd.xlsx')

    pos_train = train_df[train_df['DA>TAKE'] == 1]
    neg_train = train_df[train_df['DA>TAKE'] == 0]
    a = len(pos_train) / len(train_df) * 100
    print('pos class proportion:{}'.format(a))

    lag_values = {'yesterday': [96], \
                  'two_days': np.arange(96, 96 * 3), \
                  'same_time_before': [96, 96 * 2, 96 * 3], \
                  'around_time_before': [96, 97, 95 * 2, 96 * 2, 97 * 2]}
    lag_dict = {'take_from_system_price': lag_values['yesterday'], \
                'feed_into_system_price': lag_values['yesterday'], \
                'DA-price': lag_values['around_time_before'], \
                'system_purchase_vol': lag_values['around_time_before'], \
                'system_sell_vol': lag_values['around_time_before'], \
                'system_absolute_vol': lag_values['around_time_before'], \
                'wind_value': lag_values['around_time_before']}

    out, feat_cols = cross_validation( 'DA>TAKE', train_df, XGBClassifier, make_feat_pipeline, lag_dict)

    model_auc = out['auc']
    best_model2 = out['models'][model_auc.index(max(model_auc))]

    feat_imp = plot_feature_importance(best_model2, feat_cols)[-10:]
    print(feat_imp)

    test_x, test_y = make_feat_pipeline(lag_dict, test_df, 'DA>TAKE')
    # test_proba = best_model2.predict_proba(test_x)[:, 1]
    test_pred = best_model2.predict(test_x)

    print('test:')
    metrics = Evaluator(test_pred, test_y).classification_metrics()

    # saving results
    save_result_to_file(param.data_folder_path + '/results/classification_acc/', metrics['Accuracy'], lag_dict)
    print(metrics)
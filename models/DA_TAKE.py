from utils.import_packages import *
from xgboost import XGBClassifier, XGBRegressor
import utils.hardcode_parameters as param
from features.features import make_date_feature, make_lag_feat, standardize_feat, plot_feature_importance
from models.evaluation import cross_validation, Evaluator, save_result_to_file
from sklearn.preprocessing import StandardScaler
from data_gathering.Configure import Configuration
from models.DIFF_model import make_feat_pipeline
from data_gathering.CleanData import TimeSeriesData


if __name__ == '__main__':
    train_df, train_config = Configuration().readFile('train_imb_no_demand')
    train_df = TimeSeriesData(train_df, train_config.date_col, train_config.forecast_v).file
    test_df, test_config = Configuration().readFile('test_imb_no_demand')
    test_df = TimeSeriesData(test_df, test_config.date_col, test_config.forecast_v).file
    target = 'DA>TAKE'

    pos_train = train_df[train_df['DA>TAKE'] == 1]
    neg_train = train_df[train_df['DA>TAKE'] == 0]
    a = len(pos_train) / len(train_df) * 100
    print('pos class proportion:{}'.format(a))

    lag_dict = {'DA>TAKE': [96,96*2,96*3]}

    out, feat_cols = cross_validation(train_config.forecast_v, train_config.date_col,target, \
                                      train_df, XGBClassifier, make_feat_pipeline, lag_dict, stand=False)

    cv_auc = sum([float(auc) for auc in out['auc']])/len(out['auc'])
    cv_acc = sum([float(acc) for acc in out['acc']])/len(out['acc'])
    print('cross validation:\nauc = {}\nacc = {}\n'.format(cv_auc, cv_acc))

    # model_auc = out['auc']
    # best_model = out['models'][model_auc.index(max(model_auc))]
    best_model = out['models'][len(out)-1]

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
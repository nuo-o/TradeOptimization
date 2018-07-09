from utils.import_packages import *
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit


def plot_predictions(pred, true_v):
    results = pd.DataFrame()
    results.loc[:, 'true_values'] = true_v
    results.loc[:, 'predictions'] = pred

    f, a = plt.subplots()
    results.plot(ax=a, y='true_values')
    results.plot(ax=a, y='predictions')

    return f


def save_result_to_file(file_path, result, feat_dict, notes = None):
    with open(file_path + str(result) + str(notes)+'.txt', 'w') as f:
        for lag_column, lag_value in feat_dict.items():
            row = str(lag_column) + ':' + str(lag_value)
            f.write('%s\n' % row)


def cross_validation(valCol, dateCol, target, df, model, make_feat, feat_params, n_folds=5, classification = True, print_temp_info=True, stand = True):
    results = defaultdict(list)
    ts_split = TimeSeriesSplit(n_splits=n_folds)
    feat_cols = None

    for fold, (train_index, test_index) in enumerate(ts_split.split(df), 1):
        start_time = time.time()
        print('Fold:{}'.format(fold))
        # cv_train = df.iloc[train_index, :]
        # cv_test = df.iloc[test_index, :]

        # feat_params['df'] = cv_train
        X,y = make_feat(valCol, dateCol, feat_params, df, target, standardize = stand)
        train_x = X.iloc[train_index]
        test_x = X.iloc[test_index]
        train_y = y[train_index]
        test_y = y[test_index]
        # train_x, train_y = make_feat(valCol,dateCol, feat_params, cv_train, target, standardize = stand)
        # test_x, test_y = make_feat(valCol, dateCol, feat_params, cv_test, target, standardize = stand)
        feat_cols = train_x.columns

        cv_model = model()
        cv_model.fit(train_x, train_y)
        #             train_pred = cv_model.predict_proba(train_x)[:, 1]
        #             test_pred = cv_model.predict_proba(test_x)[:, 1]
        train_pred = cv_model.predict(train_x)
        test_pred = cv_model.predict(test_x)

        if classification:
            train_metrics = Evaluator(train_pred, train_y).classification_metrics()
            test_metrics = Evaluator(test_pred, test_y).classification_metrics()
            results['auc'].append(test_metrics['AUC'])
            results['models'].append(cv_model)
        else:
            train_metrics = Evaluator(train_pred, train_y).regression_metrics()
            test_metrics = Evaluator(test_pred, test_y).regression_metrics()
            results['test_WMAPE'].append(test_metrics['WMAPE'])
            results['models'].append(cv_model)

        if print_temp_info:
            print('train:')
            print(train_metrics)
            print('test:')
            print(test_metrics)
            print('takes:{} min\n'.format(round((time.time() - start_time) / 60, 2)))

    return results, feat_cols


class Evaluator():
    def __init__(self, prediction, true_values):
        self.pred = prediction
        self.true = true_values

    def mean_percentage_error(self):
        pe = (self.pred - self.true)/ self.true
        pe = np.sum(pe)
        mpe = 100*pe/len(self.pred)
        return mpe

    def mean_abs_percentage_error(self):
        ape = (self.pred - self.true)/self.true
        ape = np.sum(abs(ape))
        mape = 100*ape/len(self.pred)
        return mape

    def weighted_mape(self):
        ae = abs((self.pred - self.true))
        mae = np.sum(ae)/np.sum(self.true)
        wmape = 100*mae/len(self.pred)
        return wmape

    def area_under_curve(self):
        fpr, tpr, _ = roc_curve(self.true, self.pred, pos_label=1)
        return '%.4f' % auc(fpr, tpr)

    def regression_metrics(self):
        error = self.true-self.pred
        relative_error = error / self.true
        mpe = round(100*np.sum(relative_error)/len(self.pred),4)
        mape = round(100*np.sum(abs(relative_error))/len(self.pred),4)
        wmape = round(100*np.sum(abs(error))/np.sum(abs(self.true))/len(self.pred),4)
        metrics = {}
        metrics['MPE'] = mpe
        metrics['MAPE'] = mape
        metrics['WMAPE'] = wmape

        return metrics

    def classification_metrics(self):
        fpr, tpr, _ = roc_curve(self.true, self.pred, pos_label=1)
        AUC = '%.4f' % auc(fpr, tpr)
        error = self.true - self.pred
        accuracy = round(100*(1 - np.sum(abs(error)/len(self.pred))), 4)
        metrics = {}
        metrics['AUC'] = AUC
        metrics['Accuracy'] = accuracy
        return metrics
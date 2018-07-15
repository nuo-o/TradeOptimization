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

#
# def save_result_to_file(file_path, result, feat_dict, notes = None):
#     with open(file_path + str(result) + str(notes)+'.txt', 'w') as f:
#         for lag_column, lag_value in feat_dict.items():
#             row = str(lag_column) + ':' + str(lag_value)
#             f.write('%s\n' % row)


def cross_validation(X, y, cv_model, n_folds=5, classification = True, print_temp_info=True):
    results = defaultdict(list)
    ts_split = TimeSeriesSplit(n_splits=n_folds)
    feat_cols = None

    for fold, (train_index, test_index) in enumerate(ts_split.split(X), 1):
        start_time = time.time()
        print('Fold:{}'.format(fold))

        # train_x,test_x = X.iloc[train_index], X.iloc[test_index]
        # train_y,test_y = y[train_index],y[test_index]
        train_x = X.iloc[:test_index[0]]
        train_y = y[:test_index[0]]
        test_x = X.iloc[test_index[0]:test_index[-1]]
        test_y = y[test_index[0]:test_index[-1]]

        feat_cols = train_x.columns

        cv_model.fit(train_x, train_y)

        if classification:
            train_pred = cv_model.predict_proba(train_x)[:, 1]
            test_pred = cv_model.predict_proba(test_x)[:, 1]
            train_metrics = Evaluator(train_pred, train_y).classification_metrics()
            test_metrics = Evaluator(test_pred, test_y).classification_metrics()
            results['auc'].append(test_metrics['AUC'])
            results['acc'].append(test_metrics['Accuracy'])
            results['models'].append(cv_model)
        else:
            train_pred = cv_model.predict(train_x)
            test_pred = cv_model.predict(test_x)
            train_metrics = Evaluator(train_pred, train_y).regression_metrics()
            test_metrics = Evaluator(test_pred, test_y).regression_metrics()
            results['test_WMAPE(%)'].append(test_metrics['WMAPE(%)'])
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
        mae = round(np.sum(abs(relative_error)),4)
        mpe = round(100*np.sum(relative_error)/len(self.pred),4)
        mape = round(100*np.sum(abs(relative_error))/len(self.pred),4)
        wmape = round(100*np.sum(abs(error))/np.sum(abs(self.true))/len(self.pred),4)
        metrics = {}
        metrics['MAE'] = mae
        metrics['MPE(%)'] = mpe
        metrics['MAPE(%)'] = mape
        metrics['WMAPE(%)'] = wmape

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
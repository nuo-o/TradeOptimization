from utils.import_packages import *
import utils.hardcode_parameters as param
from features.make_feature import make_feat_pipeline
import pickle


if __name__ == '__main__':
    """Input: OTC price(history+deliveryday)
    Output: predict_DA_daily, predict_DA price per pte.
    Results are saved in /data/operation/operation_pred_DA.xlsx"""

    predict_date_start = datetime(2018, 7, 30)

    target = 'DA'
    forecast_v ='VWAP,DA'
    dateCol = 'DeliveryDate'
    lag_dict = {target: [96],
                'VWAP': [96, 96 * 2]}

    model = pickle.load(open("../models/DA_OTC_XGB.pickle.dat", "rb"))

    max_lag_days = 2
    lag_date_start = predict_date_start - timedelta(days=max_lag_days)
    df = pd.read_excel(param.operation_folder + '/OTC_lag_DA.xlsx', sheet_name='Sheet1')
    df = df[df[dateCol] >= lag_date_start]
    df = df.drop(['TRADEDATE'], axis = 1)

    X,_ = make_feat_pipeline(target,forecast_v,dateCol,lag_dict,df,standardize=False,dropDate=False)
    X = X[X[dateCol] >= predict_date_start]

    df = df[df[dateCol] >= predict_date_start]
    feat = X.drop([dateCol], axis =1)
    prediction = model.predict(feat)

    df['predict_DA'] = prediction
    df['predict_DA_daily'] = [prediction.mean()]*len(X)

    df = df[['VWAP','DeliveryDate','start','predict_DA','predict_DA_daily']]

    con = Configuration()
    df['DeliveryDate'] = con.add_time_to_date(df,'DeliveryDate','start')
    df = df.drop(['start'], axis=1)

    df.to_excel(param.operation_folder + '/operation_pred_DA.xlsx',index=False)



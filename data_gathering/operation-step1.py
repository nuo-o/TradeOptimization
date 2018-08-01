from utils.import_packages import *
import utils.hardcode_parameters as param
from features.make_feature import make_feat_pipeline, make_date_feature, make_lag_feat
import pickle


if __name__ == '__main__':
    target_date_start = datetime(2018, 1,1)

    model = pickle.load(open("../models/DA_OTC_XGB.pickle.dat", "rb"))
    df = pd.read_excel(param.data_folder_path + '/trade/OTC_lag_DA.xlsx', sheet_name='Sheet1')
    df = df.drop(['TRADEDATE'], axis = 1)

    target = 'DA'
    forecast_v ='VWAP,DA'
    dateCol = 'DeliveryDate'
    lag_dict = {target: [96],
                'VWAP': [96, 96 * 2]}
    max_lag_days = 2
    predict_day_start = target_date_start - timedelta(days=max_lag_days)
    df = df[df[dateCol] >= predict_day_start]
    X,_ = make_feat_pipeline(target,forecast_v,dateCol,lag_dict,df,standardize=False,dropDate=False)
    X = X[X[dateCol]>=target_date_start]
    df = df[df[dateCol]>=target_date_start]
    feat = X.drop([dateCol], axis =1)
    prediction = model.predict(feat)

    df['predict_DA'] = prediction
    df['predict_DA_daily'] = [prediction.mean()]*len(X)

    df = df[['VWAP','DeliveryDate','start','predict_DA','predict_DA_daily']]

    con = Configuration()
    df['DeliveryDate'] = con.add_time_to_date(df,'DeliveryDate','start')
    df = df.drop(['start'], axis=1)
    # merge to operation file
    df.to_excel(param.data_folder_path + '/operation_pred_DA.xlsx',index=False)



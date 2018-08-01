from utils.import_packages import *
import utils.hardcode_parameters as param


if __name__ == '__main__':
    df = pd.read_excel(param.data_folder_path + '/b_simulation.xlsx', index = False)

    new_df = df[df['Date']>=datetime(2018,1,1)]
    pred_da = pd.read_excel(param.data_folder_path + '/operation_pred_DA.xlsx', sheet_name='Sheet1')

    new_df = new_df.drop(['predict_DA','predict_DA_daily'],axis=1)
    new_df = new_df.merge(pred_da.drop(['VWAP'],axis =1), on='DeliveryDate',how= 'inner')

    # df = pd.concat([new_df, df], axis = 0)
    df = new_df
    df = df.sort_values(by=['DeliveryDate'])
    df = df.dropna(subset=['predict_DA','First_Forecast_Volume'])
    df.to_excel(param.data_folder_path + '/b_simulation_.xlsx', index = False)
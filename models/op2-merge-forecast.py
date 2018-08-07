from utils.import_packages import *
import utils.hardcode_parameters as param


if __name__ == '__main__':
    """Input file: b_simulation.xlsx, operation_pred_DA
    Output file: b_simulation_.xlsx
    This file is to merge the predicted daily DA and hourly DA with forecast volume. 
    """

    df = pd.read_excel(param.operation_folder + '/b_simulation.xlsx', index = False)

    new_df = df[df['Date']>=datetime(2018,1,1)]
    pred_da = pd.read_excel(param.operation_folder + '/operation_pred_DA.xlsx', sheet_name='Sheet1')

    new_df = new_df.drop(['predict_DA','predict_DA_daily'],axis=1)
    new_df = new_df.merge(pred_da.drop(['VWAP'], axis=1), left_on='DATETIME_START', right_on='DeliveryDate',
                          how='inner')

    # df = pd.concat([new_df, df], axis = 0)
    df = new_df
    df = df.drop('DeliveryDate_y', axis=1)
    df = df.rename(columns={'DeliveryDate_x': 'DeliveryDate'})
    df = df.sort_values(by=['DeliveryDate','PERIOD'])
    # df = df.dropna(subset=['predict_DA','First_Forecast_Volume'])
    # df = df[['Date','First_Forecast_Volume','PERIOD','DeliveryDate','predict_DA','predict_DA_daily']]
    df.to_excel(param.operation_folder + '/b_simulation_.xlsx', index = False)
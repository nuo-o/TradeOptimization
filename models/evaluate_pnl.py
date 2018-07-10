from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration
from data_gathering.CleanData import convert2RealTime


if __name__ == '__main__':
    """This file is to evaluate the pnl with our model.
    Input model results(bid-volume)"""

    configuration = Configuration()
    baseline, df_config = configuration.readFile('baseline')
    baseline[df_config.date_col] = convert2RealTime(baseline[df_config.date_col], baseline[df_config.pte_col])

    model_result, model_config = configuration.readFile('xgboost_diff_1')
    const = 2

    # compute model pnl
    result = model_result.merge(baseline, on = df_config.date_col, how = 'inner')
    result['bidVolume'] = result['First_Forecast_Volume'] + result['predict_diff']*const
    result['model_diff'] = result['bidVolume'] - result['ActualVolumes']
    result['base_diff-model_diff'] = abs(result['true_diff']) - abs(result['model_diff'])
    result['DA-Take'] = result['DayAheadPrice'] - result['Take_From']
    result['model_pnl'] = result['model_diff']*result['DA-Take']/1000

    model_total_pnl = sum(result['model_pnl'])
    base_total_pnl = sum(result['TotalPnL'])
    pnl_improvement = model_total_pnl - base_total_pnl

    model_total_diff = sum(abs(result['model_diff']))
    base_total_diff = sum(abs(result['true_diff']))
    diff_improvement = base_total_diff - model_total_diff

    print('diff_improvement:{}'.format(diff_improvement))
    print('pnl improvement:{}'.format(pnl_improvement))

    # check DA-TAKE>0 ratio
    result['DA-TAKE'] = result['DayAheadPrice'] - result['Take_From']
    DA_TAKE = [1 if v >=0 else 0 for v in result['DA-TAKE']]
    result['DA-TAKE'] = DA_TAKE
    print('{}% DA>=TAKE'.format( math.floor( sum(result['DA-TAKE'])*100/len(result))))

    # evaluate trend
    result['same_diff_trend'] = [1 if a*b>0 else 0 for (a,b) in zip(result['true_diff'], result['model_diff'])]

    # write result
    F = open(param.data_folder_path +'/results/diff_MAPE/PNL_' + str(pnl_improvement)+'.txt', 'w')
    F.writelines('Predict diff using XGBoost:\n')
    F.writelines('Baseline TotalPnl= {}\n'.format(base_total_pnl))
    F.writelines('Baseline TotalDiff= {}\n'.format(base_total_diff))
    F.writelines('\nDiff Model TotalPnl= {}\n'.format(model_total_pnl))
    F.writelines('Diff Model TotalDiff= {}\n'.format(model_total_diff))
    F.writelines('Diff improve = {}\n'.format(diff_improvement))
    F.writelines('PNL improve = {}\n'.format(pnl_improvement))
    # F.writelines('\nDiff improved={}'.format(model))
    F.close()

    savePath = param.data_folder_path + '/results/diff_MAPE/pnl_'+str(pnl_improvement)+'.xlsx'
    print('save to {}'.format(savePath))
    result[['bidVolume','ActualVolumes','First_Forecast_Volume','same_diff_trend', 'DA-TAKE', 'model_diff','model_pnl', 'TotalPnL', 'base_diff-model_diff']].to_excel(savePath, index = False)


from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
import numpy as np
from data_gathering.Configure import Configuration

if __name__ == '__main__':
    """This file is to merge power generation data from different source."""

    target = 'solar'
    if target == 'solar':
        file1 = 'solar-missing'
        file2 = 'solar-raw'
        dateCol = 'DeliveryDate'
        save_name = 'solar_all'
        save_val_name = 'solar_v'
        save_max_name = 'solar_max'
        save_min_name = 'solar_min'
    elif target == 'wind':
        file1 = 'wind-missing'
        file2 = 'wind-raw'
        dateCol = 'DeliveryDate'
        save_name = 'wind_all'
        save_val_name = 'wind_v'
        save_max_name = 'wind_max'
        save_min_name = 'wind_min'
    else:
        raise ValueError('new weather data. need to configure')

    configuration = Configuration()
    df, config = configuration.readFile(file1)
    if (config.date_col == 'nan')|(config.forecast_v == 'nan')|(config.forecast_created_t == 'nan'):
        raise ValueError('Empty entries in file configuration.')
    wf1 = TimeSeriesData(df, config.date_col, config.forecast_v, config.forecast_created_t)
    wf1.pre_process_piepline()

    df, config = configuration.readFile(file2)
    if (config.date_col == 'nan')|(config.forecast_v == 'nan')|(config.forecast_created_t == 'nan'):
        raise ValueError('Empty entries in file configuration.')
    wf2 = TimeSeriesData(df, config.date_col, config.forecast_v, config.forecast_created_t)
    wf2.pre_process_piepline()

    complete_wf = wf2.file.merge(wf1.file, left_on=wf2.dateCol, right_on=wf1.dateCol, how='outer')
    # savePath = param.weather_folder + '/solar_all.xlsx'
    # complete_wf.to_excel( savePath, index = False)

    deliveryDate,value,minValue,maxValue = [], [], [], []
    deliveryDate = [a if b.asm8 == np.datetime64('NaT') else b for (a,b) in zip( complete_wf[wf1.dateCol], complete_wf[wf2.dateCol])]
    maxValue = [math.inf if math.isnan(a) else a for a in complete_wf[wf2.valCol[1]]]
    minValue = [math.inf if math.isnan(a) else a for a in complete_wf[wf2.valCol[2]]]
    value = [a if math.isnan(b) else b for (a,b) in zip(complete_wf[wf1.valCol[0]], complete_wf[wf2.valCol[0]])]

    wf = pd.DataFrame({dateCol:deliveryDate, save_val_name:value, save_max_name:maxValue, save_min_name:minValue})
    wf = wf.sort_values(by=[dateCol], ascending=True)
    saveSubPath = '/weather-forecast/' + target + '.xlsx'
    savePath = param.data_folder_path + saveSubPath
    wf.to_excel(savePath, index=False)

    # update config file
    if len(configuration.configFile[ configuration.configFile.data_short_name == save_name]) > 0:
        print('\nWarning: Existing data short name. Configuration file not updated!!!!!!!!!!\n')
    else:
        forecast_v = ','.join([save_val_name, save_max_name, save_min_name])
        new_config = pd.DataFrame({'data_short_name': [save_name], 'data_path': [saveSubPath],\
                                   'sheet_name': ['Sheet1'], 'date_col': [dateCol], 'forecast_v': forecast_v})
        file_config = pd.concat([configuration.configFile, new_config], axis=0)
        file_config.to_excel(param.fileConfigPath, index=False)

    # one data file
    print('good')

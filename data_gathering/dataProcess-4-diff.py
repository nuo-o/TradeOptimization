from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration


def pipeline(time_series_data, imputingCols):
    time_series_data.insert_missing_time()
    time_series_data.extract_PTE()
    time_series_data.fill_nan_by_avg(imputingCols)
    return time_series_data.file


if __name__ == '__main__':
    """This file is to impute the diff data."""

    configuration = Configuration()
    diff_df, diff_config = configuration.readFile('diff-raw')

    diff = TimeSeriesData(diff_df, diff_config.date_col, diff_config.forecast_v, pteCol = diff_config.pte_col)
    clean_diff = pipeline(diff, ['First_Forecast_Volume'])
    clean_diff['ActualVolumes'] = [ math.inf if np.isnan(x) else x for x in clean_diff['ActualVolumes']]
    clean_diff['Diff'] = clean_diff['First_Forecast_Volume'] - clean_diff['ActualVolumes']
    clean_diff['MPE'] = clean_diff['Diff']/clean_diff['First_Forecast_Volume']
    clean_diff = clean_diff[(clean_diff['missing_First_Forecast_Volume']==0) & (clean_diff['ActualVolumes'] !=0)]

    clean_diff.drop(['missing_First_Forecast_Volume'], axis = 1).to_excel( param.data_folder_path +'/position/diff_clean.xlsx', index = False)


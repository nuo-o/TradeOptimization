from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration


def pipeline(time_series_data):
    # tmprt.remove_duplicate_forecast()
    # tmprt.scale_forecast_freq_by_copy()
    tmprt.insert_missing_time()
    tmprt.extract_PTE()
    tmprt.fill_nan_by_avg()
    return tmprt.file


if __name__ == '__main__':
    """This file is to impute temperature data."""

    configuration = Configuration()
    tmprt_df, tmprt_config = configuration.readFile('temperature')

    tmprt = TimeSeriesData(tmprt_df, tmprt_config.date_col, tmprt_config.forecast_v, tmprt_config.forecast_created_t)
    tmprt.remove_duplicate_forecast()
    tmprt.scale_forecast_freq_by_copy()

    tmprt.file.to_excel(param.weather_folder + '/avg_temperature.xlsx', index = False)


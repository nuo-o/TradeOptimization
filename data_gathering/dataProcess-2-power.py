from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration

if __name__ == '__main__':
    """This file is to impute missing dates and values in power forecast data."""

    target = 'solar'
    if target == 'solar':
        file = 'avg-solar'
        save_name = '/final_solar.xlsx'
    elif target == 'wind':
        file = 'avg-wind'
        save_name = '/final_wind.xlsx'

    configuration = Configuration()
    df, config = configuration.readFile(file)
    wf = TimeSeriesData(df, config.date_col, config.forecast_v)

    imputed = wf.post_process_piepeline()
    imputed.to_excel(param.weather_folder + save_name, index = False)


    print('okay')



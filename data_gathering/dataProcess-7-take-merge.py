from utils.import_packages import *
import utils.hardcode_parameters as param

if __name__ == '__main__':
    """Process TAKE price. Convert to real time and merge other data: 
    That combines:
        solar generation forecast
        wind generation forecast
        temperature forecast
        true generation forecast(lagged)
        true diff(lagged)"""

    configuration = Configuration()
    take_df, take_config = configuration.readFile('take-clean')
    solar_df, solar_config = configuration.readFile('avg-solar')
    wind_df, wind_config = configuration.readFile('avg-wind')
    temp_df, temp_config = configuration.readFile('avg-temperature')
    da_df, da_config = configuration.readFile('DA')
    da_df = TimeSeriesData(da_df, da_config.date_col, da_config.forecast_v, pteCol=da_config.pte_col).file
    plant_df, plant_config = configuration.readFile('clean-plant')
    df0 = da_df.merge(plant_df, left_on = da_config.date_col, right_on=plant_config.date_col, how = 'left')
    df0['UnavailableAmount'] = df0['UnavailableAmount'].fillna(0)
    df1 = df0.merge(solar_df, left_on=da_config.date_col, right_on=solar_config.date_col, how='left')
    df2 = df1.merge(wind_df, left_on = da_config.date_col, right_on=wind_config.date_col, how = 'left')
    df3 = df2.merge(temp_df, left_on = da_config.date_col, right_on=temp_config.date_col, how = 'left')
    df4 = df3.merge(take_df, left_on = da_config.date_col, right_on=take_config.date_col, how = 'left')

    print('merge completed')

    #impute by avg
    impute_val = ','.join([solar_config.forecast_v, wind_config.forecast_v, temp_config.forecast_v, take_config.forecast_v, plant_config.forecast_v,da_config.forecast_v])
    date_col = da_config.date_col
    pte_col = da_config.pte_col
    ts = TimeSeriesData(df4, date_col, impute_val, pteCol = pte_col, convertTime = False)
    ts.fill_nan_by_avg(impute_val.split(','))

    print('impute completed')

    #DA-TAKE
    save_file = ts.file
    save_file['DA-TAKE'] = ts.file[da_config.forecast_v] - ts.file[take_config.forecast_v]
    save_file['DA>TAKE'] = [1 if dt>0 else 0 for dt in ts.file['DA-TAKE']]

    print('saving file')
    save_file.to_excel(param.data_folder_path + '/imbalance/final-take-add-unavailabe.xlsx', index=False)

    print(','.join(ts.file.columns))













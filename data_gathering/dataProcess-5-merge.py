from utils.import_packages import *
import utils.hardcode_parameters as param

if __name__ == '__main__':
    """Generate training data for diff prediction. 
    That combines:
        solar generation forecast
        wind generation forecast
        temperature forecast
        true generation forecast"""
    #
    # configuration = Configuration()
    # solar_df, solar_config = configuration.readFile('avg-solar')
    # wind_df, wind_config = configuration.readFile('avg-wind')
    # temp_df, temp_config = configuration.readFile('avg-temperature')
    # diff_df, diff_config = configuration.readFile('clean-diff')
    #
    # # merge
    # df1 = diff_df.merge(solar_df, on=diff_config.date_col, how = 'left')
    # df2 = df1.merge(wind_df, on = diff_config.date_col, how = 'left')
    # df3 = df2.merge(temp_df, on = diff_config.date_col, how = 'left')
    #
    # #impute
    # impute_val = ','.join([solar_config.forecast_v, wind_config.forecast_v, temp_config.forecast_v])
    # df_val = ','.join([impute_val, diff_config.forecast_v])
    # date_col = diff_config.date_col
    # pte = diff_config.pte_col
    # ts = TimeSeriesData(df3, date_col, df_val, pteCol = pte, convertTime = False)
    # ts.fill_nan_by_avg(impute_val.split(','))
    #
    # checker = DataChecker(ts.file)
    # checker.check_duplicate_forecast([diff_config.date_col])
    #
    # ts.file.to_excel(param.data_folder_path + '/position/diff_all.xlsx', index=False)

    # merge with plant info
    diff_all = pd.read_excel(param.data_folder_path + '/position/diff_all.xlsx', index = False)
    plant, plant_config = Configuration().readFile('clean-plant')
    merged = plant.merge(diff_all, on = 'DeliveryDate', how = 'right')
    # merged = merged[''].fillna()
    merged.to_excel(param.data_folder_path + '/position/diff_MAPE.xlsx', index = False)
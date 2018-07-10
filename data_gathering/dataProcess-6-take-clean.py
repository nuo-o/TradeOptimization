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

    tennet_df, tennet_config = Configuration().readFile('take-raw')
    date_col, pte_col, val_col = tennet_config.date_col, tennet_config.pte_col, tennet_config.forecast_v

    tennet_df[date_col] = pd.to_datetime( tennet_df[date_col])
    ts = TimeSeriesData(tennet_df, date_col, val_col, pteCol = pte_col)
    ts.insert_missing_time()
    ts.extract_PTE()
    ts.fill_nan_by_avg()

    ts.file.to_excel(param.data_folder_path + '/imbalance/take_clean.xlsx', index = False)



from utils.import_packages import *
import utils.hardcode_parameters as param


class Configuration():
    def __init__(self):
        self.configFile = pd.read_excel(param.fileConfigPath, dtype={'date_col': str, \
                                                         'pte_col': str, \
                                                         'forecast_v': str, \
                                                         'data_short_name': str, \
                                                         'data_path': str, \
                                                         'sheet_name': str, \
                                                         'forecast_created_t': str})

    def readFile(self, short_name):
        config = self.configFile[self.configFile['data_short_name'] == short_name].iloc[0]
        path = param.data_folder_path + config.data_path

        if config.sheet_name == 'nan':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(param.data_folder_path + config.data_path, sheet_name=config.sheet_name)
        return df, config

    def add_time_to_date(self, df,dateCol, pteCol,periodByHour=4):
        timeDate = []
        for (date, pte) in zip(df[dateCol], df[pteCol]):
            timeDate.append(date + timedelta(minutes=60 / periodByHour * (pte - 1)))
        return timeDate



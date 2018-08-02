from datetime import datetime,date, timedelta

# data_folder_path = '../data_gathering/data'
data_folder_path = '//zew-v-0a2f0b00.europe.shell.com/Python_Forecast/code/dataset/OptimizeTradeStrategy/data'
hold_out_prediction_path = data_folder_path + '/results/hold-out-prediction/'
weather_folder = data_folder_path + '/weather-forecast'
day_ahead_folder = data_folder_path + '/day-ahead'
trade_folder = data_folder_path + '/position'
operation_folder = data_folder_path + '/operation'

tennet_raw_cols = ['week', 'Date', 'PTE', 'take_from_system_kWhPTE', 'feed_into_system_EURMwh', 'purchase_kWhPTE', 'sell_kWhPTE',\
                   'absolute_kWhPTE', 'imbalance_kWhPTE']

data_start = datetime(2015, 12, 31)
data_end = datetime(2018, 6, 30)
timezone = 'Europe/Amsterdam'
PTE_interval = {'minutes':15}
max_PTE = 96

missing_data_from = datetime(2016, 12, 12)
missing_data_to = datetime(2017, 12, 26)

fileConfigPath = data_folder_path + '/fileConfig.xlsx'
hold_out_date_begin = datetime(2018,1,1)

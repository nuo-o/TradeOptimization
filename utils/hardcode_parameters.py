from datetime import datetime,date, timedelta
from pathlib import Path


data_folder_path = '../data_gathering/data'
weather_folder = data_folder_path + '/weather-forecast'
day_ahead_folder = data_folder_path + '/day-ahead'
trade_folder = data_folder_path + '/position'

tennet_raw_cols = ['week', 'Date', 'PTE', 'take_from_system_kWhPTE', 'feed_into_system_EURMwh', 'purchase_kWhPTE', 'sell_kWhPTE',\
                   'absolute_kWhPTE', 'imbalance_kWhPTE']

data_start = datetime(2015, 12, 31)
data_end = datetime(2018, 6, 30)
timezone = 'Europe/Amsterdam'
PTE_interval = {'minutes':15}
max_PTE = 96

missing_data_from = datetime(2016, 12, 12)
missing_data_to = datetime(2017, 12, 26)

''' hardcode parameters for SelfChecking'''
# path = {'demand': data_folder_path + '/demand/demand-forecast.xlsx', \
#         'DA':day_ahead_folder + '/new-DA-price.xlsx', \
#         'nzwbaseline':trade_folder + '/nzwbaseline.xlsx',\
#         'solar': weather_folder + '/solar-forecast.xlsx',\
#         'solar_missing': weather_folder + '/solarmissingdata.csv',\
#         'temperature':weather_folder + '/temperaturemissingdata.csv'}
# path_parm = {'nzwbaseline':{'sheet_name':'Sheet3'}, }
# dateColName = {'demand': 'DELIVERY_DATE_LOCAL', \
#                'DA':'DeliveryDate', \
#                'nzwbaseline':'DeliveryDate', \
#                'solar':'TIMESTAMP_UTC', \
#                'solar_missing': 'DeliveryDate', \
#                'temperature': 'DeliveryDate'}
# valColName = {'demand': ['FORECAST_VALUE'], \
#               'DA': ['Value'], \
#               'nzwbaseline':['First_Forecast_Volume', 'ActualVolumes', 'DayAheadPrice'], \
#               'solar':['value', 'maxvalue', 'minvalue'], \
#               'solar_missing':['First_Field_9'], \
#               'temperature': 'First_Field_9'}
# pteColName = {'demand':'PERIOD', \
#               'nzwbaseline':'PERIOD', \
#               'DA':'start', \
#               'solar':'PERIOD',\
#               'solar_missing': '', \
#               'temperature':''}
# forecastCreatedColName = {'solar': 'CREATED_TIMESTAMP_UTC', \
#                           'solar_missing': 'First_ForecastTime', \
#                           'temperature': 'First_ForecastTime'}

fileConfigPath = data_folder_path + '/fileConfig.xlsx'

from utils.import_packages import *
import utils.hardcode_parameters as param
import pytz
from datetime import datetime,date, timedelta

from features.features import ColumnSelector, ColumnDropper, FileSaver
from sklearn.pipeline import make_pipeline

def convert_2_utc(localtime, timezone='Europe/Amsterdam'):
    utc = pytz.utc
    amsterdam = pytz.timezone(timezone)

    am_dt = [amsterdam.localize(t) for t in localtime]
    utc_time = [t.astimezone(utc) for t in am_dt]

    return utc_time


def convert_2_local(utc_time, timezone= param.timezone):
    local_time = [pytz.utc.localize(t) for t in utc_time]
    am_dt = [t.astimezone(timezone) for t in local_time]
    discard_time_zone = [ t._short_repr for t in am_dt]
    return discard_time_zone


def print_nans(df):
    nans = df[ df.isnull().any(axis = 1) ]
    num = nans.shape[0]
    print('{} number of nans.'.format(num))
    if num>0:
        print('printing head rows with null values:')
        print(nans.head())
    return nans


def fill_nans(df):
    nan_column = df.columns[ df.isna().any()].tolist()
    if len(nan_column)>0:
        print('fill nans in column {}'.format(nan_column))
    df = df.fillna(0)
    return df


def create_emtpy_time_df( day_from = date(2015, 9, 23), day_to = date(2018,6,20)):
    date_list = pd.date_range(start=day_from, end=day_to)
    PTE_cycle = np.array([np.arange(1, 97)] * len(date_list)).reshape(-1)

    empty_time_df = pd.DataFrame()
    dates = np.array([[d] * 96 for d in date_list]).reshape(-1)
    empty_time_df.loc[:, 'Date'] = dates
    empty_time_df.loc[:, 'PTE'] = PTE_cycle

    return empty_time_df


def make_date_range(start, end, interval):
    current = start
    while current < end:
        yield current
        current += timedelta(**interval)


def make_date_time_full_list(start, end, interval={'minutes': 15}):
    full_datetime = [t for t in make_date_range(start, end, interval)]
    empty_time_df = pd.DataFrame()
    empty_time_df.loc[:, 'DateTime'] = full_datetime
    empty_time_df.loc[:, 'Date'] = [d.date() for d in empty_time_df['DateTime']]

    total_days = int(len(empty_time_df) / 96)
    PTE_cycles = np.array([np.arange(1, 97)] * total_days).reshape(-1)

    empty_time_df.loc[:, 'PTE'] = PTE_cycles

    return empty_time_df








if __name__ == '__main__':

    # """clean power forecast data:
    # 1. clean historical data
    # 2. clean multiple power forecast for the same period
    # 3. merge wind and solar
    # 4. convert to local UTC
    # 5. impute"""
    #
    # solar_df = pd.read_excel(param.weather_folder + '/solar-forecast.xlsx')
    # solar_df = solar_df[( solar_df['TIMESTAMP_UTC'] >= param.data_start) & ( solar_df['TIMESTAMP_UTC'] <= param.data_end) ]
    # wind_df = pd.read_excel(param.weather_folder + '/wind-forecast.xlsx')
    # wind_df = wind_df[(wind_df['TIMESTAMP_UTC'] >= param.data_start) & (wind_df['TIMESTAMP_UTC'] <= param.data_end)]
    #
    # solar_df = solar_df.loc[solar_df.groupby(['TIMESTAMP_UTC'])['CREATED_TIMESTAMP_UTC'].idxmax().values]
    # wind_df = wind_df.loc[wind_df.groupby(['TIMESTAMP_UTC'])['CREATED_TIMESTAMP_UTC'].idxmax().values]
    #
    # power_col = ['TIMESTAMP_UTC', 'value', 'maxvalue', 'minvalue']
    # solar_col = ['solar_value', 'solar_max', 'solar_min']
    # wind_col = ['wind_value', 'wind_max', 'wind_min']
    # solar_df = ColumnSelector(power_col).fit_transform( solar_df )
    # wind_df = ColumnSelector(power_col).fit_transform(wind_df)
    # solar_df.columns = ['UTC'] + solar_col
    # wind_df.columns = ['UTC'] + wind_col
    # power_df = solar_df.merge(wind_df, on = ['UTC'], how = 'outer')
    #
    # power_df['DateTime'] = convert_2_local(power_df['UTC'])
    # power_df['DateTime'] = pd.to_datetime(power_df['DateTime'])
    #
    # # complete PTE list
    # empty_time_df = make_date_time_full_list( param.data_start, param.data_end, interval= param.PTE_interval)
    # full_power_df = empty_time_df.merge( power_df, on = 'DateTime', how = 'left')
    # full_power_df = full_power_df[4:]
    # full_power_df.to_excel(param.data_folder_path + '/temp/full_power_df.xlsx', index = False)

    """impute:
    1. the forecast power is divided by four four each PTE within an hour.
    2. impute the other missing dates with previous date data"""


    full_power_df = pd.read_excel(param.data_folder_path + '/temp/full_power_df.xlsx')
    solar_col = ['solar_value', 'solar_max', 'solar_min']
    wind_col = ['wind_value', 'wind_max', 'wind_min']
    full_power_df[solar_col + wind_col] /=4
    full_power_df = full_power_df[(full_power_df['Date'] < param.missing_data_from) | (full_power_df['Date'] > param.missing_data_to)]

    print('imputing...')
    cur_index = 0
    while cur_index < len(full_power_df):

        PTE = full_power_df.loc[cur_index]['PTE']
        cur_row = full_power_df.loc[cur_index][solar_col + wind_col]
        last_row = full_power_df.loc[cur_index-1][solar_col + wind_col]

        # assume the power forecast data is available/unavailable at the same time.
        if PTE % 4 == 1 & cur_row.isnull().any():
            # impute missing day
            if cur_index >= param.max_PTE:
                last_day_data = full_power_df.loc[cur_index - param.max_PTE][solar_col + wind_col]
                impute_data = last_day_data
                full_power_df.loc[cur_index, solar_col + wind_col] = impute_data
            else:
                cur_index +=4
                continue

        elif PTE % 4 == 1:
            impute_data = cur_row

        else:
            print(cur_index)
            print(cur_row)
            raise ValueError('Invalid missing values')

        full_power_df.loc[cur_index + 1, solar_col + wind_col] = impute_data
        full_power_df.loc[cur_index + 2, solar_col + wind_col] = impute_data
        full_power_df.loc[cur_index + 3, solar_col + wind_col] = impute_data
        cur_index += 4

    print('finish imputation')
    full_power_df.to_excel(param.weather_folder + '/imputed_power.xlsx', index=False)

    """merge DA-price & imbalance price"""
    print('merge DA-price')
    full_power_df = pd.read_excel(param.weather_folder + '/imputed_power.xlsx')
    day_ahead_price_df = pd.read_excel(param.day_ahead_folder + '/new-DA-price.xlsx')
    day_ahead_price_df.columns = ['Date','DA-price', 'PTE']
    DA_power_df = day_ahead_price_df.merge(full_power_df, on = ['Date', 'PTE'], how = 'inner')
    # DA_power_df.to_excel(param.data_folder_path + '/temp/day_ahead_merge_power.xlsx', index = False)

    print('merge imbalance price')
    imbalance_df = pd.read_csv(param.data_folder_path + '/imbalance/tennet16to18.csv')
    imb_df = imbalance_df[['week', 'Date', 'PTE', \
                           'take_from_system_kWhPTE','feed_into_system_EURMwh',\
                           'purchase_kWhPTE','sell_kWhPTE', 'absolute_kWhPTE']]
    imb_df.columns = ['week', 'Date', 'PTE', \
                      'take_from_system_price', 'feed_into_system_price', \
                      'system_purchase_vol', 'system_sell_vol', 'system_absolute_vol']
    imb_df['Date'] = imb_df['Date'].astype('datetime64[ns]')

    imb_DA_power_df = imb_df.merge(DA_power_df, on = ['Date', 'PTE'], how = 'inner')
    imb_DA_power_df.to_excel(param.data_folder_path + '/temp/imb_DA_power.xlsx', index = False)

    # imb_DA_power_df = pd.read_excel(param.data_folder_path + '/temp/imb_DA_power2.xlsx')
    # position_df = pd.read_excel(param.data_folder_path + '/position/nordzee-wind-pnl.xlsx', encoding='ISO-8859-1')
    # demand_df = pd.read_excel(param.data_folder_path + '/demand/demand-forecast.xlsx')
    # demand_df.columns = ['DELIVERY_DATE_LOCAL', 'PERIOD', 'demand']

    # position_df['DeliveryDate'] = pd.to_datetime(position_df['DeliveryDate'])
    # pos_imb_DA_pow_df = position_df.merge(imb_DA_power_df, right_on = ['Date', 'PTE'], left_on = ['PERIOD'], how = 'inner')
    # pos_imb_DA_pow_df = ColumnDropper( ['DeliveryDate', 'PERIOD'] ).fit_transform( pos_imb_DA_pow_df)
    # FileSaver( param.data_folder_path + '/temp/imb_pos.xlsx' ).transform( pos_imb_DA_pow_df)


    # print('imbalance feature + position + demand:')
    # dmd_pos_imb_DA_pow_df = demand_df.merge(pos_imb_DA_pow_df, right_on = ['Date', 'PTE'], left_on= ['DELIVERY_DATE_LOCAL', 'PERIOD'], how = 'right')
    # dmd_pos_imb_DA_pow_df = ColumnDropper( ['DELIVERY_DATE_LOCAL', 'PERIOD']).fit_transform( dmd_pos_imb_DA_pow_df )
    # FileSaver( param.data_folder_path + '/temp/imb_pos_dmd.xlsx').transform(dmd_pos_imb_DA_pow_df)
    #
    # print('imbalance + demand:')
    # dmd_imb_DA_pow_df = demand_df.merge(imb_DA_power_df, right_on = ['Date', 'PTE'], left_on = ['DELIVERY_DATE_LOCAL', 'PERIOD'], how = 'inner')
    # dmd_imb_DA_pow_df = ColumnDropper( ['DELIVERY_DATE_LOCAL', 'PERIOD']).fit_transform( dmd_imb_DA_pow_df)
    # FileSaver( param.data_folder_path + '/temp/imb_dmd.xlsx').transform(dmd_imb_DA_pow_df)

    print('done')
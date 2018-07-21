from utils.import_packages import *
import utils.hardcode_parameters as param
import re
from models.CrossValidate import Evaluator

if __name__ == '__main__':
    """This file is to extract weekend OTC price and merge with the weekday OTC prices.
    The regular expressions for 'SEQ' column is different from 2016 March."""

    wkd_df, wkd_config = Configuration().readFile('OTC-weekend')
    # filter data from 2016-01-01
    date_start = datetime(2016,1,1)
    wkd_df = wkd_df[wkd_df['TRADEDATE']>= date_start]
    pattern1 = re.compile(r'\d{2}/\d{2}-\d{2}')
    pattern2 = re.compile(r'\d{2}-\d{2}/\d{2}/\d{2}')

    new_seq = []
    for seq in wkd_df['SEQ']:
        if pattern1.search(seq):# e.g. WkEnd 09/01-10/
            r = pattern1.search(seq).group()
            trade_day2 = int(r[-2:])
            month = int(r[3:5])
            new_seq.append(datetime(2016,month,trade_day2))
        elif pattern2.search(seq):# e.g. WkEnd 02-03/07/16
            r = pattern2.search(seq).group()
            trade_day2 = int(r[-8:-6])
            month = int(r[-5:-3])
            year = int('20'+r[-2:])
            new_seq.append(datetime(year,month,trade_day2))
        else:
            print(seq)
            raise ValueError('New expression for SEQ columns')

    # aggregate weekend trades
    wkd_df['SEQ'] = new_seq
    wkd_df['QTY*PRICE'] = wkd_df['QTY']*wkd_df['PRICE']
    sum_qty_price = wkd_df.groupby('SEQ')['QTY*PRICE'].sum().reset_index(name = 'Sum_QTY*PRICE')
    sum_qty = wkd_df.groupby('SEQ')['QTY'].sum().reset_index(name='Sum_QTY')
    a = sum_qty_price.merge(sum_qty, on ='SEQ', how = 'inner')
    a['VWAP'] = a['Sum_QTY*PRICE']/a['Sum_QTY']

    wkd_vwap = []
    wkd_date = []
    for (d,v) in zip(a['SEQ'], a['VWAP']):
        wkd_date.extend([d, d-timedelta(days=1)])
        wkd_vwap.extend([v,v])

    wkd_df = pd.DataFrame({'TRADEDATE':wkd_date, 'VWAP':wkd_vwap}).sort_values(by=['TRADEDATE'])

    # merge with weekday data
    week_df, week_config = Configuration().readFile('OTC-weekday')
    week_df = week_df[week_df['TRADEDATE']>=date_start]
    OTC = pd.concat([week_df,wkd_df[['TRADEDATE', 'VWAP']]],axis=0).sort_values(by=['TRADEDATE'])
    OTC = OTC[['TRADEDATE','VWAP']]

    # merge with DA price
    DA, DA_config = Configuration().readFile('DA')
    DA = DA[DA['DeliveryDate']>= date_start]

    OTC_lag = OTC.copy()
    OTC_lag['DeliveryDate'] = [d + timedelta(days=1) for d in OTC_lag['TRADEDATE']]
    # merged = OTC.merge(DA, left_on = 'TRADEDATE', right_on='DeliveryDate', how='inner')
    merged_lag = OTC_lag.merge(DA, on='DeliveryDate',how='inner')
    merged_lag = merged_lag.dropna()

    merged_lag.to_excel(param.data_folder_path + '/trade/OTC_lag_DA.xlsx',index = False)

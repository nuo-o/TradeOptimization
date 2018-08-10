from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.DataChecker import *
import os
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from models.op3_Simulate import compute_pnl


def append_to_xlsx(path, df):
    wb = load_workbook(path)
    ws = wb['Sheet1']
    for row in dataframe_to_rows(df,index=False,header=False):
        ws.append(row)
    wb.save(path)

if __name__ == '__main__':

    """This file evaluates the total p&l against the baseline.
    Before running this file, the true price data and actual volume should be copied to operation_bid.xlsx. 
    """

    strategy = 8
    hold_df = pd.read_excel(param.operation_folder + '/operation_bid_true.xlsx')
    evaluate_start_date = datetime(2018,2,1)
    evaluate_end_date = datetime(2018,8,1)
    hold_df = hold_df[(hold_df['DeliveryDate']<evaluate_end_date) & (hold_df['DeliveryDate']>=evaluate_start_date)]
    hold_df = hold_df.dropna(subset=['First_Forecast_Volume','ActualVolumes','DayAheadPrice'])

    hold_df['our_pnl'] = compute_pnl(hold_df['our_bid'], \
                                     hold_df['ActualVolumes'],\
                                     hold_df['Take_From'],\
                                     hold_df['Feed_Into'],\
                                     hold_df['DayAheadPrice'])

    hold_df['base_pnl'] = compute_pnl(hold_df['First_Forecast_Volume'], \
                                                    hold_df['ActualVolumes'], \
                                                    hold_df['Take_From'],\
                                                    hold_df['Feed_Into'],\
                                                    hold_df['DayAheadPrice'])

    improve = hold_df['our_pnl'] - hold_df['base_pnl']
    total_improve = sum(improve)
    total_our_pnl = sum(hold_df['our_pnl'])
    total_base_pnl = sum(hold_df['base_pnl'])
    print('total baseline pnl:\t{}\n'.format(total_base_pnl))
    print('total our pnl:\t{}\n'.format(total_our_pnl))
    print('total improve:\t{}\n'.format(total_improve))

    a = hold_df[ hold_df['our_bid']!= hold_df['First_Forecast_Volume']]
    b = a[ a['base_pnl']<= a['our_pnl']]
    if max(1, len(a)) == 1:
        print('')
    improved_prct = 100*len(b)/max(1,len(a))
    print('{}% per PTE data get improved'.format( round(improved_prct, 2)))

    c = hold_df[ hold_df['base_pnl']> hold_df['our_pnl']]
    c['loss'] = c['base_pnl']-c['our_pnl']
    avg_loss = sum(c['loss'])/len(c)
    print('avg loss = {}'.format( round(avg_loss,4)))

    avg_win = sum(b['our_pnl']-b['base_pnl'])/len(c)
    print('avg win = {}'.format( round(avg_win), 4))

    # daily pnl
    base_daily_sum = hold_df.groupby('Date')['base_pnl'].sum().reset_index()
    base_daily_sum = base_daily_sum.rename(columns={'base_pnl':'base_pnl_sum'})
    our_daily_sum = hold_df.groupby('Date')['our_pnl'].sum().reset_index()
    our_daily_sum = our_daily_sum.rename(columns={'our_pnl':'our_pnl_sum'})
    base_daily_var = hold_df.groupby('Date')['base_pnl'].var().reset_index()
    base_daily_var = base_daily_var.rename(columns={'base_pnl':'base_pnl_var'})
    our_daily_var = hold_df.groupby('Date')['our_pnl'].var().reset_index()
    our_daily_var = our_daily_var.rename(columns={'our_pnl':'our_pnl_var'})
    daily_evaluate = base_daily_sum.merge(our_daily_sum, on='Date', how='inner')
    daily_evaluate = daily_evaluate.merge(base_daily_var, on='Date', how='inner')
    daily_evaluate = daily_evaluate.merge(our_daily_var, on='Date', how='inner')

    # monthly
    monthly = hold_df.copy()
    monthly['Year-month'] = [ datetime(d.year, d.month, 1) for d in monthly['Date']]
    our_monthly = monthly.groupby('Year-month')['our_pnl'].sum().reset_index()
    base_monthly = monthly.groupby('Year-month')['base_pnl'].sum().reset_index()
    monthly_evaluate = base_monthly.merge(our_monthly, on='Year-month', how='inner')

    # save result
    path_daily= param.operation_folder + '/results/strategy_' + str(strategy) + '_evaluate_daily.xlsx'
    path_monthly= param.operation_folder + '/results/strategy_' + str(strategy) + '_evaluate_monthly.xlsx'
    path_raw_result= param.operation_folder + '/results/strategy_' + str(strategy) + '.xlsx'

    if os.path.exists(path_daily):
        history_days = set(pd.read_excel(path_daily)['Date'])
        overlapped_days = [d for d in set(hold_df['Date']) if d in history_days]

        if len(overlapped_days)>0:
            raise ValueError('There are dates that has been evaluated.\nPlease remove the following days in the operation_bid.xlsx:\n{}'.format(overlapped_days))
        else: # append to existing file
            append_to_xlsx(path_daily, daily_evaluate)
            append_to_xlsx(path_monthly, monthly_evaluate)
            append_to_xlsx(path_raw_result, hold_df)
    else:
        daily_evaluate.to_excel(path_daily, index = False)
        monthly_evaluate.to_excel(path_monthly, index=False)
        hold_df.to_excel(path_raw_result, index=False)



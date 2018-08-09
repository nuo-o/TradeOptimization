from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.DataChecker import *
import os
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from models.op3_Simulate import *
from models.op4_evaluate import *
import matplotlib.pyplot as plt

# def append_to_xlsx(path, df):
#     wb = load_workbook(path)
#     ws = wb['Sheet1']
#     for row in dataframe_to_rows(df,index=False,header=False):
#         ws.append(row)
#     wb.save(path)


if __name__ == '__main__':

    evaluate_start_date = datetime(2017,12,1)
    evaluate_end_date = datetime(2018,1,1)
    strategy = 1

    imb = pd.read_excel(param.operation_folder + '/a_simulation.xlsx')
    imb = imb.dropna(subset=['Take_From','Feed_Into'])
    hold_df = pd.read_excel(param.operation_folder + '/b_simulation_.xlsx')
    hold_df = hold_df.dropna(subset=['First_Forecast_Volume', 'predict_DA','predict_DA_daily'])
    result_saved_path = param.operation_folder + '/results/'

    """parameters that needs to train"""
    num_resample = 1000
    # num_historical_days = 120
    min_bid_value_when_forecast_zero = -1000
    bid_interval_when_forecast_zero = 10

    current_experiment = 0
    experiment_result = []

    tune_historical_days = [7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    experiment_times = len(tune_historical_days)

    tune_result = []
    a_simulation = imb.copy()

    print('strategy:{}'.format(strategy))
    while current_experiment < experiment_times:
        num_historical_days = tune_historical_days[current_experiment]
        print('\nexperiment:{}'.format(current_experiment))
        print('history:{} days'.format(num_historical_days))

        row_id = 0
        best_bids = []
        last_sim_day = None
        take_kde = None
        feed_kde = None
        take_multipliers = None
        feed_multipliers = None

        np.random.seed(123)

        while row_id<len(hold_df):
            if row_id %1000==0:
                a = int(100*(row_id+1)/len(hold_df))
                print('processed:{}%'.format(a))

            d,p,v,da,da_daily =hold_df.iloc[row_id][['Date','Period','First_Forecast_Volume','predict_DA','predict_DA_daily']]

            if last_sim_day !=d:
                take_kde, take_multipliers = get_simulated_imb_price(imb,'Period',d,'DeliveryDate',num_historical_days,'Take_From')
                feed_kde, feed_multipliers = get_simulated_imb_price(imb,'Period',d,'DeliveryDate',num_historical_days,'Feed_Into')
                last_sim_day = d

            if strategy <4:
                bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)

            best_objective_v = -math.inf
            best_bid = v

            sim_take_prices = take_kde.resample(num_resample) * da_daily * take_multipliers[p]
            sim_feed_prices = feed_kde.resample(num_resample) * da_daily * feed_multipliers[p]

            if (strategy == 1):
                best_bid = search_best_quantile(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            elif (strategy == 2):
                best_bid = search_best_sum_var(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            elif strategy ==3:
                best_bid = max_mean_dev_diff(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            elif strategy ==4:
                """search space: p10, p50, p90"""
                p10,p90 = hold_df.iloc[row_id][['P10','P90']]
                bid_space = [v, p10, p90]
                best_bid = search_best_quantile(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            row_id += 1
            best_bids.append(best_bid)

        evaluate = hold_df.copy()
        evaluate['our_bid'] = best_bids
        true = evaluate[['our_bid', 'DeliveryDate']]
        evaluate = evaluate[['our_bid','DeliveryDate']].merge(a_simulation, on='DeliveryDate', how='left')
        evaluate = evaluate[
            (evaluate['DeliveryDate'] < evaluate_end_date) & (evaluate['DeliveryDate'] >= evaluate_start_date)]
        evaluate = evaluate.dropna(
            subset=['First_Forecast_Volume', 'ActualVolumes', 'DayAheadPrice', 'Take_From', 'Feed_Into', 'our_bid'])

        evaluate['our_pnl'] = compute_pnl(evaluate['our_bid'], \
                                          evaluate['ActualVolumes'], \
                                          evaluate['Take_From'], \
                                          evaluate['Feed_Into'], \
                                          evaluate['DayAheadPrice'])

        evaluate['base_pnl'] = compute_pnl(evaluate['First_Forecast_Volume'], \
                                           evaluate['ActualVolumes'], \
                                           evaluate['Take_From'], \
                                           evaluate['Feed_Into'], \
                                           evaluate['DayAheadPrice'])

        improve = evaluate['our_pnl'] - evaluate['base_pnl']
        total_improve = sum(improve)
        total_our_pnl = sum(evaluate['our_pnl'])
        total_base_pnl = sum(evaluate['base_pnl'])
        print('total baseline pnl:\t{}\n'.format(total_base_pnl))
        print('total our pnl:\t{}\n'.format(total_our_pnl))
        print('total improve:\t{}\n'.format(total_improve))

        tune_result.append(total_improve)

        current_experiment += 1

    print(tune_result)
    print(max(tune_result), min(tune_result))

    plt.figure()
    plt.plot(tune_historical_days, tune_result)
    plt.show()
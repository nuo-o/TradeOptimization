from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration
from data_gathering.CleanData import convert2RealTime
from models.Strategies import *

# def evaluate_strategy(bid, da_take_price, actual, printInfo):
#     model_diff = bid - actual
#     avgAbsDiff = sum(abs(model_diff))/len(model_diff)
#     pnl = da_take_price * model_diff
#     total_pnl = sum(pnl)
#     if printInfo:
#         print('avg_abs_diff = {}, total_pnl = {}'.format(avgAbsDiff, total_pnl))
#     return pnl, total_pnl, avgAbsDiff
#
#
# def strategy1(predict_da_take, predict_diff, forecast_v, da_take_pos_c, da_take_neg_c, da_threshold, max_bid = 25000, min_bid = 0):
#     bid = []
#     for (da_take, diff, forecast_v) in zip( predict_da_take, predict_diff, forecast_v):
#         if da_take>da_threshold:
#             new_bid = forecast_v + diff*da_take_pos_c
#             new_bid = max(min(new_bid, max_bid), min_bid)
#         else:
#             new_bid = forecast_v + diff*da_take_neg_c
#             new_bid = max(min(new_bid, max_bid), min_bid)
#         bid.append(new_bid)
#     return bid


if __name__ == '__main__':
    """This file is to evaluate the pnl with our model.
    Input model results(bid-volume)"""

    configuration = Configuration()
    baseline, baseline_config = configuration.readFile('baseline')
    baseline[baseline_config.date_col] = convert2RealTime(baseline[baseline_config.date_col], baseline[baseline_config.pte_col])
    baseline = baseline[baseline[baseline_config.date_col] >= param.hold_out_date_begin]

    take_pred, take_config = configuration.readFile('best-TAKE')

    baseline = baseline.merge(take_pred, left_on = baseline_config.date_col, right_on=take_config.date_col, how='right')

    valid_index = []
    for i, (actual,forecast,da_price, imb_price) in enumerate(zip( baseline['ActualVolumes'], baseline['First_Forecast_Volume'], baseline['Take_From'], baseline['DayAheadPrice'])):
        if math.isnan( actual + forecast ) | math.isnan( da_price + imb_price):
            continue
        else:
            valid_index.append(i)
    baseline = baseline.iloc[valid_index]
    # split evaluate-test
    evaluate_split_index = train_test_split(baseline, baseline_config.date_col, splitBySize=False, split_date=datetime(2018,3,27))
    print('train size ={}'.format(round(evaluate_split_index/len(baseline),2)))

    # evaluate strategy
    predict_DA_TAKE = take_pred[:evaluate_split_index]['predict_DA>TAKE']
    forecast_volume = baseline[:evaluate_split_index]['First_Forecast_Volume']
    true_generation = baseline[:evaluate_split_index]['ActualVolumes']
    true_da_price = baseline[:evaluate_split_index]['DayAheadPrice']
    true_imb_price = baseline[:evaluate_split_index]['Take_From']
    strategy = S1_maxPnl()
    predicted_params = {'predict_DA>TAKE': predict_DA_TAKE, \
                 'First_Forecast_Volume': forecast_volume}
    bids = strategy.get_bid_value(predicted_params)
    evaluation = strategy.evaluate(bids, \
                                   true_generation, \
                                   true_da_price, \
                                   true_imb_price)
    total_pnl = sum(evaluation['strategy_pnl'])
    baseline_pnl = sum(baseline['TotalPnL'])
    print('baseline_pnl = {}\nour_pnl = {}\nimproved = {}'.format(baseline_pnl, total_pnl, total_pnl - baseline_pnl))

    # evaluate in test set
    print('Test:')
    strategy = S1_maxPnl()
    predict_DA_TAKE = take_pred[evaluate_split_index:]['predict_DA>TAKE']
    forecast_volume = baseline[evaluate_split_index:]['First_Forecast_Volume']
    true_generation = baseline[evaluate_split_index:]['ActualVolumes']
    true_da_price = baseline[evaluate_split_index:]['DayAheadPrice']
    true_imb_price = baseline[evaluate_split_index:]['Take_From']
    predicted_params = {'predict_DA>TAKE': predict_DA_TAKE, \
                 'First_Forecast_Volume': forecast_volume}
    bids = strategy.get_bid_value(predicted_params)
    evaluation = strategy.evaluate(bids, \
                                   true_generation, \
                                   true_da_price, \
                                   true_imb_price)
    total_pnl = sum(evaluation['strategy_pnl'])
    baseline_pnl = sum(baseline['TotalPnL'])
    print('baseline_pnl = {}\nour_pnl = {}\nimproved = {}'.format(baseline_pnl, total_pnl, total_pnl - baseline_pnl))


    # save test result
    # save_result = test_result[['DeliveryDate','First_Forecast_Volume', 'ActualVolumes','Take_From', 'DayAheadPrice', 'Diff', 'TotalPnL']]
    # save_result['Model_bid'] = bid
    # save_result['Model_pnl'] = pnl
    # save_path = param.data_folder_path + '/results/test_bid_pnl_' + str(int(total_pnl)) + '.xlsx'
    # save_result.to_excel( save_path, index = False)
    # print('save test bid into {}'.format(save_path))



    # currentBest = 5146
    # if round(best_pnl - base_totalPnl,0) > currentBest:
    #     print('new best result')
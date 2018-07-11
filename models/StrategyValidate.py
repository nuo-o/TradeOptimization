from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration
from data_gathering.CleanData import convert2RealTime


def evaluate_strategy(bid, da_take_price, actual, printInfo):
    model_diff = bid - actual
    avgAbsDiff = sum(abs(model_diff))/len(model_diff)
    pnl = da_take_price * model_diff
    total_pnl = sum(pnl)
    if printInfo:
        print('avg_abs_diff = {}, total_pnl = {}'.format(avgAbsDiff, total_pnl))
    return pnl, total_pnl, avgAbsDiff


def bid_strategy( predict_da_take, predict_diff, forecast_v, da_take_pos_c, da_take_neg_c, da_threshold, max_bid = 30000, min_bid = 1000):
    bid = []
    for (da_take, diff, forecast_v) in zip( predict_da_take, predict_diff, forecast_v):
        if da_take>da_threshold:
            new_bid = forecast_v + diff*da_take_pos_c
            new_bid = max(min(new_bid, max_bid), min_bid)
        else:
            new_bid = forecast_v + diff*da_take_neg_c
            new_bid = max(min(new_bid, max_bid), min_bid)
        bid.append(new_bid)
    return bid


if __name__ == '__main__':
    """This file is to evaluate the pnl with our model.
    Input model results(bid-volume)"""

    configuration = Configuration()
    baseline, baseline_config = configuration.readFile('baseline')
    baseline[baseline_config.date_col] = convert2RealTime(baseline[baseline_config.date_col], baseline[baseline_config.pte_col])

    diff_pred, diff_config = configuration.readFile('best-diff')
    take_pred, take_config = configuration.readFile('best-TAKE')

    trend = [1 if a*b>0 else 0 for (a,b) in zip(diff_pred['true_diff'], diff_pred['predict_diff'])]
    trend_acc = sum(trend)*100/len(trend)
    print('diff trend accuracy = {}\n'.format( round(trend_acc,2)))

    # compute model pnl
    result = baseline.merge(diff_pred, left_on = baseline_config.date_col, right_on = diff_config.date_col, how = 'inner')
    result = result.merge(take_pred, left_on = baseline_config.date_col, right_on = take_config.date_col, how = 'inner')

    evaluate_split_index = train_test_split(result, baseline_config.date_col, splitBySize=False, split_date=datetime(2018,5,27))
    evaluate_result = result[:evaluate_split_index]
    test_result = result[evaluate_split_index:]

    # evaluate the best strategy
    print('Evaluation:')
    forecast_v = evaluate_result['First_Forecast_Volume']
    predict_diff = evaluate_result['predict_diff']
    actual = evaluate_result['ActualVolumes']
    predict_da_take = evaluate_result['predict_DA>TAKE']

    da_take_price = (evaluate_result['DayAheadPrice'] - evaluate_result['Take_From'])/1000

    da_threshold = 0.51
    da_pos_c = np.array([0,1,-1])*1000
    da_neg_c = np.array([0,1,-1])*1000

    # baseline
    base_avgAbsDiff = sum(abs(evaluate_result['true_diff'])) / len(evaluate_result) * 100
    base_pnl = evaluate_result['TotalPnL']
    base_totalPnl = sum(evaluate_result['TotalPnL'])
    # print('baseline:\nAvg_Abs_Diff= {}, total_pnl= {}'.format(round(base_avgAbsDiff, 2), round(base_totalPnl, 2)))

    best_pos_c = 0
    best_neg_c = 0
    best_pnl = base_totalPnl
    best_bid = []

    for pos_c in da_pos_c:
        for neg_c in da_neg_c:
            if (pos_c == 0) and (neg_c == 0):
                continue
            print('\npos_c = {}, neg_c = {}'.format(pos_c, neg_c))
            bid = bid_strategy(predict_da_take, predict_diff, forecast_v, pos_c, neg_c, da_threshold)
            pnl, total_pnl, avgAbsDiff = evaluate_strategy(bid, da_take_price, actual, printInfo=True)
            print('OurPnl - BasePnl = {}'.format( total_pnl - base_totalPnl))
            print('BaseAvgAbsDiff - OurAvgAbsDiff = {}'.format( avgAbsDiff - base_avgAbsDiff))

            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_pos_c = pos_c
                best_neg_c = neg_c
                # best_bid = bid

    if (best_pos_c == 0) & (best_neg_c == 0):
        raise ValueError('No improvement against the baseline.')

    print('\nBest Strategy:\npos = {}, neg = {}'.format(best_pos_c, best_neg_c))
    print('Best total pnl = {}'.format(best_pnl))
    print('Best total pnl improvement = {}'.format(best_pnl - base_totalPnl))

    # Test baseline
    print('\nTest')
    base_avgAbsDiff = sum(abs(test_result['true_diff'])) / len(test_result) * 100
    base_pnl = test_result['TotalPnL']
    base_totalPnl = sum(test_result['TotalPnL'])
    print('Baseline total Pnl = {}'.format(base_totalPnl))

    forecast_v = test_result['First_Forecast_Volume']
    predict_diff = test_result['predict_diff']
    actual = test_result['ActualVolumes']
    predict_da_take = test_result['predict_DA>TAKE']
    da_take_price = (test_result['DayAheadPrice'] - test_result['Take_From'])/1000

    bid = bid_strategy(predict_da_take, predict_diff, forecast_v, best_pos_c, best_neg_c, da_threshold)
    pnl, total_pnl, avgAbsDiff = evaluate_strategy(bid, da_take_price, actual, printInfo=True)
    print('TestPnl - BasePnl = {}'.format(total_pnl - base_totalPnl))
    print('BaseAvgAbsDiff - OurAvgAbsDiff = {}'.format(avgAbsDiff - base_avgAbsDiff))

    # save test result
    save_result = test_result[['DeliveryDate','First_Forecast_Volume', 'ActualVolumes','Take_From', 'DayAheadPrice', 'Diff', 'TotalPnL']]
    save_result['Model_bid'] = bid
    save_path = param.data_folder_path + '/results/test_bid_pnl_' + str(int(total_pnl)) + '.xlsx'
    save_result.to_excel( save_path, index = False)
    print('save test bid into {}'.format(save_path))


    # currentBest = 5146
    # if round(best_pnl - base_totalPnl,0) > currentBest:
    #     print('new best result')
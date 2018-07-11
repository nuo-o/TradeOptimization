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


def bid_strategy( predict_da_take, predict_diff, forecast_v, da_take_pos_c, da_take_neg_c, da_threshold, max_bid = 30000, min_bid = -30000):
    bid = []
    for (da_take, diff, forecast_v) in zip( predict_da_take, predict_diff, forecast_v):
        if da_take>da_threshold:
            new_bid = forecast_v + diff*da_take_pos_c
            new_bid = max(min(new_bid, max_bid), min_bid)
        else:
            new_bid = forecast_v + diff*da_take_pos_c
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

    forecast_v = result['First_Forecast_Volume']
    predict_diff = result['predict_diff']
    actual = result['ActualVolumes']
    predict_da_take = result['predict_DA>TAKE']
    da_take_price = (result['DayAheadPrice'] - result['Take_From'])/1000

    da_threshold = 0.5
    da_pos_c = np.array([0,1,-1])*1000
    da_neg_c = np.array([0,1,-1])*1000

    # baseline
    base_avgAbsDiff = sum(abs(result['true_diff'])) / len(result) * 100
    base_pnl = result['TotalPnL']
    base_totalPnl = sum(result['TotalPnL'])
    print('baseline:\nAvg_Abs_Diff= {}, total_pnl= {}'.format(round(base_avgAbsDiff, 2), round(base_totalPnl, 2)))

    best_strategy = []
    best_pnl = base_totalPnl

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

    print('\nBest Strategy:\npos = {}, neg = {}'.format(pos_c, neg_c))
    print('Best total pnl = {}'.format(best_pnl))
    print('Best total pnl improvement = {}'.format(best_pnl - base_totalPnl))

    currentBest = 5146
    if round(best_pnl - base_totalPnl,0) > currentBest:
        print('new best result')
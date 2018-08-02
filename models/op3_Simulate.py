from utils.import_packages import *
import utils.hardcode_parameters as param
from scipy.stats.kde import gaussian_kde
from data_gathering.DataChecker import *


def compute_pnl(bid, actual_vol, take_price, feed_price, da_price):
    PNL = []

    if len(bid)==len(actual_vol)==len(da_price):
        for b,a,take,feed,da in zip(bid, actual_vol, take_price, feed_price, da_price):
            vol_diff = b - a

            if b>a:#defit
                price_diff = da-take
            else:#surplus
                price_diff = da-feed

            pnl = vol_diff*price_diff/1000
            PNL.append(pnl)
    else:
        raise ValueError('wrong length of data for computing pnl')
    return PNL


def search_best_quantile(da, sim_take_prices, sim_feed_prices, mpe, v, bid_space):
    best_objective_v = -math.inf
    best_bid =  v
    actual_v = v - mpe*v

    for b in bid_space:
        sim_PNL = []
        for (take,feed) in zip(sim_take_prices, sim_feed_prices):
            sim_pnl = compute_pnl([b],[actual_v],[take],[feed],[da])
            sim_PNL.append(sim_pnl)
        objective_v = np.sort(sim_PNL)[:max(1, int(len(sim_PNL)*0.1))].mean()

        if best_objective_v < objective_v:
            best_objective_v = objective_v
            best_bid = b

    return best_bid


def search_best_sum_var(da, sim_take_prices, sim_feed_prices, mpe, v, bid_space):
    best_objective_v = -math.inf
    best_bid = v
    actual_v = v - v*mpe

    for b in bid_space:
        sim_PNL = []
        for (take,feed) in zip(sim_take_prices, sim_feed_prices):
            sim_pnl = compute_pnl([b], [actual_v], [take], [feed], [da])
            sim_PNL.append(sim_pnl)
        sim_PNL = np.array(sim_PNL)
        objective_v = sim_PNL.sum()/ sim_PNL.var()

        if best_objective_v < objective_v:
            best_objective_v = objective_v
            best_bid = b

    return best_bid


def search_best_mean_var(da, sim_take_prices, sim_feed_prices, mpe, v, bid_space):
    best_objective_v = -math.inf
    best_bid = v

    actual_v = v - v * mpe

    for b in bid_space:
        sim_PNL = []
        for (take, feed) in zip(sim_take_prices, sim_feed_prices):
            sim_pnl = compute_pnl([b],[actual_v],[take],[feed],[da])
            sim_PNL.append(sim_pnl)
        objective_v = np.array(sim_PNL).mean() / np.array(sim_PNL).var()

        if best_objective_v < objective_v:
            best_objective_v = objective_v
            best_bid = b

    return best_bid


def get_simulated_imb_price(history, pteCol, curTime, dateTimeCol,history_days,imbPriceType):
    history_start = curTime - timedelta(days=history_days)
    history = history[(history[dateTimeCol]<curTime)&(history[dateTimeCol]>=history_start)]

    # daily Imb/DA
    imb_daily_name = imbPriceType+'_daily'
    imb_multiplier_name = imbPriceType+'/'+imb_daily_name

    ratio = history[imb_daily_name]/history['DayAheadPrice_daily']
    ratio_kde = gaussian_kde(ratio)

    # hourly Imb/daily_imb
    history[imb_multiplier_name] = history[imbPriceType]/history[imb_daily_name]
    multiplier = history.groupby(pteCol)[imb_multiplier_name].mean()

    return ratio_kde,multiplier


def build_search_bid_space(v, a, b, min_ratio=0.5, max_ratio=1.5,interval=0.1):
    if v == 0:
        bid_space = np.arange(a, 0, b)
    elif v > 0:

        bid_space = np.arange(int(v * min_ratio), min(int(v * max_ratio), 25000), min(max(int(v * interval), 5), 25000))
    else:
        bid_space = np.arange(max(-1000, int(v * 2)), int(v * 0.5), max(int(v * 0.1), 5))

    return bid_space


if __name__ == '__main__':

    imb = pd.read_excel(param.operation_folder + '/a_simulation.xlsx')
    hold_df = pd.read_excel(param.operation_folder + '/b_simulation_.xlsx')
    result_saved_path = param.operation_folder + '/results/'

    np.random.seed(123)

    hold_df = hold_df.dropna(subset=['First_Forecast_Volume', 'predict_DA','predict_DA_daily'])
    imb = imb.dropna(subset=['Take_From','Feed_Into'])

    strategy = 1
    num_resample = 1000
    num_historical_days = 60
    min_bid_value_when_forecast_zero = -1000
    bid_interval_when_forecast_zero = 10
    experiment_times = 1
    current_experiment = 0
    experiment_result = []

    while current_experiment < experiment_times:
        current_experiment += 1
        print('\nexperiment:{}'.format(current_experiment))

        row_id = 0
        best_bids = []
        last_sim_day = None
        take_kde = None
        feed_kde = None
        take_multipliers = None
        feed_multipliers = None

        while row_id<len(hold_df):
            if row_id %1000==0:
                print('processed:{}%'.format(int(100*(row_id+1)/len(hold_df))))

            d,p,v,da,da_daily =hold_df.iloc[row_id][['Date','PERIOD','First_Forecast_Volume','predict_DA','predict_DA_daily']]
            row_id += 1

            if last_sim_day !=d:
                take_kde, take_multipliers = get_simulated_imb_price(imb,'PERIOD',d,'DeliveryDate',num_historical_days,'Take_From')
                feed_kde, feed_multipliers = get_simulated_imb_price(imb,'PERIOD',d,'DeliveryDate',num_historical_days,'Feed_Into')
                last_sim_day = d

            bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)

            best_objective_v = -math.inf
            best_bid = v

            if (strategy == 1):
                sim_take_prices = take_kde.resample(num_resample)*da_daily*take_multipliers[p]
                sim_feed_prices = feed_kde.resample(num_resample)*da_daily*feed_multipliers[p]
                best_bid = search_best_quantile(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            elif (strategy == 2):
                sim_take_prices = take_kde.resample(num_resample)*da_daily*take_multipliers[p]
                sim_feed_prices = feed_kde.resample(num_resample)*da_daily*feed_multipliers[p]

                best_bid = search_best_sum_var(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            # elif strategy ==3:
            #     sim_take_prices = take_kde.resample(num_resample) * da_daily * take_multipliers[p]
            #     sim_feed_prices = feed_kde.resample(num_resample) * da_daily * feed_multipliers[p]
            #
            #     best_bid = search_best_mean_var(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

            # elif strategy == 4: # use DA>TAKE prediction as a heuristic strategy
            #     if random.uniform(0,1)>= da_prob:
            #         if da_prob >=0.5:
            #             best_bid = v * 1.5
            #         else:
            #             best_bid = v * 0.5
            #
            # elif strategy == 5:
            #     if da_prob >= 0.5:
            #         best_bid = v * 1.5
            #     else:
            #         best_bid = v * 0.5

            best_bids.append(best_bid)

        hold_df['our_bid'] = best_bids
        hold_df.to_excel(param.operation_folder + '/operation_bid.xlsx', index = False)
        print('save predicted bid value to :{}'.format(param.operation_folder + '/operation_bid.xlsx'))

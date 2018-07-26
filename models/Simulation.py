from utils.import_packages import *
import utils.hardcode_parameters as param
from scipy.stats.kde import gaussian_kde
from models.StrategyValidate import compute_pnl


def search_best_quantile(da, sim_imb_prices, mpe, v, bid_space):
    best_objective_v = -math.inf
    best_bid =  v

    actual_v = v - mpe*v

    for b in bid_space:
        sim_pnl = (da-sim_imb_prices)*(b - actual_v)
        objective_v = np.sort(sim_pnl)[:max(1, int(len(sim_pnl)*0.1))].mean()

        if best_objective_v < objective_v:
            best_objective_v = objective_v
            best_bid = b

    return best_bid


def search_best_pnl_var(da,sim_imb_prices,mpe,v,bid_space):
    best_objective_v = -math.inf
    best_bid = v

    actual = v - v*mpe
    for b in bid_space:
        sim_pnl = (da - sim_imb_prices) * (b - actual)
        objective_v = sim_pnl.mean() / sim_pnl.var()

        if best_objective_v < objective_v:
            best_objective_v = objective_v
            best_bid = b

    return best_bid


def sample_ratioed_imb_prices(num_samples, ratio, da_daily, multiplier, da, kde):
    num_pos_sample = int(num_samples*ratio)
    num_neg_sample = num_samples - num_pos_sample

    sim_imb_prices = []
    num_p = 0
    num_n = 0

    while (num_p + num_n) < num_samples:
        p = last_kde.resample(1).reshape(-1)[0] * da_daily * multiplier
        # sim_imb_price = last_kde.resample(1)*da_daily*multiplier

        if (da > p) & (num_p < num_pos_sample):
            sim_imb_prices.append(p)
            num_p +=1

        elif ( da < p) & (num_n < num_neg_sample):
            sim_imb_prices.append(p)
            num_n +=1

    return sim_imb_prices


def get_simulated_imb_price(history, pteCol, curTime, dateTimeCol,history_days):
    history_start = curTime - timedelta(days=history_days)
    history = history[(history[dateTimeCol]<curTime)&(history[dateTimeCol]>=history_start)]

    # daily Imb/DA
    ratio = history['Take_From_daily']/history['DayAheadPrice_daily']
    ratio_kde = gaussian_kde(ratio)

    # hourly Imb/daily_imb
    history['Take_From_hourly/Take_From_daily'] = history['Take_From']/history['Take_From_daily']
    multiplier = history.groupby(pteCol)['Take_From_hourly/Take_From_daily'].mean()

    return ratio_kde,multiplier


def get_price_ratio_dist(history, target, dateCol):
    daily_imb = history.groupby(dateCol)[target].mean().reset_index()
    daily_imb_KDE = gaussian_kde(daily_imb['TAKE/DA'])
    return daily_imb_KDE


def get_pte_multiplier(history, target, pteCol):
    # pte_rt_price/daily_rt_price
    pte_values = set(history[pteCol])
    pte_multiplier = {}

    hourly = history.groupby(pteCol)[target].mean()

    for pte in pte_values:
        pte_multiplier[pte] = hourly[pteCol]

    return pte_multiplier


def build_search_bid_space(v, a, b, min_ratio=0.5, max_ratio=1.5,interval=0.1):
    if v == 0:
        bid_space = np.arange(a, 0, b)
    elif v > 0:

        bid_space = np.arange(int(v * min_ratio), min(int(v * max_ratio), 25000), min(max(int(v * interval), 5), 25000))
    else:
        bid_space = np.arange(max(-1000, int(v * 2)), int(v * 0.5), max(int(v * 0.1), 5))

    return bid_space


if __name__ == '__main__':

    # con = Configuration()
    # imb, imb_config = con.readFile('baseline')
    # imb = imb[['Take_From','DeliveryDate','DayAheadPrice','ActualVolumes','First_Forecast_Volume','PERIOD']]
    # imb = imb.rename(columns={'DeliveryDate':'Date'})
    # imb['DeliveryDate'] = con.add_time_to_date(imb,'Date','PERIOD')
    # imb = imb.dropna()
    #
    # daily_da = imb.groupby('Date')['DayAheadPrice'].mean()
    # imb = imb.join(daily_da,on='Date',rsuffix='_daily')
    # daily_imb = imb.groupby('Date')['Take_From'].mean()
    # imb = imb.join(daily_imb,on='Date',rsuffix='_daily')
    #
    # hold_df = imb[imb['DeliveryDate']>=param.hold_out_date_begin]
    # predict_da_take_prob = pd.read_excel(param.data_folder_path+'/results/hold-out-prediction/TAKE_AUC_0.5893.xlsx')
    # hold_df = hold_df.merge(predict_da_take_prob,on='DeliveryDate',how='inner')
    #
    # est_DA, _ = Configuration().readFile('predict-DA')
    # mpe, _ = Configuration().readFile('predict-MPE')
    # hold_df = hold_df.merge(est_DA, on='DeliveryDate', how='inner')
    # hold_df = hold_df.merge(mpe,on='DeliveryDate',how='inner')
    # est_DA_daily = hold_df.groupby('Date')['predict_DA'].mean().reset_index()
    # est_DA_daily = est_DA_daily.rename(columns={'predict_DA':'predict_DA_daily'})
    # hold_df = hold_df.merge(est_DA_daily,on='Date',how='inner')
    # #
    # imb.to_excel(param.data_folder_path + '/a_simulation.xlsx', index = False)
    # hold_df.to_excel(param.data_folder_path + '/b_simulation.xlsx',index = False)
    # #

    imb = pd.read_excel(param.data_folder_path + '/a_simulation.xlsx')
    hold_df = pd.read_excel(param.data_folder_path + '/b_simulation.xlsx')

    strategy = 3
    num_resample = 1000
    num_historical_days = 60
    use_strategy_prob = 0
    min_bid_value_when_forecast_zero = -1000
    bid_interval_when_forecast_zero = 10
    experiment_times = 2
    current_experiment = 0
    experiment_result = []

    while current_experiment < experiment_times:
        current_experiment += 1
        print('\nexperiment:{}'.format(current_experiment))

        row_id = 0
        best_bids = []
        last_d = None
        last_kde = None
        last_multipliers = None
        if_use_strategy = np.random.uniform(0,1,len(hold_df))
        if_use_strategy = if_use_strategy>=use_strategy_prob
        different_bid = []

        while row_id<len(hold_df):
            d,p,v,mpe,da,da_daily,da_prob=hold_df.iloc[row_id][['Date','PERIOD','First_Forecast_Volume','predict_diff','predict_DA','predict_DA_daily','predict_DA>TAKE_proba']]

            if last_d !=d:
                last_kde,last_multipliers = get_simulated_imb_price(imb,'PERIOD', d,'DeliveryDate', num_historical_days)
                last_d = d

            bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)

            best_objective_v = -math.inf
            best_bid = v

            if (strategy == 1):
                sim_imb_prices = last_kde.resample(num_resample) * da_daily * last_multipliers[p]
                best_bid = search_best_quantile(da, sim_imb_prices, mpe, v, bid_space)

            elif (strategy == 2):
                sim_imb_prices = last_kde.resample(num_resample) * da_daily * last_multipliers[p]
                best_bid = search_best_pnl_var(da, sim_imb_prices, mpe, v, bid_space)

            elif strategy == 3: # use DA>TAKE prediction as a heuristic strategy
                if random.uniform(0,1)>= da_prob:
                    if da_prob >=0.5:
                        best_bid = v * 1.5
                    else:
                        best_bid = v * 0.5

            elif strategy == 4:
                if da_prob >= 0.5:
                    best_bid = v * 1.5
                else:
                    best_bid = v * 0.5

            best_bids.append(best_bid)
            row_id +=1

        hold_df['our_bid'] = best_bids

        # evaluation
        hold_df['baseline_pnl'] = compute_pnl(hold_df['First_Forecast_Volume'], hold_df['ActualVolumes'], \
                                              hold_df['true_DA'],hold_df['Take_From'])

        hold_df['our_pnl'] = compute_pnl(hold_df['our_bid'], hold_df['ActualVolumes'], \
                                         hold_df['true_DA'], hold_df['Take_From'])
        total_baseline_pnl = sum(hold_df['baseline_pnl'])
        improvement = sum(hold_df['our_pnl']) - total_baseline_pnl
        print('total_baseline_pnl = {}'.format(sum(hold_df['baseline_pnl'])))
        print('our_pnl-baseline_pnl= {}'.format(improvement))
        experiment_result.append(improvement*100/abs(total_baseline_pnl))

        # save result to file
        # daily pnl
        daily_base_pnl = hold_df.groupby('Date')['baseline_pnl'].mean().reset_index()
        daily_base_pnl = daily_base_pnl.rename(columns={'baseline_pnl':'base_pnl_daily'})
        daily_our_pnl = hold_df.groupby('Date')['our_pnl'].mean().reset_index()
        daily_our_pnl = daily_our_pnl.rename(columns={'our_pnl':'our_pnl_daily'})

        daily_base_pnl_var = hold_df.groupby('Date')['baseline_pnl'].var().reset_index()
        daily_base_pnl_var = daily_base_pnl_var.rename(columns={'baseline_pnl':'base_daily_pnlvar'})
        daily_our_pnl_var = hold_df.groupby('Date')['our_pnl'].var().reset_index()
        daily_our_pnl_var = daily_our_pnl_var.rename(columns={'our_pnl':'our_daily_pnlvar'})

        # daily variance
        daily_pnl = daily_base_pnl.merge(daily_our_pnl,on='Date',how='inner')
        daily_pnl_var = daily_base_pnl_var.merge(daily_our_pnl_var,on='Date',how='inner')
        evaluation = daily_pnl.merge(daily_pnl_var, on='Date',how='inner')

        path1=param.hold_out_prediction_path + 'strategy_'+ str(strategy) +'_exp'+ str(current_experiment)+'_evaluate.xlsx'
        evaluation.to_excel(path1, index = False)
        path2=param.hold_out_prediction_path + 'strategy_' + str(strategy) + '_exp' + str(current_experiment) + '.xlsx'
        hold_df.to_excel(path2, index=False)
        print('save result to:\n{}\n{}'.format(path1,path2))

        # print
        a = hold_df[ hold_df['our_bid']!= hold_df['First_Forecast_Volume']]
        b = a[ a['baseline_pnl']<= a['our_pnl']]
        print('{}% used strategy'.format( round(100*len(a)/len(hold_df),2)))
        print('{}% get improved'.format( round(100* len(b)/len(a), 2)))

        c = hold_df[ hold_df['baseline_pnl']> hold_df['our_pnl']]
        c['loss'] = c['baseline_pnl']-c['our_pnl']
        avg_loss = sum(c['loss'])/len(c)
        print('avg loss = {}'.format( round(avg_loss,4)))

        avg_win = sum(b['our_pnl']-b['baseline_pnl'])/len(c)
        print('avg win = {}'.format( round(avg_win), 4))

    print('improvement percentage:')
    print(experiment_result)
    print('avg = {}, max = {}, min = {}'. format(sum(experiment_result)/len(experiment_result), \
                                                 max(experiment_result),\
                                                 min(experiment_result)))
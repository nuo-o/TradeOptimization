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
        objective_v = sim_PNL.mean() / sim_PNL.var()

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
    #
    # con = Configuration()
    # imb, imb_config = con.readFile('baseline')
    # imb = imb[['Take_From','Feed_Into','DeliveryDate','DayAheadPrice','ActualVolumes','First_Forecast_Volume','PERIOD']]
    # imb = imb.rename(columns={'DeliveryDate':'Date','TotalPnL':'base_pnl'})
    # imb['DeliveryDate'] = con.add_time_to_date(imb,'Date','PERIOD')
    # print(len(imb))
    # imb = imb.dropna()
    # print(len(imb))
    #
    # daily_da = imb.groupby('Date')['DayAheadPrice'].mean()
    # imb = imb.join(daily_da,on='Date',rsuffix='_daily')
    # daily_take = imb.groupby('Date')['Take_From'].mean()
    # imb = imb.join(daily_take, on='Date', rsuffix='_daily')
    # daily_feed = imb.groupby('Date')['Feed_Into'].mean()
    # imb = imb.join(daily_feed, on='Date', rsuffix='_daily')
    #
    # hold_df = imb[imb['DeliveryDate']>=param.hold_out_date_begin]
    # # predict_da_take_prob = pd.read_excel(param.data_folder_path+'/results/hold-out-prediction/TAKE_AUC_0.5893.xlsx')
    # # hold_df = hold_df.merge(predict_da_take_prob,on='DeliveryDate',how='inner')
    #
    # est_DA, _ = Configuration().readFile('predict-DA')
    # mpe, _ = Configuration().readFile('predict-MPE')
    # hold_df = hold_df.merge(est_DA, on='DeliveryDate', how='inner')
    # hold_df = hold_df.merge(mpe,on='DeliveryDate',how='inner')
    # est_DA_daily = hold_df.groupby('Date')['predict_DA'].mean().reset_index()
    # est_DA_daily = est_DA_daily.rename(columns={'predict_DA':'predict_DA_daily'})
    # hold_df = hold_df.merge(est_DA_daily,on='Date',how='inner')
    #
    # imb.to_excel(param.data_folder_path + '/a_simulation.xlsx', index = False)
    # hold_df.to_excel(param.data_folder_path + '/b_simulation.xlsx',index = False)

#############################
    imb = pd.read_excel(param.data_folder_path + '/a_simulation.xlsx')
    hold_df = pd.read_excel(param.data_folder_path + '/b_simulation_.xlsx')
    evaluate = False

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
        different_bid = []

        tempt = datetime(2018,7,1)
        while row_id<len(hold_df):
            #
            # if row_id %10000:
            #     print('processed:{}%'.format(int(100*(row_id+1)/len(hold_df))))

            d,p,v,da,da_daily =hold_df.iloc[row_id][['Date','PERIOD','First_Forecast_Volume','predict_DA','predict_DA_daily']]

            if d > tempt:
                print()
                tempt = d
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

            elif strategy ==3:
                sim_take_prices = take_kde.resample(num_resample) * da_daily * take_multipliers[p]
                sim_feed_prices = feed_kde.resample(num_resample) * da_daily * feed_multipliers[p]

                best_bid = search_best_mean_var(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)

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
            row_id +=1

        hold_df['our_bid'] = best_bids
        if not evaluate:
            hold_df.to_excel(param.data_folder_path + '/operation_bid.xlsx', index = False)
            print('save predicted bid value to :{}'.format(param.data_folder_path + '/operation_bid.xlsx'))
        else:
            hold_df['our_pnl'] = compute_pnl(best_bids, \
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
            print('total baseline pnl:\n{}'.format(total_base_pnl))
            print('total our pnl:\n{}'.format(total_our_pnl))
            print('total improve:\n{}'.format(total_improve))
            experiment_result.append(total_improve*100/abs(total_base_pnl))

            a = hold_df[ hold_df['our_bid']!= hold_df['First_Forecast_Volume']]
            b = a[ a['base_pnl']<= a['our_pnl']]
            print('{}% get improved'.format( round(100* len(b)/len(a), 2)))

            c = hold_df[ hold_df['base_pnl']> hold_df['our_pnl']]
            c['loss'] = c['base_pnl']-c['our_pnl']
            avg_loss = sum(c['loss'])/len(c)
            print('avg loss = {}'.format( round(avg_loss,4)))

            avg_win = sum(b['our_pnl']-b['base_pnl'])/len(c)
            print('avg win = {}'.format( round(avg_win), 4))

            # save result to file
            # daily pnl
            base_daily_sum = hold_df.groupby('Date')['base_pnl'].sum().reset_index()
            base_daily_sum = base_daily_sum.rename(columns={'base_pnl':'base_pnl_sum'})
            our_daily_sum = hold_df.groupby('Date')['our_pnl'].sum().reset_index()
            our_daily_sum = our_daily_sum.rename(columns={'our_pnl':'our_pnl_sum'})
            base_daily_var = hold_df.groupby('Date')['base_pnl'].var().reset_index()
            base_daily_var = base_daily_var.rename(columns={'base_pnl':'base_pnl_var'})
            our_daily_var = hold_df.groupby('Date')['our_pnl'].var().reset_index()
            our_daily_var = our_daily_var.rename(columns={'our_pnl':'our_pnl_var'})

            saved_result = base_daily_sum.merge(our_daily_sum, on='Date',how='inner')
            saved_result = saved_result.merge(base_daily_var, on='Date',how='inner')
            saved_result = saved_result.merge(our_daily_var, on='Date',how='inner')

            path1=param.hold_out_prediction_path + 'strategy_'+ str(strategy) +'_exp'+ str(current_experiment)+'_evaluate.xlsx'
            saved_result.to_excel(path1, index = False)
            path2=param.hold_out_prediction_path + 'strategy_' + str(strategy) + '_exp' + str(current_experiment) + '.xlsx'
            hold_df.to_excel(path2, index=False)
            print('save result to:\n{}\n{}'.format(path1,path2))
    if evaluate:
        print('improvement percentage:')
        print(experiment_result)
        print('avg = {}, max = {}, min = {}'. format(sum(experiment_result)/len(experiment_result), \
                                                     max(experiment_result),\
                                                     min(experiment_result)))
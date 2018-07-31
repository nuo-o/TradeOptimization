from utils.import_packages import *
import utils.hardcode_parameters as param
from scipy.stats.kde import gaussian_kde
from models.Simulation import search_best_quantile
from openpyxl import load_workbook


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
    #
    # imb.to_excel(param.data_folder_path + '/a_simulation.xlsx', index = False)
    # hold_df.to_excel(param.data_folder_path + '/b_simulation.xlsx',index = False)
    imb = pd.read_excel(param.data_folder_path + '/a_simulation.xlsx',sheet_name='df')
    hold_df = pd.read_excel(param.data_folder_path + '/b_simulation_.xlsx',sheet_name='Sheet1')

    predict_da = pd.read_excel(param.data_folder_path + '/operation_pred_DA.xlsx', sheet_name = 'Sheet1')
    hold_df = hold_df.merge(predict_da, on='DeliveryDate',how='inner')
    hold_df = hold_df.dropna().reset_index()

    num_resample = 1000
    num_historical_days = 60
    min_bid_value_when_forecast_zero = -1000
    bid_interval_when_forecast_zero = 10
    experiment_result = []
    best_bids = []

    last_sim_day = None
    take_kde = None
    feed_kde = None
    take_multipliers = None
    feed_multipliers = None
    row_id = 0

    while row_id < len(hold_df):
        d,p,v,da,da_daily = hold_df.iloc[row_id][['Date','PERIOD','First_Forecast_Volume','predict_DA','predict_DA_daily']]

        take_kde, take_multipliers = get_simulated_imb_price(imb,'Period',d,'DeliveryDate',num_historical_days,'Take_From')
        feed_kde, feed_multipliers = get_simulated_imb_price(imb,'Period',d,'DeliveryDate',num_historical_days,'Feed_Into')

        bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)

        sim_take_prices = take_kde.resample(num_resample)*da_daily*take_multipliers[p]
        sim_feed_prices = feed_kde.resample(num_resample)*da_daily*feed_multipliers[p]

        best_bid = search_best_quantile(da, sim_take_prices, sim_feed_prices, 0, v, bid_space)
        # m = best_bid - v
        best_bids.append(best_bid)
        row_id +=1

    hold_df['our_bid']=best_bids
    hold_df.to_excel(param.data_folder_path+'/results/hold-out-prediction/operation_bid.xlsx',index = False)

    evaluate_df = pd.read_excel(param.data_folder_path + '/operation_evaluate.xlsx')
    evaluate_df['our_bid'] = best_bids

    evaluate_df['our_pnl'] = compute_pnl(best_bids, evaluate_df['ActualVolumes'], evaluate_df['Take_From'],\
                                         evaluate_df['Feed_Into'], evaluate_df['DayAheadPrice'])
    print('total_our_pnl={}'.format(sum(evaluate_df['our_pnl'])))
    print('total_baseline={}'.format(sum(evaluate_df['base_pnl'])))
    print('total_improve= {} %'.format((100*sum(evaluate_df['our_pnl'])-sum(evaluate_df['base_pnl']))/abs(sum(evaluate_df['base_pnl']))))

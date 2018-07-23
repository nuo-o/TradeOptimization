from utils.import_packages import *
import utils.hardcode_parameters as param
from scipy.stats.kde import gaussian_kde


def get_simulated_imb_price(history, target, dateCol, pteCol, curTime, dateTimeCol):
    history_start = curTime - timedelta(days=60)
    history = history[(history[dateTimeCol]<curTime)&(history[dateTimeCol]>=history_start)]

    # daily Imb/DA
    ratio = history['Take_From_daily']/history['DayAheadPrice_daily']
    ratio_kde = gaussian_kde(ratio)

    # hourly Imb/daily_imb
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



if __name__ == '__main__':
    """Simulate real-time price:
    fit DA_Price/RT_Price distribution using recent 2-month-daily-imbalance price
    convert daily to pte price
    repeat for 100 scenarios
    compute optimal bid ranging in forecast+-1000 with max pnl/var(pnl)
    """
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
    # imb.to_excel(param.data_folder_path + '/simulation_df.xlsx', index = False)
    objective_func =2
    num_resample = 100


    imb = pd.read_excel(param.data_folder_path + '/simulation_df.xlsx')
    hold_df = imb[imb['DeliveryDate']>=param.hold_out_date_begin]
    mpe, _ = Configuration().readFile('predict-MPE')
    est_DA, _ = Configuration().readFile('predict-DA')
    hold_df = hold_df.merge(mpe,on='DeliveryDate',how='inner')
    hold_df = hold_df.merge(est_DA, on='DeliveryDate', how='inner')
    hold_df['predict-DA-daily'] = hold_df.groupby('Date')['predict_DA'].mean().reset_index()['predict_DA']

    row_id = 0
    best_bids = []
    last_d = None
    last_kde = None
    last_multipliers = None

    while row_id<len(hold_df):
        d,p,v,mpe,da,da_daily=hold_df.iloc[row_id][['Date','PERIOD','First_Forecast_Volume','predict_diff','predict_DA','predict-DA-daily']]

        if last_d !=d:
            last_kde,last_multipliers = get_simulated_imb_price(imb,'Take_From','Date','PERIOD',d,'DeliveryDate')
            last_d = d

        # simulate imbalance price
        sim_imb_prices = last_kde.resample(num_resample)*da_daily*last_multipliers[p]

        if v == 0:
            bid_space = np.arange( -1000, 0, 10)
        elif v>0:
            bid_space = np.arange( int(v*0.5), int(v * 1.5), min(max(int(v * 0.1),5),25000))
        else:
            bid_space = np.arange( max(-1000,int(v*2)), int(v*0.5), max(int(v*0.1),5))

        best_objective_v = -math.inf
        best_bid = v
        for b in bid_space:
            sim_pnl = (da - sim_imb_prices) * mpe * v

            if objective_func == 1:
                objective_v = sim_pnl.mean()/sim_pnl.var()

                if best_objective_v < objective_v:
                    best_objective_v = objective_v
                    best_bid = b

            elif objective_func ==2:
                objective_v = np.sort(sim_pnl)[:max(1, int(len(sim_pnl)*0.05))].mean()

                if best_objective_v < objective_v:
                    best_objective_v = objective_v
                    best_bid = b

        best_bids.append(best_bid)
        row_id +=1

    hold_df['our_bid'] = best_bids

    # evaluation
    hold_df['DA-TAKE'] = hold_df['true_DA']-hold_df['Take_From']
    hold_df['baseline_pnl'] = hold_df['DA-TAKE']*hold_df['true_diff']
    hold_df['our_pnl'] = hold_df['DA-TAKE']*(hold_df['our_bid']-hold_df['ActualVolumes'])

    # daily pnl
    daily_base_pnl = hold_df.groupby('Date')['baseline_pnl'].mean().reset_index()
    daily_base_pnl = daily_base_pnl.rename(columns={'baseline_pnl':'base_pnl_daily'})
    daily_our_pnl = hold_df.groupby('Date')['our_pnl'].mean().reset_index()
    daily_our_pnl = daily_our_pnl.rename(columns={'our_pnl':'our_pnl_daily'})

    daily_base_pnl_var = hold_df.groupby('Date')['baseline_pnl'].var().reset_index()
    daily_base_pnl_var = daily_base_pnl_var.rename(columns={'baseline_pnl':'base_pnl_var'})
    daily_our_pnl_var = hold_df.groupby('Date')['our_pnl'].var().reset_index()
    daily_our_pnl_var = daily_our_pnl_var.rename(columns={'our_pnl':'our_pnl_var'})

    daily_pnl = daily_base_pnl.merge(daily_our_pnl,on='Date',how='inner')
    daily_pnl_var = daily_base_pnl_var.merge(daily_our_pnl_var,on='Date',how='inner')
    evaluation = daily_pnl.merge(daily_pnl_var, on='Date',how='inner')

    evaluation.to_excel(param.data_folder_path + '/results/hold-out-prediction/simulation_2_evaluate.xlsx', index = False)
    hold_df.to_excel(param.data_folder_path + '/results/hold-out-prediction/simulation_2.xlsx', index=False)

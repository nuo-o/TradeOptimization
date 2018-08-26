from utils.import_packages import *
import utils.hardcode_parameters as param
from models.op3_Simulate import *

"""This program is to evaluate if simulated imbalance price has correct relationship with DA:
That is to evaluate if DA>TAKE and DA<FEED is correctly predicted"""

def prepare():
    imb = pd.read_excel(param.operation_folder + '/a_simulation.xlsx')
    hold_df = pd.read_excel(param.operation_folder + '/b_simulation_.xlsx')

    np.random.seed(123)

    evaluate_start_date = datetime(2018,2,1)
    evaluate_end_date = datetime(2018,8,1)
    hold_df = hold_df[(hold_df['DeliveryDate']<evaluate_end_date) & (hold_df['DeliveryDate']>=evaluate_start_date)]
    hold_df = hold_df.dropna(subset=['First_Forecast_Volume', 'predict_DA','predict_DA_daily'])
    imb = imb.dropna(subset=['Take_From','Feed_Into'])
    return imb, hold_df

if __name__ == '__main__':

    imb_origin, hold_df_origin = prepare()
    # simulation
    row_id =0
    DA_TAKE_pos_ratio = []
    DA_FEED_neg_ratio = []
    true_DA_TAKE_pos = []
    true_DA_FEED_neg = []
    last_sim_day = None
    saved_fig_count = 0

    row_id = 0

    take_quantiles = {}
    feed_quantiles = {}
    predicted_da = []
    true_data = {}
    bid_ratio = {}

    curTime = [datetime(2018, 2, 15), datetime(2018, 5, 23), datetime(2018, 3, 9), datetime(2018, 7, 28)]
    fig, position = plt.subplots( ncols = 3, nrows=2)
    posx,posy = 0, 0

    while row_id < 96*5:

        d, p, v, da, da_daily = hold_df_origin.iloc[row_id][
            ['Date', 'PERIOD', 'First_Forecast_Volume', 'predict_DA', 'predict_DA_daily']]

        if last_sim_day != d:
            simType = 'imb_daily/da_pte'
            take_kde, take_multipliers = get_simulated_imb_price(imb_origin, 'Period', d, 'DeliveryDate', num_historical_days,
                                                                 'Take_From',simType=simType)
            feed_kde, feed_multipliers = get_simulated_imb_price(imb_origin, 'Period', d, 'DeliveryDate', num_historical_days,
                                                                 'Feed_Into',simType=simType)
            last_sim_day = d
            sns.kdeplot(take_kde.resample(num_resample)[0], ax=position[posx][posy])
            posy += 1
            if posy >= 3 :
                posy = 0
                posx += 1

            row_id += 1
            continue
        else:
            row_id +=1
            continue

        if p >= 97:
            p = 96

        sampled_take = take_kde.resample(num_resample)
        sampled_feed = feed_kde.resample(num_resample)
        sim_take_prices = sampled_take * da_daily * take_multipliers[p]
        sim_feed_prices = sampled_feed * da_daily * feed_multipliers[p]

        for q in np.arange(10, 100, 10):
            quantile_this_pte = np.percentile(sim_take_prices[0], q) - da
            take_quantiles.setdefault(q, []).append(quantile_this_pte)
            quantile_this_pte = np.percentile(sim_feed_prices[0], q) - da
            feed_quantiles.setdefault(q, []).append(quantile_this_pte)

        true_row = imb_origin[ (imb_origin['Date'] ==d) & (imb_origin['Period'] ==p)].iloc[0]
        true_data.setdefault( 'Take_From', []).append(true_row['Take_From'])
        true_data.setdefault( 'Feed_Into', []).append(true_row['Feed_Into'])
        true_data.setdefault( 'TAKE-DA', []).append(true_row['Take_From']-true_row['DayAheadPrice'])
        true_data.setdefault( 'FEED-DA', []).append(true_row['Feed_Into']-true_row['DayAheadPrice'])

        # bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)

        # b = max_mean(da, sim_take_prices,sim_feed_prices , 0, v, bid_space)
        # bid_ratio.setdefault('max mean', []).append(b / v)

        # for percentile in [5, 10, 15, 20]:
        #     b = value_at_risk(da, sim_take_prices, sim_feed_prices, 0, v, bid_space, percentile=percentile)
        #     bid_ratio.setdefault(str(percentile) + ' percentile', []).append(b/v)
        #
        # predicted_da.extend([da])
        row_id += 1
    plt.suptitle('Simulated "{}" Distributions for Six Sample Days'.format(simType), fontSize='x-large')
    plt.show()

    # plt.title('Simulted Distribution of Take_daily/DA_pte')
    # plt.xlabel('Take_daily/DA_pte')
    # plt.ylabel('Probability')
    # plt.show()

    # plot take price quantiles
    colors = sorted(sns.color_palette('Oranges',len(take_quantiles.keys())), reverse=True)
    for q, c in zip(take_quantiles.keys(), colors):
        # plt.plot( np.arange(1, 97, 1), take_quantiles[q], color = c,label = 'Quantile ' + str(q),alpha = 0.7)
        plt.plot( np.arange(1, 97, 1), feed_quantiles[q], color = c,label ='Quantile' + str(q), alpha = 0.7)
    # sns.pointplot(x = np.arange(1, 97, 1), y = true_prices, color = 'b')
    # plt.plot( np.arange(1, 97, 1), true_data['TAKE-DA'], color = 'black', linestyle = 'dashed', label = 'True Price')
    plt.plot(np.arange(1, 97, 1), true_data['FEED-DA'], color='black', linestyle='dashed', label='True Price')

    plt.legend(loc = 'lower right', fontsize = 'medium')
    plt.title('Different quantiles for simulated FEED-DA on {}.'.format(last_sim_day._short_repr))
    plt.xlabel('Time Horizon (pte)')
    plt.ylabel('FEED-DA Price(euro)')
    # plt.savefig(param.operation_folder + '/take-quantiles.png')
    plt.show()

    # plt.figure()
    # colors1 = sorted(sns.color_palette('Blues', 2*len(take_quantiles.keys())), reverse=True)
    # colors2 = sorted(sns.color_palette('Oranges',2*len(feed_quantiles.keys())), reverse=True)

    # x = np.arange(1, 97, 1)
    # fig = plt.figure(1)
    # # ax1 = fig.add_subplot(3, 1, 1)
    # m = 1
    # for k, c in zip(bid_ratio.keys(), sns.color_palette("hls", len(bid_ratio))):
    #     ax = fig.add_subplot(len(bid_ratio), 1, m)
    #     m += 1
    #     ax.plot(x, bid_ratio[k], color = c, label = k, alpha = 0.5)
    #     # plt.legend(loc='best', fontsize = 'medium', bbox_to_anchor = (1,1), borderaxespad = 1)
    #     plt.legend(loc='best', fontsize = 'large')
    #     plt.ylabel('Bid/Forecast')
    #
    # plt.xlabel('Time(PTE)')
    # plt.tight_layout()
    # fig.suptitle('Bid/Forecast Volume Ratio with Different Risk on ' + last_sim_day._short_repr, fontsize = 'x-large')
    #
    # ax.legend(loc='best', fontsize='small', bbox_to_anchor=(1, 1), borderaxespad=0)
    # plt.title('Bid volumes with different q on ' + last_sim_day._short_repr)
    # plt.xlabel('Time (pte)')
    # plt.ylabel('Bid volume / Forecast Volume')
    #
    # for q, c1, c2 in zip(take_quantiles.keys(), colors1[3:], colors2[3:]):
    #     plt.plot(np.arange(1, 97, 1), take_quantiles[q], color=c1, label ='TAKE-DA Quantile ' + str(q))
    #     plt.plot(np.arange(1, 97, 1), feed_quantiles[q], color=c2, label = 'Feed-DA Quantile ' +str(q))
    #     # sns.pointplot(x = np.arange(1, 97, 1), y = true_prices, color = 'b')
    # plt.plot( np.arange(1, 97, 1), true_data['TAKE-DA'], color ='red', linestyle = 'dashed', label = 'True Take-DA')
    # plt.plot( np.arange(1, 97, 1), true_data['FEED-DA'], color='black', linestyle='dashed', label = 'True Feed-DA')
    # # plt.plot( np.arange(1, 97, 1), true_data['DayAheadPrice'], color = 'yellow',linestyle='dashed', label = 'True DayAheadPrice')
    # # plt.plot( np.arange(1, 97, 1), predicted_da, color = 'red', label = 'Predicted DA')
    #
    #
    # ax3.title('Bids given by different strategies.'.format(last_sim_day._short_repr))
    # ax3._label('Time Horizon (pte)')
    # plt.xlabel('Time Horizon (pte)')
    # plt.ylabel('Bid/Forecast power volume (KW)')
    # plt.savefig(param.operation_folder + '/feed-quantiles.png')
    # plt.show()
    #



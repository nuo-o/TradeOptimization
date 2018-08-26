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

    row_id = 96*4

    take_quantiles = {}
    feed_quantiles = {}
    predicted_da = []
    true_data = {}
    bid_ratio = {}

    curTime = [datetime(2018, 2, 15), datetime(2018, 5, 23), datetime(2018, 3, 9), datetime(2018, 7, 28)]

    for curtime in curTime:
        # hold_df = hold_df_origin[hold_df_origin['Date'] == curtime]

        num_history_days = [10]
        # colors1 = sorted(sns.color_palette('Oranges', len(num_history_days)), reverse=True)

        history_start = curtime - timedelta(days=int(10))
        history = imb_origin[(imb_origin['DeliveryDate']<curtime)&(imb_origin['DeliveryDate']>=history_start)]

        result = history[['Take_From', 'Feed_Into', 'DayAheadPrice', 'Period', 'Take_From_daily', 'Feed_Into_daily', 'DayAheadPrice_daily']]

        c1, c2, c3, c4, c5 = [], [], [], [], []
        da_mean = []
        take_mean = []
        feed_mean = []

        for i in np.arange(1, 97, 1):
            m = result[result['Period'] == i]
            # ratio = m['Take_From_daily']/m['DayAheadPrice_daily']
            c2.append(np.corrcoef( m['Take_From'], m['Take_From_daily'] )[1][0])
            c3.append(np.corrcoef( m['Take_From_daily'], m['DayAheadPrice'])[1][0])
            c5.append(np.corrcoef( m['Take_From'], m['DayAheadPrice_daily'])[1][0])

            if len(c1)==0:
                c1.extend(m['Take_From_daily'])
                c4.extend(m['DayAheadPrice_daily'])

        x = np.arange(1, 97, 1)
        fig = plt.figure()
        ax = fig.add_subplot(4,1,1)
        ax.plot( np.arange(1, 11, 1), c1, color = 'red', label = 'Take_daily')
        ax.plot( np.arange(1, 11, 1), c4, color = 'green', label = 'DA_daily')
        ax.set_xlabel('Historical Days (DAY)')
        ax.set_ylabel('Price (EURO)')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(4, 1, 2)
        ax.plot(x, c2, color='green', label = 'coeff(Take_pte, Take_daily)')
        ax.set_xlabel('Time (PTE)')
        ax.set_ylabel('Coefficient')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(4, 1, 3)
        ax.plot(x, c3, color='red', label='coeff(Take_daily, DA_pte)')
        ax.set_xlabel('Time (PTE)')
        ax.set_ylabel('Coefficient')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(4, 1, 4)
        ax.plot(x, c5, color='green', label='coeff(Take_pte, DA_daily)')
        ax.set_xlabel('Time (PTE)')
        ax.set_ylabel('Coefficient')
        ax.legend(loc='upper right')
        plt.suptitle('Correlations between Prices on historical days of {}'.format(str(curtime.date())))
        plt.tight_layout()
        plt.show()


        # c2 = np.corrcoef( m['Feed_Into'], m['DayAheadPrice'])[1][0]
        # correlation_take.extend([c1])
        # correlation_feed.extend([c2])
        # da_mean.append( m.DayAheadPrice.mean() )
        # take_mean.append(m.Take_From.mean() )
        # feed_mean.append(m.Feed_Into.mean() )

    # ax1 = fig.add_subplot(3, 1, 1)
    # m = imb[['Period', 'Take_From_daily', 'DayAheadPrice', 'DayAheadPrice_daily', 'Feed_Into_daily', 'Take_From', 'Feed_Into']]
    # m['Take_daily/DA'] = imb['Take_From_daily']/imb['DayAheadPrice']
    #
    # s = m[m['Take_daily/DA']<80]
    # ax1.scatter( s['Period'], s['Take_daily/DA'], alpha = 0.1, label = 'Daily Take_From/ Quarterly DA')
    # ax1.legend(loc='upper right')
    # # ax1.title('Daily Take_From/ Quarterly DA')
    #
    # x = imb['Period']
    # y = imb['Take_From']/imb['Take_From_daily']
    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.scatter( x, y, alpha = 0.1, label = 'Quarterly Take_From/ Daily Take_From')
    # ax2.legend(loc='upper right')
    #
    # m['Take/DA'] = imb['Take_From']/imb['DayAheadPrice']
    # s = m[m['Take/DA'] < 100]
    # ax3 = fig.add_subplot(3, 1, 3)
    # ax3.scatter(s['Period'], s['Take/DA'], alpha = 0.1, label = 'Quarterly Take_From/ Quarterly DA')
    # ax3.set_xlabel('Time(PTE)')
    # ax3.legend(loc='upper right')
    #
    # plt.suptitle('Scatter Graphs for Different Price Ratios')
    # plt.show()

    # while row_id < 96*5:
    #
    #     d, p, v, da, da_daily = hold_df.iloc[row_id][
    #         ['Date', 'PERIOD', 'First_Forecast_Volume', 'predict_DA', 'predict_DA_daily']]
    #
    #     # if last_sim_day != d:
    #     #     take_kde, take_multipliers = get_simulated_imb_price(imb, 'Period', d, 'DeliveryDate', num_historical_days,
    #     #                                                          'Take_From')
    #     #     feed_kde, feed_multipliers = get_simulated_imb_price(imb, 'Period', d, 'DeliveryDate', num_historical_days,
    #     #                                                          'Feed_Into')
    #     #     last_sim_day = d
    #     #
    #     # if p >= 97:
    #     #     p = 96
    #     #
    #     # sampled_take = take_kde.resample(num_resample)
    #     # sampled_feed = feed_kde.resample(num_resample)
    #     # sim_take_prices = sampled_take * da_daily * take_multipliers[p]
    #     # sim_feed_prices = sampled_feed * da_daily * feed_multipliers[p]
    #     #
    #     # for q in np.arange(10, 100, 10):
    #     #     quantile_this_pte = np.percentile(sim_take_prices[0], q) - da
    #     #     take_quantiles.setdefault(q, []).append(quantile_this_pte)
    #     #     quantile_this_pte = np.percentile(sim_feed_prices[0], q) - da
    #     #     feed_quantiles.setdefault(q, []).append(quantile_this_pte)
    #     #
    #     # true_row = imb[ (imb['Date'] ==d) & (imb['Period'] ==p)].iloc[0]
    #     # true_data.setdefault( 'Take_From', []).append(true_row['Take_From'])
    #     # true_data.setdefault( 'Feed_Into', []).append(true_row['Feed_Into'])
    #     # true_data.setdefault( 'TAKE-DA', []).append(true_row['Take_From']-true_row['DayAheadPrice'])
    #     # true_data.setdefault( 'FEED-DA', []).append(true_row['Feed_Into']-true_row['DayAheadPrice'])
    #     #
    #     # bid_space = build_search_bid_space(v, min_bid_value_when_forecast_zero, bid_interval_when_forecast_zero)
    #
    #     # b = max_mean(da, sim_take_prices,sim_feed_prices , 0, v, bid_space)
    #     # bid_ratio.setdefault('max mean', []).append(b / v)
    #
    #     # for percentile in [5, 10, 15, 20]:
    #     #     b = value_at_risk(da, sim_take_prices, sim_feed_prices, 0, v, bid_space, percentile=percentile)
    #     #     bid_ratio.setdefault(str(percentile) + ' percentile', []).append(b/v)
    #     #
    #     # predicted_da.extend([da])
    #     row_id += 1

    # # plot take price quantiles
    # colors = sorted(sns.color_palette('Blues', 2*len(quantile_per_hour.keys())), reverse=True)
    # for q, c in zip(quantile_per_hour.keys(), colors):
    #     plt.plot( np.arange(1, 97, 1), quantile_per_hour[q], color = c,label = 'Quantile ' + str(q))
    # # sns.pointplot(x = np.arange(1, 97, 1), y = true_prices, color = 'b')
    # plt.plot( np.arange(1, 97, 1), true_data['Take_From'], color = 'black', linestyle = 'dashed', label = 'True Price')
    #
    # plt.legend(loc = 'best', fontsize = 'x-small')
    # plt.title('Different quantiles for simulated price on {}.'.format(last_sim_day._short_repr))
    # plt.xlabel('Time Horizon (pte)')
    # plt.ylabel('Predicted Take-from-system Price (euro)')
    # plt.savefig(param.operation_folder + '/take-quantiles.png')
    # plt.show()
    # plt.figure()
    # colors1 = sorted(sns.color_palette('Blues', 2*len(take_quantiles.keys())), reverse=True)
    # colors2 = sorted(sns.color_palette('Oranges',2*len(feed_quantiles.keys())), reverse=True)
    #
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

    # for q, c1, c2 in zip(take_quantiles.keys(), colors1[3:], colors2[3:]):
    #     plt.plot(np.arange(1, 97, 1), take_quantiles[q], color=c1, label ='TAKE-DA Quantile ' + str(q))
    #     plt.plot(np.arange(1, 97, 1), feed_quantiles[q], color=c2, label = 'Feed-DA Quantile ' +str(q))
    #     # sns.pointplot(x = np.arange(1, 97, 1), y = true_prices, color = 'b')
    # plt.plot( np.arange(1, 97, 1), true_data['TAKE-DA'], color ='red', linestyle = 'dashed', label = 'True Take-DA')
    # plt.plot( np.arange(1, 97, 1), true_data['FEED-DA'], color='black', linestyle='dashed', label = 'True Feed-DA')
    # # plt.plot( np.arange(1, 97, 1), true_data['DayAheadPrice'], color = 'yellow',linestyle='dashed', label = 'True DayAheadPrice')
    # # plt.plot( np.arange(1, 97, 1), predicted_da, color = 'red', label = 'Predicted DA')


    # ax3.title('Bids given by different strategies.'.format(last_sim_day._short_repr))
    # ax3._label('Time Horizon (pte)')
    # plt.xlabel('Time Horizon (pte)')
    # plt.ylabel('Bid/Forecast power volume (KW)')
    # plt.savefig(param.operation_folder + '/feed-quantiles.png')
    # plt.show()




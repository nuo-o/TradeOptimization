from utils.import_packages import *
import utils.hardcode_parameters as param


class AbstractStrategyClass():
    def __init__(self, parameters):
        pass

    def get_bid_value(predicted_params, constraint_params = None, max_bid=25000, min_bid=0):
        pass

    def evaluate(self, bid, true_generation, true_da_price, true_imb_price):
        diff = bid - true_generation

        # for i,d in enumerate(bid):
        #     if d == math.nan:
        #         print(i)
        #         break
        result = {'strategy_diff': bid - true_generation}
        result['DA_TAKE'] = true_da_price - true_imb_price
        result['strategy_pnl'] = result['DA_TAKE'] * result['strategy_diff']
        return result


class S1_maxPnl(AbstractStrategyClass):
    """This is taken only when the DA>TAKE is predicted as true"""
    def __init__(self):
        pass

    def get_bid_value(self, predicted_params,constraint_params = None, max_bid=25000, min_bid=0):
        bid = []
        predicted_DA_take = predicted_params['predict_DA>TAKE']
        forecast_power = predicted_params['First_Forecast_Volume']

        for p_da_tak, forecast_v in zip(predicted_DA_take, forecast_power):
            if p_da_tak == 1:
                bid.append(max_bid)
            else:
                bid.append(forecast_v)

        return bid
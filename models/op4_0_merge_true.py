from utils.import_packages import *
import utils.hardcode_parameters as param
from models.op3_Simulate import compute_pnl

if __name__ == '__main__':
    pred_bid = pd.read_excel(param.operation_folder + '/operation_bid.xlsx', sheet_name='Sheet1')
    pred_bid = pred_bid[['our_bid','DeliveryDate']]
    a_simulation = pd.read_excel(param.operation_folder + '/a_simulation.xlsx', sheet_name='Sheet1')

    pred_bid = pred_bid.merge(a_simulation, on='DeliveryDate', how='left')

    pred_bid.to_excel(param.operation_folder + '/operation_bid_true.xlsx', index = False)
from utils.import_packages import *
import utils.hardcode_parameters as param

if __name__ == '__main__':
    """This file is to clean DayAhead Price from 2010-1-1 to 2018-7-23
    merge with OTC price
    prepare for DA-price prediction model"""

    da = pd.read_excel(param.day_ahead_folder + '/raw-da-two-tables.xlsx',sheet_name='Sheet1')
    da = da[['DeliveryDate','Name','Value']]

    # extend 24-hour da to 96 pte
    new_value = []
    new_pte = []
    new_d = []
    for row_id in range(len(da)):
        nv = da.iloc[row_id]['Value']
        np = da.iloc[row_id]['Name']
        nd = da.iloc[row_id]['DeliveryDate']
        new_value.extend([nv]*4)
        new_pte.extend([np*4-3, np*4-2, np*4-1, np*4])
        new_d.extend([nd]*4)

    new_da = pd.DataFrame({'DeliveryDate':new_d, 'start':new_pte, 'DA':new_value})
    new_da = new_da.sort_values(by=['DeliveryDate','start'])
    new_da.to_excel(param.day_ahead_folder +'/new-DA-price.xlsx', index = False)
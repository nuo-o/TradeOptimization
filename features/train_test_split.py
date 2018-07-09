from utils.import_packages import *
import utils.hardcode_parameters as param

def train_test_split(df, dateCol, split_date = None, splitBySize = False, train_size=0.8):
    if splitBySize:
        used = set()
        unique_days = [day for day in df[dateCol] if day not in used and (used.add(day) or True)]
        split_date = unique_days[math.floor(len(unique_days) * train_size)]

    split_index = df.index[df[dateCol] == split_date].tolist()[0]
    # train_df = df.iloc[:split_index]
    # test_df = df.iloc[split_index:]

    # return train_df, test_df
    return split_index

if __name__ == '__main__':

    # make x,y
    # raw_df = pd.read_excel(param.data_folder_path + '/temp/imb_DA_power2.xlsx')
    raw_df = pd.read_excel(param.data_folder_path + '/temp/imb_dmd.xlsx')

    raw_df['DA>TAKE'] = raw_df['DA-price'] - raw_df['take_from_system_price']
    raw_df['DA>FEED'] = raw_df['DA-price'] - raw_df['feed_into_system_price']
    binary_target = [1 if v >0 else 0 for v in raw_df['DA>TAKE']]
    raw_df['DA>TAKE'] = binary_target
    binary_target = [1 if v >0 else 0 for v in raw_df['DA>FEED']]
    raw_df['DA>FEED'] = binary_target

    train_df, test_df = train_test_split(raw_df, splitBySize=True)

    train_df.to_excel(param.data_folder_path + '/train_imb_demd.xlsx', index=False)
    test_df.to_excel(param.data_folder_path + '/test_imb_demd.xlsx', index = False)
from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.DataChecker import *
import pickle

if __name__ == '__main__':
    # target = 'Feed-DA'
    target = 'Con-Feed'
    df = pd.read_excel(param.operation_folder + '/DA_Spread.xlsx', sheet_name='Sheet1')

    for month in range(1, 13):
        cur_month_df = df[df['Month'] == month]

        weekday_hour = {1:[0]*24, 2:[0]*24, 3:[0]*24, 4:[0]*24, 5:[0]*24, 6:[0]*24,7:[0]*24}

        for weekday in range(1, 8):
            for hour in range(0, 24):
                a =cur_month_df[(cur_month_df['Hour']== hour) & (cur_month_df['WKDY']==weekday)]
                weekday_hour[weekday][hour] = np.array(a[target]).mean()

        savePath = param.operation_folder + '/'+ target + '_month_' + str(month) + '.pickle'
        with open( savePath, 'wb') as handle:
            pickle.dump(weekday_hour, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open( savePath, 'rb') as handle:
        #     b = pickle.load(handle)
        #
        # print(a==b)



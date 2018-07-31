from utils.import_packages import *
import utils.hardcode_parameters as param

from data_gathering.Configure import Configuration


class DataChecker():
    '''This class should be used as the first step of processing data.
    It checks duplicate rows, missing dates and missing values.'''
    def __init__(self, df):
        self.df = df

    def check_missing_dates(self, date_column, printInfo=True):
        if date_column not in self.df.columns:
            raise ValueError('no ' + str(date_column) + 'column in this file. Check the spelling.')

        first_day = self.df.loc[0, date_column]
        end_day = self.df.loc[self.df.index[-1], date_column]
        num_day = (end_day - first_day).days
        dates = [end_day - timedelta(days=x) for x in range(0, num_day)]
        obsv_dates = set(self.df[date_column])
        missing_dates = set(dates) - obsv_dates

        print('\nData range: From {} to {}\n'.format(first_day, end_day))

        if printInfo:
            if len(missing_dates) > 0:
                print('Missing dates...')
                for d in missing_dates:
                    print(d)
            else:
                print('No missing date')
        print()
        return missing_dates

    def get_date_of_missing_features(self, date_column, valCol, printInfo = True):
        if (date_column not in self.df.columns) | (valCol not in self.df.columns):
            raise ValueError('no '+ str(valCol) + ' column. ')

        valCol = valCol.split(',')

        missing_data_dates = []

        for (d, v) in zip(self.df[date_column].values, valCol):
            if math.isnan(v):
                missing_data_dates.append(d)

        missing_data_dates = set(missing_data_dates)

        if printInfo:
            if len(missing_data_dates)>0:
                print()
                print('Missing {} values in following dates...'.format(valCol))
                for d in missing_data_dates:
                    print(d)
            else:
                print('no missing values')

        return missing_data_dates

    def check_duplicate_forecast(self, col = None, printInfo = True):
        duplicate = self.df.duplicated(subset=col)
        result = duplicate[duplicate == True]

        if printInfo:
            print()
            if len(result)>0:
                print('Duplicate ratio = {} %'.format(len(result)*100/len(self.df)))
                print('example:{}'.format( self.df.loc[result._index[0], col]))
            else:
                print('No duplicates\n')

        return result


if __name__ == '__main__':

    # file = 'baseline'
    # print('check file:{}...'.format(file))
    # df, config = Configuration().readFile(file)

    df = pd.read_excel(param.data_folder_path + '/results/hold-out-prediction/strategy_1_exp1.xlsx')
    # df = pd.read_excel( param.data_folder_path + '/plant/clean_plant.xlsx', sheet_name='Sheet1')
    # df = pd.read_csv( param.data_folder_path + '/plant/w.csv')
    # if no realtime column, first convert
    # ts =TimeSeriesData(df, config.date_col, config.forecast_v, pteCol = config.pte_col)
    # df = ts.file
    # else
    # df[config.date_col] = pd.to_datetime(df[config.date_col])

    # checker = DataChecker(df)
    # a = checker.check_missing_dates(config.date_col)
    # checker.check_duplicate_forecast([config.date_col])

    checker = DataChecker(df)
    a = checker.check_missing_dates( 'DeliveryDate' )
    checker.check_duplicate_forecast( ['DeliveryDate'])
    # forecast_v = config.forecast_v.split(',')
    #
    # for col in forecast_v:
    #     a = checker.get_date_of_missing_features(config.date_col, col, printInfo=False)
    #     print('{} missing ratio = {}'.format(forecast_v, len(a)*100/len(df)))

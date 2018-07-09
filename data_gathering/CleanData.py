from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.DataChecker import DataChecker
import pytz

def convert2RealTime(dateVals, pteVals):
    timeDate = []

    for (date,pte) in zip( dateVals,pteVals):
        timeDate.append( date+timedelta(minutes=15*(pte-1)))
    return timeDate


class TimeSeriesData():
    def __init__(self, df, date_column, val_column, created_time_column=None, periodByHour = 4, pteCol = None, convertTime = True):
        df[date_column] = pd.to_datetime(df[date_column])
        selected_feat = []
        self.valCol = [x for x in val_column.split(',')]
        selected_feat.extend(self.valCol)

        if (pteCol != None) :
            if convertTime:
                timeDate = []
                for (date, pte) in zip(df[date_column], df[pteCol]):
                    timeDate.append( date+timedelta(minutes=60/periodByHour*(pte-1)))
                df[date_column] = timeDate
            selected_feat.append(pteCol)
            self.pteCol = pteCol

        selected_feat.append(date_column)

        if created_time_column != None:
            selected_feat.append(created_time_column)

        if math.isnan(val_column):
            self.file = df
        else:
            self.file = df[selected_feat]

        self.dateCol = date_column
        self.fctTimeCol = created_time_column
        self.file[self.dateCol] = pd.to_datetime( self.file[self.dateCol] )
        self.periodByHour = periodByHour

    def pre_process_piepline(self):
        # if ('UTC' in self.dateCol) | ('utc' in self.dateCol):
        #     self.convert_UTC_2_local( 'DeliveryDate', param.timezone )
        #     self.file = self.file.drop( self.dateCol, axis = 1)
        #     self.dateCol = 'DeliveryDate'
        #     self.file[self.dateCol] = self.file[self.dateCol]

        self.remove_duplicate_forecast()

    def post_process_piepeline(self, scale_period = 4, interval = '0.25H'):
        # self.scale_forecast_frequency(scale_period, interval)
        self.insert_missing_time()
        self.extract_PTE()
        self.fill_nan_by_avg()
        return self.file

    def scale_forecast_freq_by_avg(self, interval, scale_period = 4):
        scaled_df = pd.DataFrame()
        self.file[self.valCol]/=scale_period

        print('scale forecast frequency by average')
        for index, row in self.file.iterrows():
            expected_datetime = pd.DatetimeIndex(start=self.file.iloc[index][self.dateCol], periods=self.periodByHour,
                                                 freq=interval).values
            scaled_df = scaled_df.append([row]*self.periodByHour, ignore_index=True)
            scaled_df.iloc[self.periodByHour*index:(self.periodByHour*index+self.periodByHour)][self.dateCol] = expected_datetime

            if index % 1000 == 0:
                print('processed {}/{} rows'.format( index /1000 ,int(len(self.file)/1000)))

        self.file = scaled_df

    def scale_forecast_freq_by_copy(self, interval = '0.25H'):
        scaled_df = pd.DataFrame()
        print('scale forecast frequency by copy:')

        for index, row in self.file.iterrows():
            expected_datetime = pd.DatetimeIndex(start=self.file.iloc[index][self.dateCol], periods=self.periodByHour,
                                                 freq=interval).values
            scaled_df = scaled_df.append([row]*self.periodByHour, ignore_index=True)
            scaled_df.iloc[self.periodByHour*index:(self.periodByHour*index+self.periodByHour)][self.dateCol] = expected_datetime

            if index % 1000 == 0:
                print('processed {}/{} rows'.format( index /1000 ,int(len(self.file)/1000)))

        self.file = scaled_df

    def insert_missing_time(self, interval='0.25H'):
        expected_datetime = pd.DatetimeIndex(start=self.file.iloc[0][self.dateCol], end=self.file.iloc[-1][self.dateCol],freq=interval)
        ideal_datetime_df = pd.DataFrame({self.dateCol: expected_datetime})

        self.file = ideal_datetime_df.merge(self.file, on=self.dateCol, how='left')
        # full_power_df[self.timeName] = pd.to_datetime(full_power_df[self.timeName])
        return self.file

    def extract_PTE(self):
        minutePerPTE = 60 / self.periodByHour
        raw_time = self.file[self.dateCol]
        pte = [int((t.hour*60 + t.minute)/minutePerPTE + 1) for t in raw_time]
        self.file['PTE'] = pte
        self.pteCol = 'PTE'

    def get_avg_values(self, valCol):
        avg_val_dict = {}

        for i in range(1,self.periodByHour*24+1):
            filteredPTEval = self.file[self.file[self.pteCol] == i][valCol].values
            filteredPTEval = [ 0 if (math.isnan(x) | math.isinf(x)) else x for x in filteredPTEval]
            avg_val_dict[i] = sum(filteredPTEval)/len(filteredPTEval)
        return avg_val_dict

    def fill_nan_by_avg(self, imputingCols = None):
        if imputingCols == None:
            imputingCols = self.valCol

        for col in imputingCols:
            self.file['missing_' + col] = [1 if (math.isnan(x)|math.isinf(x)) else 0 for x in self.file[col]]
            imputed_dict = self.get_avg_values(col)

            for pte in imputed_dict:
                imputed_val = imputed_dict[pte]
                pteFile = self.file[self.file[self.pteCol] == pte]
                impute_index = self.file.index[ (self.file[self.pteCol] == pte) & ( np.isnan(self.file[col])) | (np.isinf(self.file[col]))]
                self.file.loc[impute_index, col] = imputed_val

    def impute(self, splitHour = 4):
        self.file[self.valCol] /= splitHour
        cur_index = 1
        isMissing = []

        while cur_index <= len(self.file):

            PTE = self.file.loc[cur_index]['PTE']
            cur_row = self.file.loc[cur_index][self.valCol]
            last_row = self.file.loc[cur_index - 1][self.valCol]

            # assume the power forecast data is available/unavailable at the same time.
            if PTE % 4 == 1 & cur_row.isnull().any():
                # impute missing day
                isMissing.append(1)
                if cur_index >= param.max_PTE:
                    last_day_data = self.file.loc[cur_index - param.max_PTE][self.valCol]
                    impute_data = last_day_data
                    self.file.loc[cur_index, self.valCol] = impute_data

                else:
                    cur_index += 4
                    continue

            elif PTE % 4 == 1:
                isMissing.append(0)
                impute_data = cur_row

            else:
                print(cur_index)
                print(cur_row)
                raise ValueError('Invalid missing values')

            isMissing.extend([0,0,0])
            self.file.loc[cur_index + 1, self.valCol] = impute_data
            self.file.loc[cur_index + 2, self.valCol] = impute_data
            self.file.loc[cur_index + 3, self.valCol] = impute_data
            cur_index += 4
            self.file['isMissing'] = isMissing

    def remove_duplicate_forecast(self):
        if self.fctTimeCol == None:
            pass

        # result = DataChecker( self.file ).check_duplicate_forecast( self.dateCol, self.fctTimeCol )
        result = DataChecker(self.file).check_duplicate_forecast(self.dateCol)

        if len(result>0):
            print('remove duplicates')
            self.file[self.fctTimeCol] = pd.to_datetime(self.file[self.fctTimeCol])
            self.file = self.file.loc[ self.file.groupby( self.dateCol)[self.fctTimeCol].idxmax().values ]
            self.file = self.file.sort_values(by=[self.dateCol], ascending=True)
            self.file = self.file.reset_index(drop=True)

    def convert_UTC_2_local(self, newColName, timezone):
        utc_time = self.file[self.dateCol]
        local_time = [pytz.utc.localize(t) for t in utc_time]
        am_dt = [t.astimezone(timezone) for t in local_time]
        discard_time_zone = [t._short_repr for t in am_dt]

        self.file[newColName] = discard_time_zone
        self.file[newColName] = pd.to_datetime( self.file[newColName] )

from utils.import_packages import *
from data_gathering.CleanData import convert_PTE_2_rt
import utils.hardcode_parameters as param

def round_time_2_nearest_PTE(t):
    return

if __name__ == '__main__':
    """Process plant availability data. This indicates when and how unavailability of plants. 
    Maybe extract forecastTime and compute how long been unavailable in the future
    Maybe take advantage of closed power type in the future"""

    configuration = Configuration()
    plant_df,plant_config = configuration.readFile('plant')

    # remove unavailableAmount = 0
    plant_df = plant_df[plant_df['UnavailableAmount'] != 0]
    plant_df['Date'] = [d._date_repr for d in plant_df[plant_config.date_col]]
    plant_df = plant_df.loc[ plant_df.groupby(['Date', 'Plant Name'])[plant_config.date_col].idxmax().values]

    # round time
    plant_df[plant_config.date_col] = plant_df[plant_config.date_col].dt.round('15min')

    # impute missing hours if unavilableAmount>0
    for index, (startTime, endTime, amount) in enumerate(zip(plant_df['Date'], plant_df[plant_config.date_col], plant_df['UnavailableAmount'])):
        expected_time = pd.DatetimeIndex(start = startTime,end = endTime, freq = '0.25H' )
        expected_time = pd.DataFrame({plant_config.date_col:expected_time})

        expected_time['UnavailableAmount'] = [amount]*len(expected_time)
        if index == 0:
            unavailable_df = expected_time
        else:
            unavailable_df = pd.concat([unavailable_df, expected_time], axis = 0)


    unavailable_df.to_excel(param.data_folder_path + '/plant/clean_plant.xlsx', index = False)

    print('all done')








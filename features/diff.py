from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration
from features.train_test_split import train_test_split
from features.features import LagDataGenerator

def feat_pipeline(df):
    # hot encoding categorical features

    #


    return x,y

if __name__ == '__main__':
    """This file is to merge (solar_generation_forecast, wind_generation_forecast,temperature,diff).
    The resulting file is the train data to predict forecast error: diff"""

    configuration = Configuration()
    df, df_config = configuration.readFile('final-diff')

    train,test = train_test_split(df, splitBySize=True, train_size=0.8)



    train.to_excel(param.data_folder_path +'/features/train_diff.xlsx', index = False)
    test.to_excel(param.data_folder_path + '/features/test_diff.xlsx', index = False)



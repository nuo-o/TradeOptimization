from utils.import_packages import *
import utils.hardcode_parameters as param
from data_gathering.CleanData import TimeSeriesData
from data_gathering.Configure import Configuration
from data_gathering.CleanData import convert2RealTime

# def save_result_to_file(file_path, result, feat_dict, notes = None):
#     with open(file_path + str(result) + str(notes)+'.txt', 'w') as f:
#         for lag_column, lag_value in feat_dict.items():
#             row = str(lag_column) + ':' + str(lag_value)
#             f.write('%s\n' % row)


def get_pos_class_ratio(df, pos_label, target):
    pos_ratio = df[df[target]==pos_label]
    return len(pos_ratio)/len(df)
from utils.import_packages import *
import utils.hardcode_parameters as param


if __name__ == '__main__':
    strategy = 3
    df = pd.read_excel( param.data_folder_path + '/results/hold-out-prediction/strategy_'+str(strategy)+'_exp1.xlsx')

    df['Year-month'] = [ datetime(d.year, d.month, 1) for d in df['Date']]
    our_monthly = df.groupby('Year-month')['our_pnl'].sum().reset_index()
    base_monthly = df.groupby('Year-month')['base_pnl'].sum().reset_index()

    monthly = base_monthly.merge(our_monthly, on='Year-month',how='inner')
    monthly.to_excel(param.data_folder_path + '/results/evaluation/strategy_'+str(strategy)+'_monthly.xlsx', index = False)


import pandas as pd
import numpy as np



df['lag_prc'] = df.groupby('permno')['prc'].shift(1)  
df['lag_shrout'] = df.groupby('permno')['shrout'].shift(1)
df['lagME'] = df['lag_prc'] * df['lag_shrout']

# Keep data from 1977 onwards to allow at least 11 years data for logit predictions 
df = df[df['Date_monthly'].dt.year >= 1977]

# Drop rows where lagged price is less than 1
df = df[df['lag_prc'] >= 1]


# Define function to save summary statistics to CSV and LaTeX
def save_summary_stats(group_val, lag):
    group_data = df[df['group'] == group_val]
    summary = group_data[['ret', 'NI_MTA_qavg', 'Cash_MTA_q', 'TL_MTA_q', 'ret_avg', 
                          'RSIZ', 'SIGMA', 'MB', 'PriceFitLog', 'age']].describe().T
    summary.to_csv(f'SumStatsLag{lag}Group{group_val}.csv', header=True)
    summary.to_latex(f'SumStatsLag{lag}Group{group_val}.tex', header=True)

# Save summary for Group 1 and Group 2
save_summary_stats(1, model_lag)
save_summary_stats(2, model_lag)

# Calculate equal-weighted returns (mean)
df['ret_EW'] = df.groupby(['Date_monthly', 'group'])['ret'].transform('mean')
# Calculate value-weighted returns 
df['lagME_sum'] = df.groupby(['Date_monthly', 'group'])['lagME'].transform(lambda x: np.sum(x * (df.loc[x.index, 'ret'].notna())))
df['ret_VW'] = df.groupby(['Date_monthly', 'group']).apply(
    lambda group: np.sum(group['ret'] * group['lagME']) / group['lagME_sum']).reset_index(level=[0, 1], drop=True)

# excess returns by subtracting 'rf'
df['ex_ret_EW'] = df['ret_EW'] - df['rf']
df['ex_ret_VW'] = df['ret_VW'] - df['rf']

# Collapse data by Date_monthly and group, calculating various aggregates
df_collapsed = df.groupby(['Date_monthly', 'group']).agg(
    ret_EW=('ret_EW', 'mean'),
    ret_VW=('ret_VW', 'mean'),
    mktrf=('mktrf', 'mean'),
    smb=('smb', 'mean'),
    hml=('hml', 'mean'),
    rf=('rf', 'mean'),
    umd=('umd', 'mean'),
    rmw=('rmw', 'mean'),
    cma=('cma', 'mean'),
    count_ret_EW=('ret_EW', 'count'),
    count_ret_VW=('ret_VW', 'count'),
    year=('year', 'mean')  
).reset_index()

# Calculate excess returns for each group
df_collapsed['ex_ret_EW'] = df_collapsed['ret_EW'] - df_collapsed['rf']
df_collapsed['ex_ret_VW'] = df_collapsed['ret_VW'] - df_collapsed['rf']

# Summary statistics for excess returns (ex_ret_EW and ex_ret_VW) by group
print(df_collapsed[df_collapsed['group'] == 1]['ex_ret_EW'].describe())
print(df_collapsed[df_collapsed['group'] == 2]['ex_ret_EW'].describe())
print(df_collapsed[df_collapsed['group'] == 1]['ex_ret_VW'].describe())
print(df_collapsed[df_collapsed['group'] == 2]['ex_ret_VW'].describe())



import pandas as pd
import numpy as np
import statsmodels.api as sm


df['Date_monthly'] = pd.to_datetime(df['Date_monthly'])
df = df.sort_values(['permno', 'Date_monthly'])
df.set_index(['permno', 'Date_monthly'], inplace=True)


df['BM'] = df['BE'] / df['ME']
df['BM'] = np.log(df['BM'])

# Calculate momret (momentum return)
df['momret'] = (1 + df.groupby('permno')['ret'].shift(2)) * (1 + df.groupby('permno')['ret'].shift(3)) * \
               (1 + df.groupby('permno')['ret'].shift(4)) * (1 + df.groupby('permno')['ret'].shift(5)) * \
               (1 + df.groupby('permno')['ret'].shift(6)) * (1 + df.groupby('permno')['ret'].shift(7)) * \
               (1 + df.groupby('permno')['ret'].shift(8)) * (1 + df.groupby('permno')['ret'].shift(9)) * \
               (1 + df.groupby('permno')['ret'].shift(11)) * (1 + df.groupby('permno')['ret'].shift(12)) - 1

# Drop small stocks (assuming prc < 1 are small stocks)
df = df[df['prc'] >= 1]

# Calculate lagged return (lret)
df['lret'] = df.groupby('Date_monthly')['ret'].shift(1)

# Define the list of independent variables for Fama-Macbeth regression
independent_vars = ['momret', 'ME', 'BM', 'lret']

# Fama-MacBeth procedure: run cross-sectional regressions for each period
time_periods = df.index.get_level_values('Date_monthly').drop_duplicates()

coefficients = []
standard_errors = []

for period in time_periods:

    df_period = df.loc[df.index.get_level_values('Date_monthly') == period]

    X = df_period[independent_vars]
    X = sm.add_constant(X)  
    y = df_period['ret']
    
    model = sm.OLS(y, X).fit()
    
    coefficients.append(model.params)
    standard_errors.append(model.bse)

coefficients_df = pd.DataFrame(coefficients, index=time_periods)
standard_errors_df = pd.DataFrame(standard_errors, index=time_periods)

# Compute the Fama-MacBeth average coefficients and standard errors
avg_coefficients = coefficients_df.mean()
avg_standard_errors = standard_errors_df.mean()
t_stats = avg_coefficients / avg_standard_errors


fmb_results = pd.DataFrame({
    'Coefficient': avg_coefficients,
    'Standard Error': avg_standard_errors,
    't-stat': t_stats
})

print(fmb_results)

fmb_results.to_csv('FMRegResultsLag1.csv', float_format='%9.3f')
fmb_results.to_latex('FMRegResultsLag1.tex', float_format='%9.3f')

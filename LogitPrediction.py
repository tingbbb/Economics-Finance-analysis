

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime


df['Date_monthly'] = pd.to_datetime(df['Date_monthly'])
df.set_index(['permno', 'Date_monthly'], inplace=True)

# Initialize columns for predicted probabilities and groups with predicted probabilities larger than threshold
df['PHAT'] = np.nan
df['group'] = np.nan

# Define the percentile for cutoff
percentile = 95  
# Define prediction in one month or three months, model_lag = 1 or 3
model_lag = 1
# Loop through years (1977 to 2013)
for year in range(1977, 2014):
    print(f"Year: {year}")
    
    # Step 1: Fit the logit model before the current year
    train_data = df[df.index.get_level_values('Date_monthly').dt.year < year]
    X = train_data[['NI_MTA_qavg', 'TL_MTA_q', 'ret_avg', 'RSIZ', 'MB', 'SIGMA', 'age']]
    X['age_TL_MTA_q'] = X['TL_MTA_q'] * X['age']
    X = sm.add_constant(X)
    y = train_data[f'exitin{model_lag}'] 

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit()
    
    # Step 2: Predict the exit probability for the current year (i)
    test_data = df[df.index.get_level_values('Date_monthly').dt.year == year]
    X_test = test_data[['NI_MTA_qavg', 'TL_MTA_q', 'ret_avg', 'RSIZ', 'MB', 'SIGMA', 'age']]
    X_test['age_TL_MTA_q'] = X_test['TL_MTA_q'] * X_test['age']
    X_test = sm.add_constant(X_test)
    
    y_pred_prob = logit_result.predict(X_test)
    
    predicted_var = f'exitin{model_lag}_prob'
    df.loc[df.index.get_level_values('Date_monthly').dt.year == year, 'PHAT'] = y_pred_prob
    df.loc[df.index.get_level_values('Date_monthly').dt.year == year, predicted_var] = y_pred_prob
    
    # Step 3: Sort using predicted probabilies and select stock group with predicted probabilities above a cutoff
    df['cutoff'] = df.groupby(df.index.get_level_values('Date_monthly'))['PHAT'].transform(
        lambda x: x.quantile(percentile / 100))
    # Assign groups based on predicted probabilities
    df.loc[(df.index.get_level_values('Date_monthly').dt.year == year) & 
           (df['PHAT'] >= cutoff_value), 'group'] = 1

    # Clean up temporary variables
    df.drop(columns=[predicted_var], inplace=True)

# Final cleaning, in case there are NaN values in 'group'
df.loc[(df['group'].isnull())&(df['PHAT'].notnull()),'group'] = 2
df['group'].fillna(3, inplace=True)

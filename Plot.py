import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Function to fit models, compute alphas, and handle residuals
def calculate_alpha(group_value, factors, model_num, df):

    group_data = df[df['group'] == group_value]

    X = group_data[factors]
    X = sm.add_constant(X)  
    y = group_data['ex_ret_VW']
    

    model = sm.OLS(y, X, missing='drop')  
    results = model.fit()

    group_data['error'] = results.resid

    alpha = results.params['const'] + group_data['error']

    alpha = np.log(1 + alpha)

    group_data['alpha'] = alpha
    group_data['alpha_cumulative'] = group_data['alpha'].cumsum()

    return group_data[['Date_monthly', 'group', 'alpha', 'alpha_cumulative']]

# Define the factor sets for the 3-factor, 4-factor, and 5-factor models
factors_3f = ['mktrf', 'smb', 'hml']
factors_4f = ['mktrf', 'smb', 'hml', 'umd']
factors_5f = ['mktrf', 'smb', 'hml', 'umd', 'rmw', 'cma']

all_alpha_results = pd.DataFrame()
# Loop over the groups and models
for group_value in [1, 2]:
    alpha_3f = calculate_alpha(group_value, factors_3f, 3, df_collapsed)
    alpha_4f = calculate_alpha(group_value, factors_4f, 4, df_collapsed)
    alpha_5f = calculate_alpha(group_value, factors_5f, 5, df_collapsed)
    
    all_alpha_results = pd.concat([all_alpha_results, alpha_3f, alpha_4f, alpha_5f])

# short the stocks with large predicted probabilities
all_alpha_results.loc[all_alpha_results['group'] == 1, 'alpha'] = -all_alpha_results.loc[all_alpha_results['group'] == 1, 'alpha']
all_alpha_results.loc[all_alpha_results['group'] == 1, 'alpha_cumulative'] = -all_alpha_results.loc[all_alpha_results['group'] == 1, 'alpha_cumulative']

# Plot the alpha values for group 1 and group 2
def plot_alpha_comparison(df, model_num, title):
    plt.figure(figsize=(10, 6))
    
    for group_value in [1, 2]:
        group_data = df[df['group'] == group_value]
        plt.plot(group_data['Date_monthly'], group_data['alpha_cumulative'], label=f'Group {group_value}', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(f'Alpha (Group {model_num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'alpha{model_num}FVW.png', dpi=300)
    plt.close()

# Plot the results for 3-factor, 4-factor, and 5-factor models
plot_alpha_comparison(all_alpha_results, 3, '1-month lagged, VW, 3-factor model')
plot_alpha_comparison(all_alpha_results, 4, '1-month lagged, VW, 4-factor model')
plot_alpha_comparison(all_alpha_results, 5, '1-month lagged, VW, 5-factor model')






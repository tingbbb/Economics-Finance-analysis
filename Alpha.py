import pandas as pd
import statsmodels.api as sm
import numpy as np


def run_regression(dependent, independent, group_value, model_name):
    data = df[df['group'] == group_value]
    
    X = data[independent]
    X = sm.add_constant(X)  
    y = data[dependent]
    
    model = sm.OLS(y, X, missing='drop')  
    results = model.fit()
    
    result_summary = results.summary2().tables[1]
    
    result_summary.to_csv(f"{model_name}.csv")
    result_summary.to_latex(f"{model_name}.tex")
    
    return results

# Define the independent variables for each model
factors_3f = ['mktrf', 'smb', 'hml']
factors_4f = ['mktrf', 'smb', 'hml', 'umd']
factors_5f = ['mktrf', 'smb', 'hml', 'umd', 'rmw', 'cma']

# Run regressions for Value-Weighted (VW) Returns
# VW 3-Factor Model for Group 1
run_regression('ex_ret_VW', factors_3f, 1, 'VW_FF3F_G1')
# VW 3-Factor Model for Group 2
run_regression('ex_ret_VW', factors_3f, 2, 'VW_FF3F_G2')

# VW 4-Factor Model for Group 1
run_regression('ex_ret_VW', factors_4f, 1, 'VW_FF4F_G1')
# VW 4-Factor Model for Group 2
run_regression('ex_ret_VW', factors_4f, 2, 'VW_FF4F_G2')

# VW 5-Factor Model for Group 1
run_regression('ex_ret_VW', factors_5f, 1, 'VW_FF5F_G1')
# VW 5-Factor Model for Group 2
run_regression('ex_ret_VW', factors_5f, 2, 'VW_FF5F_G2')

# Run regressions for Equal-Weighted (EW) Returns
# EW 3-Factor Model for Group 1
run_regression('ex_ret_EW', factors_3f, 1, 'EW_FF3F_G1')
# EW 3-Factor Model for Group 2
run_regression('ex_ret_EW', factors_3f, 2, 'EW_FF3F_G2')

# EW 4-Factor Model for Group 1
run_regression('ex_ret_EW', factors_4f, 1, 'EW_FF4F_G1')
# EW 4-Factor Model for Group 2
run_regression('ex_ret_EW', factors_4f, 2, 'EW_FF4F_G2')

# EW 5-Factor Model for Group 1
run_regression('ex_ret_EW', factors_5f, 1, 'EW_FF5F_G1')
# EW 5-Factor Model for Group 2
run_regression('ex_ret_EW', factors_5f, 2, 'EW_FF5F_G2')



